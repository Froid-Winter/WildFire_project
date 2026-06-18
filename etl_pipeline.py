"""
Wildfire Weather Data — PySpark ETL Pipeline
==============================================
1. Load 1.83M weather records from CSV
2. Clean missing values (median imputation)
3. Engineer synthetic fire-severity target (7 classes A–G)
4. Train/Val/Test split
5. Write processed CSV splits

Usage:
    python etl_pipeline.py
"""

import sys
import os
import math
import warnings

import numpy as np
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from config import *

warnings.filterwarnings("ignore")

# ── Environment setup ────────────────────────────────────────────────────────
import pyspark

os.environ["SPARK_HOME"] = os.path.dirname(pyspark.__file__)
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

try:
    from pyspark import SparkContext

    if SparkContext._active_spark_context is not None:
        SparkContext._active_spark_context.stop()
        SparkContext._active_spark_context = None
except Exception:
    pass


# ── 1. Start Spark ───────────────────────────────────────────────────────────
def start_spark():
    spark = (
        SparkSession.builder.appName(f"{SPARK_APP_NAME}_ETL")
        .config("spark.sql.shuffle.partitions", SPARK_SHUFFLE_PARTITIONS)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY)
        .config("spark.executor.memory", SPARK_EXECUTOR_MEMORY)
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "2g")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    print(f"Spark version: {spark.version}")
    return spark


# ── 2. Load CSV ──────────────────────────────────────────────────────────────
def load_data(spark):
    df = (
        spark.read.option("header", "true")
        .option("delimiter", ",")
        .option("mode", "PERMISSIVE")
        .schema(SCHEMA)
        .csv(DATA_PATH)
    )
    count = df.count()
    print(f"Rows loaded: {count:,}")
    return df


# ── 3. Clean missing values ──────────────────────────────────────────────────
def clean_data(df):
    medians = {}
    for col in FEATURE_COLS:
        med = df.select(F.percentile_approx(col, 0.5, 10000)).collect()[0][0]
        medians[col] = med if med is not None else 0.0

    df_clean = df.select(
        *[F.coalesce(F.col(c), F.lit(medians[c])).alias(c) for c in FEATURE_COLS],
        F.col("OBJECTID"),
    )

    remaining = (
        df_clean.select(
            [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in FEATURE_COLS]
        )
        .toPandas()
        .T[0]
        .sum()
    )
    print(f"Nulls remaining: {remaining}")
    return df_clean


# ── 4. Synthetic target ──────────────────────────────────────────────────────
def engineer_target(df):
    df_labeled = (
        df.withColumn(
            "temp_anomaly", F.col("temp_mean_0") - F.col("temp_mean_180")
        )
        .withColumn(
            "drought_ratio",
            F.col("prcp_sum_0") / (F.col("prcp_sum_180") + 1e-6),
        )
        .withColumn(
            "wind_hazard",
            F.col("wspd_mean_0") / (F.col("wspd_mean_180") + 1e-6),
        )
        .withColumn(
            "severity_score",
            F.col("temp_anomaly") * 0.40
            + (1 - F.col("drought_ratio")) * 0.35
            + F.col("wind_hazard") * 0.25,
        )
    )

    quantiles = df_labeled.approxQuantile(
        "severity_score", [0.0, 1 / 7, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 6 / 7, 1.0], 0.01
    )
    print("Quantile boundaries:", [f"{q:.4f}" for q in quantiles])

    bins = quantiles[1:-1]
    bucket = F.when(F.col("severity_score") <= bins[0], 0)
    for i in range(1, len(bins)):
        bucket = bucket.when(F.col("severity_score") <= bins[i], i)
    bucket = bucket.otherwise(NUM_CLASSES - 1)

    df_labeled = df_labeled.withColumn(TARGET_COL, bucket.cast("int"))

    class_dist = df_labeled.groupBy(TARGET_COL).count().orderBy(TARGET_COL)
    class_dist.show()

    return df_labeled


# ── 5. Train / Val / Test split ──────────────────────────────────────────────
def split_data(df):
    train_frac = 1 - TEST_FRAC - VAL_FRAC
    val_frac = VAL_FRAC / (1 - train_frac)

    df_train, df_temp = df.randomSplit([train_frac, 1 - train_frac], seed=SEED)
    df_val, df_test = df_temp.randomSplit([val_frac, 1 - val_frac], seed=SEED)

    for name, d in [
        ("Train", df_train),
        ("Val", df_val),
        ("Test", df_test),
    ]:
        print(f"{name:6s}: {d.count():>8,} rows")

    return df_train, df_val, df_test


# ── 6. Save to CSV ───────────────────────────────────────────────────────────
def save_splits(df_train, df_val, df_test):
    output_dir = "processed_data"
    os.makedirs(output_dir, exist_ok=True)
    cols_to_save = FEATURE_COLS + [TARGET_COL, "OBJECTID"]

    for name, d in [("train", df_train), ("val", df_val), ("test", df_test)]:
        path = os.path.join(output_dir, name)
        (
            d.select(*cols_to_save)
            .coalesce(4)
            .write.mode("overwrite")
            .csv(path, header=True)
        )
        print(f"{name} -> {path} [CSV]")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    spark = start_spark()
    df = load_data(spark)
    df_clean = clean_data(df)
    df_labeled = engineer_target(df_clean)
    df_train, df_val, df_test = split_data(df_labeled)
    save_splits(df_train, df_val, df_test)
    spark.stop()
    print("Done.")
