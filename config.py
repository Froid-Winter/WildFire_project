from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType

DATA_PATH = "archive/US_wildfire_weather_data.csv"
FEATURE_COLS = [
    "temp_mean_0",  "prcp_sum_0",  "wspd_mean_0",
    "temp_mean_10", "prcp_sum_10", "wspd_mean_10",
    "temp_mean_30", "prcp_sum_30", "wspd_mean_20",
    "temp_mean_60", "prcp_sum_60", "wspd_mean_60",
    "temp_mean_180","prcp_sum_180","wspd_mean_180",
]

TEMP_COLS   = ["temp_mean_0", "temp_mean_10", "temp_mean_30", "temp_mean_60", "temp_mean_180"]
PRCP_COLS   = ["prcp_sum_0",  "prcp_sum_10",  "prcp_sum_30",  "prcp_sum_60",  "prcp_sum_180"]
WSPD_COLS   = ["wspd_mean_0", "wspd_mean_10", "wspd_mean_20", "wspd_mean_60", "wspd_mean_180"]

NUM_CLASSES = 7
CLASS_NAMES = ["A", "B", "C", "D", "E", "F", "G"]

SEED = 42
TEST_FRAC = 0.2
VAL_FRAC  = 0.1

SPARK_APP_NAME = "WildfireSeverityClassifier"
SPARK_SHUFFLE_PARTITIONS = 200
SPARK_DRIVER_MEMORY = "4g"
SPARK_EXECUTOR_MEMORY = "4g"

SCHEMA = StructType([
    StructField("temp_mean_0",   DoubleType(), True),
    StructField("prcp_sum_0",    DoubleType(), True),
    StructField("wspd_mean_0",   DoubleType(), True),
    StructField("temp_mean_10",  DoubleType(), True),
    StructField("prcp_sum_10",   DoubleType(), True),
    StructField("wspd_mean_10",  DoubleType(), True),
    StructField("temp_mean_30",  DoubleType(), True),
    StructField("prcp_sum_30",   DoubleType(), True),
    StructField("wspd_mean_20",  DoubleType(), True),
    StructField("temp_mean_60",  DoubleType(), True),
    StructField("prcp_sum_60",   DoubleType(), True),
    StructField("wspd_mean_60",  DoubleType(), True),
    StructField("temp_mean_180", DoubleType(), True),
    StructField("prcp_sum_180",  DoubleType(), True),
    StructField("wspd_mean_180", DoubleType(), True),
    StructField("OBJECTID",      IntegerType(), False),
])

TARGET_COL = "fire_severity_index"

DNN_LAYERS   = [256, 128, 64]
DROPOUT_RATE = 0.3
BATCH_SIZE   = 2048
EPOCHS       = 50
LEARNING_RATE = 1e-3
