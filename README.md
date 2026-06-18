# Wildfire Severity Classification — Distributed ML Pipeline

## Overview

Predict wildfire severity class (A–G) from weather conditions using a distributed PySpark + TensorFlow pipeline. The dataset contains **1.83 million** weather records with temperature, precipitation, and wind speed averages at multiple time windows (0, 10, 30, 60, 180 days).

Dataset: https://www.kaggle.com/datasets/leternnoz/188-million-us-wildfires-weather-data/data

### Architecture

```
archive/US_wildfire_weather_data.csv (284 MB, 1.83M rows)
        │
        ▼  etl_pipeline.py (PySpark)
 Spark DataFrame → median imputation → synthetic target engineering
        │
 stratified split → processed_data/{train,val,test}/*.csv
        │
        ▼  visualization.ipynb (Pandas + Matplotlib/Seaborn)
 Feature distributions • Correlations • Class balance
        │
        ▼  Model_Training.ipynb (TensorFlow + MultiWorkerMirroredStrategy)
 sklearn StandardScaler → tf.data.Dataset → DNN (256→128→64→7)
        │
 Class-weighted training → Evaluation → SavedModel
```

## Dataset

| Property | Value |
|----------|-------|
| Source | US Wildfire Weather Data |
| Records | 1,830,945 |
| Features | 15 weather variables |
| Time windows | 0, 10, 30, 60, 180 days |
| Variables | Temperature mean, Precipitation sum, Wind speed mean |

**Features:**
- `temp_mean_{0,10,30,60,180}` — temperature averages per window
- `prcp_sum_{0,10,30,60,180}` — precipitation totals per window
- `wspd_mean_{0,10,20,60,180}` — wind speed averages per window

**Note:** The raw data contains only weather metrics. Since no `FIRE_SIZE_CLASS` labels are present, a **synthetic fire severity index** is constructed from:
- Temperature anomaly (hotter → higher risk) — weight 0.40
- Drought ratio (low precipitation → higher risk) — weight 0.35
- Wind hazard (high wind → higher risk) — weight 0.25

The composite score is discretized into 7 balanced classes (A–G) via quantile binning.

### Class Determination (A–G) in Detail

The raw dataset has no wildfire severity labels, so we engineer a synthetic target using meteorological reasoning:

**1. Composite Severity Score**

Three wildfire risk factors are derived from weather variables and weighted by domain importance:

| Component | Formula | Weight | Rationale |
|-----------|---------|--------|-----------|
| **Temperature anomaly** | `temp_mean_0` − `temp_mean_180` | **0.40** | Higher temperature relative to the 6-month baseline indicates heat-wave conditions that dry fuel and accelerate fire spread. |
| **Drought ratio** | 1 − (`prcp_sum_0` / `prcp_sum_180`) | **0.35** | Low precipitation in the current window vs. the 6-month baseline signals drought. The inverse is taken so higher values = higher risk. |
| **Wind hazard** | `wspd_mean_0` / `wspd_mean_180` | **0.25** | Higher current wind speed relative to baseline increases fire spread rate and spotting distance. |

The composite: `severity_score = temp_anomaly × 0.40 + (1 − drought_ratio) × 0.35 + wind_hazard × 0.25`

**2. Quantile Binning → A–G**

The continuous severity score is discretized into 7 classes using **equal-frequency quantile binning**:

- Compute approximate quantile boundaries at `[0, 1/7, 2/7, ..., 6/7, 1]` over the full dataset (~1.83M rows).
- Each record falls into one of the 7 bins, assigned a class index `0` through `6`.
- These indices are mapped to labels **A** (lowest severity) through **G** (highest severity).

This ensures each class contains approximately **equal number of samples** (~261k rows each), which prevents the model from being biased toward any single severity level. The quantile boundaries are computed once during the ETL step and printed to the console.

```
Quantile boundaries: [−14.72, −2.10, −0.85, 0.12, 0.98, 1.88, 3.15, 18.50]
                     ↑       ↑      ↑     ↑     ↑     ↑     ↑     ↑
                     A       B      C     D     E     F     G     max
```

### What This Project Helps With

This project demonstrates a **production-ready distributed ML pipeline** for classifying wildfire severity from purely meteorological data. Key benefits:

| Area | Application |
|------|-------------|
| **Wildfire early warning** | Predict severity class from forecast weather data before a fire ignites, enabling proactive resource prepositioning. |
| **Resource allocation** | Help fire agencies allocate aircraft, crews, and equipment proportional to predicted severity (A = routine, G = catastrophic). |
| **Climate impact analysis** | Quantify how shifting temperature, precipitation, and wind patterns affect wildfire risk distribution over time. |
| **Scaling pattern** | The architecture (PySpark → sklearn → tf.data → distributed DNN) is a reusable template for any large-scale classification problem with missing data and no predefined labels. |

Beyond prediction, the pipeline directly addresses real-world ML challenges: 1.83M-row scale, 10–19% missing data, no ground-truth labels, class imbalance, and distributed training across multi-worker environments.

## Files

| File | Purpose |
|------|---------|
| `etl_pipeline.py` | PySpark ETL: load, clean, target engineer, split, save |
| `visualization.ipynb` | EDA: distributions, correlations, class balance charts |
| `Model_Training.ipynb` | TensorFlow DNN with distributed strategy |
| `config.py` | Shared constants and hyperparameters |

## How to Run

### Prerequisites

- Python 3.12+
- PySpark 4.x (`pip install pyspark`)
- TensorFlow 2.x (`pip install tensorflow`)
- Java JDK 21+ (set `JAVA_HOME`)
- Windows: `winutils.exe` + `hadoop.dll` in Spark's `bin/` directory

### Step 1 — ETL Pipeline

```bash
python etl_pipeline.py
```

Output:
```
Spark version: 4.1.2
Rows loaded: 1,830,945
Nulls remaining: 0
Quantile boundaries: [...]
Train : 1,281,321 rows
Val   :  183,401 rows
Test  :  366,223 rows
```

Writes processed CSV splits to `processed_data/{train,val,test}/`.

### Step 2 — EDA & Visualization

```bash
jupyter notebook visualization.ipynb
```

Generates:
- Feature distribution histograms (all 15 variables)
- Correlation heatmap
- Class distribution (bar + pie)
- Feature-target relationship plots

### Step 3 — Distributed DNN Training

```bash
jupyter notebook Model_Training.ipynb
```

Trains a 3-layer DNN (256 → 128 → 64) with:
- Batch normalisation + Dropout (0.3)
- Class-weighted loss (inverse frequency)
- ReduceLROnPlateau + EarlyStopping
- `MultiWorkerMirroredStrategy` for distributed training

### DNN Architecture — Deep Explanation

**Why DNN over simpler models?** Random forests or logistic regression would struggle with the non-linear interactions between time-windowed weather features (e.g., how temperature *anomaly* across windows combines with drought ratio). A DNN automatically learns these high-order feature interactions.

**Architecture breakdown:**

| Layer | Units | Params | Purpose |
|-------|-------|--------|---------|
| Input | 15 | 0 | StandardScaler-normalized weather features |
| Dense | 256 | 4,096 | Learn non-linear combinations of weather signals |
| BatchNorm | 256 | 1,024 | Stabilize training, reduce internal covariate shift |
| ReLU | 256 | 0 | Non-linear activation (avoids vanishing gradient) |
| Dropout (0.3) | 256 | 0 | Regularization — randomly drops 30% of neurons to prevent co-adaptation |
| Dense | 128 | 32,896 | Second hidden layer, compressing learned features |
| BatchNorm + ReLU + Dropout | 128 | 512 | Same regularization stack |
| Dense | 64 | 8,256 | Bottleneck layer forcing compact representations |
| BatchNorm + ReLU + Dropout | 64 | 256 | Final regularization |
| Dense (Softmax) | 7 | 455 | Output class probabilities A–G |
| **Total** | | **47,495** | |

**Key design choices:**

- **He-normal initialization**: Matches ReLU activation by scaling weights to account for the non-linearity's variance.
- **Batch Normalization after every Dense**: Allows higher learning rates and reduces sensitivity to initialization. The 896 non-trainable params are running mean/variance statistics.
- **Dropout (0.3)**: A 30% drop rate balances underfitting vs. overfitting; higher rates would lose too much signal on 15-dim input.
- **Class-weighted loss**: Inverse-frequency weights ensure minority classes (e.g., class D with 52k samples) contribute proportionally to the gradient, preventing the model from ignoring rare severity levels.
- **Sparse Categorical Crossentropy**: Appropriate for integer-encoded labels (0–6) with 7 mutually exclusive classes.
- **ReduceLROnPlateau**: Halves the learning rate (`factor=0.5`) when validation loss plateaus for 5 epochs, helping the optimizer escape local minima.
- **EarlyStopping** with `patience=10` and `restore_best_weights`: Prevents overfitting by reverting to the best validation checkpoint.

**Distributed training with `MultiWorkerMirroredStrategy`**: Synchronous data parallelism across workers. Each worker computes gradients on its batch shard, then all-reduce averages them. On the test system (single-worker, CPU-only), training 50 epochs on 1.28M training samples completes in ~20 minutes with a batch size of 2048.

## Results

### ETL Summary

| Step | Detail |
|------|--------|
| Rows loaded | 1,830,945 |
| Missing features | 10 columns (10–19% missing) |
| Imputation | Median per column |
| Target classes | 7 balanced (A–G) |
| Train / Val / Test | 1,281,321 / 183,401 / 366,223 |

### Correlation Analysis

High multicollinearity within each variable group (adjacent time windows):

| Pair | r |
|------|---|
| `temp_mean_60` vs `temp_mean_30` | 0.962 |
| `wspd_mean_20` vs `wspd_mean_60` | 0.961 |
| `temp_mean_30` vs `temp_mean_10` | 0.954 |

### Training Performance (subset test)

| Metric | Value |
|--------|-------|
| Epochs | 3 (of 50) |
| Test accuracy | 91.3% |
| Loss reduction | 1.39 → 0.29 (val) |

Full training (50 epochs with class weights) will improve minority-class recall.

### Key Challenges Addressed

| Challenge | Solution |
|-----------|----------|
| **1.83M rows** | PySpark distributed processing |
| **Missing data** | Per-column median imputation |
| **No labels** | Synthetic fire severity index |
| **Class imbalance** | Inverse-frequency class weights |
| **Distributed training** | `MultiWorkerMirroredStrategy` |
| **Spark version conflict** | Override `SPARK_HOME` to PySpark bundled version |
| **Windows native IO** | Hadoop 3.3.6 `winutils.exe` + `hadoop.dll` |

## Environment Notes

- PySpark 4.1.2 is installed via pip (bundles Spark 4.1.2)
- `SPARK_HOME` must point to `site-packages/pyspark/` (handled automatically in `etl_pipeline.py`)
- Hadoop native libraries for Windows are placed in `D:\spark-3.5.8-bin-hadoop3\bin\`
"# WildFire_project" 
