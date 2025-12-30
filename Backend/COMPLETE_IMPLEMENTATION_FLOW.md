# FactoryGuard AI - Complete Implementation Flow

## Overview

This document provides a comprehensive overview of the complete FactoryGuard AI implementation, covering all weeks from data ingestion to model explainability. It explains the end-to-end flow, showing how data transforms through each stage and how outputs from one week become inputs for the next.

---

## Table of Contents

1. [Week 1: Data Pipeline](#week-1-data-pipeline)
2. [Week 2: Model Training](#week-2-model-training)
3. [Week 3: Model Explainability](#week-3-model-explainability)
4. [Complete Flow Diagram](#complete-flow-diagram)
5. [Input-Output Mapping](#input-output-mapping)
6. [Integration Points](#integration-points)

---

## Week 1: Data Pipeline

### Objective
Transform raw sensor data into clean, feature-engineered datasets ready for machine learning.

### Input → Output Flow

```
Raw CSV File (sensor_data.csv)
    ↓
[Data Ingestion]
    ↓
Time-ordered DataFrame with parsed timestamps
    ↓
[Data Cleaning]
    ↓
Clean DataFrame (outliers removed, missing values interpolated)
    ↓
[Feature Engineering]
    ↓
Feature-engineered DataFrame with:
  - Lag features (t-1, t-2)
  - Rolling statistics (1h, 4h, 8h windows)
  - Exponential Moving Averages (EMA)
  - Target: failure_within_24h
    ↓
[Train/Test Split]
    ↓
Train DataFrame (80%) + Test DataFrame (20%)
    ↓
[Validation]
    ↓
Validated datasets saved as Parquet files
```

### Key Processes

#### 1. Data Ingestion (`src/data/ingest.py`)
- **Input**: Raw CSV file
- **Process**: 
  - Load CSV into pandas DataFrame
  - Parse timestamp column to datetime
  - Sort by timestamp per machine (prevents cross-machine leakage)
- **Output**: Time-ordered DataFrame

#### 2. Data Cleaning (`src/data/clean.py`)
- **Input**: Time-ordered DataFrame
- **Process**:
  - Remove outliers using IQR method (per machine)
  - Interpolate missing values (time-aware forward/backward fill)
- **Output**: Clean DataFrame with no missing values

#### 3. Feature Engineering (`src/data/features.py`)
- **Input**: Clean DataFrame
- **Process**:
  - **Lag Features**: t-1, t-2 for each sensor
  - **Rolling Features**: Mean and std for 1h, 4h, 8h windows
  - **EMA Features**: Exponential moving averages (alphas: 0.3, 0.5, 0.7)
  - **Target Creation**: Binary target (failure within 24 hours)
- **Output**: DataFrame with ~100+ engineered features + target

#### 4. Train/Test Split (`src/data/split.py`)
- **Input**: Feature-engineered DataFrame
- **Process**:
  - Time-based split (80/20) to prevent temporal leakage
  - Ensures all training data comes before test data
- **Output**: Train DataFrame (80%) and Test DataFrame (20%)

#### 5. Validation (`src/data/validate.py`)
- **Input**: Train and Test DataFrames
- **Process**:
  - Check time ordering per machine
  - Verify train/test time separation
  - Detect feature leakage
  - Validate target shift
- **Output**: Validated datasets saved as versioned Parquet files

### Week 1 Outputs

**Location**: `data/artifacts/`

- `train_YYYYMMDD_HHMMSS.parquet` - Training dataset
- `test_YYYYMMDD_HHMMSS.parquet` - Test dataset

**Contents**:
- All engineered features (lag, rolling, EMA)
- Target column: `failure_within_24h`
- Metadata columns: `timestamp`, `machine_id`

---

## Week 2: Model Training

### Objective
Train multiple models, evaluate performance, and select the best model for production.

### Input → Output Flow

```
Train/Test Parquet Files (from Week 1)
    ↓
[Feature Preparation]
    ↓
X_train, y_train, X_test, y_test
    ↓
[Baseline Model Training]
    ↓
Logistic Regression with class weighting
    ↓
[Random Forest Training]
    ↓
Random Forest with hyperparameter optimization
    ↓
[XGBoost Training (Primary)]
    ↓
XGBoost with RandomizedSearchCV optimization
    ↓
[Model Evaluation]
    ↓
Metrics: Recall, F1, Precision, Confusion Matrix
    ↓
[Model Comparison]
    ↓
Best model selected (by Recall & F1)
    ↓
[Model Persistence]
    ↓
Saved models, metrics, configs in artifacts/
```

### Key Processes

#### 1. Feature Preparation (`src/training/train.py::prepare_features`)
- **Input**: Train and Test DataFrames from Week 1
- **Process**:
  - Extract feature columns (exclude metadata and target)
  - Separate features (X) from target (y)
  - Fill missing values with 0
  - Log class distribution
- **Output**: 
  - `X_train`, `y_train`, `X_test`, `y_test`

#### 2. Baseline Model (`src/models/baseline.py`)
- **Input**: X_train, y_train
- **Process**:
  - Pipeline: StandardScaler → LogisticRegression
  - Use `class_weight='balanced'` for class imbalance
  - Fit on training data
- **Output**: Trained baseline model
- **Evaluation**: Test on X_test, y_test

#### 3. Random Forest (`src/models/random_forest.py`)
- **Input**: X_train, y_train
- **Process**:
  - RandomForestClassifier with `class_weight='balanced'`
  - RandomizedSearchCV for hyperparameter optimization
  - Optimize for F1-score using 5-fold CV
- **Output**: Trained Random Forest model

#### 4. XGBoost (Primary Model) (`src/models/xgboost_model.py`)
- **Input**: X_train, y_train
- **Process**:
  - Calculate `scale_pos_weight` from class distribution
  - RandomizedSearchCV with 50 iterations
  - Parameter grid: max_depth, learning_rate, n_estimators, etc.
  - Optimize for F1-score using 5-fold CV
- **Output**: Trained XGBoost model with best parameters

#### 5. Model Evaluation (`src/models/evaluate.py`)
- **Input**: Trained model, X_test, y_test
- **Process**:
  - Generate predictions and probabilities
  - Calculate metrics: Recall, Precision, F1, ROC-AUC
  - Generate confusion matrix and classification report
- **Output**: Dictionary with all metrics

#### 6. Model Comparison (`src/models/evaluate.py::compare_models`)
- **Input**: Results from all models
- **Process**:
  - Create comparison DataFrame
  - Sort by Recall (primary metric for rare events)
- **Output**: Comparison DataFrame

### Week 2 Outputs

**Location**: `data/artifacts/models_YYYYMMDD_HHMMSS/`

- Model files (`.joblib`):
  - `baseline_model.joblib`
  - `random_forest_model.joblib`
  - `xgboost_model.joblib`
- Evaluation results (`.json`):
  - `baseline_results.json`
  - `random_forest_results.json`
  - `xgboost_results.json`
- Model comparison: `model_comparison.csv`
- Training configuration: `training_config.json`
- Feature importance: `xgboost_feature_importance.csv`

---

## Week 3: Model Explainability

### Objective
Explain WHY the model predicts failures using SHAP (SHapley Additive exPlanations), providing both global and local interpretability.

### Input → Output Flow

```
Trained XGBoost Model (from Week 2)
    ↓
Test Dataset (from Week 1)
    ↓
[Model & Data Loading]
    ↓
Loaded Model + Prepared Test Features
    ↓
[SHAP Explainer Initialization]
    ↓
SHAP TreeExplainer with background dataset
    ↓
[SHAP Values Computation]
    ↓
SHAP values array (n_samples, n_features, n_classes)
    ↓
[Validation]
    ↓
Validated SHAP explanations
    ↓
[Global Interpretability]
    ↓
- SHAP summary plot
- Feature importance bar plot
- Feature importance CSV
    ↓
[Local Interpretability]
    ↓
- Force plots for high-risk machines
- Waterfall plots for individual predictions
    ↓
[Human-Readable Insights]
    ↓
- Text report with explanations
- JSON file with detailed explanations
```

### Key Processes

#### 1. Model & Data Loading (`src/utils/model_loader.py`)
- **Input**: Model directory, test dataset
- **Process**:
  - Find latest model directory
  - Load XGBoost model using `XGBoostModel.load()`
  - Load test dataset from parquet
  - Prepare features aligned with model
- **Output**: Loaded model, X_test, y_test

#### 2. SHAP Explainer Initialization (`src/explainability/shap_explainer.py`)
- **Input**: Trained model, background dataset
- **Process**:
  - Initialize `shap.TreeExplainer` for XGBoost
  - Align features with model
  - Set up caching directory
- **Output**: SHAPExplainer instance

#### 3. SHAP Values Computation (`src/explainability/shap_explainer.py::compute_shap_values`)
- **Input**: Test data, cache key
- **Process**:
  - Compute SHAP values using TreeExplainer
  - Cache values for reuse
  - Handle binary classification format
- **Output**: SHAP values array

#### 4. Validation (`src/explainability/shap_explainer.py::validate_explanations`)
- **Input**: SHAP values, predictions
- **Process**:
  - Validate SHAP sum property: `sum(SHAP) + expected_value ≈ prediction`
  - Check error thresholds (max < 0.5, mean < 0.1)
  - Assert logical consistency
- **Output**: Validation results

#### 5. Global Interpretability (`src/explainability/plots.py`)
- **SHAP Summary Plot**: Overall feature importance visualization
- **Feature Importance Bar Plot**: Top features ranked by importance
- **Feature Importance CSV**: Export for further analysis

#### 6. Local Interpretability (`src/explainability/plots.py`)
- **Force Plots**: Interactive HTML showing feature contributions
- **Waterfall Plots**: Sequential visualization of contributions

#### 7. Human-Readable Insights (`src/explainability/insights.py`)
- **Feature Parsing**: Extract sensor type, operation, window from feature names
- **Prediction Explanation**: Natural language explanation for each prediction
- **Report Generation**: Comprehensive text report with risk distribution

### Week 3 Outputs

**Location**: `data/artifacts/models_YYYYMMDD_HHMMSS/explainability/`

- Global Interpretability:
  - `shap_summary_plot.png`
  - `feature_importance_bar.png`
  - `shap_feature_importance.csv`
- Local Interpretability:
  - `force_plot_high_risk_{i}_instance_{idx}.html` (5 files)
  - `waterfall_plot_high_risk_{i}_instance_{idx}.png` (5 files)
- Human-Readable Insights:
  - `human_readable_insights.txt`
  - `high_risk_explanations.json`
- Cache:
  - `shap_cache/shap_values_*.npy`

---

## Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    WEEK 1: DATA PIPELINE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Raw CSV                                                        │
│    ↓                                                            │
│  [Ingestion] → Time-ordered DataFrame                           │
│    ↓                                                            │
│  [Cleaning] → Clean DataFrame                                   │
│    ↓                                                            │
│  [Feature Engineering] → Engineered DataFrame                   │
│    ↓                                                            │
│  [Split] → Train (80%) + Test (20%)                             │
│    ↓                                                            │
│  [Validation] → Validated Parquet Files                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────────────┐
                    │   OUTPUTS:    │
                    │ train.parquet │
                    │  test.parquet │
                    └───────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                  WEEK 2: MODEL TRAINING                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Train/Test Parquet Files                                       │
│    ↓                                                            │
│  [Feature Preparation] → X_train, y_train, X_test, y_test       │
│    ↓                                                            │
│  [Baseline Training] → Logistic Regression                      │
│    ↓                                                            │
│  [Random Forest Training] → Random Forest                       │
│    ↓                                                            │
│  [XGBoost Training] → XGBoost (Primary)                         │
│    ↓                                                            │
│  [Evaluation] → Metrics (Recall, F1, Precision)                 │
│    ↓                                                            │
│  [Comparison] → Best Model Selected                             │
│    ↓                                                            │
│  [Persistence] → Saved Models & Metrics                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                    ┌────────────────┐
                    │   OUTPUTS:     │
                    │  xgboost.joblib│
                    │  metrics.json  │
                    │  comparison.csv│
                    └────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│              WEEK 3: MODEL EXPLAINABILITY                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Trained Model + Test Data                                      │
│    ↓                                                            │
│  [Model Loading] → Loaded Model & Features                      │
│    ↓                                                            │
│  [SHAP Initialization] → TreeExplainer                          │
│    ↓                                                            │
│  [SHAP Computation] → SHAP Values                               │
│    ↓                                                            │
│  [Validation] → Validated Explanations                          │
│    ↓                                                            │
│  [Global Interpretability] → Summary Plots                      │
│    ↓                                                            │
│  [Local Interpretability] → Force/Waterfall Plots               │
│    ↓                                                            │
│  [Human-Readable Insights] → Text Report                        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────────────┐
                    │   OUTPUTS:    │
                    │  summary.png  │
                    │  force_*.html │
                    │  insights.txt │
                    └───────────────┘
```

---

## Input-Output Mapping

### Week 1 → Week 2

| Week 1 Output | Week 2 Input | Usage |
|--------------|--------------|-------|
| `train_YYYYMMDD_HHMMSS.parquet` | `train_df` | Training data for all models |
| `test_YYYYMMDD_HHMMSS.parquet` | `test_df` | Evaluation data for all models |
| Feature names | `feature_names_` | Model feature alignment |
| Target column | `target_col` | Model training target |

### Week 2 → Week 3

| Week 2 Output | Week 3 Input | Usage |
|--------------|--------------|-------|
| `xgboost_model.joblib` | `model` | SHAP explainer initialization |
| `feature_names_` (from model) | `feature_names` | Feature alignment |
| Test dataset (from Week 1) | `X_test` | SHAP value computation |
| Model parameters | `model.model` | TreeExplainer initialization |

### Week 1 → Week 3

| Week 1 Output | Week 3 Input | Usage |
|--------------|--------------|-------|
| `test_YYYYMMDD_HHMMSS.parquet` | `test_df` | Test data for explainability |
| Feature names | `feature_names` | Feature alignment and parsing |
| Target column | `y_test` | Validation of explanations |

---

## Integration Points

### 1. Feature Consistency
- **Week 1** creates feature names
- **Week 2** stores feature names in model (`feature_names_`)
- **Week 3** uses model's feature names for alignment
- **Ensures**: Same features used throughout pipeline

### 2. Data Versioning
- **Week 1** saves datasets with timestamps
- **Week 2** saves models in timestamped directories
- **Week 3** finds latest model directory automatically
- **Ensures**: Reproducibility and version tracking

### 3. Model Compatibility
- **Week 2** saves models with feature names and parameters
- **Week 3** loads models and extracts feature names
- **Ensures**: Model and explainer use same features

### 4. Validation Chain
- **Week 1** validates data (no leakage, time ordering)
- **Week 2** validates models (metrics, confusion matrix)
- **Week 3** validates explanations (SHAP sum property, logical consistency)
- **Ensures**: Quality at each stage

---

## How Outputs Become Inputs

### Example: Feature Engineering Flow

1. **Week 1**: Creates feature `temperature_rolling_mean_4h`
   - Input: Raw temperature sensor data
   - Process: Calculate rolling mean over 4-hour window
   - Output: Feature column in train/test DataFrames

2. **Week 2**: Model learns from this feature
   - Input: `temperature_rolling_mean_4h` from Week 1
   - Process: XGBoost learns feature importance
   - Output: Model with feature in `feature_names_`

3. **Week 3**: Explains this feature's contribution
   - Input: Feature name from model, feature value from test data
   - Process: SHAP computes contribution, parser extracts "temperature", "rolling_mean", "4h"
   - Output: "Temperature average over 4h increased failure risk (mean: 85.3°C)"

### Example: Prediction Flow

1. **Week 1**: Creates target `failure_within_24h`
   - Input: Raw failure events
   - Process: Shift failure events 24 hours forward
   - Output: Binary target column

2. **Week 2**: Model predicts this target
   - Input: Features from Week 1, target from Week 1
   - Process: XGBoost learns to predict `failure_within_24h`
   - Output: Prediction probability (0-1)

3. **Week 3**: Explains the prediction
   - Input: Prediction from Week 2, features from Week 1
   - Process: SHAP decomposes prediction into feature contributions
   - Output: "Failure risk is HIGH (87.3%) due to: temperature exceeding threshold..."

---

## Execution Commands

### Complete Pipeline (Week 1 + Week 2)

```bash
python run_week1_week2.py data/raw/sensor_data.csv
```

### Explainability Only (Week 3)

```bash
python run_explainability.py
```

### Individual Weeks

```bash
# Week 1 only
python run_pipeline.py data/raw/sensor_data.csv

# Week 2 only (requires Week 1 outputs)
python -m src.training.train

# Week 3 only (requires Week 1 & 2 outputs)
python run_explainability.py
```

---

## File Structure

```
FactoryGuard AI/
├── src/
│   ├── data/              # Week 1: Data pipeline
│   │   ├── ingest.py
│   │   ├── clean.py
│   │   ├── features.py
│   │   ├── split.py
│   │   ├── validate.py
│   │   └── pipeline.py
│   ├── models/            # Week 2: Model implementations
│   │   ├── baseline.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   └── evaluate.py
│   ├── training/          # Week 2: Training orchestration
│   │   └── train.py
│   ├── explainability/    # Week 3: Explainability
│   │   ├── shap_explainer.py
│   │   ├── plots.py
│   │   └── insights.py
│   ├── utils/             # Shared utilities
│   │   ├── logger.py
│   │   └── model_loader.py
│   └── config/            # Configuration
│       └── settings.py
├── data/
│   ├── raw/               # Input CSV files
│   └── artifacts/         # All outputs
│       ├── train_*.parquet
│       ├── test_*.parquet
│       └── models_*/
│           ├── *.joblib
│           ├── *.json
│           └── explainability/
├── run_pipeline.py        # Week 1 only
├── run_week1_week2.py     # Week 1 + Week 2
├── run_explainability.py  # Week 3
├── WEEK1_WEEK2_FLOW.md    # Week 1 & 2 documentation
├── WEEK3_EXPLAINABILITY_FLOW.md  # Week 3 documentation
├── COMPLETE_IMPLEMENTATION_FLOW.md  # This file
└── requirements.txt
```

---

## Key Design Principles

### 1. Modularity
- Each week is self-contained but integrates with others
- Clear interfaces between modules
- Reusable components (model loader, logger, etc.)

### 2. Reproducibility
- Random seeds (42) for all random operations
- Versioned outputs (timestamps in filenames)
- Caching for expensive computations
- Feature alignment ensures consistency

### 3. Validation
- Data validation (Week 1): No leakage, time ordering
- Model validation (Week 2): Metrics, confusion matrix
- Explanation validation (Week 3): SHAP sum property, logical consistency

### 4. Trust & Transparency
- Human-readable insights (Week 3)
- Validation at each stage
- Clear documentation of assumptions
- Explainable predictions

### 5. Production-Ready
- Error handling and logging
- Configuration management
- Scalable design (sampling, caching)
- No notebooks for production logic

---

## Summary

This complete implementation flow shows:

1. **Week 1** transforms raw data into ML-ready datasets
2. **Week 2** trains and evaluates models, selecting the best
3. **Week 3** explains model predictions using SHAP

Each week builds on the previous, with clear inputs and outputs. The system is:
- ✅ Reproducible (seeds, versioning, caching)
- ✅ Validated (checks at each stage)
- ✅ Trustworthy (explainable, transparent)
- ✅ Production-ready (error handling, logging, modular)

The complete pipeline enables end-to-end machine learning for predictive maintenance, from raw sensor data to actionable insights.

