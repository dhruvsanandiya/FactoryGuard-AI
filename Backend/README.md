# FactoryGuard AI - Complete Predictive Maintenance System

Production-grade IoT Predictive Maintenance system with data pipeline (Week 1), model training (Week 2), model explainability (Week 3), and production REST API (Week 4) for rare-event prediction.

## Overview

FactoryGuard AI is a comprehensive machine learning system for predictive maintenance that:
- **Week 1**: Transforms raw sensor data into ML-ready datasets
- **Week 2**: Trains and evaluates multiple models for failure prediction
- **Week 3**: Explains model predictions using SHAP for trust and transparency
- **Week 4**: Production REST API with <50ms latency for real-time predictions

## Project Structure

```
FactoryGuard AI/
├── src/
│   ├── data/              # Week 1: Data pipeline
│   │   ├── ingest.py      # Data ingestion and timestamp parsing
│   │   ├── clean.py       # Time-aware cleaning and interpolation
│   │   ├── features.py    # Feature engineering (lags, rolling stats, EMAs, targets)
│   │   ├── split.py       # Time-based train/test split
│   │   ├── validate.py    # Leakage detection and validation
│   │   └── pipeline.py    # Main pipeline orchestrator
│   ├── models/            # Week 2: Model implementations
│   │   ├── baseline.py    # Logistic Regression baseline
│   │   ├── random_forest.py # Random Forest model
│   │   ├── xgboost_model.py # XGBoost model (primary)
│   │   └── evaluate.py    # Evaluation metrics
│   ├── training/          # Week 2: Training orchestration
│   │   └── train.py       # Model training orchestrator
│   ├── explainability/   # Week 3: Model explainability
│   │   ├── shap_explainer.py # SHAP TreeExplainer implementation
│   │   ├── plots.py       # Visualization functions
│   │   └── insights.py    # Human-readable insights generator
│   ├── api/              # Production REST API
│   │   ├── app.py         # Flask application factory
│   │   ├── routes.py      # API endpoints
│   │   ├── schemas.py     # Request/response validation
│   │   └── inference.py  # Model inference engine
│   ├── utils/             # Shared utilities
│   │   ├── logger.py      # Structured logging
│   │   └── model_loader.py # Model and data loading utilities
│   └── config/            # Configuration
│       └── settings.py     # Configuration management
├── data/
│   ├── raw/               # Input CSV files
│   ├── artifacts/         # Output: datasets, models, explainability
│   └── output/            # Intermediate outputs
├── tests/                # Unit and integration tests
│   ├── test_api.py      # API endpoint tests
│   └── test_inference.py # Inference engine tests
├── run_week1.py          # Week 1: Data pipeline
├── run_week2.py          # Week 2: Model training
├── run_week3.py          # Week 3: Model explainability
├── run_api.py            # Week 4: Production API server
├── run_week1_week2.py    # Week 1 + Week 2 combined
├── test_api.py           # API testing script
├── requirements.txt       # Dependencies
├── API_README.md         # Complete API documentation
└── logs/                  # Application logs
```

## Features

### Week 1: Data Pipeline
- **Leakage-Free**: Strict time-based ordering and validation
- **Time-Aware Processing**: All operations respect time-series ordering per machine
- **Feature Engineering**: Lag features, rolling statistics, EMAs
- **Production-Ready**: Comprehensive logging, error handling, and validation

### Week 2: Model Training
- **Multiple Models**: Baseline, Random Forest, XGBoost
- **Class Imbalance Handling**: Automatic weighting for rare events
- **Hyperparameter Optimization**: RandomizedSearchCV for best performance
- **Comprehensive Evaluation**: Recall, Precision, F1, ROC-AUC metrics

### Week 3: Model Explainability
- **SHAP Integration**: TreeExplainer for fast, exact SHAP values
- **Global Interpretability**: Feature importance plots and summaries
- **Local Interpretability**: Force plots and waterfall plots for individual predictions
- **Human-Readable Insights**: Natural language explanations for engineers

### Week 4: Production API
- **Low-Latency**: <50ms inference target, model loads once at startup
- **RESTful**: JSON input/output, standard HTTP status codes
- **SHAP Explanations**: Per-request local explanations with top risk factors
- **Production-Ready**: Error handling, request timing, health checks
- **Comprehensive Testing**: Unit tests for inference and API endpoints

## Installation

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Complete Pipeline (All Weeks)

```bash
# Week 1: Data Pipeline
python run_week1.py data/raw/sensor_data.csv

# Week 2: Model Training (automatically uses Week 1 outputs)
python run_week2.py

# Week 3: Model Explainability (automatically uses Week 2 outputs)
python run_week3.py

# Week 4: Production API (requires trained model from Week 2)
python run_api.py

# Or use the module directly
python -m src.api.app
```

## Week 4: Production API

The FactoryGuard AI REST API provides real-time predictions for production use.

### Quick Start

```bash
# Start API server
python run_api.py

# Test API (in another terminal)
python test_api.py

# Or use curl/Postman
curl http://localhost:5000/api/v1/health
```

**See [WEEK4_API_DEPLOYMENT.md](WEEK4_API_DEPLOYMENT.md) for detailed Week 4 documentation.**  
**See [API_README.md](API_README.md) for complete API reference.**

## Usage Examples

### Python API

**Week 1: Data Pipeline**
```python
from pathlib import Path
from src.data.pipeline import run_pipeline

train_df, test_df = run_pipeline(
    input_file=Path("data/raw/sensor_data.csv"),
    output_dir=Path("data/artifacts"),
    failure_col="Machine failure",
    test_size=0.2,
    validate=True
)
```

**Week 2: Model Training**
```python
from src.training.train import train_all_models
import pandas as pd

train_df = pd.read_parquet("data/artifacts/train_*.parquet")
test_df = pd.read_parquet("data/artifacts/test_*.parquet")

results = train_all_models(
    train_df=train_df,
    test_df=test_df,
    optimize=True
)
```

**Week 3: Explainability**
```python
from src.utils.model_loader import load_model, load_test_data, prepare_explainability_data
from src.explainability.shap_explainer import SHAPExplainer

# Load model and data
model, model_dir = load_model(model_type="xgboost")
test_df = load_test_data()
X_test, y_test = prepare_explainability_data(test_df, model)

# Initialize SHAP explainer
explainer = SHAPExplainer(model, X_test.sample(100))
shap_values = explainer.compute_shap_values(X_test)
```

**Week 4: Production API**
```python
import requests

# Make prediction via API
response = requests.post(
    'http://localhost:5000/api/v1/predict',
    json={
        'machine_id': 'M_204',
        'temperature': 82.4,
        'pressure': 1.9,
        'vibration': 0.02
    }
)
result = response.json()
print(f"Failure probability: {result['failure_probability']:.2%}")
print(f"Risk level: {result['risk_level']}")
```

## Week 1: Data Pipeline

### Pipeline Steps

1. **Data Ingestion**: Load CSV, parse timestamps, sort by time per machine
2. **Data Cleaning**: Remove outliers, interpolate missing values (time-aware)
3. **Feature Engineering**: 
   - Lag features (t-1, t-2)
   - Rolling statistics (mean, std for 1h, 4h, 8h)
   - Exponential Moving Averages (alphas: 0.3, 0.5, 0.7)
   - Binary target: failure within 24 hours
4. **Train/Test Split**: Time-based split (80/20, no random split)
5. **Validation**: Leakage checks and data quality validation

### Outputs

- `train_YYYYMMDD_HHMMSS.parquet` - Training dataset
- `test_YYYYMMDD_HHMMSS.parquet` - Test dataset

Saved to `data/artifacts/`

## Week 2: Model Training

### Models Implemented

1. **Baseline (Logistic Regression)**
   - StandardScaler + LogisticRegression
   - Class weighting for imbalance handling
   - Reproducible baseline for comparison

2. **Random Forest**
   - Handles class imbalance with `class_weight='balanced'`
   - Hyperparameter optimization via RandomizedSearchCV
   - Optimized for F1-score

3. **XGBoost (Primary Model)**
   - Automatic `scale_pos_weight` calculation
   - Hyperparameter optimization (50 iterations)
   - Feature importance analysis
   - Optimized for F1-score, evaluated on Recall

### Evaluation Metrics

- **Recall**: Primary metric (catches all failures)
- **Precision**: Reduces false alarms
- **F1-Score**: Balanced metric for optimization
- **ROC-AUC**: Overall model performance
- **Confusion Matrix**: Detailed performance breakdown

### Why Recall is Prioritized

For rare events (<1% failure rate):
- **False Negatives are costly**: Missing a failure leads to unplanned downtime
- **Accuracy is misleading**: 99% accuracy with 0% recall is useless
- **Recall = Safety**: High recall means fewer missed failures

### Outputs

All models and metrics saved to `data/artifacts/models_YYYYMMDD_HHMMSS/`:
- Trained models (`.joblib`)
- Evaluation results (`.json`)
- Model comparison (`.csv`)
- Feature importance (`.csv`)
- Training configuration (`.json`)

## Week 3: Model Explainability

### Features

1. **SHAP TreeExplainer**
   - Fast, exact SHAP values for XGBoost
   - Caching for reuse
   - Feature order consistency

2. **Global Interpretability**
   - SHAP summary plot
   - Feature importance bar plot
   - Feature importance CSV export

3. **Local Interpretability**
   - Force plots (interactive HTML) for high-risk machines
   - Waterfall plots for individual predictions

4. **Human-Readable Insights**
   - Natural language explanations
   - Feature parsing (sensor type, operation, window)
   - Risk level classification (HIGH/MEDIUM/LOW)
   - Top contributing factors

### Outputs

All outputs saved to `data/artifacts/models_*/explainability/`:
- `shap_summary_plot.png` - Global feature importance
- `feature_importance_bar.png` - Top features visualization
- `shap_feature_importance.csv` - Feature importance scores
- `force_plot_high_risk_*.html` - Interactive force plots (5 files)
- `waterfall_plot_high_risk_*.png` - Waterfall plots (5 files)
- `human_readable_insights.txt` - Comprehensive text report
- `high_risk_explanations.json` - Detailed explanations

## Data Requirements

Input CSV must contain:
- `timestamp`: Timestamp column (will be parsed to datetime)
- `machine_id`: Machine identifier (or `Product ID` as fallback)
- Sensor columns: Numeric sensor readings (e.g., temperature, pressure, vibration)
- `Machine failure` or `failure`: Binary failure indicator (0/1 or True/False)

## Configuration

Modify `src/config/settings.py` to adjust:
- Feature engineering parameters (lag windows, rolling windows, EMA alphas)
- Prediction horizon (default: 24 hours)
- Train/test split ratio (default: 0.2)
- Data validation thresholds
- Logging levels

## Validation

The pipeline includes explicit leakage checks:
- Time ordering validation per machine
- Train/test time separation validation
- Feature leakage detection
- Target shift validation
- SHAP explanation validation (Week 3)

## Documentation

Comprehensive documentation available:

- **`WEEK1_DATA_PIPELINE_FLOW.md`** - Detailed Week 1 flow and explanations
- **`WEEK2_MODEL_TRAINING_FLOW.md`** - Detailed Week 2 flow and explanations
- **`WEEK3_EXPLAINABILITY_FLOW.md`** - Detailed Week 3 flow and explanations
- **`WEEK4_API_DEPLOYMENT.md`** - Detailed Week 4 API deployment and usage
- **`API_README.md`** - Complete API reference documentation
- **`COMPLETE_IMPLEMENTATION_FLOW.md`** - End-to-end flow with integration points

## Key Design Principles

1. **Modularity**: Each week is self-contained but integrates seamlessly
2. **Reproducibility**: Random seeds (42), versioned outputs, caching
3. **Validation**: Data validation, model validation, explanation validation
4. **Trust & Transparency**: Human-readable insights, explainable predictions
5. **Production-Ready**: Error handling, logging, scalable design

## Notes

- All operations are performed per-machine to prevent cross-machine leakage
- Time-series ordering is strictly preserved
- Missing values are interpolated using time-aware methods
- Rolling features use backward-looking windows only
- All models use `random_state=42` for reproducibility
- Hyperparameter optimization uses RandomizedSearchCV (faster than GridSearch)
- SHAP values are cached for reuse and reproducibility

## Troubleshooting

### Week 1 Issues
- **Missing timestamps**: System will create synthetic timestamps
- **Missing machine_id**: System will use `Product ID` as fallback
- **Outlier removal too aggressive**: Adjust IQR multiplier in settings

### Week 2 Issues
- **Low Recall**: Increase `scale_pos_weight` or class weight
- **High False Positives**: Acceptable trade-off for high recall
- **Slow Training**: Use `--no-optimize` flag for faster training

### Week 3 Issues
- **SHAP computation slow**: Reduce background sample size
- **Memory issues**: Sample test data for SHAP computation
- **Feature mismatch**: Ensure feature names match between model and data

### Week 4 Issues
- **API not starting**: Check if model files exist in artifacts directory
- **Connection refused**: Ensure API server is running on correct port
- **High latency**: Disable SHAP explanations or reduce background data size
- **Feature mismatch**: Ensure model was trained with same feature engineering pipeline

## Dependencies

See `requirements.txt` for complete list. Key dependencies:
- pandas >= 2.0.0
- numpy >= 1.24.0
- scikit-learn >= 1.3.0
- xgboost >= 2.0.0
- shap >= 0.42.0
- matplotlib >= 3.7.0
- flask >= 2.3.0
- flask-cors >= 4.0.0
- pydantic >= 2.0.0
- requests >= 2.31.0
- pytest >= 7.4.0

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]
