<div align="center">

# 🏭 FactoryGuard AI - Backend Documentation


![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Framework-Flask-red.svg)
![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

**Production-grade IoT Predictive Maintenance System Backend**

[Features](#-features) • [API Documentation](#-api-documentation) • [Installation](#-installation) • [Usage](#-usage) • [Project Structure](#-project-structure)

</div>

---

## 📋 Overview

FactoryGuard AI Backend is a comprehensive machine learning system for predictive maintenance that transforms raw sensor data into actionable failure predictions. The system consists of four integrated components:

- **Week 1**: Data Pipeline - Transforms raw sensor data into ML-ready datasets
- **Week 2**: Model Training - Trains and evaluates multiple models for failure prediction
- **Week 3**: Model Explainability - Explains model predictions using SHAP for trust and transparency
- **Week 4**: Production REST API - Low-latency API (<50ms) for real-time predictions

---

## ✨ Features

### 🔄 Week 1: Data Pipeline

- **Leakage-Free Processing**: Strict time-based ordering and validation
- **Time-Aware Operations**: All operations respect time-series ordering per machine
- **Advanced Feature Engineering**: 
  - Lag features (t-1, t-2)
  - Rolling statistics (mean, std for 1h, 4h, 8h windows)
  - Exponential Moving Averages (alphas: 0.3, 0.5, 0.7)
  - Binary target: failure within 24 hours
- **Production-Ready**: Comprehensive logging, error handling, and validation
- **Time-Based Split**: 80/20 train/test split (no random split)

### 🤖 Week 2: Model Training

- **Multiple Models**: 
  - Baseline (Logistic Regression) - Reproducible baseline
  - Random Forest - Hyperparameter optimized
  - XGBoost (Primary) - Optimized for rare events
- **Class Imbalance Handling**: Automatic weighting for rare events (<1% failure rate)
- **Hyperparameter Optimization**: RandomizedSearchCV for best performance
- **Comprehensive Evaluation**: 
  - Recall (primary metric - catches all failures)
  - Precision (reduces false alarms)
  - F1-Score (balanced metric)
  - ROC-AUC (overall performance)
  - Confusion Matrix (detailed breakdown)

### 🔍 Week 3: Model Explainability

- **SHAP Integration**: TreeExplainer for fast, exact SHAP values
- **Global Interpretability**: 
  - Feature importance plots
  - Feature importance summaries
  - CSV exports for analysis
- **Local Interpretability**: 
  - Force plots (interactive HTML) for high-risk machines
  - Waterfall plots for individual predictions
- **Human-Readable Insights**: 
  - Natural language explanations
  - Feature parsing (sensor type, operation, window)
  - Risk level classification (HIGH/MEDIUM/LOW)
  - Top contributing factors

### 🚀 Week 4: Production API

- **Low-Latency**: <50ms inference target, model loads once at startup
- **RESTful Design**: JSON input/output, standard HTTP status codes
- **SHAP Explanations**: Per-request local explanations with top risk factors
- **Production-Ready**: 
  - Error handling and validation
  - Request timing and logging
  - Health checks
  - CORS support
- **Comprehensive Testing**: Unit tests for inference and API endpoints

---

## 🏗️ Project Structure

```
Backend/
├── src/                              # Source code
│   ├── api/                          # REST API implementation
│   │   ├── __init__.py
│   │   ├── app.py                    # Flask application factory
│   │   ├── routes.py                 # API endpoints (health, predict, model/info)
│   │   ├── schemas.py                # Pydantic request/response validation
│   │   └── inference.py              # Model inference engine
│   ├── data/                         # Week 1: Data pipeline
│   │   ├── __init__.py
│   │   ├── ingest.py                 # Data ingestion and timestamp parsing
│   │   ├── clean.py                  # Time-aware cleaning and interpolation
│   │   ├── features.py               # Feature engineering (lags, rolling stats, EMAs)
│   │   ├── split.py                  # Time-based train/test split
│   │   ├── validate.py               # Leakage detection and validation
│   │   └── pipeline.py               # Main pipeline orchestrator
│   ├── models/                       # Week 2: Model implementations
│   │   ├── __init__.py
│   │   ├── baseline.py               # Logistic Regression baseline
│   │   ├── random_forest.py          # Random Forest model
│   │   ├── xgboost_model.py          # XGBoost model (primary)
│   │   └── evaluate.py               # Evaluation metrics
│   ├── training/                     # Week 2: Training orchestration
│   │   ├── __init__.py
│   │   └── train.py                  # Model training orchestrator
│   ├── explainability/               # Week 3: Model explainability
│   │   ├── __init__.py
│   │   ├── shap_explainer.py         # SHAP TreeExplainer implementation
│   │   ├── plots.py                  # Visualization functions
│   │   └── insights.py               # Human-readable insights generator
│   ├── utils/                        # Shared utilities
│   │   ├── __init__.py
│   │   ├── logger.py                 # Structured logging
│   │   └── model_loader.py           # Model and data loading utilities
│   └── config/                       # Configuration
│       ├── __init__.py
│       └── settings.py               # Configuration management
├── data/                             # Data directories
│   ├── raw/                          # Input CSV files (sensor_data.csv)
│   ├── artifacts/                    # Output: datasets, models, explainability
│   │   ├── train_*.parquet           # Training datasets
│   │   ├── test_*.parquet            # Test datasets
│   │   └── models_*/                 # Model artifacts
│   │       ├── *.joblib            # Trained models
│   │       ├── *.json               # Evaluation results
│   │       └── explainability/      # SHAP outputs
│   └── output/                       # Intermediate outputs
│       └── cleaned_data.csv          # Cleaned data
├── tests/                            # Unit and integration tests
│   ├── __init__.py
│   ├── test_api.py                   # API endpoint tests
│   └── test_inference.py             # Inference engine tests
├── run_api.py                        # Week 4: Production API server
├── run_data_cleaning.py              # Week 1: Data pipeline
├── run_model_training.py             # Week 2: Model training
├── run_model_testing.py              # Week 3: Model testing & explainability
├── run_test_api.py                   # API testing script
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
└── COMPLETE_IMPLEMENTATION_FLOW.md   # Complete implementation guide
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Create virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Dependencies

Key dependencies include:
- `pandas >= 2.0.0` - Data processing
- `numpy >= 1.24.0` - Numerical operations
- `scikit-learn >= 1.3.0` - Machine learning
- `xgboost >= 2.0.0` - Gradient boosting
- `shap >= 0.42.0` - Model explainability
- `flask >= 2.3.0` - Web framework
- `pydantic >= 2.0.0` - Data validation
- `pytest >= 7.4.0` - Testing framework

See `requirements.txt` for complete list.

---

## 📖 Usage

### Complete Pipeline (All Weeks)

```bash
# Week 1: Data Pipeline
python run_data_cleaning.py

# Week 2: Model Training (automatically uses Week 1 outputs)
python run_model_training.py

# Week 3: Model Testing & Explainability (automatically uses Week 2 outputs)
python run_model_testing.py

# Week 4: Production API (requires trained model from Week 2)
python run_api.py

# Test API (in another terminal)
python run_test_api.py
```

### Week 1: Data Pipeline

**Pipeline Steps:**
1. **Data Ingestion**: Load CSV, parse timestamps, sort by time per machine
2. **Data Cleaning**: Remove outliers, interpolate missing values (time-aware)
3. **Feature Engineering**: 
   - Lag features (t-1, t-2)
   - Rolling statistics (mean, std for 1h, 4h, 8h)
   - Exponential Moving Averages (alphas: 0.3, 0.5, 0.7)
   - Binary target: failure within 24 hours
4. **Train/Test Split**: Time-based split (80/20, no random split)
5. **Validation**: Leakage checks and data quality validation

**Outputs:**
- `train_YYYYMMDD_HHMMSS.parquet` - Training dataset
- `test_YYYYMMDD_HHMMSS.parquet` - Test dataset

Saved to `data/artifacts/`

### Week 2: Model Training

**Models Implemented:**

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

**Evaluation Metrics:**
- **Recall**: Primary metric (catches all failures)
- **Precision**: Reduces false alarms
- **F1-Score**: Balanced metric for optimization
- **ROC-AUC**: Overall model performance
- **Confusion Matrix**: Detailed performance breakdown

**Why Recall is Prioritized:**
For rare events (<1% failure rate):
- **False Negatives are costly**: Missing a failure leads to unplanned downtime
- **Accuracy is misleading**: 99% accuracy with 0% recall is useless
- **Recall = Safety**: High recall means fewer missed failures

**Outputs:**
All models and metrics saved to `data/artifacts/models_YYYYMMDD_HHMMSS/`:
- Trained models (`.joblib`)
- Evaluation results (`.json`)
- Model comparison (`.csv`)
- Feature importance (`.csv`)
- Training configuration (`.json`)

### Week 3: Model Explainability

**Features:**
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

**Outputs:**
All outputs saved to `data/artifacts/models_*/explainability/`:
- `shap_summary_plot.png` - Global feature importance
- `feature_importance_bar.png` - Top features visualization
- `shap_feature_importance.csv` - Feature importance scores
- `force_plot_high_risk_*.html` - Interactive force plots (5 files)
- `waterfall_plot_high_risk_*.png` - Waterfall plots (5 files)
- `human_readable_insights.txt` - Comprehensive text report
- `high_risk_explanations.json` - Detailed explanations

### Week 4: Production API

The FactoryGuard AI REST API provides real-time predictions for production use.

**Quick Start:**
```bash
# Start API server
python run_api.py

# Test API (in another terminal)
python run_test_api.py

# Or use curl/Postman
curl http://localhost:5000/api/v1/health
```

---

## 🔌 API Documentation

### Base URL
```
http://localhost:5000
```

### Endpoints

#### 1. Root Endpoint
```http
GET /
```
Returns API information and available endpoints.

**Response:**
```json
{
  "name": "FactoryGuard AI API",
  "version": "1.0.0",
  "description": "Production REST API for real-time machine failure prediction",
  "status": "running",
  "endpoints": {
    "root": "/",
    "health": "/api/v1/health",
    "predict": "/api/v1/predict",
    "model_info": "/api/v1/model/info"
  }
}
```

#### 2. Health Check
```http
GET /api/v1/health
```
Checks API health and model status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_type": "xgboost",
  "shap_enabled": true
}
```

#### 3. Prediction
```http
POST /api/v1/predict
```
Makes a failure prediction for a machine.

**Request Body:**
```json
{
  "machine_id": "M_204",
  "temperature": 82.4,
  "pressure": 1.9,
  "vibration": 0.02
}
```

**Response:**
```json
{
  "failure_probability": 0.85,
  "risk_level": "HIGH",
  "top_risk_factors": [
    {
      "feature": "temperature_lag_1",
      "contribution": 0.15,
      "explanation": "Temperature from previous time step is elevated"
    }
  ],
  "shap_explanations": [
    {
      "feature": "temperature_lag_1",
      "shap_value": 0.15,
      "explanation": "Temperature from previous time step is elevated"
    }
  ]
}
```

#### 4. Model Information
```http
GET /api/v1/model/info
```
Returns detailed model information.

**Response:**
```json
{
  "model_type": "xgboost",
  "model_path": "data/artifacts/models_20250101_120000",
  "shap_enabled": true,
  "feature_count": 45,
  "training_date": "2025-01-01T12:00:00"
}
```

### Python Client Example

```python
import requests

# Health check
response = requests.get('http://localhost:5000/api/v1/health')
print(response.json())

# Make prediction
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
print(f"Top risk factors: {result['top_risk_factors']}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:5000/api/v1/health

# Prediction
curl -X POST http://localhost:5000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "machine_id": "M_204",
    "temperature": 82.4,
    "pressure": 1.9,
    "vibration": 0.02
  }'

# Model info
curl http://localhost:5000/api/v1/model/info
```

---

## 🧪 Testing

### Run Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_api.py
pytest tests/test_inference.py

# Run with coverage
pytest --cov=src tests/
```

### Test API Manually

```bash
# Start API server
python run_api.py

# In another terminal, run test script
python run_test_api.py
```

---

## ⚙️ Configuration

Modify `src/config/settings.py` to adjust:
- Feature engineering parameters (lag windows, rolling windows, EMA alphas)
- Prediction horizon (default: 24 hours)
- Train/test split ratio (default: 0.2)
- Data validation thresholds
- Logging levels
- API settings

---

## 🔍 Validation

The pipeline includes explicit leakage checks:
- Time ordering validation per machine
- Train/test time separation validation
- Feature leakage detection
- Target shift validation
- SHAP explanation validation (Week 3)

---

## 📊 Data Requirements

Input CSV must contain:
- `timestamp`: Timestamp column (will be parsed to datetime)
- `machine_id`: Machine identifier (or `Product ID` as fallback)
- Sensor columns: Numeric sensor readings (e.g., temperature, pressure, vibration)
- `Machine failure` or `failure`: Binary failure indicator (0/1 or True/False)

---

## 🐛 Troubleshooting

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

---

## 📚 Key Design Principles

1. **Modularity**: Each week is self-contained but integrates seamlessly
2. **Reproducibility**: Random seeds (42), versioned outputs, caching
3. **Validation**: Data validation, model validation, explanation validation
4. **Trust & Transparency**: Human-readable insights, explainable predictions
5. **Production-Ready**: Error handling, logging, scalable design

---

## 📝 Notes

- All operations are performed per-machine to prevent cross-machine leakage
- Time-series ordering is strictly preserved
- Missing values are interpolated using time-aware methods
- Rolling features use backward-looking windows only
- All models use `random_state=42` for reproducibility
- Hyperparameter optimization uses RandomizedSearchCV (faster than GridSearch)
- SHAP values are cached for reuse and reproducibility

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](../LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Dhruv Sanandiya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and ensure tests pass
4. **Commit your changes** with clear messages
   ```bash
   git commit -m "Add: your feature description"
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request** with a detailed description

### Contribution Guidelines

- Follow PEP 8 style guidelines
- Write clear, concise commit messages
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass before submitting PR
- Maintain backward compatibility when possible

---

## 👤 Author

**Dhruv Sanandiya**

- GitHub: [@yourusername](https://github.com/yourusername)
- Email: [your-email@example.com](mailto:your-email@example.com)

---

## 🙏 Acknowledgments

- Built as part of the Infotact Internship program
- Inspired by real-world industrial IoT predictive maintenance challenges
- Thanks to the open-source community for excellent ML and web frameworks

---

## 📈 Performance Metrics

- **Inference Latency**: <50ms per prediction
- **API Response Time**: <50ms target
- **Model Performance**: Optimized for high recall (catches all failures)
- **Data Processing**: Time-aware, leakage-free pipeline
- **Model Explainability**: SHAP-based explanations for every prediction

---

<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

Made with ❤️ for predictive maintenance

</div>
