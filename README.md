# 🏭 FactoryGuard AI

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![FastAPI](https://img.shields.io/badge/Framework-Flask-red.svg)
![ML](https://img.shields.io/badge/ML-XGBoost-orange.svg)

**Production-grade IoT Predictive Maintenance System powered by Machine Learning**

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Project Structure](#-project-structure) • [Contributing](#-contributing)

</div>

---

## 📋 Overview

**FactoryGuard AI** is a comprehensive machine learning system designed for predictive maintenance in industrial IoT environments. The system transforms raw sensor data into actionable insights, predicting machine failures before they occur, thereby reducing unplanned downtime and maintenance costs.

### 🎯 Key Capabilities

- **Data Pipeline**: Transforms raw sensor data into ML-ready datasets with time-aware processing
- **Model Training**: Multiple ML models (XGBoost, Random Forest, Baseline) optimized for rare-event prediction
- **Explainability**: SHAP-based model interpretability for transparent, trustworthy predictions
- **Production API**: Low-latency REST API (<50ms) for real-time failure predictions
- **Comprehensive Validation**: Leakage detection, time-series validation, and model evaluation

---

## ✨ Features

### 🔄 Week 1: Data Pipeline
- ✅ **Leakage-Free Processing**: Strict time-based ordering and validation
- ✅ **Time-Aware Operations**: All operations respect time-series ordering per machine
- ✅ **Advanced Feature Engineering**: Lag features, rolling statistics, exponential moving averages
- ✅ **Production-Ready**: Comprehensive logging, error handling, and validation

### 🤖 Week 2: Model Training
- ✅ **Multiple Models**: Baseline (Logistic Regression), Random Forest, XGBoost
- ✅ **Class Imbalance Handling**: Automatic weighting for rare events (<1% failure rate)
- ✅ **Hyperparameter Optimization**: RandomizedSearchCV for optimal performance
- ✅ **Comprehensive Evaluation**: Recall, Precision, F1-Score, ROC-AUC metrics
- ✅ **Recall-Prioritized**: Optimized to catch all failures (high recall for safety)

### 🔍 Week 3: Model Explainability
- ✅ **SHAP Integration**: TreeExplainer for fast, exact SHAP values
- ✅ **Global Interpretability**: Feature importance plots and summaries
- ✅ **Local Interpretability**: Force plots and waterfall plots for individual predictions
- ✅ **Human-Readable Insights**: Natural language explanations for engineers

### 🚀 Week 4: Production API
- ✅ **Low-Latency**: <50ms inference target, model loads once at startup
- ✅ **RESTful Design**: JSON input/output, standard HTTP status codes
- ✅ **SHAP Explanations**: Per-request local explanations with top risk factors
- ✅ **Production-Ready**: Error handling, request timing, health checks
- ✅ **Comprehensive Testing**: Unit tests for inference and API endpoints

---

## 🏗️ Project Structure

```
FactoryGuard-AI/
├── Backend/                          # Backend application
│   ├── src/                          # Source code
│   │   ├── api/                      # REST API implementation
│   │   │   ├── app.py                # Flask application factory
│   │   │   ├── routes.py             # API endpoints
│   │   │   ├── schemas.py            # Request/response validation
│   │   │   └── inference.py          # Model inference engine
│   │   ├── data/                     # Data pipeline
│   │   │   ├── ingest.py             # Data ingestion
│   │   │   ├── clean.py              # Data cleaning
│   │   │   ├── features.py           # Feature engineering
│   │   │   ├── split.py              # Train/test split
│   │   │   ├── validate.py           # Data validation
│   │   │   └── pipeline.py           # Pipeline orchestrator
│   │   ├── models/                   # ML models
│   │   │   ├── baseline.py           # Logistic Regression baseline
│   │   │   ├── random_forest.py      # Random Forest model
│   │   │   ├── xgboost_model.py      # XGBoost model (primary)
│   │   │   └── evaluate.py           # Evaluation metrics
│   │   ├── training/                 # Training orchestration
│   │   │   └── train.py              # Model training orchestrator
│   │   ├── explainability/           # Model explainability
│   │   │   ├── shap_explainer.py     # SHAP TreeExplainer
│   │   │   ├── plots.py              # Visualization functions
│   │   │   └── insights.py           # Human-readable insights
│   │   ├── utils/                    # Shared utilities
│   │   │   ├── logger.py             # Structured logging
│   │   │   └── model_loader.py       # Model loading utilities
│   │   └── config/                   # Configuration
│   │       └── settings.py           # Configuration management
│   ├── data/                         # Data directories
│   │   ├── raw/                      # Input CSV files
│   │   ├── artifacts/                # Output: datasets, models, explainability
│   │   └── output/                   # Intermediate outputs
│   ├── tests/                        # Unit and integration tests
│   │   ├── test_api.py               # API endpoint tests
│   │   └── test_inference.py         # Inference engine tests
│   ├── run_api.py                    # API server launcher
│   ├── run_data_cleaning.py          # Data cleaning script
│   ├── run_model_training.py         # Model training script
│   ├── run_model_testing.py          # Model testing script
│   ├── run_test_api.py               # API testing script
│   ├── requirements.txt              # Python dependencies
│   ├── README.md                     # Backend documentation
│   └── COMPLETE_IMPLEMENTATION_FLOW.md # Implementation guide
├── LICENSE                           # MIT License
└── README.md                         # This file
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/FactoryGuard-AI.git
cd FactoryGuard-AI
```

2. **Navigate to Backend directory**
```bash
cd Backend
```

3. **Create virtual environment** (recommended)
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On Linux/Mac
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## 📊 Key Metrics

- **Inference Latency**: <50ms per prediction
- **Model Performance**: Optimized for high recall (catches all failures)
- **API Response Time**: <50ms target
- **Data Processing**: Time-aware, leakage-free pipeline
- **Model Explainability**: SHAP-based explanations for every prediction

---

## 🛠️ Technology Stack

- **Language**: Python 3.8+
- **ML Framework**: XGBoost, scikit-learn
- **Web Framework**: Flask
- **Data Processing**: pandas, numpy
- **Explainability**: SHAP
- **Validation**: Pydantic
- **Testing**: pytest

---

## 📚 Documentation

Comprehensive documentation is available in the `Backend/` directory:

- **[Backend README](Backend/README.md)** - Complete backend documentation
- **[Implementation Flow](Backend/COMPLETE_IMPLEMENTATION_FLOW.md)** - End-to-end implementation guide

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

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

## 👤 Author

**Dhruv Sanandiya**

- GitHub: [@dhruvsanandiya](https://github.com/dhruvsanandiya)
- Email: [sanandiyadhruv77@gmail.com](mailto:sanandiyadhruv77@gmail.com)

---

## 🙏 Acknowledgments

- Built as part of the Infotact Internship program
- Inspired by real-world industrial IoT predictive maintenance challenges
- Thanks to the open-source community for excellent ML and web frameworks

---

## 📈 Roadmap

- [ ] Real-time streaming data support
- [ ] Multi-model ensemble predictions
- [ ] Advanced anomaly detection
- [ ] Dashboard for visualization
- [ ] Docker containerization
- [ ] Kubernetes deployment guides
- [ ] CI/CD pipeline integration

---

<div align="center">

**⭐ Star this repo if you find it helpful! ⭐**

Made with ❤️ for predictive maintenance

</div>
