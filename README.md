# Air Quality Forecasting with Machine Learning

A comprehensive machine learning project for forecasting air pollution using the UCI Air Quality dataset.

## Quick Start

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download Dataset
```bash
PYTHONPATH=. python3 - <<'PY'
from src.data_loader import download_uc_air_quality
download_uc_air_quality('data')
PY
```

### Run Baseline Models
```bash
PYTHONPATH=. python3 scripts/run_baselines.py
PYTHONPATH=. python3 scripts/run_regression_engineered.py
PYTHONPATH=. python3 scripts/run_classification.py
```

### Open Notebooks
```bash
jupyter notebook
# Then open notebooks/EDA.ipynb and notebooks/Experiments.ipynb
```

## Project Structure

```
.
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── src/
│   ├── __init__.py
│   ├── data_loader.py                # Download & load UCI dataset
│   ├── preprocessing.py              # Data cleaning & datetime merging
│   ├── features.py                   # Lag & moving average features
│   └── models/
│       ├── __init__.py
│       ├── baseline.py               # Regression baselines + temporal split
│       └── classification.py         # Classification pipeline (low/mid/high CO)
├── scripts/
│   ├── run_baselines.py              # Evaluate baseline regressors
│   ├── run_regression_engineered.py  # Regression with features
│   └── run_classification.py         # Classification with features
├── notebooks/
│   ├── EDA.ipynb                     # Exploratory data analysis
│   └── Experiments.ipynb             # Model evaluation & comparison
├── tests/
│   └── test_preprocessing.py         # Unit tests for data loading/preprocessing
└── data/                             # (Downloaded dataset goes here)
```

## Main Objectives

### 1. Exploratory Data Analysis (EDA)
- ✅ Time patterns, seasonal effects, correlations
- ✅ Missing value analysis (sentinel value -200)
- ✅ Temporal split: 2004 training, 2005 testing

### 2. Data Preprocessing
- ✅ Handle missing values
- ✅ Merge Date + Time into unified datetime
- ✅ Normalize continuous features
- ✅ Create temporal features (hour, weekday, month)

### 3. Feature Engineering
- ✅ Lag variables (1, 6, 12, 24 hours)
- ✅ Moving averages (3, 6, 12 hour windows)
- ✅ Temporal granularity analysis

### 4. Regression Models
Predict 5 pollutant concentrations (CO, NMHC, C6H6, NOx, NO2) at horizons 1h, 6h, 12h, 24h.

**Algorithms implemented:**
- Linear Regression
- Random Forest Regressor

**Metrics:** RMSE, visual residual analysis

**Results (Feature-Engineered, CO(GT)):**
| Horizon | LR RMSE | RF RMSE | Naive RMSE |
|---------|---------|---------|-----------|
| 1h      | 0.6648  | 0.8125  | 0.7966    |
| 6h      | 1.3076  | 1.2324  | 1.8049    |
| 12h     | 1.1210  | 1.3950  | 1.7981    |
| 24h     | 1.2582  | 1.4639  | 1.2778    |

### 5. Classification Models
Discretize CO(GT) into **low** (<1.5), **mid** (1.5–2.5), **high** (>2.5).

**Algorithms implemented:**
- Logistic Regression
- Random Forest Classifier
- Gradient Boosting Classifier

**Metrics:** Accuracy, weighted F1-score

**Results (Feature-Engineered):**
| Horizon | LR Acc. | RF Acc. | GB Acc. | Naive Acc. |
|---------|---------|---------|---------|-----------|
| 1h      | 0.7498  | 0.7303  | 0.7502  | 0.7588    |
| 6h      | 0.5824  | 0.5844  | 0.5781  | 0.4002    |
| 12h     | 0.5973  | 0.5627  | 0.5710  | 0.3862    |
| 24h     | 0.5535  | 0.5161  | 0.5530  | 0.6036    |

### 6. Anomaly & Event Detection
- Implemented residual-based anomaly detection in EDA
- Temporal patterns (weekday/weekend, temperature extremes) correlated with pollution spikes

### 7. Discussion
**Key Findings:**
- **Feature engineering improves regression**: Lag and moving average features capture temporal dependencies.
- **Short-term vs. long-term forecasts**: RMSE increases with horizon; 24h forecasts approach naive baseline performance.
- **Seasonal effects strong**: Hourly and weekly patterns evident; weekday rush hours show elevated pollution.
- **Classification challenges**: Imbalanced class distribution and concept drift reduce classification accuracy at longer horizons.
- **Sensor reliability**: Missing values (~26% in some columns) suggest sensor drift or maintenance events.

**Limitations & Improvements:**
1. **Sensor drift**: Implement drift detection and correction algorithms.
2. **Class imbalance**: Use SMOTE or class weights to improve minority class prediction.
3. **External variables**: Incorporate traffic volume, weather forecasts, or policy events.
4. **Ensemble methods**: Combine multiple models for robust predictions.
5. **Hyperparameter tuning**: Grid search or Bayesian optimization for model selection.

## Data Dictionary

| Variable | Description |
|----------|-------------|
| CO(GT) | True hourly averaged CO concentration (mg/m³) |
| PT08.S1(CO) | CO sensor response |
| NMHC(GT) | True hourly averaged NMHC concentration |
| PT08.S2(NMHC) | NMHC sensor response |
| C6H6(GT) | True hourly averaged benzene concentration |
| NOx(GT) | True hourly averaged NOx concentration |
| PT08.S3(NOx) | NOx sensor response |
| NO2(GT) | True hourly averaged NO₂ concentration |
| PT08.S4(NO2) | NO₂ sensor response |
| PT08.S5(O3) | O₃ sensor response |
| T | Temperature (°C) |
| RH | Relative humidity (%) |
| AH | Absolute humidity |

Missing values represented by sentinel value **-200**.

## Dependencies

- pandas, numpy: data manipulation
- scikit-learn: ML models & preprocessing
- matplotlib, seaborn: visualisation
- xgboost: gradient boosting
- jupyter: notebook environment
- pytest: unit testing

## Usage Examples

### Baseline Regression (Simple Features)
```python
from src.models.baseline import prepare_df, temporal_train_test_split, evaluate_regressors
df = prepare_df('data/AirQualityUCI.csv')
train, test = temporal_train_test_split(df)
feature_cols = ['hour', 'weekday', 'month', 'T', 'RH', 'AH']
results = evaluate_regressors(train, test, feature_cols, 'CO(GT)', [1, 6, 12, 24])
```

### Feature Engineering
```python
from src.features import engineer_features
df_engineered = engineer_features(df, target_col='CO(GT)', lag_steps=[1,6,12,24], ma_windows=[3,6,12])
# Now includes columns: CO(GT)_lag1, CO(GT)_lag6, ..., CO(GT)_ma3, CO(GT)_ma6, ...
```

### Classification
```python
from src.models.classification import evaluate_classifiers, discretize_co
results = evaluate_classifiers(train, test, feature_cols, 'CO(GT)', [1, 6, 12, 24])
# Categories: low (<1.5), mid (1.5-2.5), high (>2.5)
```

## Testing

Run unit tests to verify data loading and preprocessing:
```bash
PYTHONPATH=. python3 -m pytest tests/test_preprocessing.py -v
```

Tests cover:
- CSV loading with numeric coercion
- Missing value replacement (sentinel -200 → NaN)
- Datetime merging (Date + Time)
- Temporal train/test split
