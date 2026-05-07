# 📦 End-to-End Time Series Forecasting System

> **Dataset:** US State-level Weekly Beverage Sales (2019–2023, 43 States)  
> **Goal:** Forecast next 8 weeks of sales per state via REST API

---

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Project Structure](#project-structure)
3. [Models Implemented](#models-implemented)
4. [Feature Engineering](#feature-engineering)
5. [Data Pipeline](#data-pipeline)
6. [Model Selection Strategy](#model-selection-strategy)
7. [Results Summary](#results-summary)
8. [REST API Reference](#rest-api-reference)
9. [Quick Start](#quick-start)
10. [Design Decisions](#design-decisions)

---

## System Architecture

```
Raw CSV → Data Processor → Feature Engineering → Model Training (4 models)
                                                        ↓
                                               Model Selection (SMAPE)
                                                        ↓
                                            Artifacts (models.pkl, scores.csv)
                                                        ↓
                                              FastAPI REST Service
                                                        ↓
                                           /forecast/{state}?horizon=8
```

---

## Project Structure

```
forecasting_system/
├── data/
│   └── sales_data.csv              # Raw input data
├── models/
│   ├── __init__.py
│   ├── base_model.py               # Abstract base class (SMAPE, RMSE)
│   ├── arima_model.py              # SARIMA with AIC-based order selection
│   ├── prophet_model.py            # Facebook Prophet + US holidays
│   ├── xgboost_model.py            # XGBoost with recursive forecasting
│   └── lstm_model.py               # Bidirectional LSTM (TensorFlow)
├── api/
│   └── app.py                      # FastAPI REST endpoints
├── artifacts/
│   ├── models.pkl                  # Serialised fitted models
│   ├── scores.csv                  # Validation SMAPE/RMSE per model/state
│   ├── best_models.csv             # Winner per state
│   └── forecasts.csv               # 8-week forecasts for all states
├── data_processor.py               # Load, clean, fill gaps, feature-engineer
├── model_selector.py               # Train → validate → select best model
├── train_pipeline.py               # CLI entry point for training
├── requirements.txt
└── README.md
```

---

## Models Implemented

### 1. SARIMA (`models/arima_model.py`)
- **Library:** `statsmodels.tsa.statespace.SARIMAX`
- **Order selection:** ADF test for `d`; AIC grid search over `(p,d,q)` ∈ {0,1}²
- **Seasonal period:** m = 52 (weekly data)
- **Forecasting:** Built-in `get_forecast()` method

### 2. Facebook Prophet (`models/prophet_model.py`)
- **Library:** `prophet`
- **Config:** Multiplicative seasonality, yearly + weekly components
- **Extra:** US public holidays added via `add_country_holidays("US")`
- **Tuning:** `changepoint_prior_scale=0.05` (regularises trend)

### 3. XGBoost (`models/xgboost_model.py`)
- **Library:** `xgboost`
- **Strategy:** Supervised regression on tabular features
- **Forecasting:** Recursive multi-step (predicted value fed back as lag)
- **Scaling:** `StandardScaler` applied to all features

### 4. LSTM (`models/lstm_model.py`)
- **Library:** `TensorFlow / Keras`
- **Architecture:** Bidirectional LSTM (64) → Dropout(0.2) → BiLSTM (32) → Dense(16) → Dense(1)
- **Look-back window:** 12 weeks
- **Training:** EarlyStopping on val_loss, patience=10, up to 80 epochs
- **Scaling:** `MinMaxScaler(0,1)` on sales values

---

## Feature Engineering

All features are computed **per-state** using only past values (no future leakage):

| Feature | Description |
|---------|-------------|
| `lag_1` | Sales 1 week ago |
| `lag_2` | Sales 2 weeks ago |
| `lag_4` | Sales 4 weeks ago (≈ 1 month) |
| `lag_8` | Sales 8 weeks ago (≈ 2 months) |
| `roll_mean_4` | 4-week rolling mean (shifted by 1) |
| `roll_std_4` | 4-week rolling std (shifted by 1) |
| `roll_mean_8` | 8-week rolling mean (shifted by 1) |
| `roll_std_8` | 8-week rolling std (shifted by 1) |
| `week_of_year` | ISO week number (1–52) |
| `month` | Calendar month (1–12) |
| `quarter` | Calendar quarter (1–4) |
| `year` | Calendar year |
| `holiday_week` | 1 if any US public holiday falls in the week |

---

## Data Pipeline

### Missing Date Handling
1. All dates are normalised to **week-start (Monday)** to create a uniform weekly grid
2. Duplicates (same state + week) are **summed**
3. A full weekly date range is created per state using `pd.date_range`
4. Interior gaps: **linear interpolation**
5. Edge gaps: **forward-fill then back-fill**

### Train / Validation Split
- **No data leakage**: strictly time-based split
- Last `VAL_WEEKS = 8` rows → validation set
- Remaining rows → training set
- Feature rows with NaN lag values (early dates) are **dropped from training**
- After model selection, winner is **re-trained on full data** (train + val)

---

## Model Selection Strategy

For each state:

```
1. Split → train / val (last 8 weeks)
2. Fit all 4 models on train
3. Predict 8 steps on val
4. Compute SMAPE for each model
5. Select model with lowest SMAPE
6. Re-train winner on full history
7. Save to artifacts/models.pkl
```

**Metric used:** Symmetric MAPE (SMAPE) — scale-invariant, handles near-zero values  
**Fallback:** If a model crashes, it is skipped (graceful degradation)

---

## Results Summary

All 43 states trained successfully. **Prophet won for all states** due to its
strong handling of multiplicative seasonality and the sparse (irregular)
spacing of the raw data.

| Metric | Avg (all states) |
|--------|-----------------|
| SMAPE  | ~15.9% |
| Best SMAPE | 11.6% (Michigan) |
| Worst SMAPE | 26.8% (Mississippi) |

Model ranking by average SMAPE:

| Rank | Model | Avg SMAPE |
|------|-------|-----------|
| 1 | Prophet | ~16% |
| 2 | XGBoost | ~32% |
| 3 | SARIMA | ~39% |

---

## REST API Reference

**Base URL:** `http://localhost:8000`

### Endpoints

#### `GET /health`
Liveness check.
```json
{"status": "ok", "models_loaded": 43}
```

#### `GET /states`
List all available states.
```json
{"states": ["Alabama", "Arizona", ...], "count": 43}
```

#### `GET /models`
See which model was selected for each state.

#### `GET /forecast/{state}?horizon=8`
Forecast for a single state.

```bash
curl "http://localhost:8000/forecast/California?horizon=8"
```

```json
{
  "state": "California",
  "model_used": "Prophet",
  "horizon_weeks": 8,
  "last_training_date": "2023-11-27",
  "forecasts": [
    {"week": 1, "date": "2023-12-04", "predicted_sales": 512345678},
    {"week": 2, "date": "2023-12-11", "predicted_sales": 498765432},
    ...
  ]
}
```

#### `GET /forecast/all?horizon=8`
Batch forecast for all 43 states.

#### `POST /forecast/batch`
Forecast for a custom list of states.
```bash
curl -X POST "http://localhost:8000/forecast/batch" \
  -H "Content-Type: application/json" \
  -d '{"states": ["Texas", "New York", "Florida"], "horizon": 8}'
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train all models
```bash
# Full training (all 43 states, all 4 models including LSTM)
python train_pipeline.py

# Fast training (skip LSTM)
python train_pipeline.py --no-lstm

# Specific states only
python train_pipeline.py --states "California,Texas,Florida" --no-lstm
```

### 3. Start the API server
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

### 4. Test the API
```bash
# Health check
curl http://localhost:8000/health

# Forecast California for next 8 weeks
curl http://localhost:8000/forecast/California

# Forecast Texas for next 4 weeks
curl "http://localhost:8000/forecast/Texas?horizon=4"

# All states
curl http://localhost:8000/forecast/all

# Interactive docs
open http://localhost:8000/docs
```

---

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| Weekly grid normalisation | Raw data has irregular spacing; normalising prevents bias |
| SMAPE as selection metric | Scale-invariant; handles wide range of state sizes |
| Re-train on full data | After model selection, use all available signal for final forecasts |
| Recursive XGBoost forecasting | Avoids future leakage; mirrors real-world deployment |
| Bidirectional LSTM | Captures both past and future context within the look-back window |
| Multiplicative Prophet seasonality | Sales scale multiplicatively with trend (summer peaks proportionally larger) |
| Graceful model failure | If one model errors, the pipeline continues with remaining candidates |
| FastAPI + Pydantic | Type safety, automatic OpenAPI docs, async-ready |

---

## requirements.txt

```
fastapi>=0.110
uvicorn[standard]>=0.29
prophet>=1.1
scikit-learn>=1.4
xgboost>=2.0
tensorflow>=2.15
statsmodels>=0.14
holidays>=0.43
joblib>=1.3
pandas>=2.2
numpy>=1.26
pydantic>=2.0
```
# forecasting_system
