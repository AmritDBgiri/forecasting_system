"""
api/app.py
-----------
Production-ready FastAPI service exposing the trained forecasting models.

Endpoints
---------
GET  /health                       → liveness check
GET  /states                       → list of available states
GET  /models                       → which model won per state
GET  /forecast/{state}             → 8-week forecast for a state
GET  /forecast/{state}?horizon=N   → N-week forecast (1–52)
GET  /forecast/all                 → forecasts for every state (batch)
POST /forecast/batch               → arbitrary list of states
"""

import logging
import os
import pickle
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── path setup ─────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from data_processor import load_and_clean, add_features

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── globals (loaded at startup) ────────────────────────────────────────────
MODELS: dict = {}
BEST_MODEL_MAP: dict = {}
LAST_DATES: dict = {}
STATES: list = []


def _load_artifacts():
    artifact_dir = ROOT / "artifacts"
    model_file = artifact_dir / "models.pkl"
    best_file = artifact_dir / "best_models.csv"

    if not model_file.exists():
        raise FileNotFoundError(
            f"Trained models not found at {model_file}. "
            "Run train_pipeline.py first."
        )

    with open(model_file, "rb") as f:
        MODELS.update(pickle.load(f))

    best_df = pd.read_csv(best_file)
    BEST_MODEL_MAP.update(
        dict(zip(best_df["State"], best_df["Best_Model"]))
    )

    # Compute last training date per state from raw data
    csv = ROOT / "data/sales_data.csv"
    if csv.exists():
        df = load_and_clean(csv)
        for state, grp in df.groupby("State"):
            if state in MODELS:
                LAST_DATES[state] = grp["Date"].max()

    STATES.extend(sorted(MODELS.keys()))
    logger.info("Loaded %d state models", len(MODELS))


# ── lifespan ───────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_artifacts()
    yield


# ── app ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sales Forecasting API",
    description=(
        "End-to-end beverage sales forecasting service. "
        "Returns weekly predictions per US state using the best "
        "automatically selected model (SARIMA / Prophet / XGBoost / LSTM)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── schemas ────────────────────────────────────────────────────────────────
class WeekForecast(BaseModel):
    week: int
    date: str
    predicted_sales: float


class ForecastResponse(BaseModel):
    state: str
    model_used: str
    horizon_weeks: int
    last_training_date: str
    forecasts: list[WeekForecast]


class BatchRequest(BaseModel):
    states: list[str]
    horizon: Optional[int] = 8


# ── helpers ────────────────────────────────────────────────────────────────
def _run_forecast(state: str, horizon: int) -> ForecastResponse:
    if state not in MODELS:
        raise HTTPException(404, detail=f"No model found for state '{state}'.")
    if not (1 <= horizon <= 52):
        raise HTTPException(400, detail="horizon must be between 1 and 52.")

    model = MODELS[state]
    preds = model.predict(horizon=horizon)

    last_date = LAST_DATES.get(state, pd.Timestamp("2023-12-01"))
    dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-MON",
    )

    return ForecastResponse(
        state=state,
        model_used=BEST_MODEL_MAP.get(state, model.name),
        horizon_weeks=horizon,
        last_training_date=last_date.strftime("%Y-%m-%d"),
        forecasts=[
            WeekForecast(
                week=i + 1,
                date=d.strftime("%Y-%m-%d"),
                predicted_sales=round(float(p), 2),
            )
            for i, (d, p) in enumerate(zip(dates, preds))
        ],
    )


# ── routes ─────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "models_loaded": len(MODELS)}


@app.get("/states", tags=["System"])
def list_states():
    return {"states": STATES, "count": len(STATES)}


@app.get("/models", tags=["System"])
def list_best_models():
    return {
        "best_models": [
            {"state": s, "model": m} for s, m in BEST_MODEL_MAP.items()
        ]
    }


@app.get("/forecast/all", response_model=list[ForecastResponse], tags=["Forecast"])
def forecast_all(horizon: int = Query(8, ge=1, le=52)):
    return [_run_forecast(s, horizon) for s in STATES]


@app.get("/forecast/{state}", response_model=ForecastResponse, tags=["Forecast"])
def forecast_state(state: str, horizon: int = Query(8, ge=1, le=52)):
    # Normalise state name (title case)
    state = state.strip().title()
    return _run_forecast(state, horizon)


@app.post("/forecast/batch", response_model=list[ForecastResponse], tags=["Forecast"])
def forecast_batch(request: BatchRequest):
    horizon = request.horizon or 8
    results = []
    errors = []
    for s in request.states:
        s = s.strip().title()
        try:
            results.append(_run_forecast(s, horizon))
        except HTTPException as e:
            errors.append({"state": s, "error": e.detail})
    if errors and not results:
        raise HTTPException(400, detail={"errors": errors})
    return results
