"""
model_selector.py
------------------
Trains all four models on a state's training data, evaluates them on the
validation split, and returns the best model together with a results table.
"""

import logging
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from data_processor import time_series_split
from models.base_model import BaseForecaster
from models.arima_model import SARIMAForecaster
from models.prophet_model import ProphetForecaster
from models.xgboost_model import XGBoostForecaster
from models.lstm_model import LSTMForecaster

logger = logging.getLogger(__name__)

VAL_WEEKS = 8      # hold-out period matches forecast horizon


@dataclass
class ModelResult:
    model_name: str
    smape: float
    rmse: float
    model_obj: Any = field(repr=False)
    val_predictions: np.ndarray = field(repr=False)
    val_actuals: np.ndarray = field(repr=False)


def _safe_fit_predict(
    forecaster: BaseForecaster,
    train: pd.DataFrame,
    val: pd.DataFrame,
) -> ModelResult | None:
    """Fit a model and score it on validation; return None on failure."""
    try:
        forecaster.fit(train)
        preds = forecaster.predict(horizon=len(val))
        actuals = val["Sales"].values

        smape = forecaster.smape(actuals, preds)
        rmse = forecaster.rmse(actuals, preds)

        logger.info("  [%s] SMAPE=%.2f%%  RMSE=%.0f", forecaster.name, smape, rmse)
        return ModelResult(
            model_name=forecaster.name,
            smape=smape,
            rmse=rmse,
            model_obj=forecaster,
            val_predictions=preds,
            val_actuals=actuals,
        )
    except Exception as exc:
        logger.warning("  [%s] FAILED: %s", forecaster.name, exc)
        logger.debug(traceback.format_exc())
        return None


def select_best_model(
    df_featured: pd.DataFrame,
    state: str,
    run_lstm: bool = True,
) -> tuple[BaseForecaster, pd.DataFrame, list[ModelResult]]:
    """
    For a single *state*, train all models, pick the winner by lowest SMAPE.

    Parameters
    ----------
    df_featured : DataFrame with feature columns from data_processor.add_features
    state       : state name string
    run_lstm    : set False to skip the (slow) LSTM during quick runs

    Returns
    -------
    best_model  : fitted BaseForecaster ready to call .predict()
    score_table : DataFrame comparing all models
    all_results : list of ModelResult (including failed ones as None)
    """
    logger.info("── %s ──", state)
    train, val = time_series_split(df_featured, state, val_weeks=VAL_WEEKS)

    candidates = [
        SARIMAForecaster(),
        ProphetForecaster(),
        XGBoostForecaster(),
    ]
    if run_lstm:
        candidates.append(LSTMForecaster())

    results = []
    for model in candidates:
        r = _safe_fit_predict(model, train, val)
        if r is not None:
            results.append(r)

    if not results:
        raise RuntimeError(f"All models failed for state: {state}")

    best = min(results, key=lambda r: r.smape)
    logger.info("  ✓ Best model for %s → %s (SMAPE %.2f%%)", state, best.model_name, best.smape)

    # Re-train winner on FULL data (train + val) for final forecasting
    full_data = df_featured[df_featured["State"] == state].dropna()
    best.model_obj.fit(full_data)

    score_df = pd.DataFrame(
        [{"Model": r.model_name, "SMAPE (%)": round(r.smape, 2), "RMSE": round(r.rmse, 0)}
         for r in results]
    ).sort_values("SMAPE (%)").reset_index(drop=True)

    return best.model_obj, score_df, results
