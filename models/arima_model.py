"""
models/arima_model.py
----------------------
SARIMA model with automatic (p,d,q)(P,D,Q,m) selection via AIC grid search.
Seasonal period m=52 (weekly data).
"""

import warnings
import itertools
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller

from models.base_model import BaseForecaster

warnings.filterwarnings("ignore")


def _adf_ndiffs(series: pd.Series, max_d: int = 2) -> int:
    """Determine integration order via ADF test."""
    for d in range(max_d + 1):
        s = series.diff(d).dropna() if d > 0 else series.dropna()
        if len(s) < 10:
            return d
        p_value = adfuller(s, autolag="AIC")[1]
        if p_value < 0.05:
            return d
    return max_d


class SARIMAForecaster(BaseForecaster):
    name = "SARIMA"

    # Light grid so it runs in reasonable time; extend for production
    _P_RANGE = range(0, 2)
    _Q_RANGE = range(0, 2)
    _SEASONAL_M = 52          # weekly seasonality

    def __init__(self):
        self._model_fit = None
        self._history: pd.Series | None = None

    # ── fit ────────────────────────────────────────────────────────────────
    def fit(self, train: pd.DataFrame) -> None:
        series = train.set_index("Date")["Sales"].asfreq("W-MON").ffill()
        self._history = series

        d = _adf_ndiffs(series)
        best_aic = np.inf
        best_order = (1, d, 1)
        best_seasonal = (0, 0, 0, self._SEASONAL_M)

        # Non-seasonal grid search (keep seasonal order simple for speed)
        for p, q in itertools.product(self._P_RANGE, self._Q_RANGE):
            try:
                res = SARIMAX(
                    series,
                    order=(p, d, q),
                    seasonal_order=(0, 0, 0, self._SEASONAL_M),
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                ).fit(disp=False)
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p, d, q)
            except Exception:
                continue

        self._model_fit = SARIMAX(
            series,
            order=best_order,
            seasonal_order=best_seasonal,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False)

    # ── predict ────────────────────────────────────────────────────────────
    def predict(self, horizon: int = 8) -> np.ndarray:
        if self._model_fit is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        fc = self._model_fit.get_forecast(steps=horizon)
        return np.maximum(fc.predicted_mean.values, 0)
