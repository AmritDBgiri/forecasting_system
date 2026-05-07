"""
models/prophet_model.py
------------------------
Facebook Prophet with US holiday effects and weekly/yearly seasonality.
"""

import warnings
import numpy as np
import pandas as pd
from prophet import Prophet

from models.base_model import BaseForecaster

warnings.filterwarnings("ignore")


class ProphetForecaster(BaseForecaster):
    name = "Prophet"

    def __init__(self):
        self._model: Prophet | None = None
        self._last_date: pd.Timestamp | None = None

    # ── fit ────────────────────────────────────────────────────────────────
    def fit(self, train: pd.DataFrame) -> None:
        prophet_df = (
            train[["Date", "Sales"]]
            .rename(columns={"Date": "ds", "Sales": "y"})
            .copy()
        )
        self._last_date = prophet_df["ds"].max()

        self._model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode="multiplicative",
            changepoint_prior_scale=0.05,
        )
        self._model.add_country_holidays(country_name="US")
        self._model.fit(prophet_df)

    # ── predict ────────────────────────────────────────────────────────────
    def predict(self, horizon: int = 8) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        future = self._model.make_future_dataframe(periods=horizon, freq="W")
        forecast = self._model.predict(future)
        return np.maximum(forecast["yhat"].values[-horizon:], 0)
