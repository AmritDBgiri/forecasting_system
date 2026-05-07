"""
models/xgboost_model.py
------------------------
XGBoost regressor with full feature set (lags, rolling stats, calendar,
holiday flag).  Recursive multi-step forecasting for the horizon.
"""

import numpy as np
import pandas as pd
import holidays
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler

from models.base_model import BaseForecaster

US_HOLIDAYS = holidays.US(years=range(2018, 2026))

FEATURE_COLS = [
    "lag_1", "lag_2", "lag_4", "lag_8",
    "roll_mean_4", "roll_std_4",
    "roll_mean_8", "roll_std_8",
    "week_of_year", "month", "quarter", "year", "holiday_week",
]


def _is_holiday_week(date: pd.Timestamp) -> int:
    monday = date - pd.Timedelta(days=date.weekday())
    week_days = [monday + pd.Timedelta(days=i) for i in range(7)]
    return int(any(d.date() in US_HOLIDAYS for d in week_days))


class XGBoostForecaster(BaseForecaster):
    name = "XGBoost"

    def __init__(self):
        self._model = XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method="hist",
            verbosity=0,
        )
        self._scaler = StandardScaler()
        self._history: pd.DataFrame | None = None   # needed for recursive pred

    # ── fit ────────────────────────────────────────────────────────────────
    def fit(self, train: pd.DataFrame) -> None:
        self._history = train[["Date", "Sales"] + FEATURE_COLS].copy()
        data = train.dropna(subset=FEATURE_COLS).copy()

        X = data[FEATURE_COLS].values
        y = data["Sales"].values

        X_scaled = self._scaler.fit_transform(X)
        self._model.fit(X_scaled, y)

    # ── predict ────────────────────────────────────────────────────────────
    def predict(self, horizon: int = 8) -> np.ndarray:
        """Recursive (one-step-ahead) multi-step forecast."""
        if self._history is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        history = self._history.copy()
        preds = []

        for step in range(horizon):
            last_date = history["Date"].max()
            next_date = last_date + pd.Timedelta(weeks=1)
            sales_series = history["Sales"].values

            row = {
                "Date": next_date,
                "lag_1":       sales_series[-1],
                "lag_2":       sales_series[-2] if len(sales_series) >= 2 else sales_series[-1],
                "lag_4":       sales_series[-4] if len(sales_series) >= 4 else sales_series[-1],
                "lag_8":       sales_series[-8] if len(sales_series) >= 8 else sales_series[-1],
                "roll_mean_4": float(np.mean(sales_series[-4:])),
                "roll_std_4":  float(np.std(sales_series[-4:]) if len(sales_series) >= 4 else 0),
                "roll_mean_8": float(np.mean(sales_series[-8:])),
                "roll_std_8":  float(np.std(sales_series[-8:]) if len(sales_series) >= 8 else 0),
                "week_of_year": next_date.isocalendar().week,
                "month":        next_date.month,
                "quarter":      next_date.quarter,
                "year":         next_date.year,
                "holiday_week": _is_holiday_week(next_date),
            }

            X_pred = np.array([[row[c] for c in FEATURE_COLS]])
            X_pred_scaled = self._scaler.transform(X_pred)
            pred = float(self._model.predict(X_pred_scaled)[0])
            pred = max(pred, 0)
            preds.append(pred)

            # Append predicted row back into history for next step
            new_row = pd.DataFrame([{"Date": next_date, "Sales": pred, **{c: row[c] for c in FEATURE_COLS}}])
            history = pd.concat([history, new_row], ignore_index=True)

        return np.array(preds)
