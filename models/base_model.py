"""
models/base_model.py
---------------------
Abstract base class that every forecasting model must implement.
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """Common interface for all forecasting models."""

    name: str = "BaseForecaster"

    @abstractmethod
    def fit(self, train: pd.DataFrame) -> None:
        """Train the model on *train* data."""
        ...

    @abstractmethod
    def predict(self, horizon: int) -> np.ndarray:
        """Return an array of *horizon* future point forecasts."""
        ...

    # ── Shared metric ──────────────────────────────────────────────────────
    @staticmethod
    def smape(actual: np.ndarray, predicted: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error (0-100 scale)."""
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        denom = (np.abs(actual) + np.abs(predicted)) / 2
        mask = denom > 0
        return float(
            100 * np.mean(np.abs(actual[mask] - predicted[mask]) / denom[mask])
        )

    @staticmethod
    def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
        actual = np.asarray(actual, dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        return float(np.sqrt(np.mean((actual - predicted) ** 2)))
