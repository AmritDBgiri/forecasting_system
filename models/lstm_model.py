"""
models/lstm_model.py
---------------------
Bidirectional LSTM with dropout for weekly sales forecasting.
Uses a sliding-window approach (look-back = 12 weeks) and
recursive multi-step prediction.
"""
import gpu_init
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.set_memory_growth(gpus[0], True)

warnings.filterwarnings("ignore")

# Lazy-import TensorFlow to avoid slowing down other modules
def _get_tf():
    import os
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    return tf

from models.base_model import BaseForecaster

LOOKBACK = 12       # weeks of history fed into LSTM


class LSTMForecaster(BaseForecaster):
    name = "LSTM"

    def __init__(self):
        self._model = None
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._last_window: np.ndarray | None = None   # shape (LOOKBACK,)

    # ── helpers ────────────────────────────────────────────────────────────
    @staticmethod
    def _make_sequences(data: np.ndarray, lookback: int):
        X, y = [], []
        for i in range(len(data) - lookback):
            X.append(data[i : i + lookback])
            y.append(data[i + lookback])
        return np.array(X), np.array(y)

    # ── fit ────────────────────────────────────────────────────────────────
    def fit(self, train: pd.DataFrame) -> None:
        tf = _get_tf()
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping

        sales = train.sort_values("Date")["Sales"].values.reshape(-1, 1)
        scaled = self._scaler.fit_transform(sales).flatten()

        X, y = self._make_sequences(scaled, LOOKBACK)
        X = X.reshape(X.shape[0], X.shape[1], 1)   # (samples, timesteps, features)

        self._model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True, input_shape=(LOOKBACK, 1))),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        self._model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss="mse")

        early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
        self._model.fit(
            X, y,
            epochs=80,
            batch_size=16,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=0,
        )

        # Store the last window for recursive forecasting
        self._last_window = scaled[-LOOKBACK:]

    # ── predict ────────────────────────────────────────────────────────────
    def predict(self, horizon: int = 8) -> np.ndarray:
        if self._model is None or self._last_window is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        window = self._last_window.copy()
        preds_scaled = []

        for _ in range(horizon):
            x = window[-LOOKBACK:].reshape(1, LOOKBACK, 1)
            pred_scaled = float(self._model.predict(x, verbose=0)[0, 0])
            preds_scaled.append(pred_scaled)
            window = np.append(window, pred_scaled)

        preds = self._scaler.inverse_transform(
            np.array(preds_scaled).reshape(-1, 1)
        ).flatten()
        return np.maximum(preds, 0)
