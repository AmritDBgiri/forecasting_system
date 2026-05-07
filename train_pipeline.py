"""
train_pipeline.py
------------------
End-to-end training pipeline.

Usage
-----
    python train_pipeline.py [--states CA,TX,...] [--no-lstm] [--data PATH]

Outputs
-------
artifacts/models.pkl        : dict {state: fitted_model}
artifacts/scores.csv        : model comparison scores per state
artifacts/best_models.csv   : which model won per state
artifacts/forecasts.csv     : 8-week-ahead forecasts for all states
"""
import gpu_init
import argparse
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from data_processor import load_and_clean, add_features
from model_selector import select_best_model

# ── logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

HORIZON = 8     # weeks to forecast


def generate_forecast_dates(last_train_date: pd.Timestamp, horizon: int) -> list[str]:
    """Return a list of ISO date strings for the forecast horizon."""
    dates = pd.date_range(
        start=last_train_date + pd.Timedelta(weeks=1),
        periods=horizon,
        freq="W-MON",
    )
    return [d.strftime("%Y-%m-%d") for d in dates]


def run(states=None, run_lstm=True, data_path=None):
    # ── 1. Load & feature-engineer ─────────────────────────────────────────
    csv = data_path or Path(__file__).parent / "data/sales_data.csv"
    logger.info("Loading data from %s", csv)
    df_clean = load_and_clean(csv)
    df_feat = add_features(df_clean)

    all_states = sorted(df_feat["State"].unique())
    if states:
        all_states = [s for s in all_states if s in states]

    logger.info("Training on %d states | LSTM=%s", len(all_states), run_lstm)

    # ── 2. Train & select best model per state ─────────────────────────────
    models = {}
    all_scores = []
    best_model_records = []
    all_forecasts = []

    for state in all_states:
        try:
            best_model, score_df, _ = select_best_model(
                df_feat, state, run_lstm=run_lstm
            )
            models[state] = best_model

            # Score table
            score_df["State"] = state
            all_scores.append(score_df)

            # Which model won
            best_model_records.append(
                {"State": state, "Best_Model": best_model.name}
            )

            # Generate 8-week forecast
            preds = best_model.predict(horizon=HORIZON)
            last_date = df_feat[df_feat["State"] == state]["Date"].max()
            dates = generate_forecast_dates(last_date, HORIZON)
            for d, p in zip(dates, preds):
                all_forecasts.append(
                    {"State": state, "Forecast_Date": d, "Predicted_Sales": round(p)}
                )

        except Exception as exc:
            logger.error("State %s failed: %s", state, exc)

    # ── 3. Save artifacts ──────────────────────────────────────────────────
    # 3a. Serialise models
    with open(ARTIFACT_DIR / "models.pkl", "wb") as f:
        pickle.dump(models, f)

    # 3b. Score CSV
    score_master = pd.concat(all_scores, ignore_index=True)
    score_master.to_csv(ARTIFACT_DIR / "scores.csv", index=False)

    # 3c. Best models CSV
    best_df = pd.DataFrame(best_model_records)
    best_df.to_csv(ARTIFACT_DIR / "best_models.csv", index=False)

    # 3d. Forecast CSV
    forecast_df = pd.DataFrame(all_forecasts)
    forecast_df.to_csv(ARTIFACT_DIR / "forecasts.csv", index=False)

    logger.info("Training complete. Artifacts saved to %s/", ARTIFACT_DIR)
    logger.info(
        "Best model distribution:\n%s",
        best_df["Best_Model"].value_counts().to_string(),
    )
    return models, forecast_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train forecasting pipeline")
    parser.add_argument("--states", default=None, help="Comma-separated list of states")
    parser.add_argument("--no-lstm", action="store_true", help="Skip LSTM (faster)")
    parser.add_argument("--data", default=None, help="Path to sales CSV")
    args = parser.parse_args()

    state_list = [s.strip() for s in args.states.split(",")] if args.states else None
    run(states=state_list, run_lstm=not args.no_lstm, data_path=args.data)
