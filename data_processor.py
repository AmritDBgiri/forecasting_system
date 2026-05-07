"""
data_processor.py
-----------------
Loads the raw CSV, cleans it, handles missing dates / values,
and engineers all required features:
  - Lag features  (t-1, t-2, t-4, t-8 periods → weekly data)
  - Rolling mean / std  (4-week, 8-week windows)
  - Calendar features   (week-of-year, month, quarter, year)
  - US holiday flag
"""

import pandas as pd
import numpy as np
import holidays
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# ── US public holidays helper ──────────────────────────────────────────────
US_HOLIDAYS = holidays.US(years=range(2018, 2026))


def _is_holiday_week(date: pd.Timestamp) -> int:
    """Return 1 if any day in the ISO week of *date* is a US public holiday."""
    monday = date - pd.Timedelta(days=date.weekday())
    week_days = [monday + pd.Timedelta(days=i) for i in range(7)]
    return int(any(d.date() in US_HOLIDAYS for d in week_days))


# ── Core loader ────────────────────────────────────────────────────────────
def load_and_clean(csv_path: str | Path) -> pd.DataFrame:
    """
    Read the raw CSV, normalise columns, and return a tidy weekly DataFrame
    with one row per (State, Date) sorted chronologically.
    """
    df = pd.read_csv(csv_path)

    # --- 1. Column names ---
    df.columns = df.columns.str.strip()

    # --- 2. Numeric sales ---
    df["Sales"] = (
        df["Total"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
        .astype(float)
    )
    df.drop(columns=["Total", "Category"], inplace=True)

    # --- 3. Parse dates (mixed formats present in the raw file) ---
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)

    # --- 4. Normalise to week-start (Monday) so we get a clean weekly grid ---
    df["Date"] = df["Date"] - pd.to_timedelta(df["Date"].dt.weekday, unit="D")

    # --- 5. Deduplicate (sum if same state/week appears twice) ---
    df = df.groupby(["State", "Date"], as_index=False)["Sales"].sum()

    # --- 6. Fill missing weeks with 0 then interpolate ---
    df = _fill_missing_weeks(df)

    df.sort_values(["State", "Date"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info(
        "Loaded %d rows | %d states | %s → %s",
        len(df),
        df["State"].nunique(),
        df["Date"].min().date(),
        df["Date"].max().date(),
    )
    return df


def _fill_missing_weeks(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure every state has a continuous weekly date range; fill gaps."""
    min_date = df["Date"].min()
    max_date = df["Date"].max()
    full_weeks = pd.date_range(min_date, max_date, freq="W-MON")

    filled_parts = []
    for state, grp in df.groupby("State"):
        grp = grp.set_index("Date").reindex(full_weeks)
        grp["State"] = state
        # Linear interpolation for interior gaps; forward/back fill at edges
        grp["Sales"] = (
            grp["Sales"]
            .interpolate(method="linear")
            .ffill()
            .bfill()
        )
        filled_parts.append(grp.reset_index().rename(columns={"index": "Date"}))

    return pd.concat(filled_parts, ignore_index=True)


# ── Feature engineering ────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all ML / DL features to a cleaned (State, Date, Sales) DataFrame.
    Features are computed *per state* so there is no cross-state leakage.
    """
    parts = []
    for state, grp in df.groupby("State"):
        grp = grp.sort_values("Date").copy()

        # Lag features (in weeks)
        for lag in [1, 2, 4, 8]:
            grp[f"lag_{lag}"] = grp["Sales"].shift(lag)

        # Rolling statistics (min_periods so early rows keep some signal)
        for window in [4, 8]:
            grp[f"roll_mean_{window}"] = (
                grp["Sales"].shift(1).rolling(window, min_periods=2).mean()
            )
            grp[f"roll_std_{window}"] = (
                grp["Sales"].shift(1).rolling(window, min_periods=2).std().fillna(0)
            )

        # Calendar features
        grp["week_of_year"] = grp["Date"].dt.isocalendar().week.astype(int)
        grp["month"] = grp["Date"].dt.month
        grp["quarter"] = grp["Date"].dt.quarter
        grp["year"] = grp["Date"].dt.year

        # Holiday flag
        grp["holiday_week"] = grp["Date"].apply(_is_holiday_week)

        parts.append(grp)

    result = pd.concat(parts, ignore_index=True)
    result.sort_values(["State", "Date"], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


# ── Train / validation split (no leakage) ─────────────────────────────────
def time_series_split(
    df: pd.DataFrame,
    state: str,
    val_weeks: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For a single state, return (train_df, val_df) using the last *val_weeks*
    rows as validation.  All feature rows with NaN lag values are dropped from
    training to prevent leakage.
    """
    grp = df[df["State"] == state].sort_values("Date").copy()
    grp.dropna(inplace=True)          # drop early rows where lags are NaN
    train = grp.iloc[:-val_weeks]
    val = grp.iloc[-val_weeks:]
    return train, val
