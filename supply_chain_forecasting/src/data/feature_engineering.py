"""
feature_engineering.py
Builds lag features, rolling statistics, calendar features, and promo flags
for SKU-level demand forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Full feature engineering pipeline for one or all SKUs."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(["sku_id", "date"], inplace=True)

    # ── Calendar features
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"]        = df["date"].dt.month
    df["quarter"]      = df["date"].dt.quarter
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["year"]         = df["date"].dt.year

    # ── Lag features (grouped by SKU)─
    for lag in [7, 14, 21, 28]:
        df[f"lag_{lag}d"] = df.groupby("sku_id")["demand"].shift(lag)

    # ── Rolling statistics 
    for window in [7, 14, 28]:
        grp = df.groupby("sku_id")["demand"]
        df[f"roll_mean_{window}d"] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"roll_std_{window}d"] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
        )
        df[f"roll_max_{window}d"] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max()
        )

    # ── Exponential weighted mean 
    df["ewm_14d"] = df.groupby("sku_id")["demand"].transform(
        lambda x: x.shift(1).ewm(span=14, min_periods=1).mean()
    )

    # ── Promo lag / lead 
    df["promo_lag_7d"]  = df.groupby("sku_id")["promo_flag"].shift(7).fillna(0)
    df["promo_lead_7d"] = df.groupby("sku_id")["promo_flag"].shift(-7).fillna(0)

    # ── Drop rows with NaN lags 
    df.dropna(subset=[f"lag_{lag}d" for lag in [7, 14, 21, 28]], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


FEATURE_COLS = [
    "day_of_week", "day_of_month", "week_of_year", "month", "quarter",
    "is_weekend", "year",
    "lag_7d", "lag_14d", "lag_21d", "lag_28d",
    "roll_mean_7d", "roll_std_7d", "roll_max_7d",
    "roll_mean_14d", "roll_std_14d", "roll_max_14d",
    "roll_mean_28d", "roll_std_28d", "roll_max_28d",
    "ewm_14d",
    "promo_flag", "promo_lag_7d", "promo_lead_7d",
]

TARGET_COL = "demand"


def load_and_engineer(raw_path: str = "data/raw/sku_demand.csv") -> pd.DataFrame:
    df = pd.read_csv(raw_path, parse_dates=["date"])
    df = build_features(df)
    out_path = Path("data/processed/sku_features.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[✓] Feature-engineered dataset: {df.shape} → {out_path}")
    return df


if __name__ == "__main__":
    load_and_engineer()
