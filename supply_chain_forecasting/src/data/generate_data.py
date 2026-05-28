"""
generate_data.py
Generates synthetic SKU-level demand data with realistic supply chain patterns.
Saves raw CSV to data/raw/sku_demand.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

CATEGORIES = {
    "Apparel":       ["SKU_APP_001", "SKU_APP_002", "SKU_APP_003", "SKU_APP_004", "SKU_APP_005",
                      "SKU_APP_006", "SKU_APP_007", "SKU_APP_008", "SKU_APP_009", "SKU_APP_010"],
    "Footwear":      ["SKU_FTW_001", "SKU_FTW_002", "SKU_FTW_003", "SKU_FTW_004", "SKU_FTW_005",
                      "SKU_FTW_006", "SKU_FTW_007", "SKU_FTW_008", "SKU_FTW_009", "SKU_FTW_010"],
    "Electronics":   ["SKU_ELC_001", "SKU_ELC_002", "SKU_ELC_003", "SKU_ELC_004", "SKU_ELC_005",
                      "SKU_ELC_006", "SKU_ELC_007", "SKU_ELC_008", "SKU_ELC_009", "SKU_ELC_010"],
    "Home_Decor":    ["SKU_HOM_001", "SKU_HOM_002", "SKU_HOM_003", "SKU_HOM_004", "SKU_HOM_005",
                      "SKU_HOM_006", "SKU_HOM_007", "SKU_HOM_008", "SKU_HOM_009", "SKU_HOM_010"],
    "Accessories":   ["SKU_ACC_001", "SKU_ACC_002", "SKU_ACC_003", "SKU_ACC_004", "SKU_ACC_005",
                      "SKU_ACC_006", "SKU_ACC_007", "SKU_ACC_008", "SKU_ACC_009", "SKU_ACC_010"],
    "Sports":        ["SKU_SPT_001", "SKU_SPT_002", "SKU_SPT_003", "SKU_SPT_004", "SKU_SPT_005",
                      "SKU_SPT_006", "SKU_SPT_007", "SKU_SPT_008", "SKU_SPT_009", "SKU_SPT_010"],
}

DATE_RANGE = pd.date_range(start="2022-01-01", end="2024-12-31", freq="D")

PROMO_EVENTS = [
    ("2022-11-25", "2022-11-28"),  # Black Friday 2022
    ("2022-12-20", "2022-12-25"),  # Christmas 2022
    ("2023-07-15", "2023-07-17"),  # Summer Sale
    ("2023-11-24", "2023-11-27"),  # Black Friday 2023
    ("2023-12-20", "2023-12-25"),  # Christmas 2023
    ("2024-07-14", "2024-07-16"),  # Summer Sale 2024
    ("2024-11-29", "2024-12-02"),  # Black Friday 2024
    ("2024-12-20", "2024-12-25"),  # Christmas 2024
]


def is_promo(date):
    for start, end in PROMO_EVENTS:
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return 1
    return 0


def generate_sku_demand(sku_id, category, dates):
    n = len(dates)
    base_demand = np.random.randint(80, 250)
    trend = np.linspace(0, np.random.uniform(-10, 30), n)

    # Seasonal pattern: annual cycle
    day_of_year = np.array([d.day_of_year for d in dates])
    season = 20 * np.sin(2 * np.pi * day_of_year / 365 + np.random.uniform(0, np.pi))

    # Weekly pattern: weekends spike
    dow = np.array([d.dayofweek for d in dates])
    weekly = np.where(dow >= 5, np.random.uniform(10, 25), np.random.uniform(-5, 5))

    # Promo lift
    promo_flag = np.array([is_promo(d) for d in dates])
    promo_lift = promo_flag * np.random.uniform(40, 100)

    # Stockout simulation: random zero-demand days (~2%)
    stockout = np.random.choice([0, 1], size=n, p=[0.02, 0.98])

    # Noise
    noise = np.random.normal(0, 12, n)

    demand = (base_demand + trend + season + weekly + promo_lift + noise) * stockout
    demand = np.clip(demand, 0, None).round().astype(int)

    # Supply chain metadata
    lead_time = np.random.randint(7, 21)
    unit_cost = round(np.random.uniform(5, 80), 2)
    holding_cost_pct = round(np.random.uniform(0.15, 0.30), 2)
    service_level = round(np.random.uniform(0.90, 0.99), 2)

    df = pd.DataFrame({
        "date": dates,
        "sku_id": sku_id,
        "category": category,
        "demand": demand,
        "promo_flag": promo_flag,
        "lead_time_days": lead_time,
        "unit_cost_usd": unit_cost,
        "holding_cost_pct": holding_cost_pct,
        "service_level_target": service_level,
    })
    return df


def main():
    all_dfs = []
    for category, skus in CATEGORIES.items():
        for sku in skus:
            df = generate_sku_demand(sku, category, DATE_RANGE)
            all_dfs.append(df)

    full_df = pd.concat(all_dfs, ignore_index=True)
    full_df.sort_values(["sku_id", "date"], inplace=True)

    out_path = Path("data/raw/sku_demand.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"[✓] Generated {len(full_df):,} rows across {full_df['sku_id'].nunique()} SKUs → {out_path}")
    return full_df


if __name__ == "__main__":
    main()
