"""
train_evaluate.py
Full end-to-end training pipeline:
  1. Load + engineer features
  2. Train LSTM, XGBoost, ARIMA per SKU
  3. Evaluate on held-out test set
  4. Generate 28-day forward forecasts
  5. Run inventory optimization
  6. Save all outputs
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import time
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")

from src.data.generate_data import main as generate_data
from src.data.feature_engineering import load_and_engineer, FEATURE_COLS, TARGET_COL
from src.models.lstm_model import SimpleLSTM
from src.models.ml_models import XGBDemandForecaster, ARIMAForecaster
from src.models.metrics import evaluate_all, summarize_results
from src.optimization.inventory_optimizer import run_optimization


TRAIN_CUTOFF = "2024-09-30"
TEST_CUTOFF  = "2024-12-31"
FORECAST_HORIZON = 28
SAVED_MODELS = Path("models/saved")
OUTPUTS      = Path("outputs")


def train_and_evaluate():
    t0 = time.time()
    print("\n" + "="*60)
    print("  SUPPLY CHAIN FORECASTING & OPTIMIZATION PIPELINE")
    print("="*60)

    # ── Step 1: Data 
    print("\n[1/6] Generating synthetic supply chain data...")
    raw_path = Path("data/raw/sku_demand.csv")
    if not raw_path.exists():
        generate_data()
    else:
        print(f"      Using existing data at {raw_path}")

    print("\n[2/6] Feature engineering...")
    feat_path = Path("data/processed/sku_features.csv")
    if not feat_path.exists():
        df = load_and_engineer(str(raw_path))
    else:
        df = pd.read_csv(feat_path, parse_dates=["date"])
        print(f"      Loaded features: {df.shape}")

    skus = sorted(df["sku_id"].unique())
    print(f"      SKUs: {len(skus)}")

    # ── Step 2: Train / Test split 
    df_train = df[df["date"] <= TRAIN_CUTOFF].copy()
    df_test  = df[(df["date"] > TRAIN_CUTOFF) & (df["date"] <= TEST_CUTOFF)].copy()
    print(f"\n[3/6] Train: {df_train['date'].min().date()} → {df_train['date'].max().date()}"
          f"  |  Test: {df_test['date'].min().date()} → {df_test['date'].max().date()}")

    # ── Step 3: XGBoost
    print("\n[4/6] Training models...")
    print("      → XGBoost (all SKUs)...")
    xgb_model = XGBDemandForecaster(FEATURE_COLS)
    xgb_model.fit(df_train)
    xgb_model.save(str(SAVED_MODELS / "xgb_model.pkl"))

    xgb_preds_test = xgb_model.predict(df_test)
    xgb_metrics    = evaluate_all(df_test[TARGET_COL].values, xgb_preds_test, "XGBoost")

    # ── Step 4: ARIMA + LSTM per SKU
    arima_model = ARIMAForecaster()
    lstm_results, arima_results = [], []
    lstm_forecasts, xgb_forecasts, arima_forecasts = {}, {}, {}

    print(f"      → ARIMA + LSTM per SKU ({len(skus)} SKUs)...")

    sku_meta_rows = []

    for i, sku in enumerate(skus):
        sku_train = df_train[df_train["sku_id"] == sku].copy()
        sku_test  = df_test[df_test["sku_id"] == sku].copy()

        if sku_train.empty or sku_test.empty:
            continue

        demand_series = sku_train[TARGET_COL].values

        # ARIMA
        arima_model.fit_sku(sku, sku_train.set_index("date")[TARGET_COL])
        arima_fc = arima_model.predict_sku(sku, steps=len(sku_test))
        ar_m = evaluate_all(sku_test[TARGET_COL].values, arima_fc, "ARIMA")
        ar_m["sku_id"] = sku
        arima_results.append(ar_m)
        arima_forecasts[sku] = arima_model.predict_sku(sku, FORECAST_HORIZON)

        # LSTM
        lstm = SimpleLSTM(seq_len=28, hidden_size=32, epochs=20, lr=0.001)
        lstm.fit(demand_series)
        lstm_test_preds = []
        test_series = demand_series.copy()
        for _ in range(len(sku_test)):
            p = lstm.predict_next(test_series, steps=1)[0]
            lstm_test_preds.append(p)
            test_series = np.append(test_series, p)
        lstm_test_preds = np.clip(lstm_test_preds, 0, None)
        lt_m = evaluate_all(sku_test[TARGET_COL].values, lstm_test_preds, "LSTM")
        lt_m["sku_id"] = sku
        lstm_results.append(lt_m)

        full_series = df[df["sku_id"] == sku][TARGET_COL].values
        lstm_forecasts[sku] = np.clip(lstm.predict_next(full_series, FORECAST_HORIZON), 0, None)
        lstm.save(str(SAVED_MODELS / f"lstm_{sku}.pkl"))

        # XGBoost future forecast: repeat last known features
        last_feat = sku_test[FEATURE_COLS].iloc[-1:].copy()
        xgb_forecasts[sku] = float(xgb_model.predict(last_feat)[0]) * FORECAST_HORIZON

        # Metadata for optimization
        meta = sku_train[["sku_id", "category", "lead_time_days",
                           "unit_cost_usd", "holding_cost_pct",
                           "service_level_target"]].iloc[0].to_dict()
        sku_meta_rows.append(meta)

        if (i + 1) % 10 == 0:
            print(f"         {i+1}/{len(skus)} SKUs processed...")

    arima_model.save(str(SAVED_MODELS / "arima_model.pkl"))

    # ── Step 5: Aggregate metrics
    print("\n[5/6] Evaluating & comparing models...")

    arima_df = pd.DataFrame(arima_results)
    lstm_df  = pd.DataFrame(lstm_results)

    overall_arima_mape = arima_df["MAPE"].mean()
    overall_lstm_mape  = lstm_df["MAPE"].mean()
    overall_xgb_mape   = xgb_metrics["MAPE"]

    summary_rows = [
        {"model": "ARIMA (baseline)", "MAPE": round(overall_arima_mape, 2),
         "RMSE": round(arima_df["RMSE"].mean(), 2), "MAE": round(arima_df["MAE"].mean(), 2)},
        {"model": "LSTM",             "MAPE": round(overall_lstm_mape, 2),
         "RMSE": round(lstm_df["RMSE"].mean(), 2),  "MAE": round(lstm_df["MAE"].mean(), 2)},
        {"model": "XGBoost",          "MAPE": round(overall_xgb_mape, 2),
         "RMSE": round(xgb_metrics["RMSE"], 2),     "MAE": round(xgb_metrics["MAE"], 2)},
    ]
    summary_df = pd.DataFrame(summary_rows).sort_values("MAPE")
    print("\n" + summary_df.to_string(index=False))

    best_model = summary_df.iloc[0]["model"]
    best_mape  = summary_df.iloc[0]["MAPE"]
    mape_improvement = round(((overall_arima_mape - best_mape) / overall_arima_mape) * 100, 1)
    print(f"\n  Best model: {best_model}  |  MAPE improvement over ARIMA: {mape_improvement}%")

    # ── Step 6: Optimization 
    print("\n[6/6] Running inventory optimization...")

    sku_meta_df = pd.DataFrame(sku_meta_rows)

    fc_records = []
    for sku in skus:
        fc_28d = lstm_forecasts.get(sku, np.zeros(FORECAST_HORIZON))
        demand_std = float(np.std(fc_28d)) if fc_28d.sum() > 0 else 5.0
        fc_records.append({
            "sku_id":          sku,
            "forecast_demand": float(np.sum(fc_28d)),
            "demand_std":      demand_std,
        })
    forecast_agg_df = pd.DataFrame(fc_records)

    opt_df = run_optimization(forecast_agg_df, sku_meta_df)
    risk_summary = opt_df["fulfilment_risk"].value_counts()
    print(f"  Fulfilment Risk Distribution:\n{risk_summary.to_string()}")

    # ── Save outputs 
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    (OUTPUTS / "forecasts").mkdir(exist_ok=True)
    (OUTPUTS / "reports").mkdir(exist_ok=True)

    # Per-SKU forecast table
    fc_detail_rows = []
    start_date = pd.Timestamp("2025-01-01")
    for sku in skus:
        fc_arr = lstm_forecasts.get(sku, np.zeros(FORECAST_HORIZON))
        for d, val in enumerate(fc_arr):
            fc_detail_rows.append({
                "sku_id":          sku,
                "forecast_date":   start_date + pd.Timedelta(days=d),
                "lstm_forecast":   round(max(val, 0), 1),
                "arima_forecast":  round(max(arima_forecasts.get(sku, [0]*FORECAST_HORIZON)[d], 0), 1),
            })
    forecast_detail_df = pd.DataFrame(fc_detail_rows)
    forecast_detail_df.to_csv(OUTPUTS / "forecasts" / "28day_forecast.csv", index=False)

    # Optimization output
    opt_df.to_csv(OUTPUTS / "reports" / "inventory_optimization.csv", index=False)

    # Model comparison
    summary_df.to_csv(OUTPUTS / "reports" / "model_comparison.csv", index=False)

    # Per-SKU ARIMA metrics
    arima_df.to_csv(OUTPUTS / "reports" / "arima_sku_metrics.csv", index=False)

    # Feature importance
    fi = xgb_model.feature_importance().reset_index()
    fi.columns = ["feature", "importance"]
    fi.to_csv(OUTPUTS / "reports" / "feature_importance.csv", index=False)

    elapsed = round(time.time() - t0, 1)
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed}s")
    print(f"  Forecasts → {OUTPUTS}/forecasts/")
    print(f"  Reports   → {OUTPUTS}/reports/")
    print(f"{'='*60}\n")

    return {
        "summary":            summary_df,
        "optimization":       opt_df,
        "forecast_detail":    forecast_detail_df,
        "mape_improvement":   mape_improvement,
        "best_model":         best_model,
    }


if __name__ == "__main__":
    train_and_evaluate()
