"""
run_pipeline.py
Fast full pipeline runner — generates all outputs in ~3 minutes.
Uses reduced LSTM epochs for speed while maintaining quality.
"""

import sys, os, warnings, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path

from src.data.generate_data import main as generate_data
from src.data.feature_engineering import load_and_engineer, FEATURE_COLS, TARGET_COL
from src.models.lstm_model import SimpleLSTM
from src.models.ml_models import XGBDemandForecaster, ARIMAForecaster
from src.models.metrics import evaluate_all
from src.optimization.inventory_optimizer import run_optimization

TRAIN_CUTOFF     = "2024-09-30"
FORECAST_HORIZON = 28
SAVED            = Path("models/saved")
OUTPUTS          = Path("outputs")
SAVED.mkdir(parents=True, exist_ok=True)
(OUTPUTS / "forecasts").mkdir(parents=True, exist_ok=True)
(OUTPUTS / "reports").mkdir(parents=True, exist_ok=True)

t0 = time.time()
print("\n" + "="*60)
print("  SUPPLY CHAIN FORECASTING & OPTIMIZATION PIPELINE")
print("="*60)

# ── Data 
print("\n[1/6] Generating data...")
generate_data()

print("[2/6] Feature engineering...")
df = load_and_engineer("data/raw/sku_demand.csv")
skus = sorted(df["sku_id"].unique())
print(f"      {len(skus)} SKUs · {len(df):,} rows")

# ── Split 
df_train = df[df["date"] <= TRAIN_CUTOFF].copy()
df_test  = df[df["date"] >  TRAIN_CUTOFF].copy()
print(f"[3/6] Train rows: {len(df_train):,} | Test rows: {len(df_test):,}")

# ── XGBoost 
print("[4/6] Training XGBoost...")
xgb = XGBDemandForecaster(FEATURE_COLS)
xgb.fit(df_train)
xgb.save(str(SAVED / "xgb_model.pkl"))
xgb_test_preds = xgb.predict(df_test)
xgb_m = evaluate_all(df_test[TARGET_COL].values, xgb_test_preds, "XGBoost")

# ── Feature importance 
fi = xgb.feature_importance().reset_index()
fi.columns = ["feature", "importance"]
fi.to_csv(OUTPUTS / "reports" / "feature_importance.csv", index=False)

# ── Per-SKU: LSTM + ARIMA 
print(f"[5/6] Training LSTM + ARIMA per SKU ({len(skus)} SKUs)...")
arima_model = ARIMAForecaster()
lstm_metrics, arima_metrics = [], []
lstm_fcs, arima_fcs = {}, {}
sku_meta_rows = []

for i, sku in enumerate(skus):
    tr  = df_train[df_train["sku_id"] == sku]
    te  = df_test[df_test["sku_id"]   == sku]
    if tr.empty or te.empty:
        continue

    series = tr[TARGET_COL].values

    # ARIMA
    arima_model.fit_sku(sku, tr.set_index("date")[TARGET_COL])
    ar_te  = arima_model.predict_sku(sku, len(te))
    ar_m   = evaluate_all(te[TARGET_COL].values, ar_te, "ARIMA")
    ar_m["sku_id"] = sku
    arima_metrics.append(ar_m)
    arima_fcs[sku] = arima_model.predict_sku(sku, FORECAST_HORIZON)

    # LSTM (fast: seq_len=7, epochs=3)
    lstm = SimpleLSTM(seq_len=7, hidden_size=8, epochs=3, lr=0.01)
    lstm.fit(series)
    # Single-shot test preds (fast approximation)
    lt_preds = np.clip(lstm.predict_next(series, len(te)), 0, None)
    lt_m = evaluate_all(te[TARGET_COL].values, lt_preds, "LSTM")
    lt_m["sku_id"] = sku
    lstm_metrics.append(lt_m)

    full_series = df[df["sku_id"] == sku][TARGET_COL].values
    lstm_fcs[sku] = np.clip(lstm.predict_next(full_series, FORECAST_HORIZON), 0, None)
    lstm.save(str(SAVED / f"lstm_{sku}.pkl"))

    # Metadata
    meta = tr[["sku_id","category","lead_time_days",
               "unit_cost_usd","holding_cost_pct","service_level_target"]].iloc[0].to_dict()
    sku_meta_rows.append(meta)

    if (i+1) % 15 == 0:
        print(f"      {i+1}/{len(skus)} done...")

arima_model.save(str(SAVED / "arima_model.pkl"))

# ── Model comparison 
ar_df = pd.DataFrame(arima_metrics)
lt_df = pd.DataFrame(lstm_metrics)

summary = pd.DataFrame([
    {"model": "ARIMA (baseline)", "MAPE": round(ar_df["MAPE"].mean(),2),
     "RMSE": round(ar_df["RMSE"].mean(),2), "MAE": round(ar_df["MAE"].mean(),2)},
    {"model": "LSTM",             "MAPE": round(lt_df["MAPE"].mean(),2),
     "RMSE": round(lt_df["RMSE"].mean(),2), "MAE": round(lt_df["MAE"].mean(),2)},
    {"model": "XGBoost",          "MAPE": round(xgb_m["MAPE"],2),
     "RMSE": round(xgb_m["RMSE"],2),        "MAE": round(xgb_m["MAE"],2)},
]).sort_values("MAPE")

print("\n── Model Comparison ──────────────────────────────")
print(summary.to_string(index=False))
best_mape  = summary.iloc[0]["MAPE"]
arima_mape = summary[summary["model"].str.contains("ARIMA")]["MAPE"].values[0]
impr = round(((arima_mape - best_mape) / arima_mape) * 100, 1)
print(f"\nBest model: {summary.iloc[0]['model']} | MAPE improvement over ARIMA: {impr}%")
summary.to_csv(OUTPUTS / "reports" / "model_comparison.csv", index=False)

# ── Optimization 
print("\n[6/6] Inventory optimization...")
sku_meta_df = pd.DataFrame(sku_meta_rows)

fc_agg_rows = []
for sku in skus:
    fc_arr = lstm_fcs.get(sku, np.zeros(FORECAST_HORIZON))
    fc_agg_rows.append({
        "sku_id":          sku,
        "forecast_demand": float(fc_arr.sum()),
        "demand_std":      float(fc_arr.std()) if fc_arr.sum() > 0 else 5.0,
    })
fc_agg = pd.DataFrame(fc_agg_rows)
opt_df  = run_optimization(fc_agg, sku_meta_df)
opt_df.to_csv(OUTPUTS / "reports" / "inventory_optimization.csv", index=False)
print(f"  Risk breakdown:\n{opt_df['fulfilment_risk'].value_counts().to_string()}")

# ── 28-day forecast detail 
start_date = pd.Timestamp("2025-01-01")
fc_detail  = []
for sku in skus:
    lstm_arr  = lstm_fcs.get(sku, np.zeros(FORECAST_HORIZON))
    arima_arr = arima_fcs.get(sku, np.zeros(FORECAST_HORIZON))
    for d in range(FORECAST_HORIZON):
        fc_detail.append({
            "sku_id":         sku,
            "forecast_date":  start_date + pd.Timedelta(days=d),
            "lstm_forecast":  round(max(float(lstm_arr[d]),  0), 1),
            "arima_forecast": round(max(float(arima_arr[d]), 0), 1),
        })
pd.DataFrame(fc_detail).to_csv(OUTPUTS / "forecasts" / "28day_forecast.csv", index=False)

# ── Per-SKU metrics 
ar_df.to_csv(OUTPUTS / "reports" / "arima_sku_metrics.csv", index=False)
lt_df.to_csv(OUTPUTS / "reports" / "lstm_sku_metrics.csv",  index=False)

elapsed = round(time.time() - t0, 1)
print(f"\n{'='*60}")
print(f"  ✅ Pipeline complete in {elapsed}s")
print(f"  Forecasts  → outputs/forecasts/28day_forecast.csv")
print(f"  Opt table  → outputs/reports/inventory_optimization.csv")
print(f"  Model cmp  → outputs/reports/model_comparison.csv")
print(f"  Dashboard  → streamlit run src/dashboard/dashboard.py")
print(f"{'='*60}\n")
