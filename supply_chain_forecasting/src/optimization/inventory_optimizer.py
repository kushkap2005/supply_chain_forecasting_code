"""
inventory_optimizer.py
Classical supply chain optimization algorithms:
  - Economic Order Quantity (EOQ)
  - Safety Stock (service-level based)
  - Reorder Point (ROP)
  - Days of Supply (DOS)
  - Fulfilment Risk Scoring

Takes forecast outputs and SKU metadata as inputs.
"""

import numpy as np
import pandas as pd
from scipy import stats


# EOQ  (Economic Order Quantity)

def calc_eoq(annual_demand: float, order_cost: float,
             unit_cost: float, holding_cost_pct: float) -> float:
    """
    Wilson EOQ formula.
    EOQ = sqrt(2 * D * S / (H))
    where H = holding_cost_pct * unit_cost
    """
    H = holding_cost_pct * unit_cost
    if H <= 0 or annual_demand <= 0:
        return 0.0
    return float(np.sqrt(2 * annual_demand * order_cost / H))


# Safety Stock (SS)

def calc_safety_stock(demand_std_daily: float, lead_time_days: int,
                      service_level: float = 0.95) -> float:
    """
    SS = Z(service_level) * σ_demand * sqrt(lead_time)
    """
    z = stats.norm.ppf(service_level)
    return float(z * demand_std_daily * np.sqrt(lead_time_days))


# Reorder Point (ROP)

def calc_rop(avg_daily_demand: float, lead_time_days: int,
             safety_stock: float) -> float:
    """
    ROP = (avg_daily_demand * lead_time) + safety_stock
    """
    return float(avg_daily_demand * lead_time_days + safety_stock)


# 
# Days of Supply

def calc_days_of_supply(current_stock: float, avg_daily_demand: float) -> float:
    if avg_daily_demand <= 0:
        return np.inf
    return float(current_stock / avg_daily_demand)


# Fulfilment Risk Score

def fulfilment_risk_score(days_of_supply: float, lead_time_days: int,
                           safety_stock: float, current_stock: float) -> str:
    """
    Returns: 'HIGH', 'MEDIUM', 'LOW' risk for DC planning.
    """
    if days_of_supply < lead_time_days:
        return "HIGH"
    elif current_stock < safety_stock * 1.2:
        return "MEDIUM"
    else:
        return "LOW"



# Full Optimization Table

def run_optimization(forecast_df: pd.DataFrame,
                     sku_meta: pd.DataFrame,
                     order_cost: float = 50.0) -> pd.DataFrame:
    """
    Parameters
    ----------
    forecast_df : columns [sku_id, forecast_demand]
                  28-day total forecast per SKU
    sku_meta    : columns [sku_id, lead_time_days, unit_cost_usd,
                            holding_cost_pct, service_level_target]
    order_cost  : fixed cost per purchase order (USD)

    Returns
    -------
    DataFrame with full optimization outputs per SKU
    """
    records = []

    for _, row in sku_meta.iterrows():
        sku = row["sku_id"]
        fc_row = forecast_df[forecast_df["sku_id"] == sku]
        if fc_row.empty:
            continue

        total_28d_demand = float(fc_row["forecast_demand"].values[0])
        avg_daily        = total_28d_demand / 28
        annual_demand    = avg_daily * 365

        # Demand variability: use rolling std from forecast if available
        demand_std = float(fc_row.get("demand_std", pd.Series([avg_daily * 0.15])).values[0])

        lead_time    = int(row["lead_time_days"])
        unit_cost    = float(row["unit_cost_usd"])
        hold_pct     = float(row["holding_cost_pct"])
        svc_level    = float(row["service_level_target"])

        eoq = calc_eoq(annual_demand, order_cost, unit_cost, hold_pct)
        ss  = calc_safety_stock(demand_std, lead_time, svc_level)
        rop = calc_rop(avg_daily, lead_time, ss)

        # Simulated current stock (for risk scoring) — normally from WMS
        current_stock = np.random.uniform(ss * 0.5, rop * 2)
        dos = calc_days_of_supply(current_stock, avg_daily)
        risk = fulfilment_risk_score(dos, lead_time, ss, current_stock)

        records.append({
            "sku_id":                sku,
            "category":              row.get("category", ""),
            "avg_daily_demand":      round(avg_daily, 1),
            "forecast_28d_demand":   round(total_28d_demand, 0),
            "annual_demand_est":     round(annual_demand, 0),
            "lead_time_days":        lead_time,
            "service_level_target":  svc_level,
            "safety_stock_units":    round(ss, 1),
            "reorder_point_units":   round(rop, 1),
            "eoq_units":             round(eoq, 1),
            "current_stock_sim":     round(current_stock, 1),
            "days_of_supply":        round(dos, 1),
            "unit_cost_usd":         unit_cost,
            "holding_cost_pct":      hold_pct,
            "annual_holding_cost":   round(ss * unit_cost * hold_pct, 2),
            "fulfilment_risk":       risk,
        })

    result = pd.DataFrame(records)
    return result
