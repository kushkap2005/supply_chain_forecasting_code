"""
dashboard.py
Streamlit dashboard for SKU-level forecast visualization + inventory optimization.
Run: streamlit run src/dashboard/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import subprocess
import sys

# ── Page config 
st.set_page_config(
    page_title="Supply Chain Forecasting",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace; }

.metric-card {
    background: linear-gradient(135deg, #0f1923 0%, #1a2840 100%);
    border: 1px solid #2a4a6b;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
}
.risk-HIGH   { color: #ff4b4b; font-weight: 700; }
.risk-MEDIUM { color: #ffa500; font-weight: 700; }
.risk-LOW    { color: #00cc88; font-weight: 700; }
.stSelectbox label { font-family: 'Space Mono', monospace; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Load data 
@st.cache_data
def load_data():
    fc_path  = Path("outputs/forecasts/28day_forecast.csv")
    opt_path = Path("outputs/reports/inventory_optimization.csv")
    cmp_path = Path("outputs/reports/model_comparison.csv")
    fi_path  = Path("outputs/reports/feature_importance.csv")
    raw_path = Path("data/raw/sku_demand.csv")

    if not fc_path.exists():
        return None, None, None, None, None

    fc  = pd.read_csv(fc_path,  parse_dates=["forecast_date"])
    opt = pd.read_csv(opt_path)
    cmp = pd.read_csv(cmp_path)
    fi  = pd.read_csv(fi_path)
    raw = pd.read_csv(raw_path, parse_dates=["date"])
    return fc, opt, cmp, fi, raw


fc_df, opt_df, cmp_df, fi_df, raw_df = load_data()

# ── Run pipeline if outputs missing
if fc_df is None:
    st.warning("⚙️  No model outputs found. Running pipeline now — this takes ~3-5 minutes...")
    with st.spinner("Training LSTM, XGBoost, ARIMA models..."):
        result = subprocess.run(
            [sys.executable, "src/train_evaluate.py"],
            capture_output=True, text=True
        )
    if result.returncode == 0:
        st.success("✅ Pipeline complete! Refreshing...")
        st.rerun()
    else:
        st.error(f"Pipeline error:\n{result.stderr}")
        st.stop()

# ── Sidebar 
with st.sidebar:
    st.markdown("## 📦 SC Forecasting")
    st.markdown("---")

    categories = ["All"] + sorted(opt_df["category"].unique().tolist())
    sel_cat = st.selectbox("Category", categories)

    skus = opt_df["sku_id"].tolist()
    if sel_cat != "All":
        skus = opt_df[opt_df["category"] == sel_cat]["sku_id"].tolist()
    sel_sku = st.selectbox("SKU", skus)

    st.markdown("---")
    risk_filter = st.multiselect(
        "Fulfilment Risk",
        ["HIGH", "MEDIUM", "LOW"],
        default=["HIGH", "MEDIUM", "LOW"]
    )
    st.markdown("---")
    st.markdown("**Models Trained**")
    st.markdown("- 🔵 LSTM (per SKU)")
    st.markdown("- 🟠 XGBoost")
    st.markdown("- ⚪ ARIMA (baseline)")

# ── Header
st.markdown("# 📦 Supply Chain Forecasting & Optimization")
st.markdown("SKU-level demand forecasting · Inventory optimization · DC planning")
st.markdown("---")

# ── KPI Row 
best_row   = cmp_df.sort_values("MAPE").iloc[0]
arima_mape = cmp_df[cmp_df["model"].str.contains("ARIMA")]["MAPE"].values[0]
improvement = round(((arima_mape - best_row["MAPE"]) / arima_mape) * 100, 1)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total SKUs",          f"{opt_df['sku_id'].nunique()}")
col2.metric("Best Model MAPE",     f"{best_row['MAPE']:.1f}%")
col3.metric("vs ARIMA Baseline",   f"−{improvement}%", delta=f"-{improvement}%")
col4.metric("HIGH Risk SKUs",      str((opt_df["fulfilment_risk"] == "HIGH").sum()))
col5.metric("Forecast Horizon",    "28 days")

st.markdown("---")

# ── Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 SKU Forecast",
    "🏭 Inventory Optimization",
    "🤖 Model Comparison",
    "🔍 Feature Importance"
])


# TAB 1: SKU Forecast

with tab1:
    st.subheader(f"Demand Forecast — {sel_sku}")

    # Historical actuals
    hist = raw_df[raw_df["sku_id"] == sel_sku].tail(90)
    sku_fc = fc_df[fc_df["sku_id"] == sel_sku]

    fig = go.Figure()

    # Actuals
    fig.add_trace(go.Scatter(
        x=hist["date"], y=hist["demand"],
        name="Historical Demand",
        line=dict(color="#4a9eff", width=2),
        mode="lines"
    ))

    # Promo highlights
    promo_days = hist[hist["promo_flag"] == 1]
    if not promo_days.empty:
        fig.add_trace(go.Scatter(
            x=promo_days["date"], y=promo_days["demand"],
            name="Promo Days",
            mode="markers",
            marker=dict(color="#ffcc00", size=8, symbol="star"),
        ))

    # LSTM forecast
    fig.add_trace(go.Scatter(
        x=sku_fc["forecast_date"], y=sku_fc["lstm_forecast"],
        name="LSTM Forecast",
        line=dict(color="#00cc88", width=2.5, dash="dot"),
        mode="lines+markers", marker=dict(size=4)
    ))

    # ARIMA forecast
    fig.add_trace(go.Scatter(
        x=sku_fc["forecast_date"], y=sku_fc["arima_forecast"],
        name="ARIMA Forecast",
        line=dict(color="#ff6b6b", width=1.5, dash="dash"),
        mode="lines"
    ))

    # Forecast confidence band (±1 std approx)
    fc_std = sku_fc["lstm_forecast"].std() * 0.3
    fig.add_trace(go.Scatter(
        x=pd.concat([sku_fc["forecast_date"], sku_fc["forecast_date"][::-1]]),
        y=pd.concat([
            sku_fc["lstm_forecast"] + fc_std,
            (sku_fc["lstm_forecast"] - fc_std)[::-1]
        ]),
        fill="toself", fillcolor="rgba(0,204,136,0.1)",
        line=dict(color="rgba(255,255,255,0)"),
        name="±1σ Confidence",
    ))

    fig.update_layout(
        template="plotly_dark",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Date", yaxis_title="Units Demanded",
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,25,35,0.8)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # SKU summary metrics
    sku_opt = opt_df[opt_df["sku_id"] == sel_sku]
    if not sku_opt.empty:
        r = sku_opt.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("28-day Forecast",     f"{int(r['forecast_28d_demand'])} units")
        c2.metric("Safety Stock",         f"{int(r['safety_stock_units'])} units")
        c3.metric("Reorder Point",        f"{int(r['reorder_point_units'])} units")
        risk_color = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(r["fulfilment_risk"], "")
        c4.metric("Fulfilment Risk",      f"{risk_color} {r['fulfilment_risk']}")


# TAB 2: Inventory Optimization

with tab2:
    st.subheader("Inventory Optimization — DC Planning Dashboard")

    filtered_opt = opt_df[opt_df["fulfilment_risk"].isin(risk_filter)]
    if sel_cat != "All":
        filtered_opt = filtered_opt[filtered_opt["category"] == sel_cat]

    # Risk distribution chart
    col_l, col_r = st.columns([1, 2])
    with col_l:
        risk_counts = filtered_opt["fulfilment_risk"].value_counts()
        fig_pie = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            color=risk_counts.index,
            color_discrete_map={"HIGH": "#ff4b4b", "MEDIUM": "#ffa500", "LOW": "#00cc88"},
            title="Fulfilment Risk Distribution",
        )
        fig_pie.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        # Safety stock vs reorder point scatter
        fig_scatter = px.scatter(
            filtered_opt,
            x="safety_stock_units", y="reorder_point_units",
            color="fulfilment_risk",
            color_discrete_map={"HIGH": "#ff4b4b", "MEDIUM": "#ffa500", "LOW": "#00cc88"},
            hover_data=["sku_id", "days_of_supply", "eoq_units"],
            title="Safety Stock vs Reorder Point",
            labels={"safety_stock_units": "Safety Stock (units)",
                    "reorder_point_units": "Reorder Point (units)"}
        )
        fig_scatter.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", height=300)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # EOQ by category
    eoq_cat = opt_df.groupby("category")["eoq_units"].mean().reset_index()
    fig_eoq = px.bar(
        eoq_cat, x="category", y="eoq_units",
        title="Average EOQ by Category",
        color="eoq_units", color_continuous_scale="Blues",
        labels={"eoq_units": "EOQ (units)", "category": "Category"}
    )
    fig_eoq.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                          coloraxis_showscale=False)
    st.plotly_chart(fig_eoq, use_container_width=True)

    # Full table
    st.markdown("#### 📋 Full Optimization Table")
    display_cols = ["sku_id", "category", "avg_daily_demand", "forecast_28d_demand",
                    "safety_stock_units", "reorder_point_units", "eoq_units",
                    "days_of_supply", "lead_time_days", "fulfilment_risk"]

    def highlight_risk(val):
        colors = {"HIGH": "background-color: #3d0000; color: #ff4b4b",
                  "MEDIUM": "background-color: #2d1a00; color: #ffa500",
                  "LOW": "background-color: #003d22; color: #00cc88"}
        return colors.get(val, "")

    styled = (
        filtered_opt[display_cols]
        .sort_values("fulfilment_risk", key=lambda x: x.map({"HIGH": 0, "MEDIUM": 1, "LOW": 2}))
        .style.applymap(highlight_risk, subset=["fulfilment_risk"])
        .format({
            "avg_daily_demand":    "{:.1f}",
            "forecast_28d_demand": "{:.0f}",
            "safety_stock_units":  "{:.1f}",
            "reorder_point_units": "{:.1f}",
            "eoq_units":           "{:.1f}",
            "days_of_supply":      "{:.1f}",
        })
    )
    st.dataframe(styled, use_container_width=True, height=400)


# TAB 3: Model Comparison

with tab3:
    st.subheader("Model Performance Comparison")

    col_a, col_b = st.columns(2)
    with col_a:
        fig_bar = px.bar(
            cmp_df.sort_values("MAPE"),
            x="model", y="MAPE",
            color="model",
            color_discrete_sequence=["#00cc88", "#4a9eff", "#ff6b6b"],
            title="MAPE by Model (lower = better)",
            text="MAPE"
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_bar.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                              showlegend=False, yaxis_title="MAPE (%)")
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        fig_rmse = px.bar(
            cmp_df.sort_values("RMSE"),
            x="model", y="RMSE",
            color="model",
            color_discrete_sequence=["#00cc88", "#4a9eff", "#ff6b6b"],
            title="RMSE by Model (lower = better)",
            text="RMSE"
        )
        fig_rmse.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_rmse.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                               showlegend=False, yaxis_title="RMSE")
        st.plotly_chart(fig_rmse, use_container_width=True)

    st.markdown("#### Model Summary Table")
    st.dataframe(
        cmp_df.style.highlight_min(subset=["MAPE", "RMSE", "MAE"], color="#003d22"),
        use_container_width=True
    )

    # MAPE improvement callout
    arima_mape = cmp_df[cmp_df["model"].str.contains("ARIMA")]["MAPE"].values[0]
    best_mape  = cmp_df["MAPE"].min()
    impr = round(((arima_mape - best_mape) / arima_mape) * 100, 1)
    st.success(f"✅ Best model achieves **{impr}% lower MAPE** than ARIMA baseline")


# TAB 4: Feature Importance

with tab4:
    st.subheader("XGBoost Feature Importance")

    top_n = st.slider("Show top N features", 5, len(fi_df), 15)
    top_fi = fi_df.head(top_n)

    fig_fi = px.bar(
        top_fi[::-1],
        x="importance", y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Teal",
        title=f"Top {top_n} Features by Importance (XGBoost)",
    )
    fig_fi.update_layout(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        coloraxis_showscale=False, height=500,
        yaxis_title="", xaxis_title="Feature Importance Score"
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("#### Feature Groups")
    groups = {
        "📅 Calendar": [f for f in fi_df["feature"] if any(k in f for k in ["day", "week", "month", "quarter", "year", "weekend"])],
        "🔁 Lag":      [f for f in fi_df["feature"] if "lag_" in f and "promo" not in f],
        "📊 Rolling":  [f for f in fi_df["feature"] if "roll_" in f or "ewm_" in f],
        "🎯 Promo":    [f for f in fi_df["feature"] if "promo" in f],
    }
    g1, g2, g3, g4 = st.columns(4)
    for col, (grp, feats) in zip([g1, g2, g3, g4], groups.items()):
        avg_imp = fi_df[fi_df["feature"].isin(feats)]["importance"].mean()
        col.metric(grp, f"{len(feats)} features", f"avg imp: {avg_imp:.4f}")

st.markdown("---")
st.caption("Supply Chain Forecasting & Optimization Pipeline · LSTM + XGBoost + ARIMA · EOQ · Safety Stock · ROP")
