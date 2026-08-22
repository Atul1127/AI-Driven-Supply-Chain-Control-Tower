"""Executive Streamlit dashboard for the supply-chain control tower."""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Supply Chain Control Tower", page_icon="📦", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 2rem;}
</style>
""", unsafe_allow_html=True)

st.title("📦 Intelligent Supply Chain Control Tower")
st.caption("SKU-level demand forecasting • Inventory optimization • Supplier risk • Business impact")

@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()

control = load_csv("data/control_tower_inventory.csv")
forecast = load_csv("data/sku_30_day_forecast.csv")
risk = load_csv("data/supplier_risk_analysis.csv")
impact = load_csv("data/business_impact.csv")
models = load_csv("data/baseline_results.csv")

if control.empty:
    st.warning("Run `python src/run_pipeline.py` first to generate the control-tower outputs.")
    st.stop()

priority = control.get("priority", pd.Series(dtype=str)).astype(str)
inventory_status = control.get("inventory_status", pd.Series(dtype=str)).astype(str)
risk_level = control.get("risk_level", pd.Series(dtype=str)).astype(str)

kpi = st.columns(6)
kpi[0].metric("Store × SKU", f"{len(control):,}")
kpi[1].metric("High Priority", int(priority.isin(["HIGH", "URGENT"]).sum()))
kpi[2].metric("Critical", int((inventory_status == "CRITICAL").sum()))
kpi[3].metric("Reorder", int((inventory_status == "REORDER").sum()))
kpi[4].metric("Supplier Risk", int((risk_level == "HIGH").sum()))
if not impact.empty:
    row = impact.iloc[0]
    kpi[5].metric("Current Stockouts", f"{int(row.get('current_stockout_pairs', 0)):,}")
else:
    kpi[5].metric("Current Stockouts", "—")

st.divider()
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🚨 Control Tower", "📈 SKU Forecast", "📊 Model Benchmark", "🏭 Supplier Risk", "💰 Business Impact"])

with tab1:
    st.subheader("Priority actions")
    c1, c2 = st.columns(2)
    with c1:
        store_options = ["All stores"] + sorted(control["store"].dropna().unique().tolist())
        store_filter = st.selectbox("Store", store_options)
    with c2:
        priority_filter = st.selectbox("Priority", ["All priorities", "URGENT", "HIGH", "MEDIUM", "LOW"])

    view = control.copy()
    if store_filter != "All stores":
        view = view[view.store == store_filter]
    if priority_filter != "All priorities":
        view = view[view.priority == priority_filter]

    preferred = ["priority", "action", "store", "product", "category", "supplier", "inventory_status", "days_of_stock", "shortage_to_rop", "recommended_order_qty", "average_30_day_forecast", "risk_level"]
    display_cols = [c for c in preferred if c in view.columns]
    st.dataframe(view[display_cols], use_container_width=True, hide_index=True)

with tab2:
    st.subheader("30-day SKU/store demand forecast")
    if forecast.empty:
        st.info("No SKU forecast output found.")
    else:
        pairs = forecast[["store", "product"]].drop_duplicates().sort_values(["store", "product"])
        selected = st.selectbox("Store × Product", [f"{r.store} | {r.product}" for r in pairs.itertuples()])
        store, product = selected.split(" | ", 1)
        view = forecast[(forecast.store == store) & (forecast.product == product)].copy()
        view["date"] = pd.to_datetime(view["date"])
        st.line_chart(view.set_index("date")[["forecast_demand"]], height=360)
        a, b = st.columns(2)
        a.metric("30-day forecast", f"{view.forecast_demand.sum():,.0f} units")
        b.metric("Average daily demand", f"{view.forecast_demand.mean():,.1f} units")
        st.dataframe(view, use_container_width=True, hide_index=True)

with tab3:
    st.subheader("Store × SKU forecasting benchmark")
    st.caption("Same chronological test window across naive, seasonal-naive and XGBoost models.")
    if models.empty:
        st.info("No model benchmark output found. Run the pipeline first.")
    else:
        metric = st.selectbox("Metric", ["MAE", "RMSE", "MAPE"])
        chart = models[["model", metric]].set_index("model").sort_values(metric)
        st.bar_chart(chart, height=320)
        st.dataframe(models, use_container_width=True, hide_index=True)
        best = models.loc[models[metric].idxmin()]
        st.success(f"Best {metric}: **{best['model']}** ({best[metric]:.2f})")

with tab4:
    st.subheader("Supplier risk overview")
    if risk.empty:
        st.info("No supplier risk output found.")
    else:
        r1, r2, r3 = st.columns(3)
        r1.metric("Low", int((risk_level == "LOW").sum()))
        r2.metric("Medium", int((risk_level == "MEDIUM").sum()))
        r3.metric("High", int((risk_level == "HIGH").sum()))
        distribution = risk["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"], fill_value=0)
        st.bar_chart(distribution)
        st.dataframe(risk, use_container_width=True, hide_index=True)

with tab5:
    st.subheader("Business impact")
    if impact.empty:
        st.info("No business impact output found.")
    else:
        row = impact.iloc[0]
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("SKU/store pairs", f"{int(row['total_sku_store_pairs']):,}")
        b2.metric("Current stockouts", f"{int(row['current_stockout_pairs']):,}")
        b3.metric("Historical lost-sales", f"₹{float(row['historical_lost_sales_value']):,.0f}")
        b4.metric("Inventory value", f"₹{float(row['current_inventory_value']):,.0f}")
        st.caption("Historical lost-sales is simulated exposure, not savings generated by the system.")
        st.dataframe(impact, use_container_width=True, hide_index=True)

st.divider()
st.caption("Decision-support dashboard powered by SKU-level XGBoost forecasting, inventory optimization and supplier-risk analytics.")
