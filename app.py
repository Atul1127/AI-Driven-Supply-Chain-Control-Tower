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
st.caption("Forecasting • Inventory optimization • Supplier risk • Disruption intelligence • Business impact")

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
disruption = load_csv("data/disruption_detection.csv")
cluster_results = load_csv("data/disruption_model_comparison.csv")
pca = load_csv("data/disruption_pca.csv")
temporal = load_csv("data/temporal_disruption.csv")

if control.empty:
    st.warning("Run `python src/run_pipeline.py` first to generate the control-tower outputs.")
    st.stop()

priority = control.get("priority", pd.Series(dtype=str)).astype(str)
inventory_status = control.get("inventory_status", pd.Series(dtype=str)).astype(str)
risk_level = risk.get("risk_level", pd.Series(dtype=str)).astype(str) if not risk.empty else pd.Series(dtype=str)

kpi = st.columns(6)
kpi[0].metric("Store × SKU", f"{len(control):,}")
kpi[1].metric("High Priority", int(priority.isin(["HIGH", "URGENT"]).sum()))
kpi[2].metric("Critical Inventory", int((inventory_status == "CRITICAL").sum()))
kpi[3].metric("Reorder", int((inventory_status == "REORDER").sum()))
kpi[4].metric("Supplier Risk", int((risk_level == "HIGH").sum()))
kpi[5].metric("Disruptions", int((disruption.get("disruption_level", pd.Series(dtype=str)) == "CRITICAL").sum()) if not disruption.empty else 0)

st.divider()
tabs = st.tabs(["🚨 Control Tower", "⚠️ Disruption Intelligence", "📈 SKU Forecast", "📊 Model Benchmark", "🏭 Supplier Risk", "💰 Business Impact"])

with tabs[0]:
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
    st.dataframe(view[[c for c in preferred if c in view.columns]], use_container_width=True, hide_index=True)

with tabs[1]:
    st.subheader("Supply-chain disruption intelligence")
    if disruption.empty:
        st.info("No disruption output found. Run `python src/run_pipeline.py` first.")
    else:
        critical = int((disruption["disruption_level"] == "CRITICAL").sum())
        high = int((disruption["disruption_level"] == "HIGH").sum())
        anomalies = int((disruption["anomaly_status"] == "ANOMALY").sum())
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Critical", critical)
        d2.metric("High", high)
        d3.metric("Anomalies", anomalies)
        d4.metric("Suppliers", len(disruption))

        left, right = st.columns(2)
        with left:
            levels = disruption["disruption_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0)
            st.markdown("**Disruption level distribution**")
            st.bar_chart(levels)
        with right:
            st.markdown("**Anomaly score distribution**")
            st.bar_chart(disruption["anomaly_score"].value_counts(bins=10, sort=False))

        st.markdown("### Highest-priority disruption candidates")
        cols = ["supplier", "disruption_level", "disruption_score", "anomaly_status", "anomaly_score", "cluster", "average_lead_time", "on_time_percentage", "fill_percentage", "defect_percentage"]
        st.dataframe(disruption[[c for c in cols if c in disruption.columns]].head(20), use_container_width=True, hide_index=True)

        st.markdown("### Supplier drill-down")
        supplier_names = disruption["supplier"].dropna().astype(str).tolist()
        selected_supplier = st.selectbox("Supplier", supplier_names, key="disruption_supplier")
        selected = disruption[disruption["supplier"].astype(str) == selected_supplier].iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Disruption score", f"{selected['disruption_score']:.1f}")
        b.metric("Anomaly score", f"{selected['anomaly_score']:.3f}")
        c.metric("Lead time", f"{selected['average_lead_time']:.1f} days")
        d.metric("Fill rate", f"{selected['fill_percentage']:.1f}%")
        st.info(f"Cluster **{selected['cluster']}** • Status **{selected['disruption_level']}** • Anomaly **{selected['anomaly_status']}**")

        if not temporal.empty:
            st.markdown("### Temporal disruption timeline")
            supplier_temporal = temporal[temporal["supplier"].astype(str) == selected_supplier].copy()
            products = sorted(supplier_temporal["product"].dropna().astype(str).unique())
            if products:
                selected_product = st.selectbox("Affected SKU / Product", products, key="timeline_product")
                timeline = supplier_temporal[supplier_temporal["product"].astype(str) == selected_product].copy()
                timeline["date"] = pd.to_datetime(timeline["date"])
                timeline = timeline.sort_values("date")
                st.line_chart(timeline.set_index("date")[["disruption_signal"]], height=280)
                recent = timeline.tail(1).iloc[0]
                st.metric("Current temporal stage", str(recent["disruption_stage"]))
                st.caption(f"Signal is measured against the preceding 14-day operational baseline for {selected_supplier} / {selected_product}.")

        st.markdown("### Why is this supplier risky?")
        reasons = []
        if selected["average_lead_time"] > disruption["average_lead_time"].median(): reasons.append("Lead time is above the supplier population median.")
        if selected["fill_percentage"] < disruption["fill_percentage"].median(): reasons.append("Fill rate is below the supplier population median.")
        if selected["on_time_percentage"] < disruption["on_time_percentage"].median(): reasons.append("On-time delivery is below the supplier population median.")
        if selected["defect_percentage"] > disruption["defect_percentage"].median(): reasons.append("Defect rate is above the supplier population median.")
        if selected["anomaly_status"] == "ANOMALY": reasons.append("Isolation Forest identifies unusual multivariate operating behavior.")
        for reason in reasons or ["No single dominant driver; review the complete supplier profile."]:
            st.write(f"• {reason}")

        st.markdown("### Recommended response")
        level = str(selected["disruption_level"])
        if level == "CRITICAL":
            st.error("Activate alternate sourcing, expedite open orders, protect high-demand inventory, and review affected SKUs immediately.")
        elif level == "HIGH":
            st.warning("Review supplier capacity, increase monitoring frequency, and evaluate safety-stock or alternate-source actions.")
        else:
            st.info("Continue monitoring and investigate persistent deterioration before escalating.")

        st.markdown("### PCA supply-chain behavior map")
        if not pca.empty:
            chart = pca[["pc1", "pc2", "cluster"]].copy()
            chart["cluster"] = chart["cluster"].astype(str)
            st.scatter_chart(chart, x="pc1", y="pc2", color="cluster", height=420)
            st.caption(f"PCA explained variance: PC1 {pca['explained_variance_pc1'].iloc[0] * 100:.1f}% • PC2 {pca['explained_variance_pc2'].iloc[0] * 100:.1f}%")

        st.markdown("### Clustering model comparison")
        if not cluster_results.empty:
            metric_cols = ["model", "parameters", "clusters", "noise_pct", "silhouette", "davies_bouldin", "calinski_harabasz"]
            st.dataframe(cluster_results[[c for c in metric_cols if c in cluster_results.columns]].sort_values("silhouette", ascending=False), use_container_width=True, hide_index=True)

with tabs[2]:
    st.subheader("30-day SKU/store demand forecast")
    if forecast.empty:
        st.info("No SKU forecast output found.")
    else:
        pairs = forecast[["store", "product"]].drop_duplicates().sort_values(["store", "product"])
        selected = st.selectbox("Store × Product", [f"{r.store} | {r.product}" for r in pairs.itertuples()], key="forecast_pair")
        store, product = selected.split(" | ", 1)
        view = forecast[(forecast.store == store) & (forecast.product == product)].copy()
        view["date"] = pd.to_datetime(view["date"])
        st.line_chart(view.set_index("date")[["forecast_demand"]], height=360)
        a, b = st.columns(2)
        a.metric("30-day forecast", f"{view.forecast_demand.sum():,.0f} units")
        b.metric("Average daily demand", f"{view.forecast_demand.mean():,.1f} units")
        st.dataframe(view, use_container_width=True, hide_index=True)

with tabs[3]:
    st.subheader("Store × SKU forecasting benchmark")
    if models.empty:
        st.info("No model benchmark output found. Run the pipeline first.")
    else:
        metric = st.selectbox("Metric", ["MAE", "RMSE", "MAPE"], key="forecast_metric")
        chart = models[["model", metric]].set_index("model").sort_values(metric)
        st.bar_chart(chart, height=320)
        st.dataframe(models, use_container_width=True, hide_index=True)
        best = models.loc[models[metric].idxmin()]
        st.success(f"Best {metric}: **{best['model']}** ({best[metric]:.2f})")

with tabs[4]:
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

with tabs[5]:
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
st.caption("Decision-support dashboard powered by forecasting, inventory optimization, supplier-risk analytics and unsupervised disruption detection.")
