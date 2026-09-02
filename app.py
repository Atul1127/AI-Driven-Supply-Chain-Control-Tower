"""Executive Streamlit dashboard for the supply-chain control tower."""

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Supply Chain Control Tower",
    page_icon="📦",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    [data-testid="stMetricValue"] {font-size: 1.8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📦 Intelligent Supply Chain Control Tower")
st.caption(
    "SQL analytics • Financial performance • Forecasting • Inventory optimization • "
    "Supplier risk • Disruption intelligence • Decision support"
)


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
baseline = load_csv("data/baseline_results.csv")
xgb_results = load_csv("data/sku_xgboost_results.csv")
disruption = load_csv("data/disruption_detection.csv")
temporal = load_csv("data/temporal_disruption.csv")
finance = load_csv("data/finance_summary.csv")
budget = load_csv("data/budget_vs_actual.csv")
promo = load_csv("data/promotion_effectiveness.csv")
abc = load_csv("data/abc_analysis.csv")
supplier_finance = load_csv("data/supplier_stockout_impact.csv")

if control.empty:
    st.warning("Run `python src/run_pipeline.py` first to generate the control-tower outputs.")
    st.stop()

priority = control.get("priority", pd.Series(dtype=str)).astype(str)
inventory_status = control.get("inventory_status", pd.Series(dtype=str)).astype(str)
risk_level = (
    risk.get("risk_level", pd.Series(dtype=str)).astype(str)
    if not risk.empty
    else pd.Series(dtype=str)
)

# Executive KPI strip
kpi = st.columns(6)
kpi[0].metric("Store × SKU", f"{len(control):,}")
kpi[1].metric("High Priority", int(priority.isin(["HIGH", "URGENT"]).sum()))
kpi[2].metric("Critical Inventory", int((inventory_status == "CRITICAL").sum()))
kpi[3].metric("Reorder", int((inventory_status == "REORDER").sum()))
kpi[4].metric("High Supplier Risk", int((risk_level == "HIGH").sum()))
kpi[5].metric(
    "Critical Disruptions",
    int((disruption.get("disruption_level", pd.Series(dtype=str)) == "CRITICAL").sum())
    if not disruption.empty
    else 0,
)

st.divider()

tabs = st.tabs(
    [
        "🚨 Control Tower",
        "💰 Financial Analytics",
        "📈 Sales & Revenue",
        "📦 Inventory",
        "🏭 Supplier Risk",
        "⚠️ Disruption",
        "🔮 Forecast",
        "📊 Model Benchmark",
        "🎯 Recommendations",
    ]
)

with tabs[0]:
    st.subheader("Priority actions")
    c1, c2 = st.columns(2)
    with c1:
        store_options = ["All stores"] + sorted(control["store"].dropna().unique().tolist())
        store_filter = st.selectbox("Store", store_options)
    with c2:
        priority_filter = st.selectbox(
            "Priority", ["All priorities", "URGENT", "HIGH", "MEDIUM", "LOW"]
        )

    view = control.copy()
    if store_filter != "All stores":
        view = view[view.store == store_filter]
    if priority_filter != "All priorities":
        view = view[view.priority == priority_filter]

    preferred = [
        "priority", "action", "store", "product", "category", "supplier",
        "inventory_status", "days_of_stock", "shortage_to_rop", "recommended_order_qty",
        "average_30_day_forecast", "risk_level",
    ]
    st.dataframe(
        view[[c for c in preferred if c in view.columns]],
        use_container_width=True,
        hide_index=True,
    )

with tabs[1]:
    st.subheader("Financial performance")
    if finance.empty:
        st.info("Run the pipeline to generate finance outputs.")
    else:
        row = finance.iloc[0]
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Revenue", f"₹{row['revenue']:,.0f}")
        f2.metric("COGS", f"₹{row['cogs']:,.0f}")
        f3.metric("Gross Profit", f"₹{row['gross_profit']:,.0f}")
        f4.metric("Gross Margin", f"{row['gross_margin_pct']:.1f}%")

        f5, f6, f7, f8 = st.columns(4)
        f5.metric("Inventory Turnover", f"{row['inventory_turnover']:.2f}x")
        f6.metric("DIO", f"{row['days_inventory_outstanding']:.1f} days")
        f7.metric("Holding Cost", f"₹{row['annual_holding_cost']:,.0f}")
        f8.metric("Lost-sales Exposure", f"₹{row['historical_lost_sales_value']:,.0f}")
        st.caption(
            "Assumptions: unit cost = 60% of list price; budget values are planning assumptions."
        )

        if not budget.empty:
            st.markdown("### Budget vs Actual")
            st.line_chart(
                budget.set_index("month")[["budget_revenue", "actual_revenue"]]
            )
            st.dataframe(budget, use_container_width=True, hide_index=True)

            st.markdown("### Financial variance")
            variance_cols = [
                c for c in ["revenue_variance", "cogs_variance", "gross_profit_variance"]
                if c in budget.columns
            ]
            if variance_cols:
                st.bar_chart(budget.set_index("month")[variance_cols])

with tabs[2]:
    st.subheader("Sales & revenue analytics")
    if promo.empty and abc.empty:
        st.info("Run the pipeline to generate sales analytics.")
    else:
        if not promo.empty:
            st.markdown("### Promotion effectiveness")
            promo_cols = [
                c for c in [
                    "promo_label", "days", "units_sold", "revenue",
                    "average_daily_units", "average_daily_revenue", "conversion_pct",
                ] if c in promo.columns
            ]
            st.dataframe(promo[promo_cols], use_container_width=True, hide_index=True)
            if "promo_label" in promo.columns and "average_daily_revenue" in promo.columns:
                st.bar_chart(promo.set_index("promo_label")[["average_daily_revenue"]])
            st.caption(
                "Promotion analysis compares observed synthetic promotion days with non-promotion days; it is not a causal estimate."
            )

        if not abc.empty:
            st.markdown("### ABC product analysis")
            st.dataframe(abc.head(30), use_container_width=True, hide_index=True)
            abc_counts = abc["abc_class"].value_counts().reindex(
                ["A", "B", "C"], fill_value=0
            )
            st.bar_chart(abc_counts)

with tabs[3]:
    st.subheader("Inventory analytics")
    if impact.empty:
        st.info("No inventory output found.")
    else:
        row = impact.iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("SKU/store pairs", f"{int(row['total_sku_store_pairs']):,}")
        b.metric("Current Stockouts", f"{int(row['current_stockout_pairs']):,}")
        c.metric("Inventory Value", f"₹{float(row['current_inventory_value']):,.0f}")
        d.metric("Historical Stockout Days", f"{int(row['historical_stockout_days']):,}")
        st.caption("Historical lost-sales is simulated exposure, not savings generated by the system.")
        st.dataframe(impact, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Supplier performance and financial impact")
    if risk.empty:
        st.info("No supplier risk output found.")
    else:
        r1, r2, r3 = st.columns(3)
        r1.metric("Low", int((risk_level == "LOW").sum()))
        r2.metric("Medium", int((risk_level == "MEDIUM").sum()))
        r3.metric("High", int((risk_level == "HIGH").sum()))
        st.bar_chart(
            risk["risk_level"].value_counts().reindex(
                ["LOW", "MEDIUM", "HIGH"], fill_value=0
            )
        )
        st.dataframe(risk, use_container_width=True, hide_index=True)
        if not supplier_finance.empty:
            st.markdown("### Supplier → stockout → financial exposure")
            st.dataframe(supplier_finance, use_container_width=True, hide_index=True)

with tabs[5]:
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
            levels = disruption["disruption_level"].value_counts().reindex(
                ["LOW", "MEDIUM", "HIGH", "CRITICAL"], fill_value=0
            )
            st.bar_chart(levels)
        with right:
            st.bar_chart(disruption["anomaly_score"].value_counts(bins=10, sort=False))

        cols = [
            "supplier", "disruption_level", "disruption_score", "anomaly_status",
            "anomaly_score", "cluster", "average_lead_time", "on_time_percentage",
            "fill_percentage", "defect_percentage",
        ]
        st.dataframe(
            disruption[[c for c in cols if c in disruption.columns]].head(20),
            use_container_width=True,
            hide_index=True,
        )

        supplier_names = disruption["supplier"].dropna().astype(str).tolist()
        selected_supplier = st.selectbox("Supplier", supplier_names, key="disruption_supplier")
        selected = disruption[disruption["supplier"].astype(str) == selected_supplier].iloc[0]
        st.info(
            f"Cluster **{selected['cluster']}** • Status **{selected['disruption_level']}** • "
            f"Anomaly **{selected['anomaly_status']}**"
        )

        if not temporal.empty:
            supplier_temporal = temporal[
                temporal["supplier"].astype(str) == selected_supplier
            ].copy()
            products = sorted(supplier_temporal["product"].dropna().astype(str).unique())
            if products:
                selected_product = st.selectbox(
                    "Affected SKU / Product", products, key="timeline_product"
                )
                timeline = supplier_temporal[
                    supplier_temporal["product"].astype(str) == selected_product
                ].copy()
                timeline["date"] = pd.to_datetime(timeline["date"])
                st.line_chart(
                    timeline.sort_values("date").set_index("date")[["disruption_signal"]]
                )

with tabs[6]:
    st.subheader("30-day SKU/store demand forecast")
    if forecast.empty:
        st.info("No SKU forecast output found.")
    else:
        pairs = (
            forecast[["store", "product"]]
            .drop_duplicates()
            .sort_values(["store", "product"])
        )
        selected_pair = st.selectbox(
            "Store × Product",
            [f"{r.store} | {r.product}" for r in pairs.itertuples()],
            key="forecast_pair",
        )
        store, product = selected_pair.split(" | ", 1)
        view = forecast[(forecast.store == store) & (forecast.product == product)].copy()
        view["date"] = pd.to_datetime(view["date"])
        st.line_chart(view.set_index("date")[["forecast_demand"]], height=360)
        a, b = st.columns(2)
        a.metric("30-day forecast", f"{view.forecast_demand.sum():,.0f} units")
        b.metric("Average daily demand", f"{view.forecast_demand.mean():,.1f} units")

with tabs[7]:
    st.subheader("Forecast model benchmark")
    if baseline.empty or xgb_results.empty:
        st.info("Run the pipeline to generate model benchmark results.")
    else:
        benchmark = pd.concat(
            [baseline[["model", "MAE", "RMSE", "MAPE"]],
             xgb_results[["model", "MAE", "RMSE", "MAPE"]]],
            ignore_index=True,
        )
        benchmark = benchmark.drop_duplicates(subset=["model"]).sort_values("MAE")
        st.dataframe(benchmark, use_container_width=True, hide_index=True)
        st.markdown("### Error comparison")
        st.bar_chart(benchmark.set_index("model")[["MAE", "RMSE"]])
        st.caption(
            "Models are evaluated chronologically on held-out demand data. Lower MAE/RMSE is better."
        )

with tabs[8]:
    st.subheader("Decision recommendations")
    st.markdown(
        "1. **Protect revenue:** prioritize critical stockouts and high lost-sales exposure.\n"
        "2. **Protect margin:** review products with weak margin or promotion-heavy performance.\n"
        "3. **Improve working capital:** monitor DIO, inventory turnover and holding cost.\n"
        "4. **Manage suppliers:** focus on suppliers combining poor service with high financial exposure.\n"
        "5. **Plan demand:** use the 30-day forecast to inform replenishment and inventory decisions."
    )
    if not supplier_finance.empty:
        st.markdown("### Highest supplier financial exposure")
        st.dataframe(supplier_finance.head(10), use_container_width=True, hide_index=True)
    if not abc.empty:
        st.markdown("### Revenue concentration")
        st.dataframe(abc.head(10), use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "Decision-support dashboard powered by SQL analytics, financial analysis, forecasting, "
    "inventory optimization, supplier-risk analytics and disruption detection."
)
