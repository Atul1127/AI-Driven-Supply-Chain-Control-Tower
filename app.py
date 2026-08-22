"""Streamlit control tower for demand, inventory and supplier risk."""

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Supply Chain Control Tower", layout="wide")
st.title("Intelligent Supply Chain Control Tower")
st.caption("SKU/store demand forecasting • Inventory health • Supplier risk • Action priorities")

@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()

control = load_csv("data/control_tower_inventory.csv")
forecast = load_csv("data/sku_30_day_forecast.csv")
risk = load_csv("data/supplier_risk_analysis.csv")

if control.empty:
    st.warning("Run `python src/run_pipeline.py` first to generate the control-tower outputs.")
else:
    priority = control.get("priority", pd.Series(dtype=str)).astype(str)
    cols = st.columns(4)
    cols[0].metric("Store × SKU pairs", f"{len(control):,}")
    cols[1].metric("Urgent", int((priority == "URGENT").sum()))
    cols[2].metric("Critical", int((control.get("inventory_status", pd.Series(dtype=str)).astype(str) == "CRITICAL").sum()))
    cols[3].metric("High Supplier Risk", int((control.get("risk_level", pd.Series(dtype=str)).astype(str) == "HIGH").sum()))

    tab1, tab2, tab3 = st.tabs(["Control Tower", "SKU Forecast", "Supplier Risk"])
    with tab1:
        st.subheader("Priority actions")
        st.dataframe(control, use_container_width=True, hide_index=True)
    with tab2:
        st.subheader("30-day SKU/store forecasts")
        if forecast.empty:
            st.info("No SKU forecast output found.")
        else:
            pairs = forecast[["store", "product"]].drop_duplicates()
            selected = st.selectbox("Store × Product", [f"{r.store} | {r.product}" for r in pairs.itertuples()])
            store, product = selected.split(" | ", 1)
            view = forecast[(forecast.store == store) & (forecast.product == product)].copy()
            view["date"] = pd.to_datetime(view["date"])
            st.line_chart(view.set_index("date")["forecast_demand"])
            st.dataframe(view, use_container_width=True, hide_index=True)
    with tab3:
        st.subheader("Supplier risk")
        if risk.empty:
            st.info("No supplier risk output found.")
        else:
            st.dataframe(risk, use_container_width=True, hide_index=True)
