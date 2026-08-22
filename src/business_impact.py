"""Translate inventory state and optimization decisions into business KPIs."""

import pandas as pd

DATA_PATH = "data/retail_sales_data.csv"
INVENTORY_PATH = "data/inventory_optimization_results.csv"
OUTPUT_PATH = "data/business_impact.csv"


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    required = {"store", "product", "date", "demand", "lost_sales", "unit_price", "closing_stock"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    latest = df.sort_values("date").groupby(["store", "product"], as_index=False).tail(1).copy()
    latest["days_of_stock_observed"] = latest["closing_stock"] / latest["demand"].clip(lower=1)
    latest["current_stockout"] = latest["closing_stock"] <= 0

    historical_lost_sales_value = float((df["lost_sales"] * df["unit_price"]).sum())
    historical_stockout_days = int(df["stockout"].sum()) if "stockout" in df.columns else int((df["lost_sales"] > 0).sum())

    summary = {
        "total_sku_store_pairs": len(latest),
        "current_stockout_pairs": int(latest["current_stockout"].sum()),
        "historical_stockout_days": historical_stockout_days,
        "historical_lost_sales_value": round(historical_lost_sales_value, 2),
        "current_inventory_value": round((latest["closing_stock"] * latest["unit_price"]).sum(), 2),
    }

    try:
        inventory = pd.read_csv(INVENTORY_PATH)
        if {"store", "product", "recommended_order_qty"}.issubset(inventory.columns):
            summary["recommended_replenishment_units"] = round(inventory["recommended_order_qty"].sum(), 2)
        if "inventory_status" in inventory.columns:
            status = inventory["inventory_status"].astype(str)
            summary["critical_inventory_pairs"] = int((status == "CRITICAL").sum())
            summary["reorder_inventory_pairs"] = int((status == "REORDER").sum())
            summary["normal_inventory_pairs"] = int((status == "NORMAL").sum())
        if "low_coverage_7d" in inventory.columns:
            summary["low_coverage_pairs_7d"] = int(inventory["low_coverage_7d"].sum())
    except FileNotFoundError:
        pass

    result = pd.DataFrame([summary])
    result.to_csv(OUTPUT_PATH, index=False)
    print(result.to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
