"""Translate inventory decisions into measurable business impact."""

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

    latest = (
        df.sort_values("date")
        .groupby(["store", "product"], as_index=False)
        .tail(1)
        .copy()
    )
    latest["days_of_stock"] = latest["closing_stock"] / latest["demand"].clip(lower=1)
    latest["stockout_risk"] = latest["days_of_stock"] < 7
    latest["estimated_lost_sales_value"] = latest["lost_sales"] * latest["unit_price"]

    summary = {
        "total_sku_store_pairs": len(latest),
        "stockout_risk_pairs": int(latest["stockout_risk"].sum()),
        "stockout_risk_pct": round(latest["stockout_risk"].mean() * 100, 2),
        "estimated_lost_sales_value": round(latest["estimated_lost_sales_value"].sum(), 2),
        "current_inventory_value": round((latest["closing_stock"] * latest["unit_price"]).sum(), 2),
    }

    # Add optimization recommendations when available.
    try:
        inventory = pd.read_csv(INVENTORY_PATH)
        if {"store", "product", "recommended_order_qty"}.issubset(inventory.columns):
            summary["recommended_replenishment_units"] = round(inventory["recommended_order_qty"].sum(), 2)
            if "inventory_status" in inventory.columns:
                summary["critical_inventory_pairs"] = int((inventory["inventory_status"].astype(str) == "CRITICAL").sum())
                summary["reorder_inventory_pairs"] = int((inventory["inventory_status"].astype(str) == "REORDER").sum())
    except FileNotFoundError:
        pass

    result = pd.DataFrame([summary])
    result.to_csv(OUTPUT_PATH, index=False)
    print(result.to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
