"""Translate inventory state and optimization decisions into business KPIs."""

import os
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "retail_sales_data.csv")
INVENTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "inventory_optimization_results.csv")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "business_impact.csv")


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    required = {"store", "product", "date", "demand", "lost_sales", "unit_price", "closing_stock"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # Use realized sell price for lost-sales exposure when available because it
    # reflects discounts/promotions; fall back to list price for compatibility.
    lost_sales_price = df["sell_price"] if "sell_price" in df.columns else df["unit_price"]

    latest = df.sort_values("date").groupby(["store", "product"], as_index=False).tail(1).copy()
    latest["current_stockout"] = latest["closing_stock"] <= 0

    historical_lost_sales_value = float((df["lost_sales"] * lost_sales_price).sum())
    historical_stockout_days = (
        int(df["stockout"].sum())
        if "stockout" in df.columns
        else int((df["lost_sales"] > 0).sum())
    )

    # Inventory value uses the same 60% cost assumption as finance_analysis.py.
    current_inventory_value = float(
        (latest["closing_stock"] * latest["unit_price"] * 0.60).sum()
    )

    summary = {
        "total_sku_store_pairs": len(latest),
        "current_stockout_pairs": int(latest["current_stockout"].sum()),
        "historical_stockout_days": historical_stockout_days,
        "historical_lost_sales_value": round(historical_lost_sales_value, 2),
        "current_inventory_value": round(current_inventory_value, 2),
    }

    try:
        inventory = pd.read_csv(INVENTORY_PATH)
        if "recommended_order_qty" in inventory.columns:
            summary["recommended_replenishment_units"] = round(
                inventory["recommended_order_qty"].sum(), 2
            )
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
