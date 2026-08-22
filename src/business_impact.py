"""Translate inventory decisions into measurable business impact."""

import pandas as pd

DATA_PATH = "data/retail_sales_data.csv"
OUTPUT_PATH = "data/business_impact.csv"


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    latest = df.sort_values("date").groupby(["store", "product"], as_index=False).tail(1).copy()
    latest["days_of_stock"] = latest["stock_level"] / latest["demand"].clip(lower=1)
    latest["stockout_risk"] = latest["days_of_stock"] < 7
    latest["estimated_lost_sales_value"] = latest["lost_sales"] * latest["unit_price"]
    summary = pd.DataFrame([{
        "total_sku_store_pairs": len(latest),
        "stockout_risk_pairs": int(latest.stockout_risk.sum()),
        "stockout_risk_pct": round(latest.stockout_risk.mean() * 100, 2),
        "estimated_lost_sales_value": round(latest.estimated_lost_sales_value.sum(), 2),
        "inventory_value": round((latest.stock_level * latest.unit_price).sum(), 2),
    }])
    summary.to_csv(OUTPUT_PATH, index=False)
    print(summary.to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
