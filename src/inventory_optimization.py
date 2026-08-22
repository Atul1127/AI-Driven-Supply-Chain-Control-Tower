"""Forecast-driven SKU/store inventory optimization with realistic annualization."""

import numpy as np
import pandas as pd

RETAIL_PATH = "data/retail_sales_data.csv"
FORECAST_PATH = "data/sku_30_day_forecast.csv"
OUTPUT_PATH = "data/inventory_optimization_results.csv"
SERVICE_LEVEL_Z = 1.645  # 95% cycle-service level
ORDERING_COST = 2000.0
HOLDING_RATE = 0.25
ORDER_CAP_DAYS = 30


def main():
    retail = pd.read_csv(RETAIL_PATH, parse_dates=["date"])
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["date"])

    days = max(int((retail["date"].max() - retail["date"].min()).days) + 1, 1)
    grouped = retail.groupby(["store", "product"], as_index=False)
    inv = grouped.agg(
        category=("category", "first"),
        supplier=("supplier", "first"),
        total_demand=("demand", "sum"),
        unit_price=("unit_price", "mean"),
        lead_time_days=("lead_time_days", "mean"),
        current_stock=("closing_stock", "last"),
        demand_std=("demand", "std"),
        historical_stockout_days=("stockout", "sum"),
        observed_days=("date", "count"),
    )
    inv["annual_demand"] = inv["total_demand"] * (365.0 / days)
    inv["historical_stockout_rate"] = inv["historical_stockout_days"] / inv["observed_days"].clip(lower=1)

    fc = forecast.groupby(["store", "product"], as_index=False).agg(
        forecast_daily_demand=("forecast_demand", "mean"),
        forecast_30_day_demand=("forecast_demand", "sum"),
    )
    result = inv.merge(fc, on=["store", "product"], how="left")
    result["forecast_daily_demand"] = result["forecast_daily_demand"].fillna(result["annual_demand"] / 365)
    result["forecast_30_day_demand"] = result["forecast_30_day_demand"].fillna(result["forecast_daily_demand"] * ORDER_CAP_DAYS)
    result["demand_std"] = result["demand_std"].fillna(result["demand_std"].median()).fillna(0)
    result["safety_stock"] = SERVICE_LEVEL_Z * result["demand_std"] * np.sqrt(result["lead_time_days"].clip(lower=1))
    result["expected_lead_time_demand"] = result["forecast_daily_demand"] * result["lead_time_days"]
    result["reorder_point"] = result["expected_lead_time_demand"] + result["safety_stock"]

    holding = (result["unit_price"] * HOLDING_RATE).clip(lower=0.01)
    result["eoq"] = np.sqrt((2 * result["annual_demand"].clip(lower=1) * ORDERING_COST) / holding)
    result["recommended_order_qty"] = np.minimum(result["eoq"], result["forecast_30_day_demand"])

    result["inventory_status"] = np.select(
        [result.current_stock <= result.safety_stock, result.current_stock <= result.reorder_point],
        ["CRITICAL", "REORDER"],
        default="NORMAL",
    )
    result["shortage_to_rop"] = np.maximum(result["reorder_point"] - result["current_stock"], 0)
    result["recommended_order_qty"] = np.where(result.current_stock <= result.reorder_point, result.recommended_order_qty, 0)
    result["days_of_stock"] = result["current_stock"] / result["forecast_daily_demand"].clip(lower=0.01)
    result["current_stockout"] = (result["current_stock"] <= 0).astype(int)
    result["low_coverage_7d"] = (result["days_of_stock"] < 7).astype(int)

    for col in [
        "annual_demand", "forecast_daily_demand", "forecast_30_day_demand", "demand_std",
        "safety_stock", "expected_lead_time_demand", "reorder_point", "eoq",
        "recommended_order_qty", "shortage_to_rop", "days_of_stock", "historical_stockout_rate",
    ]:
        result[col] = result[col].round(2)

    result = result.sort_values(["inventory_status", "shortage_to_rop"], ascending=[True, False])
    result.to_csv(OUTPUT_PATH, index=False)
    print(result["inventory_status"].value_counts().to_string())
    print(f"Current stockouts: {int(result['current_stockout'].sum())}/{len(result)}")
    print(f"Low coverage (<7 days): {int(result['low_coverage_7d'].sum())}/{len(result)}")
    print(f"Historical stockout rate: {result['historical_stockout_rate'].mean() * 100:.2f}%")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
