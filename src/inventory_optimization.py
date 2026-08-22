"""Forecast-driven SKU/store inventory optimization."""

import numpy as np
import pandas as pd

RETAIL_PATH = "data/retail_sales_data.csv"
FORECAST_PATH = "data/sku_30_day_forecast.csv"
OUTPUT_PATH = "data/inventory_optimization_results.csv"
SERVICE_LEVEL_Z = 1.645
ORDERING_COST = 2000.0
HOLDING_RATE = 0.25
ORDER_CAP_DAYS = 30


def main():
    retail = pd.read_csv(RETAIL_PATH, parse_dates=["date"])
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["date"])
    inv = retail.groupby(["store", "product"], as_index=False).agg(category=("category", "first"), supplier=("supplier", "first"), annual_demand=("demand", "sum"), unit_price=("unit_price", "mean"), lead_time_days=("lead_time_days", "mean"), current_stock=("closing_stock", "last"), demand_std=("demand", "std"))
    fc = forecast.groupby(["store", "product"], as_index=False).agg(forecast_daily_demand=("forecast_demand", "mean"), forecast_30_day_demand=("forecast_demand", "sum"))
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
    result["inventory_status"] = np.select([result.current_stock <= result.safety_stock, result.current_stock <= result.reorder_point], ["CRITICAL", "REORDER"], default="NORMAL")
    result["shortage_to_rop"] = np.maximum(result["reorder_point"] - result["current_stock"], 0)
    result["recommended_order_qty"] = np.where(result.current_stock <= result.reorder_point, result.recommended_order_qty, 0)
    result["days_of_stock"] = result["current_stock"] / result["forecast_daily_demand"].clip(lower=0.01)
    for col in ["forecast_daily_demand", "forecast_30_day_demand", "safety_stock", "expected_lead_time_demand", "reorder_point", "eoq", "recommended_order_qty", "shortage_to_rop", "days_of_stock"]:
        result[col] = result[col].round(2)
    result.to_csv(OUTPUT_PATH, index=False)
    print(result["inventory_status"].value_counts().to_string())
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
