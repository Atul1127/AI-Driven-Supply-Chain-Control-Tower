"""Build the executive control-tower dataset from SKU forecasts and risk."""

import pandas as pd

INVENTORY_PATH = "data/inventory_optimization_results.csv"
SUPPLIER_PATH = "data/supplier_risk_analysis.csv"
FORECAST_PATH = "data/sku_30_day_forecast.csv"
OUTPUT_PATH = "data/control_tower_inventory.csv"


def main():
    inventory = pd.read_csv(INVENTORY_PATH)
    suppliers = pd.read_csv(SUPPLIER_PATH)
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["date"])

    forecast_kpi = forecast.groupby(["store", "product"], as_index=False).agg(
        average_30_day_forecast=("forecast_demand", "mean"),
        total_30_day_forecast=("forecast_demand", "sum")
    )

    control = inventory.merge(forecast_kpi, on=["store", "product"], how="left")
    control = control.merge(suppliers, on="supplier", how="left")

    control["priority"] = "LOW"
    control.loc[control["inventory_status"] == "REORDER", "priority"] = "MEDIUM"
    control.loc[control["inventory_status"] == "CRITICAL", "priority"] = "HIGH"
    control.loc[(control["inventory_status"] == "CRITICAL") & (control["risk_level"].astype(str) == "HIGH"), "priority"] = "URGENT"

    control["action"] = "Monitor"
    control.loc[control["priority"] == "MEDIUM", "action"] = "Plan replenishment"
    control.loc[control["priority"] == "HIGH", "action"] = "Expedite replenishment"
    control.loc[control["priority"] == "URGENT", "action"] = "Expedite + supplier escalation"

    control = control.sort_values(["priority", "shortage_to_rop"], ascending=[True, False])
    control.to_csv(OUTPUT_PATH, index=False)
    print(f"Rows: {len(control):,}")
    print(control["priority"].value_counts().to_string())
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
