"""Build the executive control-tower dataset from SKU forecasts and risk."""

import pandas as pd

INVENTORY_PATH = "data/inventory_optimization_results.csv"
SUPPLIER_PATH = "data/supplier_risk_analysis.csv"
FORECAST_PATH = "data/sku_30_day_forecast.csv"
OUTPUT_PATH = "data/control_tower_inventory.csv"

PRIORITY_ORDER = {"URGENT": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}


def main():
    inventory = pd.read_csv(INVENTORY_PATH)
    suppliers = pd.read_csv(SUPPLIER_PATH)
    forecast = pd.read_csv(FORECAST_PATH, parse_dates=["date"])

    forecast_kpi = forecast.groupby(["store", "product"], as_index=False).agg(
        average_30_day_forecast=("forecast_demand", "mean"),
        total_30_day_forecast=("forecast_demand", "sum"),
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

    control["priority_rank"] = control["priority"].map(PRIORITY_ORDER).fillna(9)
    control = control.sort_values(["priority_rank", "shortage_to_rop"], ascending=[True, False]).drop(columns="priority_rank")
    control.to_csv(OUTPUT_PATH, index=False)
    print(f"Rows: {len(control):,}")
    print(control["priority"].value_counts().reindex(["URGENT", "HIGH", "MEDIUM", "LOW"], fill_value=0).to_string())
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
