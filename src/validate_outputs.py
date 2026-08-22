"""Lightweight post-run validation for generated control-tower artifacts."""

from pathlib import Path
import pandas as pd

DATA = Path("data")

REQUIRED = {
    "baseline_results.csv": ["model", "MAE", "RMSE", "MAPE", "sku_store_pairs"],
    "sku_xgboost_results.csv": ["model", "MAE", "RMSE", "MAPE", "sku_store_pairs"],
    "sku_30_day_forecast.csv": ["date", "store", "product", "forecast_demand"],
    "inventory_optimization_results.csv": ["store", "product", "inventory_status"],
    "supplier_risk_analysis.csv": ["supplier", "risk_score", "risk_level"],
    "control_tower_inventory.csv": ["store", "product", "priority", "action"],
    "business_impact.csv": ["total_sku_store_pairs", "current_stockout_pairs", "historical_lost_sales_value"],
    "sku_xgboost_shap_importance.csv": ["feature", "mean_abs_shap"],
}


def main():
    errors = []
    frames = {}
    for filename, columns in REQUIRED.items():
        path = DATA / filename
        if not path.exists():
            errors.append(f"Missing output: {path}")
            continue
        df = pd.read_csv(path)
        frames[filename] = df
        missing = [c for c in columns if c not in df.columns]
        if missing:
            errors.append(f"{filename}: missing columns {missing}")
        if df.empty:
            errors.append(f"{filename}: file is empty")

    if not errors:
        baseline = frames["baseline_results.csv"].set_index("model")
        xgb = frames["sku_xgboost_results.csv"].iloc[0]
        seasonal = baseline.loc["SKU Seasonal-Naive-7-Day"]
        if not (xgb["MAE"] < seasonal["MAE"] and xgb["RMSE"] < seasonal["RMSE"] and xgb["MAPE"] < seasonal["MAPE"]):
            errors.append("XGBoost does not beat the seasonal-naive baseline on all three metrics")

        pairs = int(xgb["sku_store_pairs"])
        if pairs != 150:
            errors.append(f"Expected 150 store/SKU pairs, found {pairs}")

        forecast_pairs = frames["sku_30_day_forecast.csv"][["store", "product"]].drop_duplicates().shape[0]
        if forecast_pairs != pairs:
            errors.append(f"Forecast covers {forecast_pairs} store/SKU pairs; model reports {pairs}")

    if errors:
        print("OUTPUT VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("OUTPUT VALIDATION PASSED")
    print("- Required output files and schemas are present")
    print("- XGBoost beats the seasonal-naive baseline on MAE/RMSE/MAPE")
    print("- Forecast coverage matches the 150 store/SKU pairs")


if __name__ == "__main__":
    main()
