"""Post-run validation for generated control-tower artifacts."""

from pathlib import Path
import pandas as pd

DATA = Path("data")
EXPECTED_PAIRS = 150

REQUIRED = {
    "baseline_results.csv": ["model", "MAE", "RMSE", "MAPE", "sku_store_pairs"],
    "sku_xgboost_results.csv": ["model", "MAE", "RMSE", "MAPE", "sku_store_pairs"],
    "sku_30_day_forecast.csv": ["date", "store", "product", "forecast_demand"],
    "inventory_optimization_results.csv": ["store", "product", "inventory_status"],
    "supplier_risk_analysis.csv": ["supplier", "risk_score", "risk_level"],
    "disruption_detection.csv": ["supplier", "cluster", "anomaly_status", "anomaly_score", "disruption_score", "disruption_level"],
    "disruption_model_comparison.csv": ["model", "parameters", "silhouette", "davies_bouldin", "calinski_harabasz"],
    "disruption_pca.csv": ["supplier", "cluster", "pc1", "pc2"],
    "control_tower_inventory.csv": ["store", "product", "priority", "action"],
    "business_impact.csv": ["total_sku_store_pairs", "current_stockout_pairs", "historical_lost_sales_value"],
    "finance_summary.csv": ["revenue", "cogs", "gross_profit", "gross_margin_pct", "inventory_turnover", "days_inventory_outstanding", "annual_holding_cost"],
    "budget_vs_actual.csv": ["month", "actual_revenue", "budget_revenue", "revenue_variance", "revenue_variance_pct", "actual_cogs", "budget_cogs"],
    "promotion_effectiveness.csv": ["promo_event", "days", "units_sold", "revenue", "conversion_pct"],
    "abc_analysis.csv": ["product", "category", "revenue", "revenue_share_pct", "cumulative_share_pct", "abc_class"],
    "supplier_stockout_impact.csv": ["supplier", "lost_sales_units", "lost_sales_value", "stockout_days"],
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
        if not (
            xgb["MAE"] < seasonal["MAE"]
            and xgb["RMSE"] < seasonal["RMSE"]
            and xgb["MAPE"] < seasonal["MAPE"]
        ):
            errors.append("XGBoost does not beat the seasonal-naive baseline on all three metrics")

        pairs = int(xgb["sku_store_pairs"])
        if pairs != EXPECTED_PAIRS:
            errors.append(f"Expected {EXPECTED_PAIRS} store/SKU pairs, found {pairs}")

        forecast_pairs = frames["sku_30_day_forecast.csv"][["store", "product"]].drop_duplicates().shape[0]
        if forecast_pairs != pairs:
            errors.append(f"Forecast covers {forecast_pairs} store/SKU pairs; model reports {pairs}")

        disruption = frames["disruption_detection.csv"]
        if not disruption["disruption_score"].between(0, 100).all():
            errors.append("Disruption score must be between 0 and 100")
        if not disruption["anomaly_status"].isin(["NORMAL", "ANOMALY"]).all():
            errors.append("Unexpected anomaly status")

        finance = frames["finance_summary.csv"].iloc[0]
        if finance["revenue"] <= 0:
            errors.append("Finance revenue must be positive")
        if finance["cogs"] < 0:
            errors.append("Finance COGS cannot be negative")
        if finance["gross_profit"] != finance["revenue"] - finance["cogs"]:
            errors.append("Gross profit does not reconcile to revenue minus COGS")
        if not 0 <= finance["gross_margin_pct"] <= 100:
            errors.append("Gross margin percentage must be between 0 and 100")
        if finance["inventory_turnover"] < 0 or finance["days_inventory_outstanding"] < 0:
            errors.append("Inventory turnover and DIO must be non-negative")

        abc_classes = set(frames["abc_analysis"]["abc_class"].dropna().unique())
        if not abc_classes.issubset({"A", "B", "C"}):
            errors.append("ABC analysis contains invalid classification values")

        impact = frames["business_impact.csv"].iloc[0]
        if int(impact["total_sku_store_pairs"]) != EXPECTED_PAIRS:
            errors.append("Business impact pair count does not match expected coverage")
        if impact["historical_lost_sales_value"] < 0 or impact["current_inventory_value"] < 0:
            errors.append("Business impact financial values cannot be negative")

    if errors:
        print("OUTPUT VALIDATION FAILED")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("OUTPUT VALIDATION PASSED")
    print("- Required output files and schemas are present")
    print("- XGBoost beats the seasonal-naive baseline on MAE/RMSE/MAPE")
    print(f"- Forecast coverage matches the {EXPECTED_PAIRS} store/SKU pairs")
    print("- Finance KPIs reconcile and fall within valid ranges")
    print("- Business impact and ABC outputs are valid")
    print("- Disruption detection outputs and score ranges are valid")


if __name__ == "__main__":
    main()
