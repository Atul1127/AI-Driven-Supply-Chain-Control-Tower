"""SKU/store-level naive baselines for fair forecasting benchmarks."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/retail_sales_data.csv"
OUTPUT_PATH = "data/baseline_results.csv"


def mape(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100) if mask.any() else 0.0


def evaluate(name, actual, prediction):
    return {"model": name, "MAE": mean_absolute_error(actual, prediction), "RMSE": np.sqrt(mean_squared_error(actual, prediction)), "MAPE": mape(actual, prediction)}


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    daily = df.groupby(["store", "product", "date"], as_index=False)["demand"].sum()
    daily["lag_1"] = daily.groupby(["store", "product"]).demand.shift(1)
    daily["lag_7"] = daily.groupby(["store", "product"]).demand.shift(7)
    daily = daily.dropna()

    cutoff = daily.date.max() - pd.Timedelta(days=60)
    test = daily[daily.date > cutoff]
    if test.empty:
        raise ValueError("Temporal split produced an empty test set.")

    results = pd.DataFrame([
        evaluate("SKU Naive-1-Day", test.demand, test.lag_1),
        evaluate("SKU Seasonal-Naive-7-Day", test.demand, test.lag_7),
    ])
    results["sku_store_pairs"] = daily[["store", "product"]].drop_duplicates().shape[0]
    results.to_csv(OUTPUT_PATH, index=False)
    print(results.to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
