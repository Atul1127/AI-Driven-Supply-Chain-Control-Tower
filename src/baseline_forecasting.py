"""Simple forecasting baselines for honest model benchmarking."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = "data/retail_sales_data.csv"
OUTPUT_PATH = "data/baseline_results.csv"


def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(name, actual, prediction):
    return {
        "model": name,
        "MAE": mean_absolute_error(actual, prediction),
        "RMSE": np.sqrt(mean_squared_error(actual, prediction)),
        "MAPE": mape(actual, prediction),
    }


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    daily = df.groupby("date")["demand"].sum().sort_index().to_frame("demand")
    daily["lag_1"] = daily["demand"].shift(1)
    daily["lag_7"] = daily["demand"].shift(7)
    daily = daily.dropna()

    split = int(len(daily) * 0.80)
    test = daily.iloc[split:]

    results = pd.DataFrame([
        evaluate("Naive-1-Day", test["demand"], test["lag_1"]),
        evaluate("Seasonal-Naive-7-Day", test["demand"], test["lag_7"]),
    ])
    results.to_csv(OUTPUT_PATH, index=False)
    print(results.to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
