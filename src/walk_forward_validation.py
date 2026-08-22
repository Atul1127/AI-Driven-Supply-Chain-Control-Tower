"""Walk-forward validation utilities for time-series forecasting."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

DATA_PATH = "data/retail_sales_data.csv"
OUTPUT_PATH = "data/walk_forward_results.csv"
FEATURES = ["lag_1", "lag_7", "lag_14", "lag_30", "rolling_mean_7", "rolling_mean_30", "rolling_std_7", "day_of_week", "month", "day_of_month", "is_weekend", "promo_event", "average_discount"]


def mape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100)


def make_features(df):
    x = df.copy()
    x["lag_1"] = x.demand.shift(1)
    x["lag_7"] = x.demand.shift(7)
    x["lag_14"] = x.demand.shift(14)
    x["lag_30"] = x.demand.shift(30)
    shifted = x.demand.shift(1)
    x["rolling_mean_7"] = shifted.rolling(7).mean()
    x["rolling_mean_30"] = shifted.rolling(30).mean()
    x["rolling_std_7"] = shifted.rolling(7).std()
    x["day_of_week"] = x.index.dayofweek
    x["month"] = x.index.month
    x["day_of_month"] = x.index.day
    x["is_weekend"] = (x.day_of_week >= 5).astype(int)
    return x.dropna()


def main():
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    daily = raw.groupby("date").agg(demand=("demand", "sum"), promo_event=("promo_event", "sum"), average_discount=("discount_pct", "mean")).sort_index()
    data = make_features(daily)

    n_folds = 4
    min_train = int(len(data) * 0.60)
    remaining = len(data) - min_train
    fold_size = max(1, remaining // n_folds)
    rows = []

    for fold in range(n_folds):
        train_end = min_train + fold * fold_size
        test_end = min(len(data), train_end + fold_size)
        if test_end <= train_end:
            continue
        train, test = data.iloc[:train_end], data.iloc[train_end:test_end]
        model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42)
        model.fit(train[FEATURES], train.demand, verbose=False)
        pred = np.maximum(model.predict(test[FEATURES]), 0)
        rows.append({"fold": fold + 1, "train_rows": len(train), "test_rows": len(test), "MAE": mean_absolute_error(test.demand, pred), "RMSE": np.sqrt(mean_squared_error(test.demand, pred)), "MAPE": mape(test.demand, pred)})

    results = pd.DataFrame(rows)
    results.to_csv(OUTPUT_PATH, index=False)
    print(results.to_string(index=False))
    print("\nMean metrics:")
    print(results[["MAE", "RMSE", "MAPE"]].mean().to_string())
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
