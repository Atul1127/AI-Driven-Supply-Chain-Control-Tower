"""SKU/store-level demand forecasting with leakage-safe lag features."""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

DATA_PATH = "data/retail_sales_data.csv"
RESULTS_PATH = "data/sku_xgboost_results.csv"
FORECAST_PATH = "data/sku_30_day_forecast.csv"
OUTPUT_DIR = "images/forecasting"
FEATURES = ["lag_1", "lag_7", "lag_14", "lag_30", "rolling_mean_7", "rolling_mean_30", "rolling_std_7", "day_of_week", "month", "day_of_month", "is_weekend", "promo_event", "discount_pct"]


def mape(y, p):
    y, p = np.asarray(y), np.asarray(p)
    mask = y != 0
    return float(np.mean(np.abs((y[mask] - p[mask]) / y[mask])) * 100) if mask.any() else 0.0


def build_features(group):
    x = group.sort_values("date").copy()
    x["lag_1"] = x.demand.shift(1)
    x["lag_7"] = x.demand.shift(7)
    x["lag_14"] = x.demand.shift(14)
    x["lag_30"] = x.demand.shift(30)
    shifted = x.demand.shift(1)
    x["rolling_mean_7"] = shifted.rolling(7).mean()
    x["rolling_mean_30"] = shifted.rolling(30).mean()
    x["rolling_std_7"] = shifted.rolling(7).std()
    x["day_of_week"] = x.date.dt.dayofweek
    x["month"] = x.date.dt.month
    x["day_of_month"] = x.date.dt.day
    x["is_weekend"] = (x.day_of_week >= 5).astype(int)
    return x


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    required = {"store", "product", "category", "date", "demand", "promo_event", "discount_pct", "unit_price"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    # One observation per store/product/day. The source dataset uses closing_stock,
    # not stock_level; inventory state is intentionally handled downstream.
    daily = raw.groupby(["store", "product", "category", "date"], as_index=False).agg(
        demand=("demand", "sum"), promo_event=("promo_event", "sum"),
        discount_pct=("discount_pct", "mean"), unit_price=("unit_price", "mean")
    )

    parts = []
    for _, group in daily.groupby(["store", "product"], sort=False):
        part = build_features(group)
        if len(part) > 60:
            parts.append(part)
    if not parts:
        raise ValueError("No store/product series has enough history for lag features.")

    data = pd.concat(parts, ignore_index=True).dropna(subset=FEATURES)
    cutoff = data.date.max() - pd.Timedelta(days=60)
    train, test = data[data.date <= cutoff], data[data.date > cutoff]
    if train.empty or test.empty:
        raise ValueError("Temporal split produced an empty train or test set.")

    model = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train.demand, verbose=False)
    pred = np.maximum(model.predict(test[FEATURES]), 0)

    results = pd.DataFrame([{
        "model": "SKU-Store XGBoost", "train_rows": len(train), "test_rows": len(test),
        "MAE": mean_absolute_error(test.demand, pred),
        "RMSE": np.sqrt(mean_squared_error(test.demand, pred)), "MAPE": mape(test.demand, pred),
        "sku_store_pairs": data[["store", "product"]].drop_duplicates().shape[0]
    }])
    results.to_csv(RESULTS_PATH, index=False)

    test_out = test[["date", "store", "product", "category", "demand"]].copy()
    test_out["predicted_demand"] = pred
    test_out.to_csv("data/sku_test_predictions.csv", index=False)

    forecasts = []
    for (store, product), group in daily.groupby(["store", "product"], sort=False):
        hist = group.sort_values("date").copy()
        if len(hist) < 30:
            continue
        category = hist.category.iloc[-1]
        price = hist.unit_price.iloc[-1]
        for _ in range(30):
            date = hist.date.max() + pd.Timedelta(days=1)
            demand_series = hist.demand
            row = {
                "date": date, "store": store, "product": product, "category": category,
                "demand": np.nan, "promo_event": 0, "discount_pct": 0.0,
                "lag_1": demand_series.iloc[-1], "lag_7": demand_series.iloc[-7],
                "lag_14": demand_series.iloc[-14], "lag_30": demand_series.iloc[-30],
                "rolling_mean_7": demand_series.tail(7).mean(), "rolling_mean_30": demand_series.tail(30).mean(),
                "rolling_std_7": demand_series.tail(30).std(), "day_of_week": date.dayofweek,
                "month": date.month, "day_of_month": date.day, "is_weekend": int(date.dayofweek >= 5)
            }
            prediction = float(max(model.predict(pd.DataFrame([row])[FEATURES])[0], 0))
            forecasts.append({"date": date, "store": store, "product": product, "category": category, "forecast_demand": round(prediction, 2), "unit_price": price})
            hist = pd.concat([hist, pd.DataFrame([{"date": date, "store": store, "product": product, "category": category, "demand": prediction, "promo_event": 0, "discount_pct": 0.0, "unit_price": price}])], ignore_index=True)

    forecast_df = pd.DataFrame(forecasts)
    if forecast_df.empty:
        raise ValueError("No SKU/store forecasts were generated.")
    forecast_df.to_csv(FORECAST_PATH, index=False)
    # Compatibility output for legacy scripts while keeping SKU-level rows.
    forecast_df.to_csv("data/30_day_xgboost_forecast.csv", index=False)
    print(results.to_string(index=False))
    print(f"Saved: {RESULTS_PATH}")
    print(f"Saved: {FORECAST_PATH}")


if __name__ == "__main__":
    main()
