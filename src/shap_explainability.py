"""Generate SHAP feature importance for the SKU/store XGBoost forecaster."""

import pandas as pd
import shap
from xgboost import XGBRegressor

DATA_PATH = "data/retail_sales_data.csv"
OUTPUT_PATH = "data/sku_xgboost_shap_importance.csv"
FEATURES = ["lag_1", "lag_7", "lag_14", "lag_30", "rolling_mean_7", "rolling_mean_30", "rolling_std_7", "day_of_week", "month", "day_of_month", "is_weekend", "promo_event", "discount_pct"]


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
    raw = pd.read_csv(DATA_PATH, parse_dates=["date"])
    daily = raw.groupby(["store", "product", "category", "date"], as_index=False).agg(
        demand=("demand", "sum"), promo_event=("promo_event", "sum"), discount_pct=("discount_pct", "mean")
    )
    parts = [build_features(g) for _, g in daily.groupby(["store", "product"], sort=False) if len(g) > 60]
    data = pd.concat(parts, ignore_index=True).dropna(subset=FEATURES)
    cutoff = data.date.max() - pd.Timedelta(days=60)
    train = data[data.date <= cutoff]

    model = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42, n_jobs=-1)
    model.fit(train[FEATURES], train.demand, verbose=False)
    sample = train[FEATURES].tail(min(5000, len(train)))
    values = shap.TreeExplainer(model).shap_values(sample)
    importance = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(OUTPUT_PATH, index=False)
    print(importance.to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
