"""Generate SHAP explanations for the XGBoost forecasting model."""

import pandas as pd
import shap
from xgboost import XGBRegressor

DATA_PATH = "data/retail_sales_data.csv"
OUTPUT_PATH = "data/xgboost_shap_importance.csv"
FEATURES = ["lag_1", "lag_7", "lag_14", "lag_30", "rolling_mean_7", "rolling_mean_30", "rolling_std_7", "day_of_week", "month", "day_of_month", "is_weekend", "promo_event", "average_discount"]


def main():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    daily = df.groupby("date").agg(demand=("demand", "sum"), promo_event=("promo_event", "sum"), average_discount=("discount_pct", "mean")).sort_index()
    daily["lag_1"] = daily.demand.shift(1)
    daily["lag_7"] = daily.demand.shift(7)
    daily["lag_14"] = daily.demand.shift(14)
    daily["lag_30"] = daily.demand.shift(30)
    shifted = daily.demand.shift(1)
    daily["rolling_mean_7"] = shifted.rolling(7).mean()
    daily["rolling_mean_30"] = shifted.rolling(30).mean()
    daily["rolling_std_7"] = shifted.rolling(7).std()
    daily["day_of_week"] = daily.index.dayofweek
    daily["month"] = daily.index.month
    daily["day_of_month"] = daily.index.day
    daily["is_weekend"] = (daily.day_of_week >= 5).astype(int)
    data = daily.dropna()
    split = int(len(data) * 0.80)
    train = data.iloc[:split]
    model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, objective="reg:squarederror", random_state=42)
    model.fit(train[FEATURES], train.demand, verbose=False)
    sample = train[FEATURES].tail(min(5000, len(train)))
    values = shap.TreeExplainer(model).shap_values(sample)
    importance = pd.DataFrame({"feature": FEATURES, "mean_abs_shap": abs(values).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(OUTPUT_PATH, index=False)
    print(importance.to_string(index=False))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
