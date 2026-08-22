"""
forecasting.py
--------------
Trains a Random Forest Regressor to forecast daily units sold per store-product.
Saves model, forecasts CSV, and actual-vs-predicted chart.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import os
from sklearn.ensemble          import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model      import LinearRegression
from sklearn.model_selection   import train_test_split
from sklearn.metrics           import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing     import LabelEncoder

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE       = os.path.dirname(__file__)
DATA_PATH  = os.path.join(BASE, "..", "data",           "retail_sales_clean.csv")
MODEL_DIR  = os.path.join(BASE, "..", "models")
OUT_DIR    = os.path.join(BASE, "..", "outputs")
CHART_DIR  = os.path.join(OUT_DIR, "charts")
TABLE_DIR  = os.path.join(OUT_DIR, "tables")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs(TABLE_DIR, exist_ok=True)

# ─── Feature set ─────────────────────────────────────────────────────────────
FEATURES = [
    "month", "day", "day_of_week", "week_of_year", "quarter",
    "is_weekend", "is_month_end",
    "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_mean_30", "rolling_std_7",
    "unit_price", "discount_pct", "promo_event",
    "stock_level", "reorder_point",
    "store_enc", "category_enc", "product_enc",
]
TARGET = "units_sold"


def load_and_encode():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    le_store    = LabelEncoder()
    le_category = LabelEncoder()
    le_product  = LabelEncoder()

    df["store_enc"]    = le_store.fit_transform(df["store"])
    df["category_enc"] = le_category.fit_transform(df["category"])
    df["product_enc"]  = le_product.fit_transform(df["product"])

    # Save encoders for later use
    joblib.dump(le_store,    os.path.join(MODEL_DIR, "le_store.pkl"))
    joblib.dump(le_category, os.path.join(MODEL_DIR, "le_category.pkl"))
    joblib.dump(le_product,  os.path.join(MODEL_DIR, "le_product.pkl"))

    return df, le_store, le_category, le_product


def train_test_split_temporal(df):
    """Use last 60 days as test set (temporal split – no data leakage)."""
    cutoff = df["date"].max() - pd.Timedelta(days=60)
    train  = df[df["date"] <= cutoff].copy()
    test   = df[df["date"] >  cutoff].copy()
    print(f"Train: {train.shape[0]:,} rows  |  Test: {test.shape[0]:,} rows")
    return train, test


def train_model(train):
    X_train = train[FEATURES].fillna(0)
    y_train = train[TARGET]

    model = RandomForestRegressor(
        n_estimators = 200,
        max_depth    = 12,
        min_samples_leaf = 5,
        n_jobs       = -1,
        random_state = 42,
    )
    print("Training Random Forest …")
    model.fit(X_train, y_train)
    joblib.dump(model, os.path.join(MODEL_DIR, "rf_model.pkl"))
    print("Model saved → models/rf_model.pkl")
    return model


def evaluate(model, test):
    X_test = test[FEATURES].fillna(0)
    y_test = test[TARGET]

    preds = model.predict(X_test)
    preds = np.maximum(preds, 0)          # no negative predictions

    mae  = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2   = r2_score(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / (y_test + 1))) * 100

    print(f"\n── Model Evaluation ─────────────────────────────────────")
    print(f"  MAE  : {mae:.2f}  (units)")
    print(f"  RMSE : {rmse:.2f} (units)")
    print(f"  R²   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print(f"─────────────────────────────────────────────────────────\n")

    test = test.copy()
    test["predicted_units"] = np.round(preds).astype(int)
    test["residual"]        = test[TARGET] - test["predicted_units"]

    # Save metrics
    metrics = pd.DataFrame([{"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}])
    metrics.to_csv(os.path.join(TABLE_DIR, "model_metrics.csv"), index=False)

    return test


def plot_actual_vs_predicted(test_df):
    """Aggregate to daily total and plot."""
    daily = test_df.groupby("date").agg(
        Actual    = ("units_sold",       "sum"),
        Predicted = ("predicted_units",  "sum"),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily["date"], daily["Actual"],    label="Actual",    color="#e63946", linewidth=2)
    ax.plot(daily["date"], daily["Predicted"], label="Predicted", color="#2a9d8f", linewidth=2, linestyle="--")
    ax.fill_between(daily["date"],
                    daily["Actual"], daily["Predicted"],
                    alpha=0.12, color="orange", label="Error Band")
    ax.set_title("Actual vs Predicted Daily Units Sold (Test Period)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Units Sold")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "09_actual_vs_predicted.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_feature_importance(model):
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(15)
    fig, ax = plt.subplots(figsize=(10, 7))
    fi.plot(kind="barh", ax=ax, color="#4cc9f0")
    ax.set_title("Feature Importance (Random Forest)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "10_feature_importance.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def generate_30day_forecast(model, df, le_store, le_category, le_product):
    """
    For each store-product combo, create 30 future rows using last known
    rolling stats and lag features, then predict.
    """
    future_rows = []
    last_date   = df["date"].max()

    for (store, product), grp in df.groupby(["store", "product"]):
        grp     = grp.sort_values("date")
        last    = grp.iloc[-1]
        hist_7  = grp["units_sold"].tail(7).tolist()
        hist_30 = grp["units_sold"].tail(30).tolist()

        for d in range(1, 31):
            fut_date = last_date + pd.Timedelta(days=d)
            lag7     = hist_7[-7]  if len(hist_7)  >= 7  else np.mean(hist_7)
            lag14    = hist_7[-7]  if len(hist_7)  >= 7  else np.mean(hist_7)  # approx
            lag30    = hist_30[-30] if len(hist_30) >= 30 else np.mean(hist_30)
            rm7      = np.mean(hist_7[-7:])
            rm30     = np.mean(hist_30[-30:])
            std7     = np.std(hist_7[-7:]) if len(hist_7) >= 2 else 0

            row = {
                "date":         fut_date,
                "store":        store,
                "category":     last["category"],
                "product":      product,
                "month":        fut_date.month,
                "day":          fut_date.day,
                "day_of_week":  fut_date.dayofweek,
                "week_of_year": fut_date.isocalendar()[1],
                "quarter":      (fut_date.month - 1) // 3 + 1,
                "is_weekend":   int(fut_date.dayofweek >= 5),
                "is_month_end": int(fut_date == fut_date + pd.offsets.MonthEnd(0)),
                "lag_7":        lag7,
                "lag_14":       lag14,
                "lag_30":       lag30,
                "rolling_mean_7":  rm7,
                "rolling_mean_30": rm30,
                "rolling_std_7":   std7,
                "unit_price":   last["unit_price"],
                "discount_pct": 0,
                "promo_event":  0,
                "stock_level":  last["stock_level"],
                "reorder_point":last["reorder_point"],
                "store_enc":    le_store.transform([store])[0],
                "category_enc": le_category.transform([last["category"]])[0],
                "product_enc":  le_product.transform([product])[0],
            }
            future_rows.append(row)
            hist_7.append(rm7)    # use rolling mean as proxy
            hist_30.append(rm7)

    future_df = pd.DataFrame(future_rows)
    X_fut     = future_df[FEATURES].fillna(0)
    preds     = model.predict(X_fut)
    future_df["predicted_units"] = np.maximum(np.round(preds).astype(int), 0)

    out = os.path.join(TABLE_DIR, "30day_forecast.csv")
    future_df[["date","store","category","product","predicted_units"]].to_csv(out, index=False)
    print(f"30-day forecast saved → {out}")
    return future_df


def plot_30day_forecast(future_df):
    daily = future_df.groupby("date")["predicted_units"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(daily["date"], daily["predicted_units"], color="#4361ee", alpha=0.8, width=0.8)
    ax.plot(daily["date"], daily["predicted_units"], color="#f72585", linewidth=2, marker="o", markersize=4)
    ax.set_title("30-Day Sales Forecast (All Stores – All Products)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Date"); ax.set_ylabel("Predicted Units Sold")
    plt.xticks(rotation=30)
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "11_30day_forecast.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def run_forecasting():
    df, le_store, le_category, le_product = load_and_encode()
    train, test                            = train_test_split_temporal(df)
    model                                  = train_model(train)
    test_preds                             = evaluate(model, test)
    test_preds.to_csv(os.path.join(TABLE_DIR, "test_predictions.csv"), index=False)
    plot_actual_vs_predicted(test_preds)
    plot_feature_importance(model)
    future = generate_30day_forecast(model, df, le_store, le_category, le_product)
    plot_30day_forecast(future)
    print("\n✅ Forecasting complete.")
    return model, test_preds, future


if __name__ == "__main__":
    run_forecasting()
