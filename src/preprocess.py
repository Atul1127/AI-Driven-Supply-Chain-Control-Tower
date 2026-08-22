"""
preprocess.py
-------------
Loads raw retail CSV, cleans it, engineers features, and saves a clean version.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RAW_PATH   = os.path.join(DATA_DIR, "retail_sales_data.csv")
CLEAN_PATH = os.path.join(DATA_DIR, "retail_sales_clean.csv")


def load_data(path=RAW_PATH):
    df = pd.read_csv(path, parse_dates=["date"])
    print(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def check_quality(df):
    print("\n── Data Quality Report ──────────────────────────────────")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"\nDuplicates: {df.duplicated().sum()}")
    print(f"Date range : {df['date'].min().date()}  →  {df['date'].max().date()}")
    print(f"Stores     : {df['store'].nunique()} unique")
    print(f"Products   : {df['product'].nunique()} unique")
    print(f"Categories : {df['category'].nunique()} unique")
    print("─────────────────────────────────────────────────────────\n")


def clean(df):
    # Drop exact duplicates
    df = df.drop_duplicates()

    # Fill any missing numeric with 0 (safety)
    num_cols = df.select_dtypes(include=np.number).columns
    df[num_cols] = df[num_cols].fillna(0)

    # Ensure non-negative values
    for col in ["units_sold", "stock_level", "revenue", "demand"]:
        df[col] = df[col].clip(lower=0)

    return df


def engineer_features(df):
    """Add time, lag, and rolling features used by the forecasting model."""
    df = df.sort_values(["store", "product", "date"]).copy()

    # ── Calendar features ───────────────────────────────────────────────────
    df["year"]         = df["date"].dt.year
    df["month"]        = df["date"].dt.month
    df["day"]          = df["date"].dt.day
    df["day_of_week"]  = df["date"].dt.dayofweek          # 0=Mon … 6=Sun
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"]      = df["date"].dt.quarter
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    # ── Lag features (per store-product group) ──────────────────────────────
    grp = df.groupby(["store", "product"])["units_sold"]
    df["lag_7"]  = grp.shift(7)
    df["lag_14"] = grp.shift(14)
    df["lag_30"] = grp.shift(30)

    # ── Rolling statistics ──────────────────────────────────────────────────
    df["rolling_mean_7"]  = grp.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).mean())
    df["rolling_mean_30"] = grp.transform(lambda x: x.shift(1).rolling(30, min_periods=1).mean())
    df["rolling_std_7"]   = grp.transform(lambda x: x.shift(1).rolling(7,  min_periods=1).std().fillna(0))

    # ── Revenue-related ─────────────────────────────────────────────────────
    df["revenue_lag_7"]  = df.groupby(["store", "product"])["revenue"].shift(7)
    df["profit_margin"]  = ((df["sell_price"] - df["unit_price"] * 0.6) / df["sell_price"]).round(4)

    # ── Stock health ────────────────────────────────────────────────────────
    df["stock_to_demand_ratio"] = (df["stock_level"] / (df["demand"] + 1)).round(4)

    # Drop rows with NaN lags (first 30 days per group)
    df = df.dropna(subset=["lag_7", "lag_14", "lag_30"])

    print(f"After feature engineering: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df.reset_index(drop=True)


def preprocess_pipeline():
    df = load_data()
    check_quality(df)
    df = clean(df)
    df = engineer_features(df)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Clean dataset saved → {CLEAN_PATH}")
    return df


if __name__ == "__main__":
    preprocess_pipeline()
