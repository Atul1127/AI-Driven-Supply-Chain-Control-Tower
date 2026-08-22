"""
main.py
-------
Master runner – executes the full pipeline end-to-end.

Usage:
    python main.py

Steps:
  1. Generate synthetic retail dataset
  2. Preprocess & feature engineering
  3. Exploratory data analysis (8 charts)
  4. Sales forecasting (Random Forest + 30-day forecast)
  5. Inventory optimization (safety stock / ROP / EOQ / alerts)
  6. HTML business report generation
"""

import sys
import os
import time

# Make sure src/ is on the path when running from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_dataset        import generate
from preprocess              import preprocess_pipeline
from eda                     import run_eda
from forecasting             import run_forecasting
from inventory_optimization  import run_inventory_optimization
from generate_report         import build_report

import pandas as pd


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    start = time.time()
    print("\n🛍️  Retail Sales Forecasting & Inventory Optimization System")
    print("   Student Portfolio Project – Full Pipeline Run\n")

    # ── Step 1: Dataset ──────────────────────────────────────────────────────
    section("STEP 1 / 6 · Generating Synthetic Dataset")
    data_path = os.path.join(os.path.dirname(__file__), "data", "retail_sales_data.csv")
    if os.path.exists(data_path):
        print("Dataset already exists. Skipping generation.")
    else:
        df = generate()
        df.to_csv(data_path, index=False)
        print(f"Generated {len(df):,} rows.")

    # ── Step 2: Preprocess ───────────────────────────────────────────────────
    section("STEP 2 / 6 · Preprocessing & Feature Engineering")
    clean_df = preprocess_pipeline()

    # ── Step 3: EDA ──────────────────────────────────────────────────────────
    section("STEP 3 / 6 · Exploratory Data Analysis")
    run_eda()

    # ── Step 4: Forecasting ──────────────────────────────────────────────────
    section("STEP 4 / 6 · Sales Forecasting (Random Forest)")
    model, test_preds, future = run_forecasting()

    # ── Step 5: Inventory ────────────────────────────────────────────────────
    section("STEP 5 / 6 · Inventory Optimization")
    inv, alerts = run_inventory_optimization()

    # ── Step 6: Report ───────────────────────────────────────────────────────
    section("STEP 6 / 6 · Generating HTML Business Report")
    build_report()

    # ── Summary ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"  ✅  PIPELINE COMPLETE  ({elapsed:.0f}s)")
    print(f"{'='*60}")
    print("""
  📁 Key Outputs
  ─────────────────────────────────────────────────────
  data/retail_sales_data.csv       ← raw dataset
  data/retail_sales_clean.csv      ← cleaned + features
  outputs/charts/                  ← 14 PNG charts
  outputs/tables/model_metrics.csv ← MAE / RMSE / R² / MAPE
  outputs/tables/30day_forecast.csv← 30-day unit predictions
  outputs/tables/reorder_alerts.csv← reorder recommendations
  outputs/tables/inventory_optimization.csv ← full inv table
  models/rf_model.pkl              ← trained model
  reports/retail_report.html       ← 📊 Open this in browser!
  ─────────────────────────────────────────────────────
  🌐  Open  reports/retail_report.html  in Chrome / Firefox
    """)


if __name__ == "__main__":
    main()
