"""
generate_dataset.py
-------------------
Generates a realistic synthetic retail sales dataset for simulation.
Covers 2 years of daily sales across 5 stores, 6 categories, 30 products.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ─── Config ───────────────────────────────────────────────────────────────────
np.random.seed(42)

STORES      = ["Store_A", "Store_B", "Store_C", "Store_D", "Store_E"]
CATEGORIES  = ["Electronics", "Clothing", "Groceries", "Home & Kitchen", "Toys", "Sports"]
PRODUCTS = {
    "Electronics":     [("Laptop",1200),("Smartphone",800),("Headphones",150),("Tablet",400),("Smartwatch",250)],
    "Clothing":        [("T-Shirt",20),("Jeans",60),("Jacket",120),("Dress",80),("Shoes",90)],
    "Groceries":       [("Rice (5kg)",15),("Cooking Oil",12),("Biscuits",5),("Tea Packets",8),("Instant Noodles",3)],
    "Home & Kitchen":  [("Pressure Cooker",45),("Non-stick Pan",35),("Water Bottle",10),("Mixer Grinder",70),("Dinner Set",50)],
    "Toys":            [("Lego Set",40),("Board Game",25),("Doll",15),("RC Car",55),("Puzzle",20)],
    "Sports":          [("Cricket Bat",60),("Football",25),("Yoga Mat",30),("Dumbbells",40),("Badminton Set",35)],
}

START_DATE  = datetime(2023, 1, 1)
END_DATE    = datetime(2024, 12, 31)


def seasonal_factor(date):
    """Returns a multiplier based on month for seasonal trends."""
    month = date.month
    # Peak: Oct-Dec (festive), Mar-Apr (summer), Jul-Aug (mid-year)
    peaks = {10:1.4, 11:1.6, 12:1.8, 1:1.2, 2:1.0, 3:1.15,
             4:1.1,  5:0.95, 6:0.9,  7:1.1,  8:1.05, 9:1.0}
    return peaks.get(month, 1.0)


def day_of_week_factor(date):
    """Weekend boost for retail sales."""
    return 1.3 if date.weekday() >= 5 else 1.0


def trend_factor(date):
    """Slight upward trend over 2 years."""
    days_since_start = (date - START_DATE).days
    return 1.0 + (days_since_start / 730) * 0.15


def generate():
    records = []
    dates   = [START_DATE + timedelta(days=i) for i in range((END_DATE - START_DATE).days + 1)]

    for store in STORES:
        store_factor = np.random.uniform(0.8, 1.2)         # each store has its own scale

        for category, products in PRODUCTS.items():
            category_factor = np.random.uniform(0.9, 1.1)  # category variance

            for product_name, unit_price in products:
                base_demand = np.random.randint(5, 40)      # baseline daily units sold

                initial_stock  = np.random.randint(200, 600)
                reorder_point  = np.random.randint(50, 100)
                reorder_qty    = np.random.randint(100, 300)
                current_stock  = initial_stock

                for date in dates:
                    sf  = seasonal_factor(date)
                    dwf = day_of_week_factor(date)
                    tf  = trend_factor(date)
                    noise = np.random.normal(1.0, 0.15)     # ±15% daily noise

                    demand = int(base_demand * sf * dwf * tf * noise * store_factor * category_factor)
                    demand = max(0, demand)

                    # Promo event: random 5% of days → 1.5× boost
                    promo = 1 if np.random.rand() < 0.05 else 0
                    if promo:
                        demand = int(demand * 1.5)

                    # Stock-out logic
                    actual_sold = min(demand, current_stock)
                    stockout    = 1 if demand > current_stock else 0
                    lost_sales  = demand - actual_sold

                    current_stock -= actual_sold

                    # Reorder trigger
                    reordered = 0
                    if current_stock <= reorder_point:
                        current_stock += reorder_qty
                        reordered = 1

                    # Price with occasional discount
                    discount_pct = np.random.choice([0, 5, 10, 15, 20], p=[0.7, 0.1, 0.1, 0.05, 0.05])
                    sell_price   = round(unit_price * (1 - discount_pct / 100), 2)
                    revenue      = round(actual_sold * sell_price, 2)

                    records.append({
                        "date":          date.strftime("%Y-%m-%d"),
                        "store":         store,
                        "category":      category,
                        "product":       product_name,
                        "unit_price":    unit_price,
                        "discount_pct":  discount_pct,
                        "sell_price":    sell_price,
                        "demand":        demand,
                        "units_sold":    actual_sold,
                        "revenue":       revenue,
                        "stock_level":   current_stock,
                        "reorder_point": reorder_point,
                        "reorder_qty":   reorder_qty,
                        "reordered":     reordered,
                        "stockout":      stockout,
                        "lost_sales":    lost_sales,
                        "promo_event":   promo,
                    })

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store", "product", "date"]).reset_index(drop=True)
    return df


if __name__ == "__main__":
    print("Generating synthetic retail dataset ...")
    df = generate()
    out = os.path.join(os.path.dirname(__file__), "..", "data", "retail_sales_data.csv")
    df.to_csv(out, index=False)
    print(f"Dataset saved → {out}")
    print(f"Shape: {df.shape}")
    print(df.head())
