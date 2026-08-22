"""
generate_dataset.py
-------------------
Generates a realistic synthetic retail sales dataset.

Coverage:
- 2 years of daily sales
- 5 Indian retail stores
- 6 product categories
- 30 products
- Inventory and stockout tracking
- Supplier and lead-time information
- Promotions and discounts
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# ─── Configuration ────────────────────────────────────────────────────────────

np.random.seed(42)

STORES = [
    "Store_Mumbai",
    "Store_Delhi",
    "Store_Bengaluru",
    "Store_Hyderabad",
    "Store_Pune",
]

SUPPLIERS = [
    "Supplier_01",
    "Supplier_02",
    "Supplier_03",
    "Supplier_04",
    "Supplier_05",
    "Supplier_06",
    "Supplier_07",
    "Supplier_08",
]

CATEGORIES = [
    "Electronics",
    "Clothing",
    "Groceries",
    "Home & Kitchen",
    "Toys",
    "Sports",
]

PRODUCTS = {
    "Electronics": [
        ("Laptop", 1200),
        ("Smartphone", 800),
        ("Headphones", 150),
        ("Tablet", 400),
        ("Smartwatch", 250),
    ],
    "Clothing": [
        ("T-Shirt", 20),
        ("Jeans", 60),
        ("Jacket", 120),
        ("Dress", 80),
        ("Shoes", 90),
    ],
    "Groceries": [
        ("Rice (5kg)", 15),
        ("Cooking Oil", 12),
        ("Biscuits", 5),
        ("Tea Packets", 8),
        ("Instant Noodles", 3),
    ],
    "Home & Kitchen": [
        ("Pressure Cooker", 45),
        ("Non-stick Pan", 35),
        ("Water Bottle", 10),
        ("Mixer Grinder", 70),
        ("Dinner Set", 50),
    ],
    "Toys": [
        ("Lego Set", 40),
        ("Board Game", 25),
        ("Doll", 15),
        ("RC Car", 55),
        ("Puzzle", 20),
    ],
    "Sports": [
        ("Cricket Bat", 60),
        ("Football", 25),
        ("Yoga Mat", 30),
        ("Dumbbells", 40),
        ("Badminton Set", 35),
    ],
}

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)


# ─── Seasonal Demand ──────────────────────────────────────────────────────────

def seasonal_factor(date):
    """
    Monthly seasonal multiplier.

    Higher demand during festive periods and selected
    seasonal periods.
    """

    peaks = {
        1: 1.20,
        2: 1.00,
        3: 1.15,
        4: 1.10,
        5: 0.95,
        6: 0.90,
        7: 1.10,
        8: 1.05,
        9: 1.00,
        10: 1.40,
        11: 1.60,
        12: 1.80,
    }

    return peaks.get(date.month, 1.0)


def day_of_week_factor(date):
    """Weekend demand boost."""

    return 1.30 if date.weekday() >= 5 else 1.00


def trend_factor(date):
    """Small upward demand trend over the two-year period."""

    days_since_start = (date - START_DATE).days

    return 1.0 + (days_since_start / 730) * 0.15


# ─── Supplier Setup ───────────────────────────────────────────────────────────

def create_supplier_data():
    """
    Creates simple supplier characteristics.

    These values will later be used for supplier-risk analysis.
    """

    supplier_data = {}

    for supplier in SUPPLIERS:

        supplier_data[supplier] = {
            "lead_time_days": np.random.randint(3, 11),
            "on_time_rate": round(np.random.uniform(0.85, 0.99), 2),
            "defect_rate": round(np.random.uniform(0.01, 0.08), 2),
        }

    return supplier_data


# ─── Product-Supplier Mapping ─────────────────────────────────────────────────

def create_product_supplier_mapping():
    """
    Assigns one supplier to each product.
    """

    mapping = {}

    for category, products in PRODUCTS.items():

        for product_name, _ in products:

            mapping[product_name] = np.random.choice(SUPPLIERS)

    return mapping


# ─── Main Dataset Generator ──────────────────────────────────────────────────

def generate():

    records = []

    dates = [
        START_DATE + timedelta(days=i)
        for i in range((END_DATE - START_DATE).days + 1)
    ]

    supplier_data = create_supplier_data()

    product_supplier = create_product_supplier_mapping()

    for store in STORES:

        # Each store has a slightly different demand level.
        store_factor = np.random.uniform(0.80, 1.20)

        for category, products in PRODUCTS.items():

            # Category-level demand variation.
            category_factor = np.random.uniform(0.90, 1.10)

            for product_name, unit_price in products:

                # Product-level baseline demand.
                base_demand = np.random.randint(5, 40)

                # Initial inventory.
                initial_stock = np.random.randint(200, 600)

                # Reorder parameters.
                reorder_point = np.random.randint(50, 100)
                reorder_qty = np.random.randint(100, 300)

                current_stock = initial_stock

                # Supplier assigned to this product.
                supplier = product_supplier[product_name]

                supplier_info = supplier_data[supplier]

                lead_time_days = supplier_info["lead_time_days"]
                on_time_rate = supplier_info["on_time_rate"]
                defect_rate = supplier_info["defect_rate"]

                # Track one outstanding purchase order.
                pending_order = None

                for date in dates:

                    # ─── Receive pending order ───────────────────────────────

                    received_qty = 0
                    supplier_delay = 0

                    if pending_order is not None:

                        if date >= pending_order["arrival_date"]:

                            received_qty = pending_order["quantity"]

                            # Supplier may deliver late.
                            if np.random.rand() > on_time_rate:

                                supplier_delay = 1

                                # Delay the actual receipt by 1-3 days.
                                delayed_arrival = (
                                    pending_order["arrival_date"]
                                    + timedelta(days=np.random.randint(1, 4))
                                )

                                if date < delayed_arrival:

                                    received_qty = 0

                                else:

                                    current_stock += pending_order["quantity"]

                                    pending_order = None

                            else:

                                current_stock += pending_order["quantity"]

                                pending_order = None

                    # ─── Opening Stock ──────────────────────────────────────

                    opening_stock = current_stock

                    # ─── Demand Generation ──────────────────────────────────

                    sf = seasonal_factor(date)
                    dwf = day_of_week_factor(date)
                    tf = trend_factor(date)

                    noise = np.random.normal(1.0, 0.15)

                    demand = int(
                        base_demand
                        * sf
                        * dwf
                        * tf
                        * noise
                        * store_factor
                        * category_factor
                    )

                    demand = max(0, demand)

                    # ─── Promotion ──────────────────────────────────────────

                    promo = 1 if np.random.rand() < 0.05 else 0

                    if promo:
                        demand = int(demand * 1.5)

                    # ─── Sales / Stockout ───────────────────────────────────

                    actual_sold = min(demand, current_stock)

                    stockout = 1 if demand > current_stock else 0

                    lost_sales = demand - actual_sold

                    current_stock -= actual_sold

                    # ─── Pricing ────────────────────────────────────────────

                    discount_pct = np.random.choice(
                        [0, 5, 10, 15, 20],
                        p=[0.70, 0.10, 0.10, 0.05, 0.05],
                    )

                    sell_price = round(
                        unit_price * (1 - discount_pct / 100),
                        2,
                    )

                    revenue = round(
                        actual_sold * sell_price,
                        2,
                    )

                    # ─── Reorder Decision ───────────────────────────────────

                    reordered = 0
                    ordered_qty = 0

                    # Only place a new order if no order is already pending.
                    if (
                        current_stock <= reorder_point
                        and pending_order is None
                    ):

                        reordered = 1
                        ordered_qty = reorder_qty

                        expected_arrival = (
                            date + timedelta(days=lead_time_days)
                        )

                        pending_order = {
                            "quantity": reorder_qty,
                            "arrival_date": expected_arrival,
                        }

                    # ─── Closing Stock ──────────────────────────────────────

                    closing_stock = current_stock

                    # ─── Store Record ───────────────────────────────────────

                    records.append(
                        {
                            "date": date.strftime("%Y-%m-%d"),
                            "store": store,
                            "category": category,
                            "product": product_name,
                            "supplier": supplier,

                            "unit_price": unit_price,
                            "discount_pct": discount_pct,
                            "sell_price": sell_price,

                            "demand": demand,
                            "units_sold": actual_sold,
                            "lost_sales": lost_sales,
                            "revenue": revenue,

                            "opening_stock": opening_stock,
                            "closing_stock": closing_stock,

                            "reorder_point": reorder_point,
                            "reorder_qty": reorder_qty,

                            "reordered": reordered,
                            "ordered_qty": ordered_qty,
                            "received_qty": received_qty,

                            "stockout": stockout,
                            "promo_event": promo,

                            "lead_time_days": lead_time_days,
                            "on_time_rate": on_time_rate,
                            "defect_rate": defect_rate,
                            "supplier_delay": supplier_delay,
                        }
                    )

    # ─── DataFrame ────────────────────────────────────────────────────────────

    df = pd.DataFrame(records)

    df["date"] = pd.to_datetime(df["date"])

    df = df.sort_values(
        ["store", "product", "date"]
    ).reset_index(drop=True)

    return df


# ─── Script Entry Point ───────────────────────────────────────────────────────

if __name__ == "__main__":

    print("Generating synthetic Indian retail dataset...")

    df = generate()

    # Create data directory if it doesn't exist.
    output_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
    )

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        "retail_sales_data.csv",
    )

    df.to_csv(
        output_path,
        index=False,
    )

    print()
    print("Dataset generated successfully!")
    print(f"Saved to: {output_path}")
    print(f"Shape: {df.shape}")
    print()
    print("Columns:")
    print(df.columns.tolist())
    print()
    print("Sample:")
    print(df.head())
