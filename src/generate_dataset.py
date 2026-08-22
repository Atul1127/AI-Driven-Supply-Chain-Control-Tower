"""
generate_dataset.py
-------------------
Generates a realistic synthetic retail sales dataset.

Project:
Intelligent Demand & Supply Chain Control Tower

Coverage:
- 2 years of daily sales
- 5 Indian retail stores
- 6 product categories
- 30 products
- Demand, sales and revenue
- Inventory tracking
- Stockout and lost-sales tracking
- Supplier information
- Supplier lead time and reliability
- Promotions and discounts
- Reorder decisions
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

np.random.seed(42)


# -----------------------------------------------------------------------------
# Stores
# -----------------------------------------------------------------------------

STORES = [
    "Store_Mumbai",
    "Store_Delhi",
    "Store_Bengaluru",
    "Store_Hyderabad",
    "Store_Pune",
]


# -----------------------------------------------------------------------------
# Suppliers
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Product Categories
# -----------------------------------------------------------------------------

CATEGORIES = [
    "Electronics",
    "Clothing",
    "Groceries",
    "Home & Kitchen",
    "Toys",
    "Sports",
]


# -----------------------------------------------------------------------------
# Products
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Date Range
# -----------------------------------------------------------------------------

START_DATE = datetime(2023, 1, 1)
END_DATE = datetime(2024, 12, 31)


# =============================================================================
# DEMAND COMPONENTS
# =============================================================================

def seasonal_factor(date):
    """
    Monthly seasonal multiplier.

    Higher demand is simulated during:
    - January
    - March/April
    - July/August
    - October/November/December

    This gives the time series useful seasonal patterns for
    SARIMA and SARIMAX.
    """

    monthly_factors = {
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

    return monthly_factors[date.month]


def day_of_week_factor(date):
    """
    Weekend demand multiplier.
    """

    if date.weekday() >= 5:
        return 1.30

    return 1.00


def trend_factor(date):
    """
    Small upward demand trend over the two-year period.
    """

    days_since_start = (date - START_DATE).days

    total_days = (END_DATE - START_DATE).days

    return 1.0 + (days_since_start / total_days) * 0.15


# =============================================================================
# SUPPLIER DATA
# =============================================================================

def create_supplier_data():
    """
    Generate basic supplier characteristics.

    These characteristics will later be used for:
    - Supplier analysis
    - Supplier risk scoring
    - Inventory planning
    """

    supplier_data = {}

    for supplier in SUPPLIERS:

        supplier_data[supplier] = {
            "lead_time_days": int(np.random.randint(3, 11)),
            "on_time_rate": round(
                np.random.uniform(0.85, 0.99),
                2,
            ),
            "defect_rate": round(
                np.random.uniform(0.01, 0.08),
                2,
            ),
        }

    return supplier_data


# =============================================================================
# PRODUCT-SUPPLIER MAPPING
# =============================================================================

def create_product_supplier_mapping():
    """
    Assign one supplier to each product.

    Keeping one supplier per product keeps the project simple while
    still allowing supplier-risk analysis.
    """

    product_supplier = {}

    for category, products in PRODUCTS.items():

        for product_name, _ in products:

            product_supplier[product_name] = np.random.choice(
                SUPPLIERS
            )

    return product_supplier


# =============================================================================
# DATASET GENERATION
# =============================================================================

def generate():

    records = []

    # -------------------------------------------------------------------------
    # Generate dates
    # -------------------------------------------------------------------------

    number_of_days = (END_DATE - START_DATE).days + 1

    dates = [
        START_DATE + timedelta(days=i)
        for i in range(number_of_days)
    ]

    # -------------------------------------------------------------------------
    # Create supplier information
    # -------------------------------------------------------------------------

    supplier_data = create_supplier_data()

    # -------------------------------------------------------------------------
    # Assign products to suppliers
    # -------------------------------------------------------------------------

    product_supplier = create_product_supplier_mapping()

    # =========================================================================
    # STORE LOOP
    # =========================================================================

    for store in STORES:

        # Each store has a slightly different demand level.
        store_factor = np.random.uniform(
            0.80,
            1.20,
        )

        # =====================================================================
        # CATEGORY LOOP
        # =====================================================================

        for category, products in PRODUCTS.items():

            # Each category has slightly different demand behavior.
            category_factor = np.random.uniform(
                0.90,
                1.10,
            )

            # =================================================================
            # PRODUCT LOOP
            # =================================================================

            for product_name, unit_price in products:

                # -------------------------------------------------------------
                # Product demand characteristics
                # -------------------------------------------------------------

                base_demand = int(
                    np.random.randint(
                        5,
                        40,
                    )
                )

                # -------------------------------------------------------------
                # Initial inventory
                # -------------------------------------------------------------

                initial_stock = int(
                    np.random.randint(
                        200,
                        600,
                    )
                )

                # -------------------------------------------------------------
                # Reorder parameters
                # -------------------------------------------------------------

                reorder_point = int(
                    np.random.randint(
                        50,
                        100,
                    )
                )

                reorder_qty = int(
                    np.random.randint(
                        100,
                        300,
                    )
                )

                # -------------------------------------------------------------
                # Current inventory
                # -------------------------------------------------------------

                current_stock = initial_stock

                # -------------------------------------------------------------
                # Supplier information
                # -------------------------------------------------------------

                supplier = product_supplier[product_name]

                supplier_info = supplier_data[supplier]

                lead_time_days = supplier_info["lead_time_days"]
                on_time_rate = supplier_info["on_time_rate"]
                defect_rate = supplier_info["defect_rate"]

                # -------------------------------------------------------------
                # Purchase order currently in transit
                #
                # We keep this simple:
                # One outstanding order per product/store combination.
                # -------------------------------------------------------------

                pending_order = None

                # =============================================================
                # DAILY LOOP
                # =============================================================

                for date in dates:

                    # =========================================================
                    # 1. RECEIVE PENDING PURCHASE ORDER
                    # =========================================================

                    received_qty = 0
                    supplier_delay = 0

                    if pending_order is not None:

                        if date >= pending_order["arrival_date"]:

                            current_stock += pending_order["quantity"]

                            received_qty = pending_order["quantity"]

                            pending_order = None

                    # =========================================================
                    # 2. OPENING STOCK
                    # =========================================================

                    opening_stock = current_stock

                    # =========================================================
                    # 3. DEMAND GENERATION
                    # =========================================================

                    sf = seasonal_factor(date)

                    dwf = day_of_week_factor(date)

                    tf = trend_factor(date)

                    # Daily random variation.
                    noise = np.random.normal(
                        1.0,
                        0.15,
                    )

                    demand = int(
                        base_demand
                        * sf
                        * dwf
                        * tf
                        * noise
                        * store_factor
                        * category_factor
                    )

                    demand = max(
                        0,
                        demand,
                    )

                    # =========================================================
                    # 4. PROMOTION
                    # =========================================================

                    promo_event = (
                        1
                        if np.random.rand() < 0.05
                        else 0
                    )

                    if promo_event == 1:

                        demand = int(
                            demand * 1.50
                        )

                    # =========================================================
                    # 5. SALES AND STOCKOUT
                    # =========================================================

                    units_sold = min(
                        demand,
                        current_stock,
                    )

                    stockout = (
                        1
                        if demand > current_stock
                        else 0
                    )

                    lost_sales = (
                        demand - units_sold
                    )

                    # Remove sold inventory.
                    current_stock -= units_sold

                    # =========================================================
                    # 6. PRICE / DISCOUNT
                    # =========================================================

                    discount_pct = int(
                        np.random.choice(
                            [0, 5, 10, 15, 20],
                            p=[
                                0.70,
                                0.10,
                                0.10,
                                0.05,
                                0.05,
                            ],
                        )
                    )

                    sell_price = round(
                        unit_price
                        * (1 - discount_pct / 100),
                        2,
                    )

                    revenue = round(
                        units_sold * sell_price,
                        2,
                    )

                    # =========================================================
                    # 7. REORDER DECISION
                    # =========================================================

                    reordered = 0
                    ordered_qty = 0

                    # Only create a new order if:
                    #
                    # - inventory is below reorder point
                    # - there is no order currently in transit
                    #
                    if (
                        current_stock <= reorder_point
                        and pending_order is None
                    ):

                        reordered = 1

                        ordered_qty = reorder_qty

                        expected_arrival = (
                            date
                            + timedelta(
                                days=lead_time_days
                            )
                        )

                        # -----------------------------------------------------
                        # Simple supplier delay simulation
                        #
                        # If supplier is unreliable, some orders arrive
                        # two days later than expected.
                        # -----------------------------------------------------

                        if np.random.rand() > on_time_rate:

                            supplier_delay = 1

                            actual_arrival = (
                                expected_arrival
                                + timedelta(days=2)
                            )

                        else:

                            actual_arrival = expected_arrival

                        pending_order = {
                            "quantity": ordered_qty,
                            "arrival_date": actual_arrival,
                        }

                    # =========================================================
                    # 8. CLOSING STOCK
                    # =========================================================

                    closing_stock = current_stock

                    # =========================================================
                    # 9. STORE RECORD
                    # =========================================================

                    records.append(
                        {
                            # -------------------------------------------------
                            # Date / dimensions
                            # -------------------------------------------------

                            "date": date.strftime(
                                "%Y-%m-%d"
                            ),

                            "store": store,

                            "category": category,

                            "product": product_name,

                            "supplier": supplier,

                            # -------------------------------------------------
                            # Pricing
                            # -------------------------------------------------

                            "unit_price": unit_price,

                            "discount_pct": discount_pct,

                            "sell_price": sell_price,

                            # -------------------------------------------------
                            # Demand / sales
                            # -------------------------------------------------

                            "demand": demand,

                            "units_sold": units_sold,

                            "lost_sales": lost_sales,

                            "revenue": revenue,

                            # -------------------------------------------------
                            # Inventory
                            # -------------------------------------------------

                            "opening_stock": opening_stock,

                            "closing_stock": closing_stock,

                            "reorder_point": reorder_point,

                            "reorder_qty": reorder_qty,

                            # -------------------------------------------------
                            # Reordering
                            # -------------------------------------------------

                            "reordered": reordered,

                            "ordered_qty": ordered_qty,

                            "received_qty": received_qty,

                            # -------------------------------------------------
                            # Stockout
                            # -------------------------------------------------

                            "stockout": stockout,

                            # -------------------------------------------------
                            # Promotion
                            # -------------------------------------------------

                            "promo_event": promo_event,

                            # -------------------------------------------------
                            # Supplier
                            # -------------------------------------------------

                            "lead_time_days": lead_time_days,

                            "on_time_rate": on_time_rate,

                            "defect_rate": defect_rate,

                            "supplier_delay": supplier_delay,
                        }
                    )

    # =========================================================================
    # CREATE DATAFRAME
    # =========================================================================

    df = pd.DataFrame(records)

    # -------------------------------------------------------------------------
    # Convert date column
    # -------------------------------------------------------------------------

    df["date"] = pd.to_datetime(
        df["date"]
    )

    # -------------------------------------------------------------------------
    # Sort data
    # -------------------------------------------------------------------------

    df = df.sort_values(
        [
            "store",
            "product",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    return df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("Generating synthetic retail supply-chain dataset")
    print("=" * 70)

    df = generate()

    # -------------------------------------------------------------------------
    # Create data directory
    # -------------------------------------------------------------------------

    output_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # Save dataset
    # -------------------------------------------------------------------------

    output_path = os.path.join(
        output_dir,
        "retail_sales_data.csv",
    )

    df.to_csv(
        output_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    print()
    print("Dataset generated successfully!")
    print()
    print(f"Output file : {output_path}")
    print(f"Rows        : {len(df):,}")
    print(f"Columns     : {len(df.columns)}")
    print()

    print("Date range:")
    print(
        f"{df['date'].min().date()} "
        f"to "
        f"{df['date'].max().date()}"
    )

    print()

    print("Stores:")
    print(df["store"].nunique())

    print()

    print("Products:")
    print(df["product"].nunique())

    print()

    print("Suppliers:")
    print(df["supplier"].nunique())

    print()

    print("Columns:")
    for column in df.columns:
        print(f" - {column}")

    print()

    print("First 5 rows:")
    print(
        df.head().to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("Dataset generation complete")
    print("=" * 70)
