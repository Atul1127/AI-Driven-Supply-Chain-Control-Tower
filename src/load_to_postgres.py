"""
load_to_postgres.py
-------------------
Loads the retail supply-chain CSV dataset into PostgreSQL.

Pipeline:

CSV
 ↓
Pandas
 ↓
Data Cleaning / Mapping
 ↓
PostgreSQL
 ├── stores
 ├── suppliers
 ├── products
 ├── sales
 └── inventory
"""

import getpass
import os

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values


# =============================================================================
# CONFIGURATION
# =============================================================================

DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "supply_chain_db"
DB_USER = "postgres"

DATA_PATH = os.path.join(
    "data",
    "retail_sales_data.csv",
)


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

def connect_to_database():
    """
    Connect to PostgreSQL.

    The password is requested securely from the terminal
    instead of being stored inside the source code.
    """

    password = getpass.getpass(
        "Enter PostgreSQL password: "
    )

    connection = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=password,
    )

    return connection


# =============================================================================
# LOAD CSV
# =============================================================================

def load_csv():
    """
    Load the generated retail dataset.
    """

    print()
    print("=" * 70)
    print("Loading CSV dataset")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    print(f"Dataset loaded: {len(df):,} rows")
    print(f"Columns: {len(df.columns)}")

    return df


# =============================================================================
# CLEAN DATA
# =============================================================================

def clean_data(df):
    """
    Basic data validation before loading into PostgreSQL.
    """

    print()
    print("=" * 70)
    print("Validating dataset")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Check missing values
    # -------------------------------------------------------------------------

    null_count = df.isna().sum().sum()

    if null_count > 0:

        raise ValueError(
            f"Dataset contains {null_count} missing values."
        )

    # -------------------------------------------------------------------------
    # Check duplicates
    # -------------------------------------------------------------------------

    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:

        raise ValueError(
            f"Dataset contains {duplicate_count} duplicate rows."
        )

    # -------------------------------------------------------------------------
    # Ensure numeric columns are numeric
    # -------------------------------------------------------------------------

    numeric_columns = [
        "unit_price",
        "discount_pct",
        "sell_price",
        "demand",
        "units_sold",
        "lost_sales",
        "revenue",
        "opening_stock",
        "closing_stock",
        "reorder_point",
        "reorder_qty",
        "reordered",
        "ordered_qty",
        "received_qty",
        "stockout",
        "promo_event",
        "lead_time_days",
        "on_time_rate",
        "defect_rate",
        "supplier_delay",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column]
        )

    print("Validation successful.")
    print(f"Rows ready for loading: {len(df):,}")

    return df


# =============================================================================
# LOAD STORES
# =============================================================================

def load_stores(cursor, df):
    """
    Load unique stores.
    """

    stores = sorted(
        df["store"].unique()
    )

    values = [
        (store,)
        for store in stores
    ]

    execute_values(
        cursor,
        """
        INSERT INTO stores (store_name)
        VALUES %s
        ON CONFLICT (store_name)
        DO NOTHING
        """,
        values,
    )

    cursor.execute(
        """
        SELECT store_id, store_name
        FROM stores
        """
    )

    rows = cursor.fetchall()

    store_map = {
        store_name: store_id
        for store_id, store_name in rows
    }

    print(
        f"Stores loaded: {len(store_map)}"
    )

    return store_map


# =============================================================================
# LOAD SUPPLIERS
# =============================================================================

def load_suppliers(cursor, df):
    """
    Load unique suppliers.

    Supplier characteristics are constant for a supplier
    in our generated dataset.
    """

    supplier_df = (
        df[
            [
                "supplier",
                "lead_time_days",
                "on_time_rate",
                "defect_rate",
            ]
        ]
        .drop_duplicates(
            subset=["supplier"]
        )
        .sort_values("supplier")
    )

    values = [
        (
            row["supplier"],
            int(row["lead_time_days"]),
            float(row["on_time_rate"]),
            float(row["defect_rate"]),
        )
        for _, row in supplier_df.iterrows()
    ]

    execute_values(
        cursor,
        """
        INSERT INTO suppliers (
            supplier_name,
            lead_time_days,
            on_time_rate,
            defect_rate
        )
        VALUES %s
        ON CONFLICT (supplier_name)
        DO NOTHING
        """,
        values,
    )

    cursor.execute(
        """
        SELECT
            supplier_id,
            supplier_name
        FROM suppliers
        """
    )

    rows = cursor.fetchall()

    supplier_map = {
        supplier_name: supplier_id
        for supplier_id, supplier_name in rows
    }

    print(
        f"Suppliers loaded: {len(supplier_map)}"
    )

    return supplier_map


# =============================================================================
# LOAD PRODUCTS
# =============================================================================

def load_products(
    cursor,
    df,
    supplier_map,
):
    """
    Load unique products.
    """

    product_df = (
        df[
            [
                "product",
                "category",
                "supplier",
                "unit_price",
            ]
        ]
        .drop_duplicates(
            subset=["product"]
        )
        .sort_values("product")
    )

    values = []

    for _, row in product_df.iterrows():

        supplier_id = supplier_map[
            row["supplier"]
        ]

        values.append(
            (
                row["product"],
                row["category"],
                supplier_id,
                float(row["unit_price"]),
            )
        )

    execute_values(
        cursor,
        """
        INSERT INTO products (
            product_name,
            category,
            supplier_id,
            unit_price
        )
        VALUES %s
        ON CONFLICT (product_name)
        DO NOTHING
        """,
        values,
    )

    cursor.execute(
        """
        SELECT
            product_id,
            product_name
        FROM products
        """
    )

    rows = cursor.fetchall()

    product_map = {
        product_name: product_id
        for product_id, product_name in rows
    }

    print(
        f"Products loaded: {len(product_map)}"
    )

    return product_map


# =============================================================================
# LOAD SALES
# =============================================================================

def load_sales(
    cursor,
    df,
    store_map,
    product_map,
):
    """
    Load sales transactions.
    """

    values = []

    for _, row in df.iterrows():

        values.append(
            (
                row["date"].date(),

                store_map[
                    row["store"]
                ],

                product_map[
                    row["product"]
                ],

                float(row["unit_price"]),
                float(row["discount_pct"]),
                float(row["sell_price"]),

                int(row["demand"]),
                int(row["units_sold"]),
                int(row["lost_sales"]),
                float(row["revenue"]),

                int(row["promo_event"]),
            )
        )

    print()
    print(
        "Loading sales records..."
    )

    execute_values(
        cursor,
        """
        INSERT INTO sales (
            sale_date,
            store_id,
            product_id,
            unit_price,
            discount_pct,
            sell_price,
            demand,
            units_sold,
            lost_sales,
            revenue,
            promo_event
        )
        VALUES %s
        """,
        values,
        page_size=5000,
    )

    print(
        f"Sales loaded: {len(values):,}"
    )


# =============================================================================
# LOAD INVENTORY
# =============================================================================

def load_inventory(
    cursor,
    df,
    store_map,
    product_map,
):
    """
    Load daily inventory records.
    """

    values = []

    for _, row in df.iterrows():

        values.append(
            (
                row["date"].date(),

                store_map[
                    row["store"]
                ],

                product_map[
                    row["product"]
                ],

                int(row["opening_stock"]),
                int(row["closing_stock"]),

                int(row["reorder_point"]),
                int(row["reorder_qty"]),

                int(row["reordered"]),
                int(row["ordered_qty"]),
                int(row["received_qty"]),

                int(row["stockout"]),
                int(row["supplier_delay"]),
            )
        )

    print()
    print(
        "Loading inventory records..."
    )

    execute_values(
        cursor,
        """
        INSERT INTO inventory (
            inventory_date,
            store_id,
            product_id,
            opening_stock,
            closing_stock,
            reorder_point,
            reorder_qty,
            reordered,
            ordered_qty,
            received_qty,
            stockout,
            supplier_delay
        )
        VALUES %s
        """,
        values,
        page_size=5000,
    )

    print(
        f"Inventory records loaded: {len(values):,}"
    )


# =============================================================================
# VERIFY DATABASE
# =============================================================================

def verify_database(cursor):
    """
    Verify record counts after loading.
    """

    print()
    print("=" * 70)
    print("DATABASE VERIFICATION")
    print("=" * 70)

    tables = [
        "stores",
        "suppliers",
        "products",
        "sales",
        "inventory",
    ]

    for table in tables:

        cursor.execute(
            f"SELECT COUNT(*) FROM {table}"
        )

        count = cursor.fetchone()[0]

        print(
            f"{table:<15} : {count:,}"
        )

    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 70)
    print("RETAIL SUPPLY CHAIN ETL")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load and validate dataset
    # -------------------------------------------------------------------------

    df = load_csv()

    df = clean_data(df)

    # -------------------------------------------------------------------------
    # Connect to PostgreSQL
    # -------------------------------------------------------------------------

    print()
    print("Connecting to PostgreSQL...")

    connection = connect_to_database()

    cursor = connection.cursor()

    try:

        # -------------------------------------------------------------
        # Clear existing data.
        #
        # This makes the script safe to run again.
        # -------------------------------------------------------------

        print()
        print("Preparing database...")

        cursor.execute(
            """
            TRUNCATE TABLE
                sales,
                inventory,
                products,
                suppliers,
                stores
            RESTART IDENTITY CASCADE
            """
        )

        # -------------------------------------------------------------
        # Load dimension tables
        # -------------------------------------------------------------

        store_map = load_stores(
            cursor,
            df,
        )

        supplier_map = load_suppliers(
            cursor,
            df,
        )

        product_map = load_products(
            cursor,
            df,
            supplier_map,
        )

        # -------------------------------------------------------------
        # Load fact tables
        # -------------------------------------------------------------

        load_sales(
            cursor,
            df,
            store_map,
            product_map,
        )

        load_inventory(
            cursor,
            df,
            store_map,
            product_map,
        )

        # -------------------------------------------------------------
        # Commit transaction
        # -------------------------------------------------------------

        connection.commit()

        print()
        print("All data committed successfully.")

        # -------------------------------------------------------------
        # Verify
        # -------------------------------------------------------------

        verify_database(
            cursor
        )

    except Exception as error:

        connection.rollback()

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(error)
        print()
        print("Transaction rolled back.")
        raise

    finally:

        cursor.close()

        connection.close()

        print()
        print("Database connection closed.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    main()
