"""
inventory_optimization.py
-------------------------

Forecast-driven inventory optimization.

Pipeline:

XGBoost 30-day forecast
        ↓
Forecast daily demand
        ↓
Expected demand during supplier lead time
        ↓
Safety Stock
        ↓
Forecast-driven Reorder Point
        ↓
EOQ
        ↓
Recommended Order Quantity

The model uses:
- XGBoost forecast
- Historical demand variability
- Supplier lead time
- Current inventory
- EOQ

Business assumptions are intentionally simple
and transparent for a fresher-level project.
"""

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

RETAIL_PATH = "data/retail_sales_data.csv"

FORECAST_PATH = "data/30_day_xgboost_forecast.csv"

OUTPUT_PATH = "data/inventory_optimization_results.csv"


# -----------------------------------------------------------------------------
# Inventory assumptions
# -----------------------------------------------------------------------------

# 95% service level
SERVICE_LEVEL_Z = 1.645

# Assumed ordering cost per purchase order.
# This is a transparent business assumption for the project.
ORDERING_COST = 2000.0

# Annual holding cost as a percentage of product value.
HOLDING_RATE = 0.25

ANNUAL_DAYS = 365


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    print("=" * 70)
    print("LOADING RETAIL AND FORECAST DATA")
    print("=" * 70)

    retail = pd.read_csv(
        RETAIL_PATH,
        parse_dates=["date"],
    )

    forecast = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["date"],
    )

    print(
        f"Retail rows   : {len(retail):,}"
    )

    print(
        f"Forecast rows : {len(forecast):,}"
    )

    return retail, forecast


# =============================================================================
# CREATE PRODUCT-STORE FORECAST
# =============================================================================

def create_forecast_input(
    retail,
    forecast,
):

    print()
    print("=" * 70)
    print("PREPARING XGBOOST FORECAST")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Historical demand by store/product.
    # -------------------------------------------------------------------------

    product_store_demand = (
        retail.groupby(
            [
                "store",
                "product",
            ]
        )
        .agg(
            historical_demand=(
                "demand",
                "sum",
            )
        )
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # Calculate each store/product's historical demand share.
    # -------------------------------------------------------------------------

    total_historical_demand = (
        product_store_demand[
            "historical_demand"
        ].sum()
    )

    product_store_demand[
        "demand_share"
    ] = (
        product_store_demand[
            "historical_demand"
        ]
        / total_historical_demand
    )

    # -------------------------------------------------------------------------
    # Overall XGBoost forecast.
    # -------------------------------------------------------------------------

    average_daily_forecast = (
        forecast[
            "forecast_demand"
        ].mean()
    )

    total_30_day_forecast = (
        forecast[
            "forecast_demand"
        ].sum()
    )

    print(
        f"Average daily XGBoost forecast : "
        f"{average_daily_forecast:,.0f}"
    )

    print(
        f"Total 30-day XGBoost forecast  : "
        f"{total_30_day_forecast:,.0f}"
    )

    # -------------------------------------------------------------------------
    # Allocate overall forecast to each store/product using historical share.
    # -------------------------------------------------------------------------

    product_store_demand[
        "forecast_daily_demand"
    ] = (
        product_store_demand[
            "demand_share"
        ]
        * average_daily_forecast
    )

    return product_store_demand


# =============================================================================
# DEMAND VARIABILITY
# =============================================================================

def calculate_demand_variability(
    retail,
):

    # -------------------------------------------------------------------------
    # Store + product level variability.
    # -------------------------------------------------------------------------

    store_product_std = (
        retail.groupby(
            [
                "store",
                "product",
            ]
        )
        .agg(
            demand_std=(
                "demand",
                "std",
            )
        )
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # Product-level fallback.
    #
    # If a store/product combination has insufficient observations,
    # use the overall product variability.
    # -------------------------------------------------------------------------

    product_std = (
        retail.groupby(
            "product"
        )
        .agg(
            product_demand_std=(
                "demand",
                "std",
            )
        )
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # Overall fallback.
    # -------------------------------------------------------------------------

    overall_std = (
        retail[
            "demand"
        ].std()
    )

    return (
        store_product_std,
        product_std,
        overall_std,
    )


# =============================================================================
# CALCULATE INVENTORY METRICS
# =============================================================================

def calculate_inventory_metrics(
    retail,
    forecast_input,
):

    print()
    print("=" * 70)
    print("CALCULATING INVENTORY METRICS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Current inventory and supplier information.
    # -------------------------------------------------------------------------

    inventory_metrics = (
        retail.groupby(
            [
                "store",
                "product",
            ]
        )
        .agg(
            category=(
                "category",
                "first",
            ),

            supplier=(
                "supplier",
                "first",
            ),

            annual_demand=(
                "demand",
                "sum",
            ),

            unit_price=(
                "unit_price",
                "mean",
            ),

            lead_time_days=(
                "lead_time_days",
                "mean",
            ),

            current_stock=(
                "closing_stock",
                "last",
            ),
        )
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # Add forecast.
    # -------------------------------------------------------------------------

    inventory_metrics = inventory_metrics.merge(
        forecast_input[
            [
                "store",
                "product",
                "forecast_daily_demand",
            ]
        ],
        on=[
            "store",
            "product",
        ],
        how="left",
    )

    # -------------------------------------------------------------------------
    # Calculate demand variability at multiple levels.
    # -------------------------------------------------------------------------

    (
        store_product_std,
        product_std,
        overall_std,
    ) = calculate_demand_variability(
        retail
    )

    inventory_metrics = inventory_metrics.merge(
        store_product_std,
        on=[
            "store",
            "product",
        ],
        how="left",
    )

    inventory_metrics = inventory_metrics.merge(
        product_std,
        on="product",
        how="left",
    )

    # -------------------------------------------------------------------------
    # SAFETY STOCK FALLBACK LOGIC
    #
    # Priority:
    #
    # 1. Store/product demand standard deviation
    # 2. Product-level demand standard deviation
    # 3. Overall demand standard deviation
    #
    # This prevents meaningless zero safety stock values.
    # -------------------------------------------------------------------------

    inventory_metrics[
        "demand_std"
    ] = (
        inventory_metrics[
            "demand_std"
        ]
        .fillna(
            inventory_metrics[
                "product_demand_std"
            ]
        )
        .fillna(
            overall_std
        )
    )

    # If variability is exactly zero, use the overall variability.
    inventory_metrics[
        "demand_std"
    ] = np.where(
        inventory_metrics[
            "demand_std"
        ] <= 0,

        overall_std,

        inventory_metrics[
            "demand_std"
        ],
    )

    # -------------------------------------------------------------------------
    # Safety Stock
    #
    # Safety Stock =
    # Z × Demand Std Dev × sqrt(Lead Time)
    # -------------------------------------------------------------------------

    inventory_metrics[
        "safety_stock"
    ] = (
        SERVICE_LEVEL_Z
        * inventory_metrics[
            "demand_std"
        ]
        * np.sqrt(
            inventory_metrics[
                "lead_time_days"
            ]
        )
    )

    # -------------------------------------------------------------------------
    # EXPECTED DEMAND DURING LEAD TIME
    #
    # Forecast daily demand × supplier lead time
    # -------------------------------------------------------------------------

    inventory_metrics[
        "expected_lead_time_demand"
    ] = (
        inventory_metrics[
            "forecast_daily_demand"
        ]
        * inventory_metrics[
            "lead_time_days"
        ]
    )

    # -------------------------------------------------------------------------
    # FORECAST-DRIVEN REORDER POINT
    #
    # ROP =
    # Expected Lead-Time Demand + Safety Stock
    # -------------------------------------------------------------------------

    inventory_metrics[
        "calculated_rop"
    ] = (
        inventory_metrics[
            "expected_lead_time_demand"
        ]
        + inventory_metrics[
            "safety_stock"
        ]
    )

    # -------------------------------------------------------------------------
    # HOLDING COST
    #
    # H = Unit Price × Holding Rate
    # -------------------------------------------------------------------------

    inventory_metrics[
        "holding_cost_per_unit"
    ] = (
        inventory_metrics[
            "unit_price"
        ]
        * HOLDING_RATE
    )

    # -------------------------------------------------------------------------
    # EOQ
    #
    # EOQ = sqrt(2DS / H)
    #
    # D = annual demand
    # S = ordering cost
    # H = annual holding cost
    # -------------------------------------------------------------------------

    inventory_metrics[
        "eoq"
    ] = np.sqrt(
        (
            2
            * inventory_metrics[
                "annual_demand"
            ]
            * ORDERING_COST
        )
        /
        inventory_metrics[
            "holding_cost_per_unit"
        ]
    )

    # -------------------------------------------------------------------------
    # Handle invalid values.
    # -------------------------------------------------------------------------

    numeric_columns = [
        "forecast_daily_demand",
        "demand_std",
        "safety_stock",
        "expected_lead_time_demand",
        "calculated_rop",
        "eoq",
    ]

    for column in numeric_columns:

        inventory_metrics[column] = (
            inventory_metrics[column]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

    # -------------------------------------------------------------------------
    # INVENTORY STATUS
    #
    # CRITICAL:
    # Current stock <= Safety Stock
    #
    # REORDER:
    # Current stock <= ROP
    #
    # NORMAL:
    # Current stock > ROP
    # -------------------------------------------------------------------------

    inventory_metrics[
        "inventory_status"
    ] = np.select(
        [
            inventory_metrics[
                "current_stock"
            ]
            <= inventory_metrics[
                "safety_stock"
            ],

            inventory_metrics[
                "current_stock"
            ]
            <= inventory_metrics[
                "calculated_rop"
            ],
        ],
        [
            "CRITICAL",
            "REORDER",
        ],
        default="NORMAL",
    )

    # -------------------------------------------------------------------------
    # SHORTAGE TO ROP
    # -------------------------------------------------------------------------

    inventory_metrics[
        "shortage_to_rop"
    ] = np.maximum(
        inventory_metrics[
            "calculated_rop"
        ]
        - inventory_metrics[
            "current_stock"
        ],
        0,
    )

    # -------------------------------------------------------------------------
    # RECOMMENDED ORDER
    #
    # If current stock is below ROP:
    #
    #     Order EOQ
    #
    # Otherwise:
    #
    #     No order.
    # -------------------------------------------------------------------------

    inventory_metrics[
        "recommended_order_qty"
    ] = np.where(
        inventory_metrics[
            "current_stock"
        ]
        <= inventory_metrics[
            "calculated_rop"
        ],

        inventory_metrics[
            "eoq"
        ],

        0,
    )

    # -------------------------------------------------------------------------
    # Round quantities.
    # -------------------------------------------------------------------------

    quantity_columns = [
        "forecast_daily_demand",
        "demand_std",
        "annual_demand",
        "lead_time_days",
        "current_stock",
        "safety_stock",
        "expected_lead_time_demand",
        "calculated_rop",
        "eoq",
        "shortage_to_rop",
        "recommended_order_qty",
    ]

    for column in quantity_columns:

        inventory_metrics[column] = (
            inventory_metrics[column]
            .round(0)
            .astype(int)
        )

    # -------------------------------------------------------------------------
    # Remove helper columns from final output.
    # -------------------------------------------------------------------------

    inventory_metrics = inventory_metrics.drop(
        columns=[
            "product_demand_std",
        ]
    )

    return inventory_metrics


# =============================================================================
# SUMMARY
# =============================================================================

def create_summary(
    results,
):

    print()
    print("=" * 70)
    print("INVENTORY RISK SUMMARY")
    print("=" * 70)

    summary = (
        results[
            "inventory_status"
        ]
        .value_counts()
    )

    print(
        summary.to_string()
    )

    reorder_items = (
        results[
            "recommended_order_qty"
        ]
        > 0
    ).sum()

    total_order_qty = (
        results[
            "recommended_order_qty"
        ].sum()
    )

    critical_items = (
        results[
            "inventory_status"
        ]
        == "CRITICAL"
    ).sum()

    print()
    print(
        f"Critical items          : {critical_items}"
    )

    print(
        f"Items requiring reorder : {reorder_items}"
    )

    print(
        f"Recommended order units : "
        f"{total_order_qty:,}"
    )


# =============================================================================
# TOP REORDER RECOMMENDATIONS
# =============================================================================

def show_top_reorders(
    results,
):

    print()
    print("=" * 70)
    print("TOP REORDER RECOMMENDATIONS")
    print("=" * 70)

    reorder_df = (
        results[
            results[
                "recommended_order_qty"
            ]
            > 0
        ]
        .sort_values(
            [
                "inventory_status",
                "shortage_to_rop",
            ],
            ascending=[
                True,
                False,
            ],
        )
        .head(20)
    )

    columns = [
        "store",
        "product",
        "supplier",
        "forecast_daily_demand",
        "lead_time_days",
        "expected_lead_time_demand",
        "demand_std",
        "safety_stock",
        "calculated_rop",
        "current_stock",
        "shortage_to_rop",
        "eoq",
        "inventory_status",
        "recommended_order_qty",
    ]

    print(
        reorder_df[
            columns
        ].to_string(
            index=False
        )
    )


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(
    results,
):

    status_order = {
        "CRITICAL": 1,
        "REORDER": 2,
        "NORMAL": 3,
    }

    results[
        "status_order"
    ] = (
        results[
            "inventory_status"
        ]
        .map(
            status_order
        )
    )

    results = results.sort_values(
        [
            "status_order",
            "shortage_to_rop",
        ],
        ascending=[
            True,
            False,
        ],
    )

    results = results.drop(
        columns=[
            "status_order",
        ]
    )

    results.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Results saved to: {OUTPUT_PATH}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 70)
    print("FORECAST-DRIVEN INVENTORY OPTIMIZATION")
    print("=" * 70)

    print()
    print(
        "Business assumptions:"
    )

    print(
        f"Service level      : 95%"
    )

    print(
        f"Ordering cost      : ₹{ORDERING_COST:,.0f}"
    )

    print(
        f"Holding rate       : {HOLDING_RATE * 100:.0f}%"
    )

    print()

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    retail, forecast = load_data()

    # -------------------------------------------------------------------------
    # Prepare forecast
    # -------------------------------------------------------------------------

    forecast_input = create_forecast_input(
        retail,
        forecast,
    )

    # -------------------------------------------------------------------------
    # Calculate inventory decisions
    # -------------------------------------------------------------------------

    results = calculate_inventory_metrics(
        retail,
        forecast_input,
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    create_summary(
        results
    )

    # -------------------------------------------------------------------------
    # Top recommendations
    # -------------------------------------------------------------------------

    show_top_reorders(
        results
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    save_results(
        results
    )

    print()
    print("=" * 70)
    print(
        "FORECAST-DRIVEN INVENTORY OPTIMIZATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
