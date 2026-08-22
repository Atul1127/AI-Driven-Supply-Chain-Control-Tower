"""
inventory_optimization.py
-------------------------

Forecast-driven inventory optimization.

Pipeline:

XGBoost 30-day forecast
        ↓
Expected demand during lead time
        ↓
Safety Stock
        ↓
Reorder Point
        ↓
EOQ
        ↓
Recommended Order Quantity

The inventory decision layer now uses the
XGBoost demand forecast rather than relying
only on historical average demand.
"""

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

RETAIL_PATH = "data/retail_sales_data.csv"

FORECAST_PATH = "data/30_day_xgboost_forecast.csv"

OUTPUT_PATH = "data/inventory_optimization_results.csv"

# 95% service level
SERVICE_LEVEL_Z = 1.645

# Simple business assumptions
ORDERING_COST = 500.0

# Annual holding cost = 20% of product value
HOLDING_RATE = 0.20


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

def create_forecast_input(retail, forecast):

    print()
    print("=" * 70)
    print("PREPARING FORECAST INPUT")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # The current XGBoost forecast is at total-retail level.
    #
    # To make it usable for individual store/product inventory decisions,
    # we distribute the total forecast according to each product-store's
    # historical demand share.
    # -------------------------------------------------------------------------

    historical_product_store_demand = (
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

    total_historical_demand = (
        historical_product_store_demand[
            "historical_demand"
        ].sum()
    )

    historical_product_store_demand[
        "demand_share"
    ] = (
        historical_product_store_demand[
            "historical_demand"
        ]
        / total_historical_demand
    )

    # -------------------------------------------------------------------------
    # Average daily XGBoost forecast.
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
        f"Average daily XGBoost forecast: "
        f"{average_daily_forecast:,.0f}"
    )

    print(
        f"Total 30-day XGBoost forecast: "
        f"{total_30_day_forecast:,.0f}"
    )

    # -------------------------------------------------------------------------
    # Allocate the overall forecast to each store/product.
    # -------------------------------------------------------------------------

    historical_product_store_demand[
        "forecast_daily_demand"
    ] = (
        historical_product_store_demand[
            "demand_share"
        ]
        * average_daily_forecast
    )

    return historical_product_store_demand


# =============================================================================
# CALCULATE INVENTORY METRICS
# =============================================================================

def calculate_inventory_metrics(
    retail,
    forecast_input,
):

    print()
    print("=" * 70)
    print("CALCULATING FORECAST-DRIVEN INVENTORY METRICS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Historical inventory / supplier metrics.
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

            demand_std=(
                "demand",
                "std",
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
    # Attach XGBoost forecast.
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
    # Safety Stock
    #
    # Safety Stock =
    # Z × historical demand standard deviation × sqrt(lead time)
    #
    # We retain historical variability because it represents uncertainty
    # around demand.
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
    # This is the important new step.
    #
    # Expected lead-time demand =
    # forecast daily demand × supplier lead time
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
    # Holding cost
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
    # D = annual historical demand
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
    # Prevent invalid values.
    # -------------------------------------------------------------------------

    inventory_metrics[
        "safety_stock"
    ] = inventory_metrics[
        "safety_stock"
    ].fillna(0)

    inventory_metrics[
        "expected_lead_time_demand"
    ] = inventory_metrics[
        "expected_lead_time_demand"
    ].fillna(0)

    inventory_metrics[
        "calculated_rop"
    ] = inventory_metrics[
        "calculated_rop"
    ].fillna(0)

    inventory_metrics[
        "eoq"
    ] = inventory_metrics[
        "eoq"
    ].replace(
        [np.inf, -np.inf],
        0,
    ).fillna(0)

    # -------------------------------------------------------------------------
    # Inventory status
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
    # Shortage relative to ROP.
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
    # Recommended order quantity.
    #
    # If inventory is below ROP:
    #
    #     Order EOQ
    #
    # Otherwise:
    #
    #     Order 0
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

    return inventory_metrics


# =============================================================================
# SUMMARY
# =============================================================================

def create_summary(results):

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

    print()
    print(
        f"Items requiring reorder : {reorder_items}"
    )

    print(
        f"Recommended order units : "
        f"{total_order_qty:,}"
    )


# =============================================================================
# TOP RECOMMENDATIONS
# =============================================================================

def show_top_reorders(results):

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
            "shortage_to_rop",
            ascending=False,
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
# SAVE
# =============================================================================

def save_results(results):

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
        .map(status_order)
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
            "status_order"
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

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    retail, forecast = load_data()

    # -------------------------------------------------------------------------
    # Prepare product-store forecast
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
    # Recommendations
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
    print("FORECAST-DRIVEN INVENTORY OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
