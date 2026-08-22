"""
inventory_optimization.py
-------------------------

Inventory optimization for the
Intelligent Demand & Supply Chain Control Tower.

Calculates:

1. Average daily demand
2. Demand variability
3. Safety Stock
4. Reorder Point
5. EOQ
6. Current inventory status
7. Recommended order quantity

The goal is to convert demand forecasts into
practical inventory decisions.
"""

import os

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "data/retail_sales_data.csv"

OUTPUT_PATH = (
    "data/inventory_optimization_results.csv"
)

SERVICE_LEVEL_Z = 1.645

# Simple business assumptions.
# These are intentionally kept transparent for a fresher-level project.

ORDERING_COST = 500.0

HOLDING_RATE = 0.20

# Annual holding cost = unit price × 20%
ANNUAL_DAYS = 365


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    print(
        f"Rows loaded: {len(df):,}"
    )

    return df


# =============================================================================
# CALCULATE PRODUCT-STORE METRICS
# =============================================================================

def calculate_metrics(df):

    print()
    print("=" * 70)
    print("CALCULATING INVENTORY METRICS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Aggregate demand statistics.
    # -------------------------------------------------------------------------

    demand_metrics = (
        df.groupby(
            [
                "store",
                "product",
            ]
        )
        .agg(
            average_daily_demand=(
                "demand",
                "mean",
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

            reorder_point_existing=(
                "reorder_point",
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
    # Safety Stock
    #
    # Safety Stock = Z × Std Dev × sqrt(Lead Time)
    # -------------------------------------------------------------------------

    demand_metrics[
        "safety_stock"
    ] = (
        SERVICE_LEVEL_Z
        * demand_metrics["demand_std"]
        * np.sqrt(
            demand_metrics["lead_time_days"]
        )
    )

    # -------------------------------------------------------------------------
    # Reorder Point
    #
    # ROP =
    # Average Daily Demand × Lead Time
    # + Safety Stock
    # -------------------------------------------------------------------------

    demand_metrics[
        "calculated_rop"
    ] = (
        demand_metrics[
            "average_daily_demand"
        ]
        * demand_metrics[
            "lead_time_days"
        ]
        + demand_metrics[
            "safety_stock"
        ]
    )

    # -------------------------------------------------------------------------
    # Annual holding cost.
    # -------------------------------------------------------------------------

    demand_metrics[
        "holding_cost_per_unit"
    ] = (
        demand_metrics["unit_price"]
        * HOLDING_RATE
    )

    # -------------------------------------------------------------------------
    # EOQ
    #
    # EOQ = sqrt(2DS / H)
    # -------------------------------------------------------------------------

    demand_metrics[
        "eoq"
    ] = np.sqrt(
        (
            2
            * demand_metrics[
                "annual_demand"
            ]
            * ORDERING_COST
        )
        / demand_metrics[
            "holding_cost_per_unit"
        ]
    )

    # -------------------------------------------------------------------------
    # Round inventory quantities.
    # -------------------------------------------------------------------------

    quantity_columns = [
        "average_daily_demand",
        "demand_std",
        "annual_demand",
        "safety_stock",
        "calculated_rop",
        "eoq",
        "current_stock",
    ]

    for column in quantity_columns:

        demand_metrics[column] = (
            demand_metrics[column]
            .round(0)
            .astype(int)
        )

    # -------------------------------------------------------------------------
    # Inventory status.
    # -------------------------------------------------------------------------

    demand_metrics[
        "inventory_status"
    ] = np.select(
        [
            demand_metrics[
                "current_stock"
            ]
            <= demand_metrics[
                "safety_stock"
            ],

            demand_metrics[
                "current_stock"
            ]
            <= demand_metrics[
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
    # Recommended order quantity.
    #
    # If stock is below ROP:
    #
    # Recommended order = EOQ
    #
    # Otherwise:
    # No immediate order.
    # -------------------------------------------------------------------------

    demand_metrics[
        "recommended_order_qty"
    ] = np.where(
        demand_metrics[
            "current_stock"
        ]
        <= demand_metrics[
            "calculated_rop"
        ],

        demand_metrics[
            "eoq"
        ],

        0,
    )

    # -------------------------------------------------------------------------
    # Potential shortage.
    # -------------------------------------------------------------------------

    demand_metrics[
        "shortage_to_rop"
    ] = np.maximum(
        demand_metrics[
            "calculated_rop"
        ]
        - demand_metrics[
            "current_stock"
        ],
        0,
    )

    demand_metrics[
        "shortage_to_rop"
    ] = (
        demand_metrics[
            "shortage_to_rop"
        ]
        .round(0)
        .astype(int)
    )

    return demand_metrics


# =============================================================================
# CREATE SUMMARY
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

    print()

    reorder_items = (
        results[
            "recommended_order_qty"
        ]
        > 0
    ).sum()

    print(
        f"Items requiring reorder: {reorder_items}"
    )

    total_order_qty = (
        results[
            "recommended_order_qty"
        ]
        .sum()
    )

    print(
        f"Total recommended units: "
        f"{total_order_qty:,.0f}"
    )


# =============================================================================
# TOP REORDER ITEMS
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
        "current_stock",
        "safety_stock",
        "calculated_rop",
        "eoq",
        "shortage_to_rop",
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

def save_results(results):

    # Sort critical items first.
    status_order = {
        "CRITICAL": 1,
        "REORDER": 2,
        "NORMAL": 3,
    }

    results[
        "status_order"
    ] = results[
        "inventory_status"
    ].map(
        status_order
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
    print("INVENTORY OPTIMIZATION")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    df = load_data()

    # -------------------------------------------------------------------------
    # Calculate inventory metrics
    # -------------------------------------------------------------------------

    results = calculate_metrics(
        df
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------

    create_summary(
        results
    )

    # -------------------------------------------------------------------------
    # Show recommendations
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

    # -------------------------------------------------------------------------
    # Finish
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("INVENTORY OPTIMIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
