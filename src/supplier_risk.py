"""
supplier_risk.py
----------------

Supplier performance and risk analysis.

Metrics:
- Average lead time
- On-time delivery rate
- Defect rate
- Supplier delays
- Ordered quantity
- Received quantity
- Fill rate
- Supplier risk score
"""

import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "data/retail_sales_data.csv"

OUTPUT_PATH = "data/supplier_risk_analysis.csv"


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():

    print("=" * 70)
    print("LOADING SUPPLIER DATA")
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
# CALCULATE SUPPLIER METRICS
# =============================================================================

def calculate_supplier_metrics(df):

    print()
    print("=" * 70)
    print("CALCULATING SUPPLIER PERFORMANCE")
    print("=" * 70)

    supplier_metrics = (
        df.groupby("supplier")
        .agg(
            average_lead_time=(
                "lead_time_days",
                "mean",
            ),

            on_time_rate=(
                "on_time_rate",
                "mean",
            ),

            defect_rate=(
                "defect_rate",
                "mean",
            ),

            total_supplier_delays=(
                "supplier_delay",
                "sum",
            ),

            total_ordered=(
                "ordered_qty",
                "sum",
            ),

            total_received=(
                "received_qty",
                "sum",
            ),
        )
        .reset_index()
    )

    # -------------------------------------------------------------------------
    # Fill Rate
    # -------------------------------------------------------------------------

    supplier_metrics["fill_rate"] = (
        supplier_metrics["total_received"]
        / supplier_metrics["total_ordered"].replace(
            0,
            1,
        )
    )

    # -------------------------------------------------------------------------
    # Convert rates to percentages.
    # -------------------------------------------------------------------------

    supplier_metrics["on_time_percentage"] = (
        supplier_metrics["on_time_rate"]
        * 100
    )

    supplier_metrics["defect_percentage"] = (
        supplier_metrics["defect_rate"]
        * 100
    )

    supplier_metrics["fill_percentage"] = (
        supplier_metrics["fill_rate"]
        * 100
    )

    # -------------------------------------------------------------------------
    # Supplier Risk Score
    #
    # Higher score = higher risk.
    #
    # We deliberately use simple business rules.
    # -------------------------------------------------------------------------

    supplier_metrics["risk_score"] = (
        (100 - supplier_metrics["on_time_percentage"])
        * 0.40

        +

        supplier_metrics["defect_percentage"]
        * 5
        * 0.30

        +

        (100 - supplier_metrics["fill_percentage"])
        * 0.30
    )

    # -------------------------------------------------------------------------
    # Risk classification
    # -------------------------------------------------------------------------

    supplier_metrics["risk_level"] = pd.cut(
        supplier_metrics["risk_score"],
        bins=[
            -float("inf"),
            10,
            20,
            float("inf"),
        ],
        labels=[
            "LOW",
            "MEDIUM",
            "HIGH",
        ],
    )

    # -------------------------------------------------------------------------
    # Round values.
    # -------------------------------------------------------------------------

    supplier_metrics[
        "average_lead_time"
    ] = supplier_metrics[
        "average_lead_time"
    ].round(2)

    supplier_metrics[
        "on_time_percentage"
    ] = supplier_metrics[
        "on_time_percentage"
    ].round(2)

    supplier_metrics[
        "defect_percentage"
    ] = supplier_metrics[
        "defect_percentage"
    ].round(2)

    supplier_metrics[
        "fill_percentage"
    ] = supplier_metrics[
        "fill_percentage"
    ].round(2)

    supplier_metrics[
        "risk_score"
    ] = supplier_metrics[
        "risk_score"
    ].round(2)

    return supplier_metrics


# =============================================================================
# DISPLAY RESULTS
# =============================================================================

def display_results(results):

    print()
    print("=" * 70)
    print("SUPPLIER RISK ANALYSIS")
    print("=" * 70)

    columns = [
        "supplier",
        "average_lead_time",
        "on_time_percentage",
        "defect_percentage",
        "fill_percentage",
        "total_supplier_delays",
        "risk_score",
        "risk_level",
    ]

    print(
        results[
            columns
        ].to_string(
            index=False
        )
    )

    print()
    print("=" * 70)
    print("RISK SUMMARY")
    print("=" * 70)

    print(
        results[
            "risk_level"
        ]
        .value_counts()
        .to_string()
    )


# =============================================================================
# SAVE RESULTS
# =============================================================================

def save_results(results):

    risk_order = {
        "HIGH": 1,
        "MEDIUM": 2,
        "LOW": 3,
    }

    results["risk_order"] = (
        results["risk_level"]
        .astype(str)
        .map(risk_order)
    )

    results = results.sort_values(
        [
            "risk_order",
            "risk_score",
        ]
    )

    results = results.drop(
        columns=[
            "risk_order"
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
    print("SUPPLIER RISK ANALYSIS")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    df = load_data()

    # -------------------------------------------------------------------------
    # Calculate metrics
    # -------------------------------------------------------------------------

    results = calculate_supplier_metrics(
        df
    )

    # -------------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------------

    display_results(
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
    print("SUPPLIER RISK ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
