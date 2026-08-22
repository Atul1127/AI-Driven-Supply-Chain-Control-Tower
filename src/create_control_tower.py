import pandas as pd

RETAIL_PATH = "data/retail_sales_data.csv"
INVENTORY_PATH = "data/inventory_optimization_results.csv"
SUPPLIER_PATH = "data/supplier_risk_analysis.csv"
FORECAST_PATH = "data/30_day_xgboost_forecast.csv"

OUTPUT_PATH = "data/control_tower_inventory.csv"


def main():

    print("=" * 70)
    print("CREATING CONTROL TOWER DATASET")
    print("=" * 70)

    retail = pd.read_csv(RETAIL_PATH)

    inventory = pd.read_csv(INVENTORY_PATH)

    suppliers = pd.read_csv(SUPPLIER_PATH)

    forecast = pd.read_csv(FORECAST_PATH)

    # -------------------------------------------------------------------------
    # Product → Supplier mapping
    # -------------------------------------------------------------------------

    product_supplier = (
        retail[
            [
                "product",
                "category",
                "supplier",
            ]
        ]
        .drop_duplicates(
            subset=["product"]
        )
    )

    # -------------------------------------------------------------------------
    # Attach supplier information
    # -------------------------------------------------------------------------

    control_tower = inventory.merge(
        product_supplier,
        on="product",
        how="left",
    )

    # -------------------------------------------------------------------------
    # Attach supplier risk
    # -------------------------------------------------------------------------

    control_tower = control_tower.merge(
        suppliers,
        on="supplier",
        how="left",
    )

    # -------------------------------------------------------------------------
    # Forecast KPI
    #
    # Forecast is currently at overall daily level.
    # We calculate a 30-day average forecast as an executive KPI.
    # -------------------------------------------------------------------------

    forecast["date"] = pd.to_datetime(
        forecast["date"]
    )

    average_forecast = (
        forecast["forecast_demand"]
        .mean()
    )

    total_forecast = (
        forecast["forecast_demand"]
        .sum()
    )

    control_tower[
        "average_30_day_forecast"
    ] = round(
        average_forecast
    )

    control_tower[
        "total_30_day_forecast"
    ] = round(
        total_forecast
    )

    # -------------------------------------------------------------------------
    # Business priority
    # -------------------------------------------------------------------------

    control_tower[
        "priority"
    ] = "LOW"

    control_tower.loc[
        (
            control_tower[
                "inventory_status"
            ]
            == "CRITICAL"
        ),
        "priority",
    ] = "HIGH"

    control_tower.loc[
        (
            control_tower[
                "inventory_status"
            ]
            == "REORDER"
        ),
        "priority",
    ] = "MEDIUM"

    # High inventory risk + high supplier risk
    # becomes highest business priority.

    control_tower.loc[
        (
            control_tower[
                "inventory_status"
            ]
            == "CRITICAL"
        )
        &
        (
            control_tower[
                "risk_level"
            ].astype(str)
            == "HIGH"
        ),
        "priority",
    ] = "URGENT"

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    control_tower.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print(
        f"Rows: {len(control_tower):,}"
    )

    print(
        f"Columns: {len(control_tower.columns)}"
    )

    print()
    print(
        "Priority distribution:"
    )

    print(
        control_tower[
            "priority"
        ]
        .value_counts()
        .to_string()
    )

    print()
    print(
        f"Saved to: {OUTPUT_PATH}"
    )

    print()
    print("=" * 70)
    print("CONTROL TOWER DATASET COMPLETE")
    print("=" * 70)


if __name__ == "__main__": main()
