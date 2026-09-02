"""Finance and business analysis layer for the supply-chain control tower.

Uses the existing synthetic sales/inventory dataset and explicit assumptions to
produce finance KPIs suitable for analyst interviews: COGS, gross profit,
gross margin, inventory turnover, DIO, holding cost, budget-vs-actual,
promotion effectiveness, ABC analysis, and supplier stockout exposure.
"""

import os
import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "retail_sales_data.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

UNIT_COST_RATE = 0.60
ANNUAL_HOLDING_RATE = 0.25


def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    # The original synthetic dataset contains selling prices but not procurement cost.
    # Use a transparent 60% of list-price assumption for portfolio finance analysis.
    df["unit_cost"] = df["unit_price"] * UNIT_COST_RATE
    df["cogs"] = df["units_sold"] * df["unit_cost"]
    df["gross_profit"] = df["revenue"] - df["cogs"]
    df["gross_margin_pct"] = np.where(df["revenue"] > 0, 100 * df["gross_profit"] / df["revenue"], 0)
    df["inventory_value"] = df["closing_stock"] * df["unit_cost"]
    df["lost_sales_value"] = df["lost_sales"] * df["sell_price"]
    return df


def finance_summary(df):
    revenue = df["revenue"].sum()
    cogs = df["cogs"].sum()
    gross_profit = revenue - cogs
    avg_inventory = df["inventory_value"].mean()
    turnover = cogs / avg_inventory if avg_inventory else 0
    dio = 365 / turnover if turnover else 0
    holding_cost = avg_inventory * ANNUAL_HOLDING_RATE
    return pd.DataFrame([{
        "revenue": revenue,
        "cogs": cogs,
        "gross_profit": gross_profit,
        "gross_margin_pct": 100 * gross_profit / revenue if revenue else 0,
        "average_inventory_value": avg_inventory,
        "inventory_turnover": turnover,
        "days_inventory_outstanding": dio,
        "annual_holding_cost": holding_cost,
        "historical_lost_sales_value": df["lost_sales_value"].sum(),
    }])


def monthly_budget_vs_actual(df):
    monthly = df.assign(month=df["date"].dt.to_period("M").dt.to_timestamp()).groupby("month", as_index=False).agg(
        actual_revenue=("revenue", "sum"),
        actual_cogs=("cogs", "sum"),
        actual_gross_profit=("gross_profit", "sum"),
    )
    # Simple planning assumptions: budget is 5% above prior month for revenue and 3% for COGS.
    monthly["budget_revenue"] = monthly["actual_revenue"].shift(1) * 1.05
    monthly["budget_cogs"] = monthly["actual_cogs"].shift(1) * 1.03
    monthly.loc[0, "budget_revenue"] = monthly.loc[0, "actual_revenue"]
    monthly.loc[0, "budget_cogs"] = monthly.loc[0, "actual_cogs"]
    monthly["budget_gross_profit"] = monthly["budget_revenue"] - monthly["budget_cogs"]
    monthly["revenue_variance"] = monthly["actual_revenue"] - monthly["budget_revenue"]
    monthly["revenue_variance_pct"] = 100 * monthly["revenue_variance"] / monthly["budget_revenue"].replace(0, np.nan)
    monthly["cogs_variance"] = monthly["actual_cogs"] - monthly["budget_cogs"]
    monthly["gross_profit_variance"] = monthly["actual_gross_profit"] - monthly["budget_gross_profit"]
    monthly["margin_variance_pct"] = (
        100 * monthly["actual_gross_profit"] / monthly["actual_revenue"].replace(0, np.nan)
        - 100 * monthly["budget_gross_profit"] / monthly["budget_revenue"].replace(0, np.nan)
    )
    return monthly.fillna(0)


def promotion_effectiveness(df):
    result = df.groupby("promo_event", as_index=False).agg(
        days=("date", "count"),
        units_sold=("units_sold", "sum"),
        revenue=("revenue", "sum"),
        demand=("demand", "sum"),
    )
    result["average_daily_units"] = result["units_sold"] / result["days"]
    result["average_daily_revenue"] = result["revenue"] / result["days"]
    result["conversion_pct"] = 100 * result["units_sold"] / result["demand"].replace(0, np.nan)
    result["promo_label"] = result["promo_event"].map({0: "Non-promotion", 1: "Promotion"})
    return result.fillna(0)


def abc_analysis(df):
    product = df.groupby(["category", "product"], as_index=False).agg(revenue=("revenue", "sum"))
    product = product.sort_values("revenue", ascending=False).reset_index(drop=True)
    product["revenue_share_pct"] = 100 * product["revenue"] / product["revenue"].sum()
    product["cumulative_share_pct"] = product["revenue_share_pct"].cumsum()
    product["abc_class"] = np.select(
        [product["cumulative_share_pct"] <= 80, product["cumulative_share_pct"] <= 95],
        ["A", "B"], default="C"
    )
    return product


def supplier_stockout_impact(df):
    return df.groupby("supplier", as_index=False).agg(
        revenue=("revenue", "sum"),
        lost_sales_units=("lost_sales", "sum"),
        lost_sales_value=("lost_sales_value", "sum"),
        stockout_days=("stockout", "sum"),
        average_lead_time_days=("lead_time_days", "mean"),
        on_time_rate=("on_time_rate", "mean"),
        defect_rate=("defect_rate", "mean"),
    ).sort_values("lost_sales_value", ascending=False)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = load_data()
    finance_summary(df).to_csv(os.path.join(OUTPUT_DIR, "finance_summary.csv"), index=False)
    monthly_budget_vs_actual(df).to_csv(os.path.join(OUTPUT_DIR, "budget_vs_actual.csv"), index=False)
    promotion_effectiveness(df).to_csv(os.path.join(OUTPUT_DIR, "promotion_effectiveness.csv"), index=False)
    abc_analysis(df).to_csv(os.path.join(OUTPUT_DIR, "abc_analysis.csv"), index=False)
    supplier_stockout_impact(df).to_csv(os.path.join(OUTPUT_DIR, "supplier_stockout_impact.csv"), index=False)
    print("Finance and business analysis outputs generated successfully.")


if __name__ == "__main__":
    main()
