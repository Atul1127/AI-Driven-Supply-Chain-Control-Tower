"""Exploratory data analysis for the retail supply-chain dataset."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "retail_sales_data.csv"
)
CHART_DIR = os.path.join(
    os.path.dirname(__file__), "..", "images", "business"
)

os.makedirs(CHART_DIR, exist_ok=True)


def load_data():
    """Load the canonical generated retail dataset."""
    return pd.read_csv(DATA_PATH, parse_dates=["date"])


def plot_monthly_revenue(df):
    monthly = (
        df.groupby(df["date"].dt.to_period("M"))["revenue"]
        .sum()
        .reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly["date"], monthly["revenue"], linewidth=2.5)
    ax.set_title("Monthly Total Revenue (All Stores)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "01_monthly_revenue.png"), dpi=150)
    plt.close(fig)


def plot_category_revenue(df):
    category = df.groupby("category")["revenue"].sum().sort_values()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(category.index, category.values)
    ax.set_title("Total Revenue by Category", fontsize=16, fontweight="bold")
    ax.set_xlabel("Revenue (₹)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "02_category_revenue.png"), dpi=150)
    plt.close(fig)


def plot_store_comparison(df):
    monthly = (
        df.groupby(["store", df["date"].dt.to_period("M")])["revenue"]
        .sum()
        .reset_index()
    )
    monthly["date"] = monthly["date"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 6))
    for store, group in monthly.groupby("store"):
        ax.plot(group["date"], group["revenue"], label=store, linewidth=2)
    ax.set_title("Monthly Revenue by Store", fontsize=16, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (₹)")
    ax.legend(title="Store")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "03_store_comparison.png"), dpi=150)
    plt.close(fig)


def plot_seasonality(df):
    data = df.assign(year=df["date"].dt.year, month=df["date"].dt.month)
    pivot = data.groupby(["year", "month"])["units_sold"].sum().unstack()
    pivot = pivot.reindex(columns=range(1, 13))

    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(
        pivot,
        cmap="YlOrRd",
        annot=True,
        fmt=",",
        ax=ax,
        xticklabels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    )
    ax.set_title("Units Sold – Seasonal Heatmap", fontsize=15, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "04_seasonal_heatmap.png"), dpi=150)
    plt.close(fig)


def plot_top_products(df):
    top = df.groupby("product")["revenue"].sum().nlargest(15).sort_values()
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top.index, top.values)
    ax.set_title("Top 15 Products by Revenue", fontsize=16, fontweight="bold")
    ax.set_xlabel("Revenue (₹)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "05_top_products.png"), dpi=150)
    plt.close(fig)


def plot_stockout_rate(df):
    stockout = (
        df.groupby("category")
        .agg(total_days=("date", "count"), stockout_days=("stockout", "sum"))
        .reset_index()
    )
    stockout["stockout_rate"] = stockout["stockout_days"] / stockout["total_days"] * 100
    stockout = stockout.sort_values("stockout_rate")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(stockout["category"], stockout["stockout_rate"])
    ax.axvline(5, linestyle="--", linewidth=1.5, label="5% Threshold")
    ax.set_title("Stockout Rate by Category", fontsize=15, fontweight="bold")
    ax.set_xlabel("Stockout Rate (%)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "06_stockout_rate.png"), dpi=150)
    plt.close(fig)


def plot_revenue_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["revenue"], bins=60, kde=True, ax=axes[0])
    axes[0].set_title("Revenue Distribution")
    sns.boxplot(data=df, x="category", y="revenue", ax=axes[1])
    axes[1].set_title("Revenue by Category")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "07_revenue_distribution.png"), dpi=150)
    plt.close(fig)


def plot_correlation(df):
    columns = [
        "units_sold", "revenue", "opening_stock", "closing_stock",
        "demand", "lost_sales", "promo_event", "discount_pct",
    ]
    columns = [column for column in columns if column in df.columns]
    corr = df[columns].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, "08_correlation_heatmap.png"), dpi=150)
    plt.close(fig)


def run_eda():
    """Generate the business-analysis charts used by the repository."""
    df = load_data()
    plot_monthly_revenue(df)
    plot_category_revenue(df)
    plot_store_comparison(df)
    plot_seasonality(df)
    plot_top_products(df)
    plot_stockout_rate(df)
    plot_revenue_distribution(df)
    plot_correlation(df)
    print(f"EDA charts saved to {CHART_DIR}")


if __name__ == "__main__":
    run_eda()
