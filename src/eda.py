"""
eda.py
------
Exploratory Data Analysis – generates 8 charts saved to outputs/charts/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os

DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data",           "retail_sales_clean.csv")
CHART_DIR  = os.path.join(os.path.dirname(__file__), "..", "outputs", "charts")
os.makedirs(CHART_DIR, exist_ok=True)

PALETTE = "viridis"
sns.set_theme(style="darkgrid", palette=PALETTE)


def load():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print(f"EDA dataset: {df.shape}")
    return df


# ── Individual chart functions ──────────────────────────────────────────────

def plot_monthly_revenue(df):
    monthly = df.groupby(df["date"].dt.to_period("M"))["revenue"].sum().reset_index()
    monthly["date"] = monthly["date"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(monthly["date"], monthly["revenue"], color="#00b4d8", linewidth=2.5, marker="o", markersize=3)
    ax.fill_between(monthly["date"], monthly["revenue"], alpha=0.15, color="#00b4d8")
    ax.set_title("Monthly Total Revenue (All Stores)", fontsize=16, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "01_monthly_revenue.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_category_sales(df):
    cat = df.groupby("category")["revenue"].sum().sort_values()
    colors = sns.color_palette("Set2", len(cat))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(cat.index, cat.values, color=colors)
    ax.bar_label(bars, fmt="₹%.0f", padding=5, fontsize=9)
    ax.set_title("Total Revenue by Category", fontsize=16, fontweight="bold")
    ax.set_xlabel("Revenue (₹)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "02_category_revenue.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_store_comparison(df):
    store_monthly = df.groupby(["store", df["date"].dt.to_period("M")])["revenue"].sum().reset_index()
    store_monthly["date"] = store_monthly["date"].dt.to_timestamp()

    fig, ax = plt.subplots(figsize=(14, 6))
    for store, grp in store_monthly.groupby("store"):
        ax.plot(grp["date"], grp["revenue"], label=store, linewidth=2)
    ax.set_title("Monthly Revenue by Store", fontsize=16, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Revenue (₹)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
    ax.legend(title="Store")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "03_store_comparison.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_seasonal_heatmap(df):
    df2 = df.copy()
    df2["month_name"] = df2["date"].dt.month_name().str[:3]
    df2["year"]       = df2["date"].dt.year
    pivot = df2.groupby(["year", "month"])["units_sold"].sum().reset_index()
    pivot = pivot.pivot(index="year", columns="month", values="units_sold")

    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    fig, ax = plt.subplots(figsize=(14, 4))
    sns.heatmap(pivot, cmap="YlOrRd", annot=True, fmt=",", ax=ax,
                xticklabels=month_labels, linewidths=0.5)
    ax.set_title("Units Sold – Seasonal Heatmap (Year × Month)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Month"); ax.set_ylabel("Year")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "04_seasonal_heatmap.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_top_products(df):
    top = df.groupby("product")["revenue"].sum().nlargest(15).sort_values()

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(top.index, top.values, color=sns.color_palette("magma_r", 15))
    ax.bar_label(bars, fmt="₹%.0f", padding=3, fontsize=8)
    ax.set_title("Top 15 Products by Revenue", fontsize=16, fontweight="bold")
    ax.set_xlabel("Revenue (₹)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x/1e6:.1f}M"))
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "05_top_products.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_stockout_analysis(df):
    so = df.groupby("category").agg(
        total_days   = ("date", "count"),
        stockout_days= ("stockout", "sum")
    ).reset_index()
    so["stockout_rate"] = (so["stockout_days"] / so["total_days"] * 100).round(2)
    so = so.sort_values("stockout_rate")

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(so["category"], so["stockout_rate"],
                   color=["#e63946" if r > 5 else "#2a9d8f" for r in so["stockout_rate"]])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.axvline(5, color="orange", linestyle="--", linewidth=1.5, label="5% Threshold")
    ax.set_title("Stockout Rate by Category (%)", fontsize=15, fontweight="bold")
    ax.set_xlabel("Stockout Rate (%)"); ax.legend()
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "06_stockout_rate.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_revenue_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df["revenue"], bins=60, kde=True, color="#4cc9f0", ax=axes[0])
    axes[0].set_title("Revenue Distribution (All Records)")
    axes[0].set_xlabel("Revenue (₹)")

    daily = df.groupby("date")["revenue"].sum()
    sns.boxplot(data=df, x="category", y="revenue", palette="Set3", ax=axes[1])
    axes[1].set_title("Revenue by Category (Box Plot)")
    axes[1].set_xlabel("Category"); axes[1].set_ylabel("Revenue (₹)")
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle("Revenue Distribution Analysis", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "07_revenue_distribution.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_correlation(df):
    cols = ["units_sold", "revenue", "stock_level", "demand", "lost_sales",
            "promo_event", "is_weekend", "discount_pct", "lag_7", "rolling_mean_7"]
    cols = [c for c in cols if c in df.columns]
    corr = df[cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=15, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "08_correlation_heatmap.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def run_eda():
    df = load()
    print("\nRunning EDA …")
    plot_monthly_revenue(df)
    plot_category_sales(df)
    plot_store_comparison(df)
    plot_seasonal_heatmap(df)
    plot_top_products(df)
    plot_stockout_analysis(df)
    plot_revenue_distribution(df)
    plot_correlation(df)
    print("\n✅ All 8 EDA charts saved to outputs/charts/")


if __name__ == "__main__":
    run_eda()
