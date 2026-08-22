"""
inventory_optimization.py
--------------------------
Calculates safety stock, reorder points, EOQ, and generates
reorder recommendations from the 30-day forecast.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

BASE      = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE, "..", "data",           "retail_sales_clean.csv")
FORE_PATH = os.path.join(BASE, "..", "outputs", "tables", "30day_forecast.csv")
TABLE_DIR = os.path.join(BASE, "..", "outputs", "tables")
CHART_DIR = os.path.join(BASE, "..", "outputs", "charts")
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

# ─── Constants ────────────────────────────────────────────────────────────────
LEAD_TIME_DAYS  = 7       # days to receive stock after ordering
SERVICE_LEVEL   = 0.95    # 95% → z-score ≈ 1.65
Z_SCORE         = 1.65
HOLDING_COST_PCT= 0.25    # 25% of unit price per year
ORDERING_COST   = 50      # fixed cost per order (₹)


def load_data():
    df   = pd.read_csv(DATA_PATH, parse_dates=["date"])
    fore = pd.read_csv(FORE_PATH, parse_dates=["date"])
    return df, fore


# ── Core inventory formulas ────────────────────────────────────────────────────

def compute_safety_stock(mean_daily_demand, std_daily_demand, lead_time=LEAD_TIME_DAYS):
    """
    Safety Stock = Z × σ_demand × √(Lead Time)
    Protects against demand variability during lead time.
    """
    return Z_SCORE * std_daily_demand * np.sqrt(lead_time)


def compute_reorder_point(mean_daily_demand, safety_stock, lead_time=LEAD_TIME_DAYS):
    """
    Reorder Point (ROP) = (Mean Daily Demand × Lead Time) + Safety Stock
    Order when stock drops to this level.
    """
    return (mean_daily_demand * lead_time) + safety_stock


def compute_eoq(annual_demand, unit_price, holding_pct=HOLDING_COST_PCT, order_cost=ORDERING_COST):
    """
    Economic Order Quantity = √(2 × D × S / H)
    Optimal order quantity balancing holding and ordering costs.
    """
    H = unit_price * holding_pct
    if H <= 0 or annual_demand <= 0:
        return 0
    return np.sqrt((2 * annual_demand * order_cost) / H)


def build_inventory_table(df, fore):
    # Historical demand stats per store-product
    hist = df.groupby(["store", "product", "category"]).agg(
        mean_daily_demand = ("units_sold", "mean"),
        std_daily_demand  = ("units_sold", "std"),
        current_stock     = ("stock_level", "last"),
        unit_price        = ("unit_price",  "first"),
        reorder_qty       = ("reorder_qty", "first"),
    ).reset_index()
    hist["std_daily_demand"] = hist["std_daily_demand"].fillna(0)

    # 30-day forecast sum per store-product
    fore_sum = fore.groupby(["store", "product"])["predicted_units"].sum().reset_index()
    fore_sum.rename(columns={"predicted_units": "forecast_30d"}, inplace=True)

    inv = hist.merge(fore_sum, on=["store", "product"], how="left")
    inv["forecast_30d"] = inv["forecast_30d"].fillna(inv["mean_daily_demand"] * 30)

    # ── Calculations ──────────────────────────────────────────────────────────
    inv["safety_stock"]   = inv.apply(
        lambda r: compute_safety_stock(r["mean_daily_demand"], r["std_daily_demand"]), axis=1
    ).round(1)

    inv["reorder_point"]  = inv.apply(
        lambda r: compute_reorder_point(r["mean_daily_demand"], r["safety_stock"]), axis=1
    ).round(1)

    inv["annual_demand"]  = inv["mean_daily_demand"] * 365

    inv["eoq"]            = inv.apply(
        lambda r: compute_eoq(r["annual_demand"], r["unit_price"]), axis=1
    ).round(0).astype(int)

    # Days of stock remaining
    inv["days_of_stock"]  = (inv["current_stock"] / (inv["mean_daily_demand"] + 0.01)).round(1)

    # Recommended order quantity
    inv["recommended_order"] = inv.apply(
        lambda r: max(r["eoq"], int(r["reorder_qty"])) if r["current_stock"] <= r["reorder_point"] else 0,
        axis=1
    )

    # Alert level
    def alert(row):
        if row["current_stock"] <= row["safety_stock"]:
            return "🔴 CRITICAL"
        elif row["current_stock"] <= row["reorder_point"]:
            return "🟠 REORDER NOW"
        elif row["days_of_stock"] < 14:
            return "🟡 LOW STOCK"
        else:
            return "🟢 OK"

    inv["alert"]     = inv.apply(alert, axis=1)
    inv["order_value"] = (inv["recommended_order"] * inv["unit_price"]).round(2)

    return inv


def generate_reorder_alerts(inv):
    alerts = inv[inv["recommended_order"] > 0].sort_values("days_of_stock")
    path   = os.path.join(TABLE_DIR, "reorder_alerts.csv")
    alerts[[
        "store","category","product","current_stock","reorder_point",
        "safety_stock","recommended_order","eoq","days_of_stock",
        "order_value","alert"
    ]].to_csv(path, index=False)
    print(f"Reorder alerts saved → {path}  ({len(alerts)} products need ordering)")
    return alerts


def plot_stock_health(inv):
    """Bar chart of days-of-stock for critical items."""
    critical = inv[inv["alert"].isin(["🔴 CRITICAL", "🟠 REORDER NOW"])].head(20)
    if critical.empty:
        return

    critical = critical.sort_values("days_of_stock")
    labels   = critical["product"] + "\n(" + critical["store"] + ")"
    colors   = ["#e63946" if "CRITICAL" in a else "#f4a261" for a in critical["alert"]]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.barh(labels, critical["days_of_stock"], color=colors)
    ax.axvline(LEAD_TIME_DAYS, color="blue", linestyle="--", linewidth=1.5, label=f"Lead Time ({LEAD_TIME_DAYS}d)")
    ax.bar_label(bars, fmt="%.1f days", padding=3, fontsize=8)
    ax.set_title("Stock Health – Days of Stock Remaining (Critical Items)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Days of Stock Remaining")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "12_stock_health.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_reorder_summary(inv):
    """Category-wise reorder value."""
    cat_orders = inv[inv["recommended_order"] > 0].groupby("category")["order_value"].sum().sort_values()

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#4361ee"] * len(cat_orders)
    bars   = ax.barh(cat_orders.index, cat_orders.values, color=colors)
    ax.bar_label(bars, fmt="₹%.0f", padding=3)
    ax.set_title("Total Recommended Order Value by Category", fontsize=14, fontweight="bold")
    ax.set_xlabel("Order Value (₹)")
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "13_reorder_by_category.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def plot_safety_vs_current(inv):
    sample = inv.groupby("category").agg(
        current_stock = ("current_stock", "mean"),
        safety_stock  = ("safety_stock",  "mean"),
        reorder_point = ("reorder_point", "mean"),
    ).reset_index()

    x = np.arange(len(sample))
    w = 0.25
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - w, sample["current_stock"], width=w, label="Current Stock", color="#4cc9f0")
    ax.bar(x,     sample["safety_stock"],  width=w, label="Safety Stock",  color="#f4a261")
    ax.bar(x + w, sample["reorder_point"], width=w, label="Reorder Point", color="#e63946")
    ax.set_xticks(x); ax.set_xticklabels(sample["category"], rotation=20)
    ax.set_title("Average Current Stock vs Safety Stock vs Reorder Point (by Category)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Units")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(CHART_DIR, "14_stock_vs_safety.png")
    plt.savefig(path, dpi=150); plt.close()
    print(f"Saved: {path}")


def run_inventory_optimization():
    df, fore = load_data()
    print("Building inventory optimization table …")
    inv      = build_inventory_table(df, fore)
    inv.to_csv(os.path.join(TABLE_DIR, "inventory_optimization.csv"), index=False)
    print(f"Full inventory table saved → outputs/tables/inventory_optimization.csv")

    alerts   = generate_reorder_alerts(inv)
    plot_stock_health(inv)
    plot_reorder_summary(inv)
    plot_safety_vs_current(inv)

    # Summary stats
    print(f"\n── Inventory Summary ─────────────────────────────────────")
    print(f"  Total products tracked : {len(inv)}")
    print(f"  Critical / Reorder Now : {len(inv[inv['alert'].str.contains('CRITICAL|REORDER')])} products")
    print(f"  Total order value      : ₹{inv['order_value'].sum():,.2f}")
    print(f"─────────────────────────────────────────────────────────\n")
    print("✅ Inventory optimization complete.")
    return inv, alerts


if __name__ == "__main__":
    run_inventory_optimization()
