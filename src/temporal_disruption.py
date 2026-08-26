"""Temporal supplier/SKU disruption signals using rolling operational baselines."""

from pathlib import Path
import numpy as np
import pandas as pd

DATA_PATH = Path("data/retail_sales_data.csv")
OUTPUT_PATH = Path("data/temporal_disruption.csv")


def run():
    df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    required = {"date", "supplier", "product", "lead_time_days", "on_time_rate", "defect_rate", "ordered_qty", "received_qty", "demand", "stockout"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    keys = ["supplier", "product"]
    daily = (df.groupby(keys + ["date"], as_index=False)
             .agg(lead_time=("lead_time_days", "mean"), on_time=("on_time_rate", "mean"),
                  defect_rate=("defect_rate", "mean"), ordered=("ordered_qty", "sum"),
                  received=("received_qty", "sum"), demand=("demand", "sum"), stockout=("stockout", "mean")))
    daily["fill_rate"] = (daily["received"] / daily["ordered"].replace(0, np.nan)).fillna(0)
    daily = daily.sort_values(keys + ["date"])
    group = daily.groupby(keys, group_keys=False)
    for col in ["lead_time", "on_time", "defect_rate", "fill_rate", "demand", "stockout"]:
        daily[f"{col}_baseline"] = group[col].transform(lambda s: s.rolling(14, min_periods=5).mean().shift(1))
        daily[f"{col}_change"] = daily[col] - daily[f"{col}_baseline"]

    daily["lead_time_pct_change"] = daily["lead_time_change"] / daily["lead_time_baseline"].replace(0, np.nan)
    daily["fill_rate_pct_change"] = daily["fill_rate_change"] / daily["fill_rate_baseline"].replace(0, np.nan)
    daily["on_time_pct_change"] = daily["on_time_change"] / daily["on_time_baseline"].replace(0, np.nan)
    daily["defect_pct_change"] = daily["defect_rate_change"] / daily["defect_rate_baseline"].replace(0, np.nan)
    daily = daily.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Standardized deterioration indicators; higher is worse.
    daily["disruption_signal"] = (
        daily["lead_time_pct_change"].clip(lower=0) * 35
        + (-daily["fill_rate_pct_change"]).clip(lower=0) * 25
        + (-daily["on_time_pct_change"]).clip(lower=0) * 20
        + daily["defect_pct_change"].clip(lower=0) * 10
        + daily["stockout_change"].clip(lower=0) * 10
    ).clip(0, 100)
    daily["disruption_stage"] = pd.cut(
        daily["disruption_signal"],
        bins=[-np.inf, 20, 40, 70, np.inf],
        labels=["NORMAL", "EARLY_WARNING", "EMERGING", "CRITICAL"],
    )
    daily["is_disruption"] = daily["disruption_stage"].isin(["EMERGING", "CRITICAL"])

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved temporal disruption signals: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
