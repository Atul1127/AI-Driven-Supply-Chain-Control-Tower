"""
time_series_analysis.py
-----------------------

Time Series Exploratory Data Analysis for the
Intelligent Demand & Supply Chain Control Tower.

Analysis:
1. Daily demand aggregation
2. Trend analysis
3. Rolling mean
4. Rolling standard deviation
5. Monthly seasonality
6. Weekly seasonality
7. Stationarity test
8. ACF
9. PACF

Output:
images/time_series/
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "data/retail_sales_data.csv"

OUTPUT_DIR = "images/time_series"


# =============================================================================
# CREATE OUTPUT DIRECTORY
# =============================================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# =============================================================================
# LOAD DATA
# =============================================================================

def load_data():
    """
    Load retail sales dataset.
    """

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    print(f"Rows loaded: {len(df):,}")

    return df


# =============================================================================
# CREATE DAILY TIME SERIES
# =============================================================================

def create_daily_series(df):
    """
    Aggregate demand across all stores and products.

    This gives us the overall daily demand signal for the
    Supply Chain Control Tower.
    """

    daily_demand = (
        df.groupby("date")
        .agg(
            demand=("demand", "sum"),
            units_sold=("units_sold", "sum"),
            lost_sales=("lost_sales", "sum"),
            revenue=("revenue", "sum"),
        )
        .sort_index()
    )

    # Make sure every date exists.
    full_dates = pd.date_range(
        start=daily_demand.index.min(),
        end=daily_demand.index.max(),
        freq="D",
    )

    daily_demand = daily_demand.reindex(
        full_dates
    )

    daily_demand.index.name = "date"

    # Missing demand means zero demand for that date.
    daily_demand["demand"] = (
        daily_demand["demand"]
        .fillna(0)
    )

    daily_demand["units_sold"] = (
        daily_demand["units_sold"]
        .fillna(0)
    )

    daily_demand["lost_sales"] = (
        daily_demand["lost_sales"]
        .fillna(0)
    )

    daily_demand["revenue"] = (
        daily_demand["revenue"]
        .fillna(0)
    )

    return daily_demand


# =============================================================================
# TREND ANALYSIS
# =============================================================================

def analyze_trend(daily_demand):
    """
    Plot daily demand and rolling statistics.
    """

    print()
    print("=" * 70)
    print("TREND ANALYSIS")
    print("=" * 70)

    series = daily_demand["demand"]

    # 30-day rolling average.
    rolling_mean = (
        series.rolling(
            window=30
        )
        .mean()
    )

    # 30-day rolling standard deviation.
    rolling_std = (
        series.rolling(
            window=30
        )
        .std()
    )

    # -------------------------------------------------------------
    # Daily demand + rolling mean
    # -------------------------------------------------------------

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        series.index,
        series.values,
        linewidth=1,
        label="Daily Demand",
    )

    plt.plot(
        rolling_mean.index,
        rolling_mean.values,
        linewidth=2,
        label="30-Day Rolling Mean",
    )

    plt.title(
        "Daily Demand and 30-Day Rolling Mean"
    )

    plt.xlabel("Date")
    plt.ylabel("Demand")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "01_daily_demand_trend.png",
        ),
        dpi=150,
    )

    plt.close()

    # -------------------------------------------------------------
    # Rolling standard deviation
    # -------------------------------------------------------------

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        rolling_std.index,
        rolling_std.values,
        linewidth=1.5,
    )

    plt.title(
        "30-Day Rolling Standard Deviation"
    )

    plt.xlabel("Date")
    plt.ylabel("Rolling Standard Deviation")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "02_rolling_std.png",
        ),
        dpi=150,
    )

    plt.close()

    print(
        f"Average daily demand: {series.mean():,.0f}"
    )

    print(
        f"Minimum daily demand: {series.min():,.0f}"
    )

    print(
        f"Maximum daily demand: {series.max():,.0f}"
    )


# =============================================================================
# MONTHLY SEASONALITY
# =============================================================================

def analyze_monthly_seasonality(daily_demand):
    """
    Analyze average demand by month.
    """

    print()
    print("=" * 70)
    print("MONTHLY SEASONALITY")
    print("=" * 70)

    monthly = (
        daily_demand
        .groupby(
            daily_demand.index.month
        )["demand"]
        .mean()
    )

    print()
    print(
        monthly.to_string()
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        monthly.index,
        monthly.values,
        marker="o",
    )

    plt.title(
        "Average Daily Demand by Month"
    )

    plt.xlabel("Month")
    plt.ylabel("Average Demand")

    plt.xticks(
        range(1, 13)
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "03_monthly_seasonality.png",
        ),
        dpi=150,
    )

    plt.close()


# =============================================================================
# WEEKLY SEASONALITY
# =============================================================================

def analyze_weekly_seasonality(daily_demand):
    """
    Analyze average demand by day of week.
    """

    print()
    print("=" * 70)
    print("WEEKLY SEASONALITY")
    print("=" * 70)

    weekly = (
        daily_demand
        .groupby(
            daily_demand.index.dayofweek
        )["demand"]
        .mean()
    )

    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    weekly.index = [
        day_names[i]
        for i in weekly.index
    ]

    print()
    print(
        weekly.to_string()
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        weekly.index,
        weekly.values,
        marker="o",
    )

    plt.title(
        "Average Daily Demand by Day of Week"
    )

    plt.xlabel("Day")
    plt.ylabel("Average Demand")

    plt.xticks(
        rotation=30
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "04_weekly_seasonality.png",
        ),
        dpi=150,
    )

    plt.close()


# =============================================================================
# STATIONARITY
# =============================================================================

def stationarity_test(series):
    """
    Augmented Dickey-Fuller test.

    H0:
        The time series is non-stationary.

    If p-value < 0.05:
        Reject H0.
        Series is likely stationary.

    If p-value >= 0.05:
        Fail to reject H0.
        Differencing may be required.
    """

    print()
    print("=" * 70)
    print("STATIONARITY TEST")
    print("=" * 70)

    result = adfuller(
        series.dropna()
    )

    adf_statistic = result[0]
    p_value = result[1]

    print(
        f"ADF Statistic: {adf_statistic:.4f}"
    )

    print(
        f"p-value: {p_value:.6f}"
    )

    print()

    if p_value < 0.05:

        print(
            "Result: Series is likely stationary."
        )

    else:

        print(
            "Result: Series is likely non-stationary."
        )

        print(
            "Differencing may be required."
        )

    return p_value


# =============================================================================
# DIFFERENCING
# =============================================================================

def create_differenced_series(series):
    """
    Create first-differenced demand series.
    """

    differenced = series.diff().dropna()

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        differenced.index,
        differenced.values,
        linewidth=1,
    )

    plt.title(
        "First-Differenced Daily Demand"
    )

    plt.xlabel("Date")
    plt.ylabel("Differenced Demand")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "05_differenced_demand.png",
        ),
        dpi=150,
    )

    plt.close()

    return differenced


# =============================================================================
# ACF AND PACF
# =============================================================================

def analyze_acf_pacf(series):
    """
    Analyze autocorrelation and partial autocorrelation.

    Used to understand possible ARIMA/SARIMA parameters.
    """

    print()
    print("=" * 70)
    print("ACF / PACF ANALYSIS")
    print("=" * 70)

    # -------------------------------------------------------------
    # Use a manageable number of lags.
    # -------------------------------------------------------------

    lags = 60

    # -------------------------------------------------------------
    # ACF
    # -------------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plot_acf(
        series,
        lags=lags,
        ax=plt.gca(),
    )

    plt.title(
        "Autocorrelation Function (ACF)"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "06_acf.png",
        ),
        dpi=150,
    )

    plt.close()

    # -------------------------------------------------------------
    # PACF
    # -------------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plot_pacf(
        series,
        lags=lags,
        ax=plt.gca(),
        method="ywm",
    )

    plt.title(
        "Partial Autocorrelation Function (PACF)"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "07_pacf.png",
        ),
        dpi=150,
    )

    plt.close()

    print(
        "ACF and PACF plots generated."
    )


# =============================================================================
# SAVE DAILY SERIES
# =============================================================================

def save_daily_series(daily_demand):
    """
    Save aggregated daily demand for forecasting.
    """

    output_path = os.path.join(
        "data",
        "daily_demand.csv",
    )

    daily_demand.to_csv(
        output_path
    )

    print()
    print(
        f"Daily time series saved to: {output_path}"
    )


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 70)
    print("TIME SERIES ANALYSIS")
    print("=" * 70)

    # -------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------

    df = load_data()

    # -------------------------------------------------------------
    # Create daily series
    # -------------------------------------------------------------

    daily_demand = create_daily_series(
        df
    )

    print()
    print(
        f"Daily observations: {len(daily_demand):,}"
    )

    # -------------------------------------------------------------
    # Trend
    # -------------------------------------------------------------

    analyze_trend(
        daily_demand
    )

    # -------------------------------------------------------------
    # Monthly seasonality
    # -------------------------------------------------------------

    analyze_monthly_seasonality(
        daily_demand
    )

    # -------------------------------------------------------------
    # Weekly seasonality
    # -------------------------------------------------------------

    analyze_weekly_seasonality(
        daily_demand
    )

    # -------------------------------------------------------------
    # Stationarity
    # -------------------------------------------------------------

    series = daily_demand[
        "demand"
    ]

    p_value = stationarity_test(
        series
    )

    # -------------------------------------------------------------
    # Differencing
    # -------------------------------------------------------------

    differenced = create_differenced_series(
        series
    )

    # Test differenced series.
    stationarity_test(
        differenced
    )

    # -------------------------------------------------------------
    # ACF / PACF
    # -------------------------------------------------------------

    analyze_acf_pacf(
        differenced
    )

    # -------------------------------------------------------------
    # Save data
    # -------------------------------------------------------------

    save_daily_series(
        daily_demand
    )

    # -------------------------------------------------------------
    # Finish
    # -------------------------------------------------------------

    print()
    print("=" * 70)
    print("TIME SERIES ANALYSIS COMPLETE")
    print("=" * 70)

    print()
    print(
        "Generated files:"
    )

    print(
        " - data/daily_demand.csv"
    )

    print(
        " - images/time_series/01_daily_demand_trend.png"
    )

    print(
        " - images/time_series/02_rolling_std.png"
    )

    print(
        " - images/time_series/03_monthly_seasonality.png"
    )

    print(
        " - images/time_series/04_weekly_seasonality.png"
    )

    print(
        " - images/time_series/05_differenced_demand.png"
    )

    print(
        " - images/time_series/06_acf.png"
    )

    print(
        " - images/time_series/07_pacf.png"
    )


if __name__ == "__main__":
    main()
