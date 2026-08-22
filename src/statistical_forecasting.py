"""
statistical_forecasting.py
--------------------------

Statistical demand forecasting for the
Intelligent Demand & Supply Chain Control Tower.

Models:
1. ARIMA
2. SARIMA
3. SARIMAX

Evaluation:
- MAE
- RMSE
- MAPE

Forecast:
- 30 days

The models are trained using a time-based split.
No random train/test split is used because this is time-series data.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "data/retail_sales_data.csv"

OUTPUT_DIR = "images/forecasting"

RESULTS_PATH = "data/statistical_model_results.csv"

FORECAST_PATH = "data/30_day_statistical_forecast.csv"


# Create output directory.
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

def load_data():
    """
    Load the retail dataset and create the daily demand series.

    We also aggregate promotion and discount information because
    SARIMAX will use them as external variables.
    """

    print("=" * 70)
    print("LOADING DATA")
    print("=" * 70)

    df = pd.read_csv(
        DATA_PATH,
        parse_dates=["date"],
    )

    # -------------------------------------------------------------------------
    # Aggregate to daily level.
    # -------------------------------------------------------------------------

    daily = (
        df.groupby("date")
        .agg(
            demand=("demand", "sum"),
            promo_event=("promo_event", "sum"),
            average_discount=("discount_pct", "mean"),
        )
        .sort_index()
    )

    # Ensure continuous daily dates.
    full_dates = pd.date_range(
        start=daily.index.min(),
        end=daily.index.max(),
        freq="D",
    )

    daily = daily.reindex(
        full_dates
    )

    daily.index.name = "date"

    # Fill missing values.
    daily["demand"] = daily["demand"].fillna(0)

    daily["promo_event"] = (
        daily["promo_event"]
        .fillna(0)
    )

    daily["average_discount"] = (
        daily["average_discount"]
        .fillna(0)
    )

    print(
        f"Daily observations: {len(daily):,}"
    )

    print(
        f"Date range: "
        f"{daily.index.min().date()} "
        f"to "
        f"{daily.index.max().date()}"
    )

    return daily


# =============================================================================
# TRAIN / TEST SPLIT
# =============================================================================

def train_test_split(daily):
    """
    Perform a chronological 80/20 split.

    The future must never be used to train the model.
    """

    split_index = int(
        len(daily) * 0.80
    )

    train = daily.iloc[
        :split_index
    ]

    test = daily.iloc[
        split_index:
    ]

    print()
    print("=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    print(
        f"Training observations: {len(train)}"
    )

    print(
        f"Testing observations : {len(test)}"
    )

    print(
        f"Training period      : "
        f"{train.index.min().date()} "
        f"to "
        f"{train.index.max().date()}"
    )

    print(
        f"Testing period       : "
        f"{test.index.min().date()} "
        f"to "
        f"{test.index.max().date()}"
    )

    return train, test


# =============================================================================
# METRICS
# =============================================================================

def calculate_mape(actual, predicted):
    """
    Calculate MAPE while avoiding division by zero.
    """

    actual = np.array(actual)
    predicted = np.array(predicted)

    mask = actual != 0

    if mask.sum() == 0:
        return np.nan

    return (
        np.mean(
            np.abs(
                (actual[mask] - predicted[mask])
                / actual[mask]
            )
        )
        * 100
    )


def evaluate_model(actual, predicted):
    """
    Calculate forecasting metrics.
    """

    mae = mean_absolute_error(
        actual,
        predicted,
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted,
        )
    )

    mape = calculate_mape(
        actual,
        predicted,
    )

    return mae, rmse, mape


# =============================================================================
# ARIMA
# =============================================================================

def run_arima(train, test):
    """
    Basic ARIMA model.

    Order:
        (1, 1, 1)

    p = autoregressive component
    d = differencing
    q = moving average component
    """

    print()
    print("=" * 70)
    print("ARIMA")
    print("=" * 70)

    model = ARIMA(
        train["demand"],
        order=(1, 1, 1),
    )

    fitted_model = model.fit()

    predictions = fitted_model.forecast(
        steps=len(test)
    )

    predictions = pd.Series(
        predictions,
        index=test.index,
    )

    mae, rmse, mape = evaluate_model(
        test["demand"],
        predictions,
    )

    print(
        f"MAE : {mae:,.2f}"
    )

    print(
        f"RMSE: {rmse:,.2f}"
    )

    print(
        f"MAPE: {mape:.2f}%"
    )

    return predictions, mae, rmse, mape


# =============================================================================
# SARIMA
# =============================================================================

def run_sarima(train, test):
    """
    SARIMA model.

    Weekly seasonality is represented using period 7.

    Order:
        (1, 1, 1)

    Seasonal order:
        (1, 1, 1, 7)
    """

    print()
    print("=" * 70)
    print("SARIMA")
    print("=" * 70)

    model = SARIMAX(
        train["demand"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted_model = model.fit(
        disp=False
    )

    predictions = fitted_model.forecast(
        steps=len(test)
    )

    predictions = pd.Series(
        predictions,
        index=test.index,
    )

    mae, rmse, mape = evaluate_model(
        test["demand"],
        predictions,
    )

    print(
        f"MAE : {mae:,.2f}"
    )

    print(
        f"RMSE: {rmse:,.2f}"
    )

    print(
        f"MAPE: {mape:.2f}%"
    )

    return predictions, mae, rmse, mape


# =============================================================================
# SARIMAX
# =============================================================================

def run_sarimax(train, test):
    """
    SARIMAX model with external variables.

    Exogenous variables:
        - promotion events
        - average discount

    This allows the model to account for business factors
    that influence demand.
    """

    print()
    print("=" * 70)
    print("SARIMAX")
    print("=" * 70)

    exog_columns = [
        "promo_event",
        "average_discount",
    ]

    train_exog = train[
        exog_columns
    ]

    test_exog = test[
        exog_columns
    ]

    model = SARIMAX(
        train["demand"],
        exog=train_exog,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    fitted_model = model.fit(
        disp=False
    )

    predictions = fitted_model.forecast(
        steps=len(test),
        exog=test_exog,
    )

    predictions = pd.Series(
        predictions,
        index=test.index,
    )

    mae, rmse, mape = evaluate_model(
        test["demand"],
        predictions,
    )

    print(
        f"MAE : {mae:,.2f}"
    )

    print(
        f"RMSE: {rmse:,.2f}"
    )

    print(
        f"MAPE: {mape:.2f}%"
    )

    return predictions, mae, rmse, mape


# =============================================================================
# PLOT MODEL COMPARISON
# =============================================================================

def plot_predictions(
    test,
    predictions,
    model_name,
):
    """
    Plot actual vs predicted demand.
    """

    plt.figure(
        figsize=(14, 6)
    )

    plt.plot(
        test.index,
        test["demand"],
        label="Actual Demand",
        linewidth=1.5,
    )

    plt.plot(
        test.index,
        predictions,
        label=f"{model_name} Prediction",
        linewidth=1.5,
    )

    plt.title(
        f"Actual vs {model_name} Forecast"
    )

    plt.xlabel("Date")

    plt.ylabel("Demand")

    plt.legend()

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_actual_vs_predicted.png"
    )

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            filename,
        ),
        dpi=150,
    )

    plt.close()


# =============================================================================
# FUTURE 30-DAY FORECAST
# =============================================================================

def create_future_forecast(
    daily,
    best_model_name,
):
    """
    Train the selected statistical model on the full dataset
    and generate a 30-day forecast.

    For SARIMAX, future promotional/discount values are assumed
    to be zero because we do not know future promotions yet.
    """

    print()
    print("=" * 70)
    print("30-DAY FUTURE FORECAST")
    print("=" * 70)

    forecast_days = 30

    # -------------------------------------------------------------------------
    # ARIMA
    # -------------------------------------------------------------------------

    if best_model_name == "ARIMA":

        model = ARIMA(
            daily["demand"],
            order=(1, 1, 1),
        )

        fitted_model = model.fit()

        forecast = fitted_model.forecast(
            steps=forecast_days
        )

    # -------------------------------------------------------------------------
    # SARIMA
    # -------------------------------------------------------------------------

    elif best_model_name == "SARIMA":

        model = SARIMAX(
            daily["demand"],
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted_model = model.fit(
            disp=False
        )

        forecast = fitted_model.forecast(
            steps=forecast_days
        )

    # -------------------------------------------------------------------------
    # SARIMAX
    # -------------------------------------------------------------------------

    else:

        exog_columns = [
            "promo_event",
            "average_discount",
        ]

        train_exog = daily[
            exog_columns
        ]

        # Future external variables are unknown.
        # We use zero as a simple baseline assumption.
        future_exog = pd.DataFrame(
            {
                "promo_event": np.zeros(
                    forecast_days
                ),
                "average_discount": np.zeros(
                    forecast_days
                ),
            }
        )

        model = SARIMAX(
            daily["demand"],
            exog=train_exog,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 7),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )

        fitted_model = model.fit(
            disp=False
        )

        forecast = fitted_model.forecast(
            steps=forecast_days,
            exog=future_exog,
        )

    # -------------------------------------------------------------------------
    # Create forecast DataFrame
    # -------------------------------------------------------------------------

    future_dates = pd.date_range(
        start=daily.index.max()
        + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="D",
    )

    forecast_df = pd.DataFrame(
        {
            "date": future_dates,
            "forecast_demand": np.maximum(
                np.array(forecast),
                0,
            ),
            "model": best_model_name,
        }
    )

    forecast_df["forecast_demand"] = (
        forecast_df["forecast_demand"]
        .round()
        .astype(int)
    )

    forecast_df.to_csv(
        FORECAST_PATH,
        index=False,
    )

    print(
        f"Forecast saved to: {FORECAST_PATH}"
    )

    print()

    print(
        forecast_df.to_string(
            index=False
        )
    )

    return forecast_df


# =============================================================================
# MAIN
# =============================================================================

def main():

    print()
    print("=" * 70)
    print("STATISTICAL DEMAND FORECASTING")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load data
    # -------------------------------------------------------------------------

    daily = load_data()

    # -------------------------------------------------------------------------
    # Train/test split
    # -------------------------------------------------------------------------

    train, test = train_test_split(
        daily
    )

    # -------------------------------------------------------------------------
    # Train models
    # -------------------------------------------------------------------------

    arima_predictions, arima_mae, arima_rmse, arima_mape = run_arima(
        train,
        test,
    )

    sarima_predictions, sarima_mae, sarima_rmse, sarima_mape = run_sarima(
        train,
        test,
    )

    sarimax_predictions, sarimax_mae, sarimax_rmse, sarimax_mape = run_sarimax(
        train,
        test,
    )

    # -------------------------------------------------------------------------
    # Store results
    # -------------------------------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "model": "ARIMA",
                "MAE": arima_mae,
                "RMSE": arima_rmse,
                "MAPE": arima_mape,
            },
            {
                "model": "SARIMA",
                "MAE": sarima_mae,
                "RMSE": sarima_rmse,
                "MAPE": sarima_mape,
            },
            {
                "model": "SARIMAX",
                "MAE": sarimax_mae,
                "RMSE": sarimax_rmse,
                "MAPE": sarimax_mape,
            },
        ]
    )

    results = results.sort_values(
        "RMSE"
    )

    results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Display comparison
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results.to_string(
            index=False
        )
    )

    # -------------------------------------------------------------------------
    # Select best model
    # -------------------------------------------------------------------------

    best_model_name = (
        results.iloc[0]["model"]
    )

    print()
    print(
        f"Best statistical model: {best_model_name}"
    )

    # -------------------------------------------------------------------------
    # Plot predictions
    # -------------------------------------------------------------------------

    predictions_map = {
        "ARIMA": arima_predictions,
        "SARIMA": sarima_predictions,
        "SARIMAX": sarimax_predictions,
    }

    for model_name, predictions in predictions_map.items():

        plot_predictions(
            test,
            predictions,
            model_name,
        )

    # -------------------------------------------------------------------------
    # Create future forecast
    # -------------------------------------------------------------------------

    create_future_forecast(
        daily,
        best_model_name,
    )

    # -------------------------------------------------------------------------
    # Finish
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("FORECASTING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Results saved to: {RESULTS_PATH}"
    )

    print(
        f"30-day forecast saved to: {FORECAST_PATH}"
    )


if __name__ == "__main__":
    main()
