"""
xgboost_forecasting.py
----------------------

Machine Learning demand forecasting using XGBoost.

Features:
- Lag features
- Rolling statistics
- Calendar features
- Promotion
- Discount

Evaluation:
- MAE
- RMSE
- MAPE

The split is chronological to prevent future-data leakage.
"""

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor


warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_PATH = "data/retail_sales_data.csv"

OUTPUT_DIR = "images/forecasting"

RESULTS_PATH = "data/xgboost_results.csv"

FORECAST_PATH = "data/30_day_xgboost_forecast.csv"


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


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

    daily = (
        df.groupby("date")
        .agg(
            demand=("demand", "sum"),
            promo_event=("promo_event", "sum"),
            average_discount=("discount_pct", "mean"),
        )
        .sort_index()
    )

    full_dates = pd.date_range(
        daily.index.min(),
        daily.index.max(),
        freq="D",
    )

    daily = daily.reindex(
        full_dates
    )

    daily.index.name = "date"

    daily["demand"] = daily["demand"].fillna(0)

    daily["promo_event"] = (
        daily["promo_event"].fillna(0)
    )

    daily["average_discount"] = (
        daily["average_discount"].fillna(0)
    )

    return daily


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def create_features(df):

    data = df.copy()

    # -------------------------------------------------------------------------
    # Lag features
    # -------------------------------------------------------------------------

    data["lag_1"] = (
        data["demand"].shift(1)
    )

    data["lag_7"] = (
        data["demand"].shift(7)
    )

    data["lag_14"] = (
        data["demand"].shift(14)
    )

    data["lag_30"] = (
        data["demand"].shift(30)
    )

    # -------------------------------------------------------------------------
    # Rolling features
    #
    # Shift first to prevent using today's demand to predict today's demand.
    # -------------------------------------------------------------------------

    shifted_demand = (
        data["demand"].shift(1)
    )

    data["rolling_mean_7"] = (
        shifted_demand
        .rolling(7)
        .mean()
    )

    data["rolling_mean_30"] = (
        shifted_demand
        .rolling(30)
        .mean()
    )

    data["rolling_std_7"] = (
        shifted_demand
        .rolling(7)
        .std()
    )

    # -------------------------------------------------------------------------
    # Calendar features
    # -------------------------------------------------------------------------

    data["day_of_week"] = (
        data.index.dayofweek
    )

    data["month"] = (
        data.index.month
    )

    data["day_of_month"] = (
        data.index.day
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # -------------------------------------------------------------------------
    # Remove rows with missing lag/rolling values.
    # -------------------------------------------------------------------------

    data = data.dropna()

    return data


# =============================================================================
# METRICS
# =============================================================================

def calculate_mape(actual, predicted):

    actual = np.array(actual)

    predicted = np.array(predicted)

    mask = actual != 0

    return (
        np.mean(
            np.abs(
                (
                    actual[mask]
                    - predicted[mask]
                )
                / actual[mask]
            )
        )
        * 100
    )


# =============================================================================
# TRAIN MODEL
# =============================================================================

def train_model(train):

    features = [
        "lag_1",
        "lag_7",
        "lag_14",
        "lag_30",
        "rolling_mean_7",
        "rolling_mean_30",
        "rolling_std_7",
        "day_of_week",
        "month",
        "day_of_month",
        "is_weekend",
        "promo_event",
        "average_discount",
    ]

    X_train = train[
        features
    ]

    y_train = train[
        "demand"
    ]

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
    )

    print()
    print("=" * 70)
    print("TRAINING XGBOOST")
    print("=" * 70)

    model.fit(
        X_train,
        y_train,
        verbose=False,
    )

    return model, features


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_model(
    model,
    test,
    features,
):

    X_test = test[
        features
    ]

    y_test = test[
        "demand"
    ]

    predictions = model.predict(
        X_test
    )

    predictions = np.maximum(
        predictions,
        0,
    )

    mae = mean_absolute_error(
        y_test,
        predictions,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions,
        )
    )

    mape = calculate_mape(
        y_test,
        predictions,
    )

    print()
    print("=" * 70)
    print("XGBOOST RESULTS")
    print("=" * 70)

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
# FEATURE IMPORTANCE
# =============================================================================

def save_feature_importance(
    model,
    features,
):

    importance = pd.DataFrame(
        {
            "feature": features,
            "importance": model.feature_importances_,
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        "data/xgboost_feature_importance.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    print(
        importance.to_string(
            index=False
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        importance["feature"],
        importance["importance"],
    )

    plt.gca().invert_yaxis()

    plt.title(
        "XGBoost Feature Importance"
    )

    plt.xlabel(
        "Importance"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "xgboost_feature_importance.png",
        ),
        dpi=150,
    )

    plt.close()


# =============================================================================
# ACTUAL VS PREDICTED
# =============================================================================

def plot_predictions(
    test,
    predictions,
):

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
        label="XGBoost Prediction",
        linewidth=1.5,
    )

    plt.title(
        "Actual vs XGBoost Predicted Demand"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Demand"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "xgboost_actual_vs_predicted.png",
        ),
        dpi=150,
    )

    plt.close()


# =============================================================================
# FUTURE FORECAST
# =============================================================================

def create_future_forecast(
    daily,
    model,
    features,
):

    print()
    print("=" * 70)
    print("CREATING 30-DAY XGBOOST FORECAST")
    print("=" * 70)

    history = daily.copy()

    forecasts = []

    future_dates = pd.date_range(
        start=history.index.max()
        + pd.Timedelta(days=1),
        periods=30,
        freq="D",
    )

    for future_date in future_dates:

        # ---------------------------------------------------------------------
        # Create future row.
        #
        # Future promotion and discount values are unknown.
        # We use zero as a simple baseline assumption.
        # ---------------------------------------------------------------------

        future_row = pd.DataFrame(
            index=[future_date]
        )

        future_row["promo_event"] = 0

        future_row["average_discount"] = 0

        # ---------------------------------------------------------------------
        # Combine historical demand with future row.
        # ---------------------------------------------------------------------

        temp = pd.concat(
            [
                history,
                future_row,
            ]
        )

        # ---------------------------------------------------------------------
        # Create lag features.
        # ---------------------------------------------------------------------

        temp["lag_1"] = (
            temp["demand"].shift(1)
        )

        temp["lag_7"] = (
            temp["demand"].shift(7)
        )

        temp["lag_14"] = (
            temp["demand"].shift(14)
        )

        temp["lag_30"] = (
            temp["demand"].shift(30)
        )

        shifted = (
            temp["demand"].shift(1)
        )

        temp["rolling_mean_7"] = (
            shifted
            .rolling(7)
            .mean()
        )

        temp["rolling_mean_30"] = (
            shifted
            .rolling(30)
            .mean()
        )

        temp["rolling_std_7"] = (
            shifted
            .rolling(7)
            .std()
        )

        # ---------------------------------------------------------------------
        # Calendar features.
        # ---------------------------------------------------------------------

        temp["day_of_week"] = (
            temp.index.dayofweek
        )

        temp["month"] = (
            temp.index.month
        )

        temp["day_of_month"] = (
            temp.index.day
        )

        temp["is_weekend"] = (
            temp["day_of_week"] >= 5
        ).astype(int)

        # ---------------------------------------------------------------------
        # Extract future row.
        # ---------------------------------------------------------------------

        X_future = temp.loc[
            [future_date],
            features,
        ]

        prediction = model.predict(
            X_future
        )[0]

        prediction = max(
            0,
            prediction,
        )

        prediction = round(
            prediction
        )

        forecasts.append(
            {
                "date": future_date,
                "forecast_demand": int(
                    prediction
                ),
            }
        )

        # ---------------------------------------------------------------------
        # Add prediction to history.
        #
        # This allows the next forecast day to use the previous prediction
        # as lag_1.
        # ---------------------------------------------------------------------

        history.loc[
            future_date,
            "demand"
        ] = prediction

        history.loc[
            future_date,
            "promo_event"
        ] = 0

        history.loc[
            future_date,
            "average_discount"
        ] = 0

    forecast_df = pd.DataFrame(
        forecasts
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
    print("XGBOOST DEMAND FORECASTING")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Load
    # -------------------------------------------------------------------------

    daily = load_data()

    # -------------------------------------------------------------------------
    # Feature engineering
    # -------------------------------------------------------------------------

    data = create_features(
        daily
    )

    print()
    print(
        f"Rows after feature engineering: {len(data)}"
    )

    # -------------------------------------------------------------------------
    # Time-based split
    # -------------------------------------------------------------------------

    split_index = int(
        len(data) * 0.80
    )

    train = data.iloc[
        :split_index
    ]

    test = data.iloc[
        split_index:
    ]

    print(
        f"Training rows: {len(train)}"
    )

    print(
        f"Testing rows : {len(test)}"
    )

    # -------------------------------------------------------------------------
    # Train
    # -------------------------------------------------------------------------

    model, features = train_model(
        train
    )

    # -------------------------------------------------------------------------
    # Evaluate
    # -------------------------------------------------------------------------

    predictions, mae, rmse, mape = evaluate_model(
        model,
        test,
        features,
    )

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "model": "XGBoost",
                "MAE": mae,
                "RMSE": rmse,
                "MAPE": mape,
            }
        ]
    )

    results.to_csv(
        RESULTS_PATH,
        index=False,
    )

    # -------------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------------

    plot_predictions(
        test,
        predictions,
    )

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------

    save_feature_importance(
        model,
        features,
    )

    # -------------------------------------------------------------------------
    # Future forecast
    # -------------------------------------------------------------------------

    create_future_forecast(
        daily,
        model,
        features,
    )

    # -------------------------------------------------------------------------
    # Finish
    # -------------------------------------------------------------------------

    print()
    print("=" * 70)
    print("XGBOOST FORECASTING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
