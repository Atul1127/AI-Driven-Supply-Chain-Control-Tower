# Intelligent Demand & Supply Chain Control Tower

### Retail Demand Forecasting • Inventory Optimization • Supplier Risk

An end-to-end retail supply-chain analytics system that connects **demand forecasting, inventory decisions, and supplier risk** into one operational control-tower workflow.

> **Business question:** What demand should we expect, which inventory is at risk, and what should we reorder?

## Architecture

```text
Synthetic Retail Data
        │
        ▼
   PostgreSQL + SQL
        │
        ├───────────────┐
        ▼               ▼
 Time-Series        Business EDA
 Analysis               │
        │               │
   ┌────┴────┐          │
   ▼    ▼    ▼          │
 ARIMA SARIMA SARIMAX   │
   │    │    │          │
   └────┼────┘          │
        ▼               │
      XGBoost           │
        │               │
        ▼               │
   30-Day Forecast      │
        │               │
        ▼               │
 Inventory Optimization
   │    │    │
   ▼    ▼    ▼
 Safety  ROP  EOQ
 Stock
        │
        ▼
 Recommended Orders
        │
        ├──────────────► Supplier Risk
        │
        ▼
  CONTROL TOWER
```

## What the system does

1. **Generates** a reproducible synthetic Indian retail dataset.
2. **Loads** the data into PostgreSQL for relational business analysis.
3. **Explores** revenue, product, store, seasonality, and stockout patterns.
4. **Analyzes** demand trend, seasonality, stationarity, ACF, and PACF.
5. **Benchmarks** ARIMA, SARIMA, and SARIMAX forecasting approaches.
6. **Forecasts** demand with XGBoost using lag, rolling, calendar, promotion, and discount features.
7. **Optimizes inventory** using forecast-driven lead-time demand, safety stock, reorder point, EOQ, and an operational demand cap.
8. **Scores supplier risk** using lead time, delivery reliability, defects, delays, and fill-rate signals.
9. **Creates a control-tower dataset** combining forecast, inventory, supplier, and business priority signals.

## Dataset

The project uses synthetic retail data rather than proprietary customer or company data.

| Property | Value |
|---|---:|
| Time period | 2023–2024 |
| Frequency | Daily |
| Stores | 5 |
| Categories | 6 |
| Products | 30 |
| Suppliers | 8 |
| Records | 109,650 |

The primary dataset is `data/retail_sales_data.csv`.

## Forecasting

### Statistical baselines

- ARIMA
- SARIMA
- SARIMAX

The models use a chronological split and provide MAE, RMSE, and MAPE comparisons.

### XGBoost

The ML forecasting pipeline uses features including:

```text
lag_1, lag_7, lag_14, lag_30
rolling_mean_7, rolling_mean_30
rolling_std_7
month, day_of_week, day_of_month, is_weekend
promo_event, average_discount
```

The repository's current XGBoost results are stored in `data/xgboost_results.csv` and the 30-day forecast in `data/30_day_xgboost_forecast.csv`.

## Inventory Optimization

The forecast is converted into operational decisions:

```text
Forecast Daily Demand
        ↓
Lead-Time Demand
        ↓
Safety Stock
        ↓
Forecast-Driven ROP
        ↓
EOQ
        ↓
30-Day Operational Cap
        ↓
Recommended Order Quantity
```

The final inventory output is `data/inventory_optimization_results.csv`.

## Supplier Risk

Supplier risk is based on explainable operational indicators such as:

- Average lead time
- On-time delivery rate
- Defect rate
- Supplier delays
- Fill rate

Output: `data/supplier_risk_analysis.csv`.

## Control Tower

`src/create_control_tower.py` combines:

- Inventory status
- Recommended order quantity
- Supplier risk
- 30-day forecast KPIs
- Business priority

Output: `data/control_tower_inventory.csv`.

## Repository Structure

```text
.
├── data/
│   ├── retail_sales_data.csv
│   ├── daily_demand.csv
│   ├── 30_day_statistical_forecast.csv
│   ├── statistical_model_results.csv
│   ├── 30_day_xgboost_forecast.csv
│   ├── xgboost_results.csv
│   ├── xgboost_feature_importance.csv
│   ├── inventory_optimization_results.csv
│   ├── supplier_risk_analysis.csv
│   └── control_tower_inventory.csv
│
├── images/
│   ├── banner.png
│   ├── Interface.png
│   ├── business/
│   ├── forecasting/
│   └── time_series/
│
├── sql/
│   └── 01_business_analysis.sql
│
├── src/
│   ├── generate_dataset.py
│   ├── load_to_postgres.py
│   ├── eda.py
│   ├── time_series_analysis.py
│   ├── statistical_forecasting.py
│   ├── xgboost_forecasting.py
│   ├── inventory_optimization.py
│   ├── supplier_risk.py
│   └── create_control_tower.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/Atul1127/Intelligent-Demand-Supply-Chain-Control-Tower-Our-Architecture.git
cd Intelligent-Demand-Supply-Chain-Control-Tower-Our-Architecture
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the pipeline

### 1. Generate the dataset

```bash
python src/generate_dataset.py
```

### 2. Load PostgreSQL

Configure the database connection used by `src/load_to_postgres.py`, then run:

```bash
python src/load_to_postgres.py
```

### 3. Run SQL analysis

Execute `sql/01_business_analysis.sql` against the PostgreSQL database.

### 4. Run EDA

```bash
python src/eda.py
```

Charts are written to `images/business/`.

### 5. Run time-series analysis

```bash
python src/time_series_analysis.py
```

Charts are written to `images/time_series/`.

### 6. Run statistical forecasting

```bash
python src/statistical_forecasting.py
```

### 7. Run XGBoost forecasting

```bash
python src/xgboost_forecasting.py
```

### 8. Run inventory optimization

```bash
python src/inventory_optimization.py
```

### 9. Run supplier-risk analysis

```bash
python src/supplier_risk.py
```

### 10. Build the control tower

```bash
python src/create_control_tower.py
```

## Key outputs

```text
data/30_day_xgboost_forecast.csv
      ↓
data/inventory_optimization_results.csv
      +
data/supplier_risk_analysis.csv
      ↓
data/control_tower_inventory.csv
```

## Technology

**Python · Pandas · NumPy · Scikit-learn · XGBoost · Statsmodels · PostgreSQL · SQL · Matplotlib · Seaborn**

## Notes

- The dataset is synthetic and intended for portfolio/learning use.
- Forecasting uses chronological evaluation rather than random train/test splitting.
- Inventory formulas are simplified operational models, not production procurement policies.
- Database credentials should be supplied through environment variables or local configuration and never committed to Git.
