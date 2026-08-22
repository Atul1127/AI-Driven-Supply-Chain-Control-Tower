
# Intelligent Demand & Supply Chain Control Tower

### Retail Demand Forecasting, Inventory Optimization & Supplier Risk Analytics

An end-to-end retail supply-chain analytics project that combines **PostgreSQL, Advanced SQL, Time-Series Forecasting, XGBoost, Inventory Optimization, and Supplier Risk Analysis** to transform retail demand data into actionable replenishment decisions.

The project focuses on a practical business question:

> **What demand should we expect, which products are at risk, and how much inventory should we reorder?**

---

## Project Overview

Retail businesses need to maintain enough inventory to satisfy customer demand while avoiding unnecessary inventory holding costs.

Poor demand forecasting can result in:

- Stockouts
- Lost sales
- Excess inventory
- Higher holding costs
- Poor replenishment decisions
- Supplier-related disruptions

This project builds a simplified **Supply Chain Control Tower** that connects demand forecasting with inventory and supplier decisions.

Instead of treating forecasting as an isolated machine-learning problem, the project follows the complete analytical workflow:

```text
Retail Data
     ↓
PostgreSQL
     ↓
Advanced SQL & Business Analysis
     ↓
Time-Series Analysis
     ↓
ARIMA / SARIMA / SARIMAX
     ↓
XGBoost Demand Forecasting
     ↓
30-Day Demand Forecast
     ↓
Inventory Optimization
     ↓
Safety Stock + Lead-Time Demand
     ↓
Forecast-Driven ROP
     ↓
EOQ + Operational Order Cap
     ↓
Recommended Replenishment
     ↓
Supplier Risk Analysis
     ↓
Supply Chain Control Tower
````

---

# Key Objectives

The project is designed to answer five practical supply-chain questions:

1. **What has happened to sales and demand?**
2. **What demand should we expect in the next 30 days?**
3. **Which products/stores are at risk of stockout?**
4. **When should inventory be reordered and how much should be ordered?**
5. **Which suppliers represent higher supply-chain risk?**

---

# Architecture

```text
                    INDIAN RETAIL DATA
                           │
                           ▼
                      PostgreSQL
                           │
                  ┌────────┴────────┐
                  ▼                 ▼
            Advanced SQL       Time Series
                  │                 │
                  │          ┌──────┴──────┐
                  │          ▼      ▼      ▼
                  │        ARIMA  SARIMA  SARIMAX
                  │          │      │      │
                  │          └──────┼──────┘
                  │                 ▼
                  │              XGBoost
                  │                 │
                  └────────┬────────┘
                           ▼
                    Demand Forecast
                           │
                 ┌─────────┼─────────┐
                 ▼         ▼         ▼
             Stockout   Inventory  Supplier
               Risk      Planning    Risk
                           │
                     ┌─────┼─────┐
                     ▼     ▼     ▼
                    ROP   EOQ  Safety Stock
                           │
                           ▼
                  Recommended Orders
                           │
                           ▼
                   CONTROL TOWER
```

---

# Dataset

A synthetic Indian retail supply-chain dataset is generated specifically for this project.

### Dataset Characteristics

| Property           |     Value |
| ------------------ | --------: |
| Time Period        | 2023–2024 |
| Frequency          |     Daily |
| Stores             |         5 |
| Product Categories |         6 |
| Products           |        30 |
| Suppliers          |         8 |
| Records            |   109,650 |

### Product Categories

* Electronics
* Clothing
* Groceries
* Home & Kitchen
* Toys
* Sports

### Stores

* Bengaluru
* Delhi
* Mumbai
* Hyderabad
* Pune

The dataset contains both demand-side and supply-chain information.

### Important Fields

```text
date
store
category
product
supplier
unit_price
discount_pct
sell_price
demand
units_sold
lost_sales
revenue
opening_stock
closing_stock
reorder_point
reorder_qty
reordered
ordered_qty
received_qty
stockout
promo_event
lead_time_days
on_time_rate
defect_rate
supplier_delay
```

---

# 1. Exploratory Data Analysis

The project begins with business-oriented exploratory analysis.

Key areas include:

* Monthly revenue
* Category revenue
* Store performance
* Product performance
* Revenue distribution
* Stockout rates
* Correlation analysis
* Seasonal demand patterns

Visualizations are available under:

```text
images/business/
```

---

# 2. PostgreSQL & Advanced SQL

The dataset is loaded into **PostgreSQL** to perform business analysis directly at the database layer.

SQL analysis includes:

* Revenue analysis
* Product performance
* Store performance
* Category performance
* Inventory analysis
* Stockout analysis
* Supplier analysis
* Inventory risk classification
* Ranking
* Common Table Expressions
* Window functions
* Conditional business logic

The SQL implementation is available in:

```text
sql/
└── 01_business_analysis.sql
```

---

# 3. Time-Series Analysis

Demand is analyzed as a time series before machine-learning forecasting.

The analysis covers:

### Trend

Identifying long-term movement in demand.

### Seasonality

Identifying recurring:

* Monthly patterns
* Weekly patterns
* Retail seasonal effects

### Rolling Statistics

Rolling mean and standard deviation are used to understand changing demand behavior and volatility.

### Stationarity

Differencing is used to examine whether the demand series is stationary.

### Autocorrelation

ACF and PACF are analyzed to understand temporal relationships and support statistical model selection.

Visualizations are available under:

```text
images/time_series/
```

---

# 4. Statistical Forecasting

Three classical time-series approaches are evaluated:

```text
ARIMA
SARIMA
SARIMAX
```

The models provide a statistical forecasting baseline before applying machine learning.

The models are evaluated using:

* MAE
* RMSE
* MAPE

Forecast comparison visualizations are available under:

```text
images/forecasting/
```

---

# 5. XGBoost Demand Forecasting

XGBoost is used as the primary machine-learning forecasting model.

The model uses engineered time-series and business features.

### Features

```text
lag_1
lag_7
lag_14
lag_30

rolling_mean_7
rolling_mean_30

rolling_std_7

month
day_of_week
day_of_month
is_weekend

promo_event
average_discount
```

These features allow the model to capture:

* Recent demand
* Weekly patterns
* Longer-term demand behavior
* Seasonality
* Weekend effects
* Promotional effects
* Demand volatility

---

# Model Evaluation

The XGBoost model achieved:

| Metric | Result |
| ------ | -----: |
| MAE    | 286.68 |
| RMSE   | 368.75 |
| MAPE   |  4.88% |

### Feature Importance

The strongest features in the final model included:

```text
is_weekend
lag_7
month
day_of_week
lag_14
rolling_mean_7
lag_1
rolling_std_7
lag_30
```

Forecasting visualizations are available under:

```text
images/forecasting/
```

---

# 6. 30-Day Demand Forecast

The trained XGBoost model generates a 30-day demand forecast.

Output:

```text
data/30_day_xgboost_forecast.csv
```

The forecast provides the demand signal used by the inventory optimization layer.

Example:

```text
Date          Forecast Demand
2025-01-01        4,878
2025-01-02        4,849
2025-01-03        4,868
...
```

---

# 7. Forecast-Driven Inventory Optimization

This is the main business-decision layer of the project.

The demand forecast is directly connected to inventory planning.

```text
XGBoost Forecast
       ↓
Forecast Daily Demand
       ↓
Expected Demand During Lead Time
       ↓
Safety Stock
       ↓
Forecast-Driven ROP
       ↓
EOQ
       ↓
Operational Demand Cap
       ↓
Recommended Order
```

---

## Expected Lead-Time Demand

Expected demand during the supplier lead time is calculated as:

```text
Expected Lead-Time Demand
=
Forecast Daily Demand × Supplier Lead Time
```

For example:

```text
Forecast demand = 67 units/day
Lead time       = 9 days

Expected lead-time demand
= 67 × 9
= 603 units
```

---

# Safety Stock

Safety stock protects against demand variability during supplier lead time.

The project uses a 95% service level.

```text
Safety Stock
=
Z × Demand Standard Deviation × √Lead Time
```

Where:

```text
Z = 1.645
```

A fallback hierarchy is used when calculating demand variability:

```text
Store + Product Demand Std
            ↓
Product-Level Demand Std
            ↓
Overall Demand Std
```

This prevents unrealistic zero safety-stock values when a particular grouping has insufficient variability.

---

# Forecast-Driven Reorder Point

The reorder point is based on the forecast rather than only historical average demand.

```text
ROP
=
Expected Lead-Time Demand
+
Safety Stock
```

Therefore:

```text
Forecast
   ↓
Lead-Time Demand
   ↓
Safety Stock
   ↓
ROP
```

This connects the machine-learning forecast directly to the inventory decision.

---

# Economic Order Quantity

Classical EOQ is calculated using:

```text
EOQ
=
√(2 × D × S / H)
```

Where:

```text
D = Annual Demand
S = Ordering Cost
H = Annual Holding Cost per Unit
```

Project assumptions:

```text
Service Level = 95%
Ordering Cost = ₹2,000
Holding Rate  = 25%
```

The theoretical EOQ is retained in the output for analytical transparency.

---

# Operational Order Cap

Classical EOQ can sometimes produce very large quantities when applied to synthetic demand and simplified cost assumptions.

Therefore, the project applies a practical operational constraint:

```text
30-Day Demand Cap
=
Forecast Daily Demand × 30
```

The final recommended order quantity is:

```text
Recommended Order
=
MIN(EOQ, 30-Day Forecast Demand)
```

This preserves the EOQ calculation while preventing the system from recommending an unrealistic purchase quantity relative to near-term demand.

---

# Inventory Risk Classification

Products are classified into three inventory states:

```text
CRITICAL
REORDER
NORMAL
```

### CRITICAL

Current inventory is at or below safety stock.

### REORDER

Current inventory is above safety stock but at or below the forecast-driven reorder point.

### NORMAL

Current inventory is above the reorder point.

---

# Inventory Optimization Output

The final optimization output is saved to:

```text
data/inventory_optimization_results.csv
```

Important fields include:

```text
store
product
supplier
forecast_daily_demand
lead_time_days
expected_lead_time_demand
demand_std
safety_stock
calculated_rop
current_stock
shortage_to_rop
eoq
30_day_demand_cap
inventory_status
recommended_order_qty
```

This transforms a demand forecast into an actionable replenishment recommendation.

---

# 8. Supplier Risk Analysis

Supplier performance is analyzed using:

```text
Lead Time
On-Time Delivery Rate
Defect Rate
Supplier Delay
```

The analysis helps identify suppliers that may increase supply-chain risk.

Output:

```text
data/supplier_risk_analysis.csv
```

The goal is not to build a complex supplier-risk model, but to provide a simple and explainable risk layer for the control tower.

---

# 9. Supply Chain Control Tower

The final control-tower layer combines the outputs from:

```text
Demand Forecast
       +
Inventory Risk
       +
Replenishment Recommendations
       +
Supplier Risk
```

The control tower provides a consolidated view of:

* Demand expectations
* Inventory health
* Critical products
* Reorder requirements
* Recommended order quantities
* Supplier risk

Final output:

```text
data/control_tower_inventory.csv
```

---

# Project Outputs

The main generated datasets are:

```text
data/
├── retail_sales_data.csv
├── daily_demand.csv
├── statistical_model_results.csv
├── 30_day_statistical_forecast.csv
├── xgboost_results.csv
├── xgboost_feature_importance.csv
├── 30_day_xgboost_forecast.csv
├── inventory_optimization_results.csv
├── supplier_risk_analysis.csv
└── control_tower_inventory.csv
```

---

# Project Structure

```text
Intelligent-Demand-Supply-Chain-Control-Tower/
│
├── data/
│   ├── retail_sales_data.csv
│   ├── daily_demand.csv
│   ├── statistical_model_results.csv
│   ├── 30_day_statistical_forecast.csv
│   ├── xgboost_results.csv
│   ├── xgboost_feature_importance.csv
│   ├── 30_day_xgboost_forecast.csv
│   ├── inventory_optimization_results.csv
│   ├── supplier_risk_analysis.csv
│   └── control_tower_inventory.csv
│
├── images/
│   ├── business/
│   ├── forecasting/
│   ├── time_series/
│   ├── Interface.png
│   └── banner.png
│
├── notebooks/
│   └── retail_supply_chain_analysis.ipynb
│
├── sql/
│   └── 01_business_analysis.sql
│
├── src/
│   ├── __init__.py
│   ├── create_control_tower.py
│   ├── eda.py
│   ├── generate_dataset.py
│   ├── inventory_optimization.py
│   ├── load_to_postgres.py
│   ├── statistical_forecasting.py
│   ├── supplier_risk.py
│   ├── time_series_analysis.py
│   └── xgboost_forecasting.py
│
├── docs/
│   └── project_guide.md
│
├── requirements.txt
├── setup.py
└── README.md
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/Atul1127/Intelligent-Demand-Supply-Chain-Control-Tower-Our-Architecture.git
```

Move into the project:

```bash
cd Intelligent-Demand-Supply-Chain-Control-Tower-Our-Architecture
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

## Step 1 — Generate Dataset

```bash
python src/generate_dataset.py
```

This creates:

```text
data/retail_sales_data.csv
```

---

## Step 2 — Load Data into PostgreSQL

Configure the PostgreSQL connection in:

```text
src/load_to_postgres.py
```

Then run:

```bash
python src/load_to_postgres.py
```

---

## Step 3 — Run SQL Analysis

Open:

```text
sql/01_business_analysis.sql
```

Execute the queries inside PostgreSQL.

---

## Step 4 — Run EDA

```bash
python src/eda.py
```

---

## Step 5 — Run Time-Series Analysis

```bash
python src/time_series_analysis.py
```

This produces the time-series diagnostic outputs.

---

## Step 6 — Run Statistical Forecasting

```bash
python src/statistical_forecasting.py
```

This evaluates:

```text
ARIMA
SARIMA
SARIMAX
```

---

## Step 7 — Run XGBoost Forecasting

```bash
python src/xgboost_forecasting.py
```

This generates:

```text
data/30_day_xgboost_forecast.csv
```

---

## Step 8 — Run Inventory Optimization

```bash
python src/inventory_optimization.py
```

This uses the XGBoost forecast to calculate:

```text
Expected Lead-Time Demand
Safety Stock
ROP
EOQ
Recommended Order Quantity
```

---

## Step 9 — Run Supplier Risk Analysis

```bash
python src/supplier_risk.py
```

---

## Step 10 — Generate Control Tower Output

```bash
python src/create_control_tower.py
```

---

# Technologies Used

### Programming & Data

* Python
* Pandas
* NumPy

### Database

* PostgreSQL
* SQL

### Machine Learning

* Scikit-learn
* XGBoost

### Time Series

* Statsmodels
* ARIMA
* SARIMA
* SARIMAX

### Visualization

* Matplotlib
* Seaborn

### Development

* Jupyter Notebook
* Git
* GitHub

---

# Business Value

The project demonstrates how different analytics techniques can be connected into a practical supply-chain workflow.

### Instead of:

```text
Machine Learning Model
        ↓
Prediction
```

The project builds:

```text
Historical Data
      ↓
Business Analysis
      ↓
Demand Forecast
      ↓
Inventory Risk
      ↓
Reorder Point
      ↓
Order Quantity
      ↓
Supplier Risk
      ↓
Business Decision
```

This makes the project useful for demonstrating skills relevant to:

* Business Analyst
* Data Analyst
* Data Scientist
* Machine Learning Engineer
* Supply Chain Analytics

---

# Key Results

### Forecasting

```text
XGBoost MAPE = 4.88%
```

### Forecast Horizon

```text
30 days
```

### Inventory Planning

The system calculates:

```text
Safety Stock
Forecast-Driven ROP
Classical EOQ
30-Day Operational Cap
Recommended Order Quantity
```

### Inventory Risk

The control tower classifies products into:

```text
CRITICAL
REORDER
NORMAL
```

---

# Resume Project Description

### Intelligent Demand & Supply Chain Control Tower

Built an end-to-end retail supply-chain analytics system using **PostgreSQL, Advanced SQL, statistical time-series forecasting, and XGBoost** across 109,650 daily retail records. Compared **ARIMA, SARIMA, and SARIMAX** with machine-learning forecasting and achieved **4.88% MAPE using XGBoost**. Developed a forecast-driven inventory optimization layer combining **lead-time demand, safety stock, reorder point, EOQ, and operational demand caps** to generate replenishment recommendations, and integrated supplier lead-time, delivery, defect, and delay metrics into a supply-chain control tower.

---

# Future Improvements

Potential future extensions include:

* Real-time inventory feeds
* Automated model retraining
* Real supplier data
* Multi-echelon inventory optimization
* Promotion-aware forecasting
* Dynamic safety-stock policies
* Automated alerts for critical inventory
* Production deployment and monitoring

GitHub:

[https://github.com/Atul1127](https://github.com/Atul1127)

---

# Disclaimer

The retail dataset used in this project is **synthetically generated for educational and portfolio purposes**.

The supplier, inventory, demand, pricing, and operational assumptions do not represent any specific real-world company.

