# Intelligent Demand & Supply Chain Control Tower

### SKU-level Demand Forecasting • Inventory Optimization • Supplier Risk • Control Tower

An end-to-end retail supply-chain decision system that connects **store/SKU demand forecasting, inventory optimization, supplier risk, explainability, and operational priorities**.

> **Business question:** What will each store/SKU need, which inventory positions require action, and what should be reordered?

## Architecture

```text
Synthetic Retail Data
        │
        ├──────────────► PostgreSQL + SQL Analytics
        │
        ▼
 Store × SKU Daily Demand
        │
        ├── Naive baselines
        └── XGBoost forecasting
                │
                ▼
        30-Day SKU Forecast
                │
                ├──────────────► SHAP Explainability
                │
                ▼
        Inventory Optimization
        ├── Safety Stock
        ├── Reorder Point
        ├── EOQ
        └── Recommended Order
                │
        ┌───────┴────────┐
        ▼                ▼
 Supplier Risk     Business Impact
        │                │
        └───────┬────────┘
                ▼
        Executive Control Tower
                │
                ▼
          Streamlit Dashboard
```

## What is implemented

1. Reproducible synthetic retail data covering **2023–2024**, 5 stores, 30 products, 8 suppliers, and 109,650 daily records.
2. Store × product demand aggregation with leakage-safe lag and rolling features.
3. Comparable **1-day naive and 7-day seasonal-naive baselines**.
4. **XGBoost** SKU/store forecasting with chronological evaluation.
5. 30-day recursive forecasts for every eligible store/SKU pair.
6. Forecast-driven **Safety Stock, Reorder Point, EOQ, and replenishment recommendations**.
7. Supplier risk scoring from delivery reliability, defects, delays, and fill rate.
8. Control-tower priority logic: **URGENT → HIGH → MEDIUM → LOW**.
9. Historical stockout/lost-sales and current inventory business KPIs.
10. **SHAP** feature importance aligned with the same SKU-level XGBoost feature set.
11. Streamlit dashboard for operational filtering, forecasting, model benchmarking, supplier risk, and business impact.
12. A deterministic `src/run_pipeline.py` entry point with post-run output validation.

## Latest validated benchmark

The latest local end-to-end run used the same Store × SKU population and chronological test window for all three forecasting approaches:

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Naive-1-Day | 10.52 | 15.38 | 23.66% |
| Seasonal-Naive-7-Day | 8.93 | 13.30 | 19.83% |
| **XGBoost** | **5.79** | **8.28** | **13.38%** |

Relative to the seasonal-naive baseline, the validated XGBoost run achieved approximately **35.1% lower MAE, 37.7% lower RMSE, and 32.5% lower MAPE**.

### Latest operational snapshot

From the same successful pipeline run:

| Signal | Value |
|---|---:|
| Store × SKU pairs | 150 |
| Current stockout pairs | 44 |
| Historical stockout rate | 15.76% |
| Low coverage (<7 days) | 93 |
| Critical inventory pairs | 59 |
| Reorder inventory pairs | 48 |
| Recommended replenishment | 117,294 units |
| High supplier-risk suppliers | 0 |

These are **synthetic simulation results**, not real-world business savings.

## Forecasting

### Fair baseline comparison

The operational forecasting path evaluates baselines at the same **store × SKU level** and uses the same final chronological test window as XGBoost:

- Naive-1-Day
- Seasonal-Naive-7-Day
- XGBoost

Metrics:

- MAE
- RMSE
- MAPE

The repository does **not** claim XGBoost is best without comparing it against these baselines.

### XGBoost features

```text
lag_1, lag_7, lag_14, lag_30
rolling_mean_7, rolling_mean_30, rolling_std_7
month, day_of_week, day_of_month, is_weekend
promo_event, discount_pct
```

### Explainability

`src/shap_explainability.py` produces feature-level SHAP importance from the same SKU/store XGBoost formulation used by the operational forecast.

The latest run's strongest features were the 7-day rolling mean, 30-day rolling mean, day of week, month, and promotion signal.

## Inventory Optimization

The forecast is translated into operational decisions:

```text
Forecast Daily Demand
        ↓
Lead-Time Demand
        ↓
Safety Stock (95% service-level z)
        ↓
Reorder Point
        ↓
EOQ
        ↓
30-Day Operational Cap
        ↓
Recommended Replenishment
```

Annual demand is annualized from the actual observed date span rather than assuming the dataset contains exactly one year.

The system separately reports:

- current stockouts
- historical stockout days/rate
- low inventory coverage (<7 days)
- critical inventory pairs
- reorder pairs
- recommended replenishment units

## Supplier Risk

Supplier risk uses:

- average lead time
- on-time delivery
- defect rate
- supplier delays
- fill rate

Risk levels are **LOW / MEDIUM / HIGH** and feed the control-tower priority engine.

## Business Impact

`src/business_impact.py` reports historical and current operational signals rather than incorrectly labeling low inventory coverage as stockout risk:

- current stockout pairs
- historical stockout days
- historical lost-sales value
- current inventory value
- recommended replenishment
- critical/reorder/normal inventory counts
- low-coverage pairs

> **Important:** historical lost-sales value is simulated exposure. It is not money saved by the system.

## Streamlit Control Tower

Run:

```bash
streamlit run app.py
```

Dashboard sections:

- 🚨 Control Tower — filters and prioritized actions
- 📈 SKU Forecast — 30-day store/SKU forecast
- 📊 Model Benchmark — MAE/RMSE/MAPE comparison
- 🏭 Supplier Risk — supplier risk distribution and details
- 💰 Business Impact — stockouts, lost sales, inventory value, replenishment

## Repository Structure

```text
.
├── app.py
├── data/
│   └── retail_sales_data.csv
├── images/
│   ├── banner.png
│   ├── Interface.png
│   ├── business/
│   ├── forecasting/
│   └── time_series/
├── sql/
│   └── 01_business_analysis.sql
├── src/
│   ├── generate_dataset.py
│   ├── load_to_postgres.py
│   ├── eda.py
│   ├── time_series_analysis.py
│   ├── statistical_forecasting.py
│   ├── xgboost_forecasting.py
│   ├── baseline_forecasting.py
│   ├── sku_level_forecasting.py
│   ├── inventory_optimization.py
│   ├── supplier_risk.py
│   ├── create_control_tower.py
│   ├── business_impact.py
│   ├── shap_explainability.py
│   ├── validate_outputs.py
│   └── run_pipeline.py
├── .gitignore
├── requirements.txt
└── README.md
```

Generated CSV outputs are intentionally **not committed**. They are recreated by the pipeline.

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

## Run the operational pipeline

```bash
python src/run_pipeline.py
```

The pipeline runs, in order:

```text
1. baseline_forecasting.py
2. sku_level_forecasting.py
3. inventory_optimization.py
4. supplier_risk.py
5. create_control_tower.py
6. business_impact.py
7. shap_explainability.py
8. validate_outputs.py
```

The final validation checks required output schemas, non-empty artifacts, Store × SKU forecast coverage, and whether XGBoost beats the seasonal-naive benchmark on MAE/RMSE/MAPE.

Then launch the dashboard:

```bash
streamlit run app.py
```

## Optional analytical modules

For deeper portfolio analysis, the repository also contains:

```bash
python src/eda.py
python src/time_series_analysis.py
python src/statistical_forecasting.py
python src/xgboost_forecasting.py
```

These modules support the exploratory/statistical analysis layer; the operational control-tower pipeline is driven by the SKU-level forecasting path above.

## PostgreSQL / SQL

`src/load_to_postgres.py` and `sql/01_business_analysis.sql` provide the relational analytics layer. Database credentials should be supplied through environment variables or local configuration and never committed.

## Technology

**Python · Pandas · NumPy · Scikit-learn · XGBoost · Statsmodels · SHAP · Streamlit · PostgreSQL · SQL · Matplotlib · Seaborn**

## Important limitations

- The dataset is synthetic and intended for portfolio/learning use.
- Inventory formulas are simplified decision-support models, not production procurement policies.
- Recursive 30-day forecasts assume future promotion/discount inputs are zero unless explicitly modeled.
- Supplier risk thresholds are business-rule heuristics and should be calibrated for a real organization.
- Forecast metrics should be interpreted alongside the naive baselines rather than in isolation.
