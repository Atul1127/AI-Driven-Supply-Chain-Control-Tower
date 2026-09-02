# AI-Driven Supply Chain Control Tower

**Supply-Chain Analytics · Financial Analysis · Demand Forecasting · Inventory Optimization · Supplier Risk · Decision Dashboard**

An end-to-end supply-chain analytics project that turns store/SKU demand and supplier operations into **SQL insights, financial KPIs, forecasts, replenishment decisions, risk signals, disruption alerts, and an interactive control tower**.

> **Portfolio note:** the dataset is synthetic. The project demonstrates an end-to-end analytics/ML workflow and does not claim production deployment or real-world disruption ground truth.

## What the system does

```text
Retail Supply-Chain Data
          │
          ├── SQL Business Analytics
          ├── Financial Analysis
          │     ├── Revenue / COGS / Gross Profit / Margin
          │     ├── Budget vs Actual / Variance
          │     ├── Inventory Turnover / DIO / Holding Cost
          │     ├── Promotion Effectiveness
          │     └── ABC Product Analysis
          │
          ▼
     Store × SKU Demand
          │
          ├── Naive baselines
          └── XGBoost forecasting
                  │
                  ▼
          30-Day Demand Forecast
                  │
          ┌───────┴────────┐
          ▼                ▼
 Inventory Optimization  SHAP
          │
          ▼
 Safety Stock / ROP / EOQ
          │
          ├─────────────────────┐
          ▼                     ▼
 Supplier Risk        Disruption Intelligence
                                │
                                ▼
                     Business Impact Signals
                                │
                                ▼
                       Streamlit Control Tower
```

## Key capabilities

### 1. SQL business analytics

PostgreSQL analysis covers the core skills expected from an analyst fresher:

- JOINs and multi-table analysis
- GROUP BY and aggregations
- CASE WHEN business rules
- CTEs
- Window functions
- LAG and period-over-period growth
- RANK / PARTITION BY
- Revenue, demand, stockout and lost-sales KPIs
- Product, store and supplier performance

### 2. Financial analysis

The project now includes a dedicated finance layer using transparent assumptions because procurement cost is not present in the original synthetic dataset.

Core outputs:

- Revenue
- COGS
- Gross Profit
- Gross Margin %
- Inventory Turnover
- Days Inventory Outstanding (DIO)
- Annual inventory holding cost
- Historical lost-sales exposure
- Monthly Budget vs Actual
- Revenue, COGS and Gross Profit variance
- Margin variance
- Promotion effectiveness
- ABC product classification by cumulative revenue contribution
- Supplier-level lost-sales / stockout exposure

**Cost assumption:** unit cost is modeled as 60% of list price. This is a portfolio assumption, not a claim about real procurement costs.

### 3. Demand forecasting

- Store × SKU daily demand aggregation
- Leakage-safe lag and rolling features
- 1-day naive baseline
- 7-day seasonal-naive baseline
- XGBoost forecasting
- Chronological evaluation
- 30-day recursive forecasts
- MAE, RMSE and MAPE comparison

### 4. Inventory optimization

Forecasts are converted into operational inventory signals:

```text
Forecast Demand
      ↓
Lead-Time Demand
      ↓
Safety Stock
      ↓
Reorder Point
      ↓
EOQ
      ↓
Recommended Replenishment
```

### 5. Supplier risk

Supplier profiles use operational measures including:

- on-time delivery
- lead time
- defects
- delays
- ordered vs received quantity
- fill rate

### 6. Unsupervised disruption intelligence

Supplier behavior is segmented and analyzed with clustering, PCA and Isolation Forest methods. The output is a decision-support ranking rather than a calibrated disruption probability.

### 7. Temporal disruption monitoring

`src/temporal_disruption.py` monitors supplier × product behavior against a preceding 14-day operational baseline, covering lead time, fill rate, on-time delivery, defects and stockouts.

### 8. Business impact

The control tower connects operational issues to business consequences such as stockouts, lost-sales exposure, inventory value and replenishment needs.

## Streamlit dashboard

Run:

```bash
streamlit run app.py
```

Dashboard sections include:

1. **Control Tower** — priority actions and replenishment
2. **Disruption Intelligence** — anomalies, temporal signals and explanations
3. **SKU Forecast** — 30-day demand forecasts
4. **Model Benchmark** — MAE / RMSE / MAPE comparison
5. **Supplier Risk** — supplier performance and risk
6. **Business Impact** — stockouts, lost sales and inventory value

## Repository structure

```text
.
├── app.py
├── data/
│   └── retail_sales_data.csv
├── images/
├── sql/
│   └── 01_business_analysis.sql
├── src/
│   ├── generate_dataset.py
│   ├── load_to_postgres.py
│   ├── baseline_forecasting.py
│   ├── sku_level_forecasting.py
│   ├── inventory_optimization.py
│   ├── supplier_risk.py
│   ├── disruption_detection.py
│   ├── temporal_disruption.py
│   ├── create_control_tower.py
│   ├── business_impact.py
│   ├── finance_analysis.py
│   ├── shap_explainability.py
│   ├── validate_outputs.py
│   └── run_pipeline.py
├── .gitignore
├── requirements.txt
└── README.md
```

Generated pipeline CSVs are ignored by Git; only the source retail dataset is retained in `data/` by default.

## Run the complete pipeline

```bash
python src/run_pipeline.py
```

The reproducible pipeline now executes finance analysis after the existing operational analytics:

```text
1. baseline_forecasting.py
2. sku_level_forecasting.py
3. inventory_optimization.py
4. supplier_risk.py
5. temporal_disruption.py
6. disruption_detection.py
7. create_control_tower.py
8. business_impact.py
9. finance_analysis.py
10. shap_explainability.py
11. validate_outputs.py
```

## Data

The included dataset is **synthetic retail supply-chain data** generated for the project. It represents daily observations across:

- 5 stores
- 30 products
- 8 suppliers
- 2023–2024 dates
- 109,650 daily records

Existing fields already support pricing, discounts, promotions, demand, sales, revenue, inventory, stockouts, supplier performance and replenishment analysis.

## Evaluation and limitations

Forecasting uses chronological MAE, RMSE and MAPE comparisons. Disruption detection is unsupervised/heuristic because there is no verified historical disruption ground truth.

Financial outputs are also designed as portfolio analysis: unit cost is an explicit 60%-of-list-price assumption, and budget values are planning assumptions rather than historical company budgets.

Other limitations include synthetic data, simplified inventory formulas, no streaming production architecture, no automated retraining, and human review of recommendations.

## Technology

**Python · Pandas · NumPy · Scikit-learn · XGBoost · SHAP · Streamlit · PostgreSQL · SQL · Matplotlib · Seaborn**

## Resume-ready project description

> **AI-Driven Supply Chain Control Tower** — Built an end-to-end supply-chain analytics platform combining SQL business analysis, financial KPIs, budget-vs-actual variance, inventory optimization, demand forecasting, supplier risk, disruption detection, and an interactive Streamlit decision dashboard.
