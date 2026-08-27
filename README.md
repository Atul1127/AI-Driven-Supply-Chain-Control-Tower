# AI-Driven Supply Chain Control Tower

**Demand Forecasting · Inventory Optimization · Supplier Risk · Disruption Detection · Decision Dashboard**

An end-to-end supply-chain analytics and machine-learning project that turns store/SKU demand and supplier operations into **forecasts, replenishment decisions, risk signals, disruption alerts, and an interactive control tower**.

> **Portfolio note:** the dataset is synthetic. The project is designed to demonstrate an end-to-end ML/analytics workflow and does not claim production deployment or real-world disruption ground truth.

## What the system does

```text
Retail Supply-Chain Data
          │
          ├── SQL / EDA
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
                   ┌────────────┼────────────┐
                   ▼            ▼            ▼
                K-Means   Hierarchical    DBSCAN
                   └────────────┼────────────┘
                                ▼
                               PCA
                                │
                                ▼
                       Isolation Forest
                                │
                                ▼
                    Temporal Supplier × SKU
                         Monitoring
                                │
                                ▼
                     Disruption Prioritization
                                │
                                ▼
                      Business Impact Signals
                                │
                                ▼
                       Streamlit Control Tower
```

## Key capabilities

### 1. Demand forecasting

- Store × SKU daily demand aggregation
- Leakage-safe lag and rolling features
- 1-day naive baseline
- 7-day seasonal-naive baseline
- XGBoost forecasting
- Chronological evaluation
- 30-day recursive forecasts
- MAE, RMSE and MAPE comparison

### 2. Inventory optimization

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

### 3. Supplier risk

Supplier profiles use operational measures including:

- on-time delivery
- lead time
- defects
- delays
- ordered vs received quantity
- fill rate

### 4. Unsupervised disruption intelligence

Supplier behavior is segmented with three clustering approaches:

| Method | Purpose |
|---|---|
| **K-Means** | Identify compact supplier operating regimes |
| **Hierarchical clustering** | Explore supplier similarity structure |
| **DBSCAN** | Find density-based groups and unusual/noise points |
| **PCA** | Visualize high-dimensional supplier behavior |
| **Isolation Forest** | Detect unusual multivariate supplier behavior |

Clustering configurations are compared using:

- Silhouette score
- Davies-Bouldin index
- Calinski-Harabasz score

### 5. Temporal disruption monitoring

`src/temporal_disruption.py` monitors **supplier × product** behavior against a preceding 14-day operational baseline.

Signals include:

- lead-time deterioration
- fill-rate deterioration
- on-time deterioration
- defect-rate deterioration
- stockout deterioration

The operational signal is classified as:

```text
NORMAL → EARLY_WARNING → EMERGING → CRITICAL
```

This complements the cross-sectional supplier anomaly model: a supplier can be unusual compared with peers, or it can be actively deteriorating compared with its own recent behavior.

### 6. Disruption prioritization

The project combines multivariate anomaly severity with operational KPI deviations into a transparent **Disruption Priority Score**.

The score is a decision-support ranking, **not a calibrated probability of disruption**.

### 7. Operational explanations

The dashboard highlights potential drivers such as:

- elevated lead time
- below-median fill rate
- below-median on-time delivery
- above-median defect rate
- unusual multivariate behavior

Recommendations are deliberately presented as **decision support requiring human review**, not autonomous procurement decisions.

## Streamlit dashboard

Run:

```bash
streamlit run app.py
```

### 🚨 Control Tower

- Store/SKU priority actions
- Inventory status
- Replenishment recommendations
- Operational filtering

### ⚠️ Disruption Intelligence

- Critical/high disruption counts
- Anomaly ranking
- Supplier drill-down
- Temporal disruption timeline
- Supplier/SKU selection
- PCA behavior map
- Clustering model comparison
- Operational risk explanations
- Response guidance

### 📈 SKU Forecast

- Store/SKU selection
- 30-day forecast visualization
- Forecast totals and daily averages

### 📊 Model Benchmark

- MAE / RMSE / MAPE comparison
- Baseline vs XGBoost performance

### 🏭 Supplier Risk

- Risk distribution
- Supplier-level risk table

### 💰 Business Impact

- Current stockout pairs
- Historical stockout days
- Historical lost-sales exposure
- Current inventory value
- Replenishment signals

> Historical lost-sales is simulated exposure from the synthetic dataset; it is **not savings generated by the system**.

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
│   ├── eda.py
│   ├── time_series_analysis.py
│   ├── statistical_forecasting.py
│   ├── xgboost_forecasting.py
│   ├── baseline_forecasting.py
│   ├── sku_level_forecasting.py
│   ├── inventory_optimization.py
│   ├── supplier_risk.py
│   ├── disruption_detection.py
│   ├── temporal_disruption.py
│   ├── create_control_tower.py
│   ├── business_impact.py
│   ├── shap_explainability.py
│   ├── validate_outputs.py
│   └── run_pipeline.py
├── .gitignore
├── requirements.txt
└── README.md
```

Generated pipeline CSVs are ignored by Git; only the source retail dataset is retained in `data/` by default.

## Installation

```bash
git clone https://github.com/Atul1127/AI-Driven-Supply-Chain-Control-Tower.git
cd AI-Driven-Supply-Chain-Control-Tower
python -m venv .venv
```

### Windows Git Bash

```bash
source .venv/Scripts/activate
```

### macOS / Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the complete pipeline

```bash
python src/run_pipeline.py
```

The reproducible pipeline executes:

```text
1. baseline_forecasting.py
2. sku_level_forecasting.py
3. inventory_optimization.py
4. supplier_risk.py
5. temporal_disruption.py
6. disruption_detection.py
7. create_control_tower.py
8. business_impact.py
9. shap_explainability.py
10. validate_outputs.py
```

Validation checks output schemas, non-empty artifacts, forecast coverage, baseline comparison, and disruption output ranges.

Then launch:

```bash
streamlit run app.py
```

## Data

The included dataset is **synthetic retail supply-chain data** generated for the project. It represents daily observations across:

- 5 stores
- 30 products
- 8 suppliers
- 2023–2024 dates
- 109,650 daily records

The synthetic design allows the project to demonstrate supplier, inventory, demand, and disruption workflows without exposing private business data.

## Evaluation

### Forecasting

Models are compared on a common chronological test window using:

- MAE
- RMSE
- MAPE

XGBoost is not assumed to be the best model; it is compared against simple baselines.

### Clustering

Cluster structure is assessed with:

- Silhouette score — higher is better
- Davies-Bouldin index — lower is better
- Calinski-Harabasz score — higher is better

### Disruption detection

The disruption detector is **unsupervised/heuristic** and does not have verified historical disruption labels. Therefore the project reports anomaly scores, cluster structure, and operational deviations rather than claiming supervised precision/recall against real disruption events.

## Limitations

- Synthetic data rather than confidential production data
- No verified historical disruption ground truth
- Disruption priority thresholds are decision-support heuristics
- Temporal monitoring uses a configurable rolling baseline rather than a production-calibrated change-point model
- Inventory formulas are simplified operational models
- No streaming/event-driven production architecture
- No automated model retraining or drift monitoring
- Dashboard recommendations require human review

These limitations are intentional and are documented rather than hidden.

## Technology

**Python · Pandas · NumPy · Scikit-learn · XGBoost · Statsmodels · SHAP · Streamlit · PostgreSQL · SQL · Matplotlib · Seaborn**

## Resume-ready project description

> **AI-Driven Supply Chain Control Tower** — Built an end-to-end supply-chain ML system combining leakage-safe XGBoost demand forecasting, inventory optimization, supplier risk scoring, K-Means/Hierarchical/DBSCAN segmentation, PCA, Isolation Forest anomaly detection, and temporal supplier-SKU monitoring; integrated outputs into a Streamlit control tower for disruption prioritization and operational decision support.
