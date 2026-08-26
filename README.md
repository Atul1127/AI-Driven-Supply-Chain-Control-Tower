# AI-Driven Supply Chain Control Tower

### SKU-level Demand Forecasting • Inventory Optimization • Supplier Risk • Disruption Detection • Control Tower

An end-to-end retail supply-chain decision system that connects **store/SKU demand forecasting, inventory optimization, supplier risk, unsupervised disruption detection, temporal monitoring, explainability, and operational priorities**.

> **Business question:** What will each store/SKU need, which inventory positions require action, and are any suppliers or supplier/SKU relationships showing early signs of disruption?

## Architecture

```text
Synthetic Retail Data
        │
        ├── PostgreSQL + SQL Analytics
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
                ▼
        Inventory Optimization
                │
        ┌───────┴────────┐
        ▼                ▼
 Supplier Risk    Disruption Intelligence
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          K-Means   Hierarchical   DBSCAN
             └───────────┼───────────┘
                         ▼
                        PCA
                         │
                         ▼
                 Isolation Forest
                         │
                         ▼
                Temporal Monitoring
                         │
                         ▼
                  Impact / Actions
                         │
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
8. **K-Means, hierarchical clustering, and DBSCAN** for supplier operating-regime discovery.
9. **PCA** for lower-dimensional supply-chain behavior visualization.
10. **Isolation Forest** for multivariate supplier anomaly detection.
11. A transparent **Disruption Score** and LOW/MEDIUM/HIGH/CRITICAL alert levels.
12. **Temporal supplier × SKU monitoring** against rolling 14-day operational baselines.
13. Disruption drill-down with anomaly drivers, temporal signal, PCA map, and response guidance.
14. Historical/current business impact metrics and operational control-tower priorities.
15. SHAP feature importance aligned with the same SKU-level XGBoost feature set.
16. Streamlit dashboard for operational filtering, forecasting, model benchmarking, supplier risk, disruption intelligence, and business impact.
17. Deterministic `src/run_pipeline.py` entry point with post-run output validation.

## Disruption Detection

The disruption module operates at supplier level using operational features such as:

```text
Average lead time
Lead-time variability
On-time delivery
Defect rate
Fill rate
Supplier delay rate
Demand level / volatility
Stockout rate
```

Three clustering families are compared:

- **K-Means** — compact operating regimes
- **Hierarchical clustering** — interpretable supplier similarity structure
- **DBSCAN** — density-based regimes and noise points

Clustering quality is evaluated with:

- Silhouette score
- Davies-Bouldin index
- Calinski-Harabasz score

Isolation Forest then detects suppliers with unusual multivariate behavior.

### Temporal disruption detection

`src/temporal_disruption.py` builds daily supplier × product signals and compares current behavior with a preceding 14-day rolling baseline. It monitors:

- lead-time deterioration
- fill-rate deterioration
- on-time deterioration
- defect-rate deterioration
- stockout deterioration

Signals are classified as:

```text
NORMAL → EARLY_WARNING → EMERGING → CRITICAL
```

This distinguishes a supplier that is generally unusual from a supplier whose behavior is **actively deteriorating**.

### Operational response

The dashboard combines disruption severity with operational context and provides response guidance such as:

- activate alternate sourcing
- expedite open orders
- protect high-demand inventory
- increase monitoring
- review safety-stock coverage

These are decision-support recommendations, not autonomous procurement decisions.

## Streamlit Control Tower

Run:

```bash
streamlit run app.py
```

Dashboard sections:

- 🚨 Control Tower — filters and prioritized actions
- ⚠️ Disruption Intelligence — anomaly ranking, temporal monitoring, supplier drill-down, PCA, clustering comparison
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

Generated CSV outputs are intentionally **not committed**. They are recreated by the pipeline.

## Installation

```bash
git clone https://github.com/Atul1127/AI-Driven-Supply-Chain-Control-Tower.git
cd AI-Driven-Supply-Chain-Control-Tower
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
5. temporal_disruption.py
6. disruption_detection.py
7. create_control_tower.py
8. business_impact.py
9. shap_explainability.py
10. validate_outputs.py
```

The final validation checks required output schemas, non-empty artifacts, Store × SKU forecast coverage, forecasting benchmark quality, and disruption-score/anomaly output validity.

Then launch the dashboard:

```bash
streamlit run app.py
```

## Forecasting

The operational forecasting path evaluates baselines at the same **store × SKU level** and uses the same final chronological test window as XGBoost.

Metrics:

- MAE
- RMSE
- MAPE

The repository does **not** claim XGBoost is best without comparing it against the naive baselines.

## Inventory Optimization

```text
Forecast Daily Demand
        ↓
Lead-Time Demand
        ↓
Safety Stock
        ↓
Reorder Point
        ↓
EOQ
        ↓
Operational Cap
        ↓
Recommended Replenishment
```

## Business Impact

`src/business_impact.py` reports operational signals rather than incorrectly labeling low inventory coverage as stockout risk:

- current stockout pairs
- historical stockout days
- historical lost-sales value
- current inventory value
- recommended replenishment
- critical/reorder/normal inventory counts
- low-coverage pairs

> **Important:** historical lost-sales value is simulated exposure. It is not money saved by the system.

## Important limitations

- The dataset is synthetic and intended for portfolio/learning use.
- Disruption labels are unsupervised/heuristic signals, not verified historical disruption events.
- The temporal detector uses rolling operational baselines and should be calibrated for production alerting.
- Inventory formulas are simplified decision-support models, not production procurement policies.
- Supplier risk and disruption thresholds should be calibrated for a real organization.
- Response recommendations are decision support and require human review.
- Forecast metrics should be interpreted alongside naive baselines rather than in isolation.

## Technology

**Python · Pandas · NumPy · Scikit-learn · XGBoost · Statsmodels · SHAP · Streamlit · PostgreSQL · SQL · Matplotlib · Seaborn**
