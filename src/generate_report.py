"""
generate_report.py
------------------
Creates a clean HTML business report with embedded charts.
Open outputs/reports/retail_report.html in any browser.
"""

import pandas as pd
import os, base64, json
from datetime import datetime

BASE      = os.path.dirname(__file__)
TABLE_DIR = os.path.join(BASE, "..", "outputs", "tables")
CHART_DIR = os.path.join(BASE, "..", "outputs", "charts")
REPORT_DIR= os.path.join(BASE, "..", "reports")
os.makedirs(REPORT_DIR, exist_ok=True)


def img_b64(filename):
    path = os.path.join(CHART_DIR, filename)
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def load_tables():
    tables = {}
    for name in ["model_metrics", "reorder_alerts", "inventory_optimization", "30day_forecast"]:
        p = os.path.join(TABLE_DIR, f"{name}.csv")
        if os.path.exists(p):
            tables[name] = pd.read_csv(p)
    return tables


def df_to_html(df, max_rows=20):
    df2 = df.head(max_rows).copy()
    return df2.to_html(classes="data-table", index=False, border=0)


def build_report():
    tables = load_tables()
    metrics = tables.get("model_metrics", pd.DataFrame())
    alerts  = tables.get("reorder_alerts", pd.DataFrame())
    inv     = tables.get("inventory_optimization", pd.DataFrame())

    mae  = f"{metrics['MAE'].iloc[0]:.2f}"  if len(metrics) else "N/A"
    rmse = f"{metrics['RMSE'].iloc[0]:.2f}" if len(metrics) else "N/A"
    r2   = f"{metrics['R2'].iloc[0]:.4f}"   if len(metrics) else "N/A"
    mape = f"{metrics['MAPE'].iloc[0]:.2f}%" if len(metrics) else "N/A"

    total_products = len(inv) if len(inv) else 0
    critical_count = len(inv[inv["alert"].str.contains("CRITICAL|REORDER", na=False)]) if len(inv) else 0
    total_order_val= f"₹{inv['order_value'].sum():,.0f}" if len(inv) else "N/A"

    # Load charts
    charts = {
        "monthly":    img_b64("01_monthly_revenue.png"),
        "category":   img_b64("02_category_revenue.png"),
        "store":      img_b64("03_store_comparison.png"),
        "heatmap":    img_b64("04_seasonal_heatmap.png"),
        "top_prod":   img_b64("05_top_products.png"),
        "stockout":   img_b64("06_stockout_rate.png"),
        "rev_dist":   img_b64("07_revenue_distribution.png"),
        "corr":       img_b64("08_correlation_heatmap.png"),
        "avp":        img_b64("09_actual_vs_predicted.png"),
        "fi":         img_b64("10_feature_importance.png"),
        "forecast":   img_b64("11_30day_forecast.png"),
        "stock":      img_b64("12_stock_health.png"),
        "reorder":    img_b64("13_reorder_by_category.png"),
        "safety":     img_b64("14_stock_vs_safety.png"),
    }

    alerts_html = df_to_html(alerts[["store","category","product","current_stock",
                                     "reorder_point","recommended_order","days_of_stock","alert"]]) \
                  if len(alerts) else "<p>No alerts.</p>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Retail Sales Forecasting & Inventory Optimization – Report</title>
<style>
  :root {{
    --bg:#0f1117; --surface:#1a1d2e; --card:#212438;
    --accent:#4361ee; --accent2:#7209b7; --green:#2dc653;
    --orange:#f4a261; --red:#e63946; --text:#e2e8f0; --muted:#94a3b8;
    --border:rgba(255,255,255,0.08);
  }}
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;line-height:1.6}}
  header{{background:linear-gradient(135deg,var(--accent),var(--accent2));padding:40px;text-align:center}}
  header h1{{font-size:2.2rem;font-weight:800;letter-spacing:-1px}}
  header p{{color:rgba(255,255,255,0.8);margin-top:8px}}
  .container{{max-width:1300px;margin:0 auto;padding:30px 20px}}
  section{{margin-bottom:50px}}
  h2{{font-size:1.5rem;font-weight:700;margin-bottom:20px;padding-bottom:10px;
      border-bottom:2px solid var(--accent);color:#fff}}
  h3{{font-size:1.1rem;font-weight:600;color:var(--muted);margin-bottom:12px}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:16px;margin-bottom:30px}}
  .kpi{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:20px;text-align:center}}
  .kpi .val{{font-size:2rem;font-weight:800;color:var(--accent)}}
  .kpi .label{{font-size:.85rem;color:var(--muted);margin-top:4px}}
  .chart-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(580px,1fr));gap:20px}}
  .chart-box{{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:16px}}
  .chart-box img{{width:100%;border-radius:8px}}
  .chart-box h3{{margin-bottom:10px}}
  .data-table{{width:100%;border-collapse:collapse;font-size:.85rem}}
  .data-table th{{background:var(--accent);color:#fff;padding:10px 12px;text-align:left}}
  .data-table td{{padding:9px 12px;border-bottom:1px solid var(--border)}}
  .data-table tr:nth-child(even){{background:var(--surface)}}
  .table-wrap{{overflow-x:auto;background:var(--card);border-radius:12px;padding:16px;border:1px solid var(--border)}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.75rem;font-weight:600}}
  footer{{text-align:center;padding:30px;color:var(--muted);font-size:.85rem;
          border-top:1px solid var(--border);margin-top:40px}}
</style>
</head>
<body>
<header>
  <h1>🛍️ Retail Sales Forecasting & Inventory Optimization</h1>
  <p>Business Intelligence Report &nbsp;·&nbsp; Generated: {datetime.now().strftime("%d %B %Y, %H:%M")}</p>
</header>

<div class="container">

<!-- KPIs -->
<section>
  <h2>📊 Key Performance Indicators</h2>
  <div class="kpi-grid">
    <div class="kpi"><div class="val">{mae}</div><div class="label">Forecast MAE (units)</div></div>
    <div class="kpi"><div class="val">{r2}</div><div class="label">Model R² Score</div></div>
    <div class="kpi"><div class="val">{mape}</div><div class="label">MAPE</div></div>
    <div class="kpi"><div class="val">{total_products}</div><div class="label">Products Tracked</div></div>
    <div class="kpi" style="--accent:var(--red)"><div class="val" style="color:var(--red)">{critical_count}</div><div class="label">Need Reordering</div></div>
    <div class="kpi" style="--accent:var(--orange)"><div class="val" style="color:var(--orange)">{total_order_val}</div><div class="label">Total Order Value</div></div>
  </div>
</section>

<!-- EDA Charts -->
<section>
  <h2>📈 Sales & Revenue Analysis</h2>
  <div class="chart-grid">
    <div class="chart-box"><h3>Monthly Revenue Trend</h3><img src="{charts['monthly']}" alt="monthly revenue"></div>
    <div class="chart-box"><h3>Revenue by Category</h3><img src="{charts['category']}" alt="category revenue"></div>
    <div class="chart-box"><h3>Store Comparison</h3><img src="{charts['store']}" alt="store comparison"></div>
    <div class="chart-box"><h3>Seasonal Heatmap</h3><img src="{charts['heatmap']}" alt="heatmap"></div>
    <div class="chart-box"><h3>Top 15 Products by Revenue</h3><img src="{charts['top_prod']}" alt="top products"></div>
    <div class="chart-box"><h3>Stockout Rate by Category</h3><img src="{charts['stockout']}" alt="stockout rate"></div>
    <div class="chart-box"><h3>Revenue Distribution</h3><img src="{charts['rev_dist']}" alt="revenue dist"></div>
    <div class="chart-box"><h3>Feature Correlation Heatmap</h3><img src="{charts['corr']}" alt="correlation"></div>
  </div>
</section>

<!-- Forecasting -->
<section>
  <h2>🔮 Sales Forecasting</h2>
  <div class="chart-grid">
    <div class="chart-box"><h3>Actual vs Predicted (Test Period)</h3><img src="{charts['avp']}" alt="avp"></div>
    <div class="chart-box"><h3>Feature Importance (Random Forest)</h3><img src="{charts['fi']}" alt="feature importance"></div>
    <div class="chart-box" style="grid-column:1/-1"><h3>30-Day Sales Forecast</h3><img src="{charts['forecast']}" alt="forecast"></div>
  </div>
</section>

<!-- Inventory -->
<section>
  <h2>📦 Inventory Optimization</h2>
  <div class="chart-grid">
    <div class="chart-box"><h3>Stock Health – Critical Items</h3><img src="{charts['stock']}" alt="stock health"></div>
    <div class="chart-box"><h3>Reorder Value by Category</h3><img src="{charts['reorder']}" alt="reorder"></div>
    <div class="chart-box" style="grid-column:1/-1"><h3>Current Stock vs Safety Stock vs Reorder Point</h3><img src="{charts['safety']}" alt="safety"></div>
  </div>
</section>

<!-- Reorder Alerts Table -->
<section>
  <h2>🚨 Reorder Alerts</h2>
  <div class="table-wrap">
    {alerts_html}
  </div>
</section>

<!-- Model Metrics -->
<section>
  <h2>🤖 Model Performance</h2>
  <div class="table-wrap">
    {df_to_html(metrics) if len(metrics) else "<p>N/A</p>"}
  </div>
</section>

</div>
<footer>
  Retail Sales Forecasting & Inventory Optimization System &nbsp;·&nbsp;
  Built with Python · scikit-learn · Pandas · Matplotlib &nbsp;·&nbsp;
  Student Portfolio Project
</footer>
</body>
</html>"""

    out = os.path.join(REPORT_DIR, "retail_report.html")
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ Report saved → {out}")
    print("   Open this file in your browser to view the full report.")


if __name__ == "__main__":
    build_report()
