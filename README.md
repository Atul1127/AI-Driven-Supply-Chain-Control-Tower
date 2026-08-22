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
