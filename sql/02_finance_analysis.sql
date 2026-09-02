/* Finance & Business SQL Analysis */

-- Q1 Monthly Revenue, COGS & Gross Profit
WITH monthly AS (
SELECT DATE_TRUNC('month',date) month,
SUM(revenue) revenue,
SUM(units_sold*unit_cost) cogs
FROM retail_sales
GROUP BY 1)
SELECT month,revenue,cogs,revenue-cogs gross_profit,
ROUND((revenue-cogs)*100/revenue,2) gross_margin
FROM monthly;

-- Q2 Budget vs Actual Variance
SELECT month,budget_revenue,actual_revenue,
actual_revenue-budget_revenue variance,
ROUND((actual_revenue-budget_revenue)*100/budget_revenue,2) variance_pct
FROM budget_actual;

-- Q3 Inventory Turnover & DIO
SELECT store,
SUM(cogs)/AVG(inventory_value) inventory_turnover,
365/(SUM(cogs)/AVG(inventory_value)) dio
FROM finance_metrics GROUP BY store;

-- Q4 ABC Product Analysis
WITH rev AS (
SELECT product,SUM(revenue) revenue
FROM retail_sales GROUP BY product),
r AS (
SELECT product,revenue,
SUM(revenue) OVER(ORDER BY revenue DESC) cum_rev,
SUM(revenue) OVER() total_rev
FROM rev)
SELECT product,revenue,
ROUND(cum_rev*100/total_rev,2) cumulative_pct,
CASE WHEN cum_rev<=0.8*total_rev THEN 'A'
WHEN cum_rev<=0.95*total_rev THEN 'B'
ELSE 'C' END abc_class
FROM r;

-- Q5 Promotion Effectiveness
SELECT promo_event,
AVG(units_sold) avg_units,
AVG(revenue) avg_revenue,
ROUND(SUM(units_sold)*100/SUM(demand),2) conversion_rate
FROM retail_sales GROUP BY promo_event;

-- Q6 Supplier Financial Impact
SELECT supplier,
SUM(lost_sales) lost_units,
SUM(lost_sales*sell_price) lost_sales_value,
SUM(stockout) stockout_days
FROM retail_sales GROUP BY supplier
ORDER BY lost_sales_value DESC;