/* Finance & Business SQL Analysis
   Assumption: procurement cost = 60% of product/list price.
   This is a portfolio assumption because procurement cost is not stored in the source schema.
*/

-- Q1 Monthly Revenue, COGS, Gross Profit & Gross Margin
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', sale_date) AS month,
        SUM(revenue) AS revenue,
        SUM(units_sold * unit_price * 0.60) AS cogs
    FROM sales
    GROUP BY 1
)
SELECT
    month,
    revenue,
    cogs,
    revenue - cogs AS gross_profit,
    ROUND(100.0 * (revenue - cogs) / NULLIF(revenue, 0), 2) AS gross_margin_pct
FROM monthly
ORDER BY month;

-- Q2 Monthly Budget vs Actual Variance
-- Planning assumption: budget = 105% of prior-month revenue and 103% of prior-month COGS.
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', sale_date) AS month,
        SUM(revenue) AS actual_revenue,
        SUM(units_sold * unit_price * 0.60) AS actual_cogs
    FROM sales
    GROUP BY 1
), budgets AS (
    SELECT
        month,
        actual_revenue,
        actual_cogs,
        COALESCE(LAG(actual_revenue) OVER (ORDER BY month) * 1.05, actual_revenue) AS budget_revenue,
        COALESCE(LAG(actual_cogs) OVER (ORDER BY month) * 1.03, actual_cogs) AS budget_cogs
    FROM monthly
)
SELECT
    month,
    budget_revenue,
    actual_revenue,
    actual_revenue - budget_revenue AS revenue_variance,
    ROUND(100.0 * (actual_revenue - budget_revenue) / NULLIF(budget_revenue, 0), 2) AS revenue_variance_pct,
    budget_cogs,
    actual_cogs,
    actual_cogs - budget_cogs AS cogs_variance
FROM budgets
ORDER BY month;

-- Q3 Inventory Turnover & DIO by Store
WITH store_finance AS (
    SELECT
        s.store_id,
        s.store_name AS store,
        SUM(sa.units_sold * sa.unit_price * 0.60) AS cogs,
        AVG(i.closing_stock * p.unit_price * 0.60) AS average_inventory_value
    FROM stores s
    JOIN sales sa ON sa.store_id = s.store_id
    JOIN inventory i
        ON i.store_id = sa.store_id
       AND i.product_id = sa.product_id
       AND i.inventory_date = sa.sale_date
    JOIN products p ON p.product_id = sa.product_id
    GROUP BY s.store_id, s.store_name
)
SELECT
    store,
    cogs,
    average_inventory_value,
    ROUND(cogs / NULLIF(average_inventory_value, 0), 2) AS inventory_turnover,
    ROUND(365.0 / NULLIF(cogs / NULLIF(average_inventory_value, 0), 0), 2) AS dio
FROM store_finance
ORDER BY store;

-- Q4 ABC Product Analysis by Cumulative Revenue Contribution
WITH product_revenue AS (
    SELECT
        p.product_id,
        p.product_name AS product,
        p.category,
        SUM(s.revenue) AS revenue
    FROM sales s
    JOIN products p ON p.product_id = s.product_id
    GROUP BY p.product_id, p.product_name, p.category
), ranked AS (
    SELECT
        *,
        SUM(revenue) OVER (ORDER BY revenue DESC, product) AS cumulative_revenue,
        SUM(revenue) OVER () AS total_revenue
    FROM product_revenue
)
SELECT
    product,
    category,
    revenue,
    ROUND(100.0 * revenue / NULLIF(total_revenue, 0), 2) AS revenue_share_pct,
    ROUND(100.0 * cumulative_revenue / NULLIF(total_revenue, 0), 2) AS cumulative_share_pct,
    CASE
        WHEN cumulative_revenue <= 0.80 * total_revenue THEN 'A'
        WHEN cumulative_revenue <= 0.95 * total_revenue THEN 'B'
        ELSE 'C'
    END AS abc_class
FROM ranked
ORDER BY revenue DESC;

-- Q5 Promotion Effectiveness
SELECT
    CASE WHEN promo_event = 1 THEN 'Promotion' ELSE 'Non-promotion' END AS promo_label,
    COUNT(*) AS days,
    SUM(units_sold) AS units_sold,
    SUM(revenue) AS revenue,
    ROUND(AVG(units_sold), 2) AS average_daily_units,
    ROUND(AVG(revenue), 2) AS average_daily_revenue,
    ROUND(100.0 * SUM(units_sold) / NULLIF(SUM(demand), 0), 2) AS conversion_pct
FROM sales
GROUP BY promo_event
ORDER BY promo_event;

-- Q6 Supplier Financial Impact
SELECT
    sup.supplier_name AS supplier,
    SUM(s.lost_sales) AS lost_sales_units,
    ROUND(SUM(s.lost_sales * s.sell_price), 2) AS lost_sales_value,
    SUM(i.stockout) AS stockout_days,
    ROUND(AVG(sup.lead_time_days), 2) AS average_lead_time_days,
    ROUND(AVG(sup.on_time_rate), 2) AS on_time_rate,
    ROUND(AVG(sup.defect_rate), 2) AS defect_rate
FROM sales s
JOIN products p ON p.product_id = s.product_id
JOIN suppliers sup ON sup.supplier_id = p.supplier_id
JOIN inventory i
    ON i.store_id = s.store_id
   AND i.product_id = s.product_id
   AND i.inventory_date = s.sale_date
GROUP BY sup.supplier_id, sup.supplier_name
ORDER BY lost_sales_value DESC;
