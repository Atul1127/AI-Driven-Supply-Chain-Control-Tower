/*
===============================================================================
01_business_analysis.sql

Project:
Intelligent Demand & Supply Chain Control Tower

Purpose:
Business and supply-chain analytics using PostgreSQL.

Topics:
- JOIN
- GROUP BY
- CASE
- CTE
- Window Functions
- LAG
- RANK
- Aggregations
===============================================================================
*/


/*
===============================================================================
QUERY 1
Top Products by Revenue

Business Question:
Which products generate the highest revenue?
===============================================================================
*/

SELECT
    p.product_name,
    p.category,
    SUM(s.units_sold) AS total_units_sold,
    ROUND(SUM(s.revenue), 2) AS total_revenue
FROM sales s

JOIN products p
    ON s.product_id = p.product_id

GROUP BY
    p.product_name,
    p.category

ORDER BY
    total_revenue DESC

LIMIT 10;


/*
===============================================================================
QUERY 2
Store Performance

Business Question:
Which stores generate the highest sales and revenue?
===============================================================================
*/

SELECT
    st.store_name,
    SUM(s.units_sold) AS total_units_sold,
    ROUND(SUM(s.revenue), 2) AS total_revenue,
    ROUND(
        SUM(s.revenue)
        / NULLIF(SUM(s.units_sold), 0),
        2
    ) AS revenue_per_unit

FROM sales s

JOIN stores st
    ON s.store_id = st.store_id

GROUP BY
    st.store_name

ORDER BY
    total_revenue DESC;


/*
===============================================================================
QUERY 3
Stockout Analysis

Business Question:
Which products have the highest stockout rate?
===============================================================================
*/

SELECT
    p.product_name,
    p.category,

    COUNT(*) AS total_days,

    SUM(i.stockout) AS stockout_days,

    ROUND(
        100.0 * SUM(i.stockout)
        / COUNT(*),
        2
    ) AS stockout_rate

FROM inventory i

JOIN products p
    ON i.product_id = p.product_id

GROUP BY
    p.product_name,
    p.category

ORDER BY
    stockout_rate DESC;


/*
===============================================================================
QUERY 4
Lost Sales Analysis

Business Question:
Which products are losing the most potential sales because of stockouts?
===============================================================================
*/

SELECT
    p.product_name,
    p.category,

    SUM(s.demand) AS total_demand,

    SUM(s.units_sold) AS total_units_sold,

    SUM(s.lost_sales) AS total_lost_sales,

    ROUND(
        100.0 * SUM(s.lost_sales)
        / NULLIF(SUM(s.demand), 0),
        2
    ) AS lost_sales_rate

FROM sales s

JOIN products p
    ON s.product_id = p.product_id

GROUP BY
    p.product_name,
    p.category

ORDER BY
    total_lost_sales DESC

LIMIT 10;


/*
===============================================================================
QUERY 5
Monthly Demand Growth

Business Question:
How is demand changing month over month?

SQL Concepts:
- CTE
- DATE_TRUNC
- LAG
===============================================================================
*/

WITH monthly_demand AS (

    SELECT
        DATE_TRUNC(
            'month',
            sale_date
        ) AS month,

        SUM(demand) AS total_demand

    FROM sales

    GROUP BY
        DATE_TRUNC(
            'month',
            sale_date
        )
),

monthly_growth AS (

    SELECT
        month,
        total_demand,

        LAG(total_demand)
        OVER (
            ORDER BY month
        ) AS previous_month_demand

    FROM monthly_demand
)

SELECT
    month,
    total_demand,
    previous_month_demand,

    ROUND(
        100.0
        * (total_demand - previous_month_demand)
        / NULLIF(previous_month_demand, 0),
        2
    ) AS growth_percentage

FROM monthly_growth

ORDER BY
    month;


/*
===============================================================================
QUERY 6
Top Products Within Each Category

Business Question:
What are the top 3 products in each category?

SQL Concepts:
- CTE
- RANK()
- PARTITION BY
===============================================================================
*/

WITH product_revenue AS (

    SELECT
        p.category,
        p.product_name,

        SUM(s.revenue) AS total_revenue

    FROM sales s

    JOIN products p
        ON s.product_id = p.product_id

    GROUP BY
        p.category,
        p.product_name
),

ranked_products AS (

    SELECT
        category,
        product_name,
        total_revenue,

        RANK()
        OVER (
            PARTITION BY category
            ORDER BY total_revenue DESC
        ) AS category_rank

    FROM product_revenue
)

SELECT
    category,
    product_name,
    ROUND(total_revenue, 2) AS total_revenue,
    category_rank

FROM ranked_products

WHERE category_rank <= 3

ORDER BY
    category,
    category_rank;


/*
===============================================================================
QUERY 7
Current Inventory Risk

Business Question:
Which store-product combinations need attention?

Business Rules:

CRITICAL:
    Closing stock <= 50% of reorder point

HIGH:
    Closing stock <= reorder point

NORMAL:
    Closing stock > reorder point
===============================================================================
*/

SELECT
    st.store_name,
    p.product_name,
    p.category,

    i.inventory_date,

    i.closing_stock,
    i.reorder_point,

    CASE

        WHEN i.closing_stock
             <= i.reorder_point * 0.50
        THEN 'CRITICAL'

        WHEN i.closing_stock
             <= i.reorder_point
        THEN 'HIGH'

        ELSE 'NORMAL'

    END AS inventory_status

FROM inventory i

JOIN stores st
    ON i.store_id = st.store_id

JOIN products p
    ON i.product_id = p.product_id

WHERE i.inventory_date = (
    SELECT MAX(inventory_date)
    FROM inventory
)

ORDER BY
    CASE

        WHEN i.closing_stock
             <= i.reorder_point * 0.50
        THEN 1

        WHEN i.closing_stock
             <= i.reorder_point
        THEN 2

        ELSE 3

    END;


/*
===============================================================================
QUERY 8
Supplier Performance & Risk

Business Question:
Which suppliers have poor delivery performance?

Risk logic:

HIGH:
    On-time rate < 90%
    OR defect rate > 5%

MEDIUM:
    On-time rate < 95%
    OR defect rate > 3%

LOW:
    Otherwise
===============================================================================
*/

SELECT
    supplier_name,
    lead_time_days,
    on_time_rate,
    defect_rate,

    CASE

        WHEN on_time_rate < 0.90
             OR defect_rate > 0.05
        THEN 'HIGH'

        WHEN on_time_rate < 0.95
             OR defect_rate > 0.03
        THEN 'MEDIUM'

        ELSE 'LOW'

    END AS supplier_risk

FROM suppliers

ORDER BY
    CASE

        WHEN on_time_rate < 0.90
             OR defect_rate > 0.05
        THEN 1

        WHEN on_time_rate < 0.95
             OR defect_rate > 0.03
        THEN 2

        ELSE 3

    END;
