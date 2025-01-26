---
title: "Can this query be optimized?"
date: "2025-01-26"
id: "can-this-query-be-optimized"
---

```sql
SELECT
    p.product_name,
    c.category_name,
    AVG(s.sale_price) AS average_sale_price
FROM
    products p
JOIN
    categories c ON p.category_id = c.category_id
JOIN
    sales s ON p.product_id = s.product_id
WHERE
    s.sale_date >= DATE('now', '-3 months')
GROUP BY
    p.product_name, c.category_name
ORDER BY
    average_sale_price DESC;

```
The query, as presented, will perform adequately on small datasets, but its performance will degrade significantly with increasing data volume, particularly in the `sales` table. The primary performance bottleneck lies in the implicit full table scan of the `sales` table, even with the date filter. To optimize it effectively, we must address this using appropriate indexing and, potentially, data pre-aggregation. My experience building and maintaining e-commerce platforms, which often involves dealing with similar data structures, confirms the criticality of this optimization.

The core problem with the existing query is the `WHERE s.sale_date >= DATE('now', '-3 months')` clause. While it restricts the dataset, the database engine still needs to examine every row in the `sales` table to determine which sales fall within the specified timeframe. Without a suitable index on `sale_date`, this becomes a full table scan, which becomes prohibitively costly with millions or billions of entries. Furthermore, the calculation of `AVG(s.sale_price)` requires a full read of the qualifying rows. The `GROUP BY` operation and `ORDER BY` clause add to the workload.

Here's a refined version of the query, incorporating an index and potentially using a pre-aggregated view for further performance gains:

**First Optimization - Utilizing Index on `sale_date`:**

This modification focuses on implementing an index on the `sale_date` column of the `sales` table. This is the most direct and often most significant single optimization one can apply. Before this change, every row of the `sales` table is considered. With the index, only relevant records are accessed.

```sql
CREATE INDEX idx_sales_sale_date ON sales(sale_date);

SELECT
    p.product_name,
    c.category_name,
    AVG(s.sale_price) AS average_sale_price
FROM
    products p
JOIN
    categories c ON p.category_id = c.category_id
JOIN
    sales s ON p.product_id = s.product_id
WHERE
    s.sale_date >= DATE('now', '-3 months')
GROUP BY
    p.product_name, c.category_name
ORDER BY
    average_sale_price DESC;

```

**Commentary:**

The creation of `idx_sales_sale_date` on the `sales(sale_date)` column is paramount for efficiency. The database can now efficiently find sales within the desired 3-month window. Previously, it was required to examine every row of the `sales` table before filtering. This results in a dramatic reduction in read operations especially in large tables. The index should be regularly maintained to reflect ongoing data changes. The remainder of the query remains largely the same, as the major bottleneck was the full scan in `sales`. While this will provide substantial improvement, for very large data sets, we can consider pre-aggregation.

**Second Optimization - Materialized View (If Feasible):**

If the report is run frequently and performance is a paramount concern, generating a pre-aggregated materialized view for the sale data can drastically reduce query execution time. This approach trades increased storage costs for improved query speed. This is generally advantageous where the view is used for multiple queries and the underlying table is large.

```sql
CREATE MATERIALIZED VIEW sales_summary_last_3_months AS
SELECT
    p.product_id,
    p.product_name,
    c.category_name,
    AVG(s.sale_price) AS average_sale_price
FROM
    products p
JOIN
    categories c ON p.category_id = c.category_id
JOIN
    sales s ON p.product_id = s.product_id
WHERE
    s.sale_date >= DATE('now', '-3 months')
GROUP BY
    p.product_id, p.product_name, c.category_name;

-- Query using materialized view
SELECT
    product_name,
    category_name,
    average_sale_price
FROM
    sales_summary_last_3_months
ORDER BY
    average_sale_price DESC;

```

**Commentary:**

Here, a materialized view `sales_summary_last_3_months` is created that aggregates the sales data for the past three months. The view contains product name, category name and the calculated average sale price, aggregated by product and category. This requires that a background job (or scheduler) periodically updates this view. In some database systems, this update may happen automatically upon data modification. With the materialized view established, the subsequent query becomes extremely efficient, as it is now only reading from this pre-calculated dataset, avoiding the costly join and aggregation of the `products`, `categories`, and `sales` tables. This eliminates the primary processing load of the original query. This optimization is ideal for read-heavy reporting systems. The trade off here is increased storage requirements for materialized view and more complex data change management when the underlying data is updated.

**Third Optimization - Utilizing Window Functions for Further Aggregation:**

If we wanted to refine the report to include category-level averages in addition to the current product-level average, we could combine our previous indexing with window functions:

```sql
CREATE INDEX idx_sales_sale_date ON sales(sale_date);

SELECT
    p.product_name,
    c.category_name,
    AVG(s.sale_price) AS average_product_sale_price,
    AVG(AVG(s.sale_price)) OVER (PARTITION BY c.category_name) AS average_category_sale_price
FROM
    products p
JOIN
    categories c ON p.category_id = c.category_id
JOIN
    sales s ON p.product_id = s.product_id
WHERE
    s.sale_date >= DATE('now', '-3 months')
GROUP BY
    p.product_name, c.category_name
ORDER BY
    average_product_sale_price DESC;

```

**Commentary:**

This refined query uses a window function to calculate the average sale price across all products *within the same category*. We maintain the index on `sale_date` as described earlier. The outer `AVG` within the `OVER` clause is calculated using the results of the `AVG(s.sale_price)` which is grouped by `p.product_name` and `c.category_name`. The `PARTITION BY c.category_name`  specifies that the window function operates independently within each category. The final result is a report that not only provides product-level average sale price but also category-level averages, providing broader insights without dramatically increasing query time if the index on `sales.sale_date` is utilized. This version shows the power of window functions for sophisticated aggregations that can be added to existing queries.

**Resource Recommendations:**

To further enhance understanding of database query optimization, I recommend reviewing resources focusing on SQL indexing techniques and query planning. Exploring concepts related to materialized views, partitioning strategies, and window functions will also prove valuable. Database-specific documentation, tutorials on SQL best practices, and the "Use the Index, Luke!" online resource, while dated, provides a deep understanding of index usage. Additionally, any database performance tuning manual, usually specific to database vendor like PostgreSQL, MySQL or SQL Server, are useful. Finally, consider attending local database user group meetings which can be a great resource for specific use cases. Through continuous study and practical application, you will find further refinement of SQL efficiency.
