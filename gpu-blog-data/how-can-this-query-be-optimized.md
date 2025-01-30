---
title: "How can this query be optimized?"
date: "2025-01-30"
id: "how-can-this-query-be-optimized"
---
Here's the query we're considering:

```sql
SELECT p.product_name,
       c.category_name,
       COUNT(o.order_id) AS total_orders
FROM products p
JOIN categories c ON p.category_id = c.category_id
LEFT JOIN order_items oi ON p.product_id = oi.product_id
LEFT JOIN orders o ON oi.order_id = o.order_id
WHERE o.order_date BETWEEN '2023-01-01' AND '2023-12-31'
GROUP BY p.product_name, c.category_name
ORDER BY total_orders DESC;

```

The core inefficiency in this query lies in the unnecessary joining of the `orders` table when the aggregation only requires information from the `order_items` table and date filtering. The join pattern creates a cartesian product, potentially processing millions of rows, even for products with no orders in the specified date range, impacting performance and scalability. From prior experience optimizing similar e-commerce database schemas, the initial assessment often points to the join strategy as the primary bottleneck in these aggregation type queries.

The fundamental problem is the `LEFT JOIN orders o` and the associated `WHERE o.order_date BETWEEN '2023-01-01' AND '2023-12-31'` clause. Because it's a `LEFT JOIN`, products without associated orders will still appear in the result set, but their order count will become NULL because of the join. The subsequent `WHERE` clause then filters these NULL values, as NULL can't satisfy a between condition. Essentially, the join to `orders` becomes a method to filter, and this forces the database to perform operations related to the `orders` table, even for products with no sales in the date range. Also, if a product has multiple orders within the date range, the date constraint filters them down for each `order_items` record, and counts all the entries, instead of a single order entry which is unnecessary. The optimal approach avoids involving the `orders` table directly, and aggregates counts from `order_items`.

**Optimization Strategy:**

The solution involves restructuring the query to directly aggregate order items within the date range without joining the `orders` table.  We can achieve this by using a subquery or CTE to pre-aggregate order item counts within the date range, then join this result with the `products` and `categories` tables.  This effectively pre-filters order items by date before joining and aggregating further up the chain, resulting in significant performance gains.

**Code Example 1: Using a Subquery**

```sql
SELECT p.product_name,
       c.category_name,
       COALESCE(oi_agg.total_orders, 0) AS total_orders
FROM products p
JOIN categories c ON p.category_id = c.category_id
LEFT JOIN (
    SELECT oi.product_id,
           COUNT(oi.order_id) AS total_orders
    FROM order_items oi
    WHERE oi.order_id IN (SELECT order_id from orders where order_date BETWEEN '2023-01-01' AND '2023-12-31')
    GROUP BY oi.product_id
) AS oi_agg ON p.product_id = oi_agg.product_id
ORDER BY total_orders DESC;
```

In this first optimization, the subquery `oi_agg` calculates the total orders per product by filtering the order item entries based on order dates. Note how we added `COALESCE`, and this handles product that had no orders by converting `NULL` values to 0, meaning the product is included in the list with 0 orders. This approach minimizes unnecessary joins and pre-aggregates the order counts, ensuring performance gains, compared to the original query. This version filters the `order_items` records first, then groups them by `product_id` before joining the `products` table and `categories` table.

**Code Example 2: Using a Common Table Expression (CTE)**

```sql
WITH OrderItemsAgg AS (
    SELECT oi.product_id,
           COUNT(oi.order_id) AS total_orders
    FROM order_items oi
    WHERE oi.order_id IN (SELECT order_id from orders where order_date BETWEEN '2023-01-01' AND '2023-12-31')
    GROUP BY oi.product_id
)
SELECT p.product_name,
       c.category_name,
       COALESCE(oa.total_orders, 0) AS total_orders
FROM products p
JOIN categories c ON p.category_id = c.category_id
LEFT JOIN OrderItemsAgg oa ON p.product_id = oa.product_id
ORDER BY total_orders DESC;

```

This second example implements the same logic as the first example but utilizes a CTE (`OrderItemsAgg`) instead of a subquery. CTEs often improve readability, especially for more complex queries, and in certain database systems can lead to slightly better query plan generation due to the optimizer's ability to better reason about the temporary named result set. Again, `COALESCE` ensures products with no orders are included with a count of 0. This approach is functionally identical to the first example but offers a more organized structure.

**Code Example 3: Leveraging an Indexed Approach for Filtering**

```sql
CREATE INDEX idx_orders_date ON orders (order_date);

WITH OrderItemsAgg AS (
    SELECT oi.product_id,
           COUNT(oi.order_id) AS total_orders
    FROM order_items oi
    WHERE EXISTS (
       SELECT 1
       FROM orders o
       WHERE oi.order_id = o.order_id
       AND o.order_date BETWEEN '2023-01-01' AND '2023-12-31'
   )
    GROUP BY oi.product_id
)
SELECT p.product_name,
       c.category_name,
       COALESCE(oa.total_orders, 0) AS total_orders
FROM products p
JOIN categories c ON p.category_id = c.category_id
LEFT JOIN OrderItemsAgg oa ON p.product_id = oa.product_id
ORDER BY total_orders DESC;

```
This third optimized query includes an index creation command, to speed up the filtering of the orders. It also replaces the `IN` clause, with `EXISTS` clause. The use of `EXISTS` with a correlated subquery allows the database to more efficiently filter order items based on the presence of matching order entries within the specified date range. The `idx_orders_date` index on the `order_date` column significantly accelerates the filtering of orders based on the date range. This can be advantageous in large datasets and reduce the time needed to find qualifying rows. The query still works the same as the other examples, but leverages indexes and correlated subqueries to further optimize performance.

**Resource Recommendations**

To deepen your understanding of database optimization, I recommend exploring resources related to query execution plans, indexing strategies, and database-specific performance tuning tools. Textbooks covering relational database design principles and SQL optimization practices, from various authors like C.J. Date, or the documentation for specific database systems (e.g., PostgreSQL, MySQL, SQL Server), can be beneficial for learning to analyze and optimize different queries. Additionally, look for courses focused on advanced SQL techniques and database performance, that teach and train on advanced optimization strategies.

In my professional experience with data management and database systems, consistently analyzing query execution plans and adapting the query accordingly, especially in response to changing data volumes, proves to be the most effective method of ensuring good database performance. These examples, and a deeper understanding of the concepts, will ensure the database can scale with increased data size and user requests, which helps ensure the overall success of the applications.
