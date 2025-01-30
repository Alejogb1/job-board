---
title: "Why are shared buffer reads high in Postgres explain plan for slow queries retrieving a small number of rows?"
date: "2025-01-30"
id: "why-are-shared-buffer-reads-high-in-postgres"
---
High shared buffer reads in PostgreSQL's explain plan, even for queries returning a small number of rows, often stem from inefficient query planning related to index usage and data distribution, not necessarily a problem with the buffer pool itself.  My experience troubleshooting performance bottlenecks in large-scale data warehousing systems has shown this to be a recurring theme.  The issue isn't the quantity of data retrieved, but how PostgreSQL accesses it.  A seemingly small result set can trigger a cascade of buffer reads if the query's execution plan involves accessing numerous pages, even if only a few rows are ultimately returned.

**1. Explanation:**

PostgreSQL's buffer pool, a cache of disk pages, is crucial for performance.  When a query needs data, it first checks the buffer pool. If the necessary pages are present (a buffer hit), the data is readily available.  If not (a buffer miss), a disk read is required, leading to a shared buffer read (because shared buffers are used for data pages).  The crucial aspect is that a single page can contain multiple rows.  Therefore, even if a query only needs a few rows, it might require accessing many pages if those rows are scattered across numerous pages.  This scattering can result from several factors:

* **Poorly chosen indexes:**  If the query doesn't utilize a suitable index, PostgreSQL might resort to a full table scan, reading every page of the table, dramatically increasing shared buffer reads. Even if a suitable index exists, a poorly designed index, such as one with many NULL values, may not be effective, forcing the database to resort to full table scans on parts of the data.

* **Non-uniform data distribution:**  If the data is clustered in a way that relevant rows are spread across many pages, retrieving a small number of rows can still involve numerous page accesses.  This is common in tables with unevenly distributed key values.  Consider a table storing customer data where a small subset of customers (those with specific attributes) might be distributed over numerous database pages, requiring reads beyond just the initial few.

* **Inefficient query plan:** The query planner might choose a less-than-optimal execution plan, for instance, by performing joins in an order that leads to higher I/O.  This can be exacerbated by the presence of large tables in the query, even if they contribute only a small amount of data to the final result.


**2. Code Examples and Commentary:**

Let's illustrate these scenarios with examples.  Assume a table named `customers` with columns `customer_id` (INT, primary key), `name` (VARCHAR), `city` (VARCHAR), and `purchase_date` (DATE).

**Example 1: Full Table Scan**

```sql
-- No index used, leading to a full table scan
EXPLAIN ANALYZE SELECT name FROM customers WHERE city = 'London' LIMIT 1;
```

Without an index on the `city` column, this query will likely perform a full table scan, reading every page in the `customers` table, even though it only retrieves one row. The explain plan will show many shared buffer reads, reflecting the numerous pages accessed. This scenario highlights the importance of creating indexes on frequently queried columns.

**Example 2: Inefficient Index Usage**

```sql
-- Index exists but is not optimally used due to data distribution
CREATE INDEX idx_customer_city ON customers (city, purchase_date);
EXPLAIN ANALYZE SELECT name FROM customers WHERE city = 'London' AND purchase_date > '2023-01-01';
```

Even with an index, the order of columns in the composite index `idx_customer_city` matters. If the data is such that 'London' customers are spread across many pages, despite the index, the query might still incur high shared buffer reads because of a less-than-optimal query execution plan. A more selective index on purchase_date might perform better in this specific case.

**Example 3:  Suboptimal Query Plan (Join)**

```sql
-- Suboptimal join order can lead to high buffer reads, even with indexes
CREATE INDEX idx_orders_customer_id ON orders (customer_id);
EXPLAIN ANALYZE
SELECT c.name
FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date BETWEEN '2023-01-01' AND '2023-01-31'
LIMIT 10;
```

Suppose this query joins the `customers` and `orders` tables. Even with indexes on both tables, a poor join order chosen by the query planner (e.g., a nested loop join that iterates through a large table first) can cause a large number of shared buffer reads, despite the relatively small output size (10 rows).  Examining the execution plan and potentially using hints to guide the optimizer may resolve such issues.


**3. Resource Recommendations:**

To address these issues effectively, I suggest referring to the official PostgreSQL documentation, specifically the sections on query planning, indexing strategies, and performance tuning. Consult advanced PostgreSQL books that cover performance optimization techniques in depth.  Finally, studying the output of `EXPLAIN ANALYZE` thoroughly is crucial for understanding the execution plan and identifying bottlenecks, as it provides insights into the steps undertaken by the query optimizer and the time spent on each operation.  Analyzing this output meticulously is essential for identifying the root cause of high shared buffer reads and implementing effective solutions.  Practice writing and optimizing queries with various index scenarios.  Experimentation and careful observation of execution plans will equip you to handle such challenges effectively in the future.
