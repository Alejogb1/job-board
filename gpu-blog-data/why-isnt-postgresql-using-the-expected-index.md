---
title: "Why isn't PostgreSQL using the expected index?"
date: "2025-01-30"
id: "why-isnt-postgresql-using-the-expected-index"
---
PostgreSQL's failure to utilize an expected index often stems from a mismatch between the query's WHERE clause and the index's definition, frequently due to subtle issues involving data types, operators, or expression evaluation.  My experience troubleshooting performance bottlenecks in large-scale PostgreSQL deployments has shown that seemingly minor discrepancies can significantly impact query planning.  Let's examine this systematically.

**1. Understanding Query Planning and Index Usage:**

PostgreSQL's query planner employs a cost-based optimizer. It analyzes the query, available indexes, and table statistics to determine the most efficient execution plan.  This plan considers factors like index selectivity, the cost of index scans versus sequential scans, and the overall I/O overhead. If the planner estimates that a sequential scan (a full table scan) is cheaper than using an index, it will opt for the sequential scan, even if an index seemingly fits the WHERE clause.  This often happens because the statistics gathered on the table might not accurately represent the current data distribution, leading the optimizer to misjudge the cost.

Furthermore, the planner's assessment hinges critically on the exact form of the WHERE clause.  Any difference, however small, between the WHERE clause's conditions and the index's columns (including data types, operators, and function calls) can prevent index usage.  For instance, a case-insensitive comparison in the WHERE clause will not utilize a case-sensitive index. Similarly, using a different function on the indexed column than the one used in the index definition (e.g., `lower(column)` vs. `column`) will also render the index useless for that query.

**2. Common Causes of Index Inefficiency:**

* **Data Type Mismatches:**  A WHERE clause condition comparing an indexed column to a value of a different data type will likely prevent index usage.  Implicit type coercion might occur, but it adds overhead and can interfere with the optimizer's ability to recognize the index as suitable.

* **Operator Mismatches:**  The operators used in the WHERE clause must match those suitable for the index.  For example, using `=` in the WHERE clause while the index is built using a `LIKE` operator will not lead to index usage.

* **Function Calls:**  Functions applied to the indexed column in the WHERE clause must be the same as, or compatible with, those used during index creation. Applying a function in the WHERE clause that isn't used in the index definition will force a function scan, bypassing the index.

* **Leading Wildcards in `LIKE` Clauses:**  Queries using `LIKE` with leading wildcards (`%pattern`) cannot efficiently utilize B-tree indexes.  These necessitate a full index scan, often rendering index usage less efficient than a sequential scan.

* **Outdated Statistics:**  If the table statistics are outdated, the planner may make inaccurate estimations of the index selectivity and cost.  Running `ANALYZE` on the table is crucial for ensuring accurate statistics.

* **Complex Queries:** In intricate queries with multiple JOINs or subqueries, the planner might choose a plan that doesn't utilize all possible indexes due to the complexity of the overall optimization problem.


**3. Code Examples and Commentary:**

**Example 1: Data Type Mismatch**

```sql
-- Table creation
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    price NUMERIC(10,2)
);

-- Index creation
CREATE INDEX idx_price ON products (price);

-- Query that won't use the index due to implicit type coercion
SELECT * FROM products WHERE price = '19.99';  -- Implicit conversion from text to numeric
```

This query, although seemingly straightforward, will likely bypass the `idx_price` index because the WHERE clause compares the `NUMERIC` column `price` with a `TEXT` literal.  PostgreSQL will perform an implicit type conversion, negating the benefits of the index. The correct approach involves using a numeric literal: `SELECT * FROM products WHERE price = 19.99;`


**Example 2: Function Call Mismatch**

```sql
-- Table creation
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username TEXT
);

-- Index creation
CREATE INDEX idx_username ON users (username);

-- Query that won't use the index
SELECT * FROM users WHERE lower(username) = 'john';
```

Here, the index `idx_username` is on the `username` column directly.  The WHERE clause, however, applies the `lower()` function. This forces a function scan instead of using the index directly. To utilize the index, either create an index on `lower(username)` or modify the query to avoid the function call if case-insensitivity is not crucial.


**Example 3:  Outdated Statistics**

```sql
-- Table creation and data insertion (simplified for brevity)
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    order_date DATE
);
INSERT INTO orders (order_date) SELECT ('2024-01-01'::date + INTERVAL '1 day' * i) FROM generate_series(1,100000) i; -- insert 100k records

-- Index creation
CREATE INDEX idx_order_date ON orders (order_date);

-- Query that might not use the index due to outdated statistics
SELECT * FROM orders WHERE order_date >= '2024-06-01';

-- Update Statistics
ANALYZE orders;

--Query after updating Statistics; likely to use the index now
SELECT * FROM orders WHERE order_date >= '2024-06-01';
```

This example highlights the importance of up-to-date statistics.  After inserting a large volume of data, the planner might rely on outdated statistics, leading to incorrect cost estimations. Running `ANALYZE orders;` updates the statistics, allowing the planner to correctly assess the index's effectiveness and use it for this query.


**4. Resource Recommendations:**

The PostgreSQL documentation provides extensive detail on query planning, indexing strategies, and statistics.  Consult the official documentation for in-depth explanations of the query planner's behavior.  Furthermore, studying the `EXPLAIN` and `EXPLAIN ANALYZE` commands will be invaluable in understanding the chosen execution plan for your queries.  Exploring advanced topics like index-only scans will enhance your understanding of index optimization.  Finally, a thorough grasp of SQL optimization techniques is essential for effectively utilizing indexes.  A solid grounding in database design principles will help you avoid common pitfalls in indexing strategies.
