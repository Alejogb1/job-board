---
title: "Why does PostgreSQL underestimate the number of returned rows?"
date: "2025-01-30"
id: "why-does-postgresql-underestimate-the-number-of-returned"
---
PostgreSQL's row count estimation often underestimates the actual number of rows returned, particularly in complex queries involving joins, subqueries, and predicates on functions or expressions.  This stems fundamentally from the limitations of its query planner's statistical model, which relies on histograms and sampled data to approximate the data distribution within tables.  My experience optimizing database performance over the last decade has shown this to be a recurring challenge, especially in rapidly evolving data warehouses.

**1. The Statistical Foundation of the Problem:**

The PostgreSQL query planner employs cost-based optimization.  To determine the optimal execution plan, it needs to estimate the cost of various execution strategies.  A crucial component of this cost calculation is the estimated number of rows returned by each part of the query.  These estimates rely heavily on statistics collected during `ANALYZE` operations. These statistics typically include:

* **Table size:** The total number of rows in a table.
* **Histogram:** A statistical summary of the distribution of values for indexed columns.  This provides an approximation of data density across different value ranges.
* **Null fraction:** The proportion of NULL values for each column.

However, these statistics, even after a thorough `ANALYZE`, are inherently approximations.  The planner cannot perfectly predict the outcome of complex predicates involving functions, expressions, or correlated subqueries.  For instance, a predicate like `WHERE function(column) > 10` relies on the planner's ability to accurately estimate the distribution of `function(column)`, a task that is not always possible based solely on the column's histogram.  Similarly, joins involving multiple tables require the planner to estimate the selectivity of join conditions, which can lead to inaccuracies if the correlation between columns in different tables is not well captured in the statistics.

The resulting underestimation, while sometimes minor, can significantly impact query performance.  An underestimated row count can lead the planner to choose an execution plan that is suboptimal for the actual data volume.  For example, it might favor a nested loop join over a hash join, even though a hash join would be significantly faster if the actual row count is much higher than estimated.  This frequently happens in scenarios involving large datasets and complex queries.


**2. Code Examples Illustrating the Problem:**

Let's examine three scenarios that demonstrate the issue of row count underestimation in PostgreSQL.


**Example 1:  Predicate on a Function:**

```sql
-- Table: products
-- Columns: id (integer, primary key), price (numeric), category (text)

CREATE TABLE products (id SERIAL PRIMARY KEY, price NUMERIC, category TEXT);
INSERT INTO products (price, category) SELECT random()*1000, 'A' FROM generate_series(1,100000);
ANALYZE products;

SELECT count(*) FROM products WHERE price > (random()*500);
```

In this example, the predicate `price > (random()*500)` makes accurate estimation difficult. The `random()` function introduces unpredictability, which the planner cannot readily model using the existing statistics.  The resulting `count(*)` will likely be significantly higher than the planner's initial estimate.


**Example 2:  Complex Join:**

```sql
-- Table: orders
-- Columns: id (integer, primary key), customer_id (integer), order_date (date)

-- Table: customers
-- Columns: id (integer, primary key), city (text)


CREATE TABLE orders (id SERIAL PRIMARY KEY, customer_id INTEGER, order_date DATE);
CREATE TABLE customers (id SERIAL PRIMARY KEY, city TEXT);
INSERT INTO customers (city) SELECT 'City ' || generate_series(1, 10000) FROM generate_series(1,1);
INSERT INTO orders (customer_id, order_date) SELECT id, '2024-01-01'::date FROM customers;
ANALYZE orders;
ANALYZE customers;

SELECT count(*)
FROM orders o
JOIN customers c ON o.customer_id = c.id
WHERE c.city LIKE 'City %' AND o.order_date >= '2024-01-01';
```

This query joins two tables based on a foreign key relationship.  Even with appropriate indexes, the planner might underestimate the number of rows resulting from the join, especially if the distribution of `customer_id` in `orders` does not closely match the distribution of `id` in `customers`.  The `LIKE` predicate adds another layer of complexity to the estimation.


**Example 3:  Correlated Subquery:**

```sql
-- Table: employees
-- Columns: id (integer, primary key), department_id (integer), salary (numeric)

-- Table: departments
-- Columns: id (integer, primary key), name (text)

CREATE TABLE departments (id SERIAL PRIMARY KEY, name TEXT);
CREATE TABLE employees (id SERIAL PRIMARY KEY, department_id INTEGER, salary NUMERIC);
INSERT INTO departments (name) SELECT 'Department ' || generate_series(1, 100) FROM generate_series(1,1);
INSERT INTO employees (department_id, salary) SELECT id, random()*100000 FROM departments;
ANALYZE departments;
ANALYZE employees;


SELECT count(*)
FROM departments d
WHERE EXISTS (SELECT 1 FROM employees e WHERE e.department_id = d.id AND e.salary > 50000);
```

This query uses a correlated subquery to count departments with at least one employee earning more than 50,000.  The planner must estimate the selectivity of the subquery for each department, making accurate row count estimation particularly challenging.  Underestimation is common here because it's difficult to accurately predict the distribution of high salaries within each department without detailed analysis beyond histogram statistics.



**3. Mitigation Strategies and Resources:**

Addressing the row count underestimation issue often requires a multi-pronged approach.  Precise estimation is not always feasible, but you can improve accuracy and overall query performance through these techniques:

* **Regular `ANALYZE`:** Regularly updating statistics with `ANALYZE` is crucial for maintaining the accuracy of the planner's estimations.  The frequency of `ANALYZE` depends on the rate of data changes.
* **Detailed Statistics:** Consider collecting more detailed statistics using the `VACUUM FULL` command. This rebuilds the table's indexes and statistics, providing a more accurate representation of the data distribution. Use with caution, as it's resource intensive.
* **Index Optimization:**  Ensuring appropriate indexes are in place for columns used in `WHERE` clauses and joins is critical.  Careful index selection can drastically improve query performance, even with imperfect row count estimations.
* **Explain Analyze:** Use `EXPLAIN ANALYZE` to analyze the actual execution plan and identify potential bottlenecks.  Comparing estimated and actual row counts reveals discrepancies that highlight areas for optimization.
* **Query Rewriting:**  In some cases, rewriting the query to simplify predicates or eliminate correlated subqueries can improve the planner's ability to estimate the row count accurately.
* **pg_stat_statements:**  Monitor query performance with `pg_stat_statements` to identify frequently executed queries that exhibit significant estimation errors.  Prioritize optimization efforts on these queries.



Understanding the limitations of PostgreSQL's statistical model and employing the above strategies can help minimize the impact of row count underestimation and improve overall database performance.  Thorough testing and monitoring are essential for refining query optimization efforts. Remember that the PostgreSQL documentation and community forums offer extensive resources to aid in advanced query tuning and performance analysis.
