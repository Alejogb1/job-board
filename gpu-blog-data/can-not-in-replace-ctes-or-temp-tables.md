---
title: "Can `NOT IN` replace CTEs or temp tables effectively?"
date: "2025-01-30"
id: "can-not-in-replace-ctes-or-temp-tables"
---
Directly, `NOT IN` often represents a performance bottleneck when used for exclusion, especially as dataset sizes increase, making it an unsuitable replacement for the more performant and readable strategies often achieved through Common Table Expressions (CTEs) or temporary tables. I've witnessed this first-hand in numerous database optimization projects.

The core issue stems from how `NOT IN` is typically processed by relational database management systems (RDBMS). For each row in the outer query, the database must perform a nested scan across the entire subquery result set to determine if the value is *not* present. This operation scales poorly, resulting in a computational complexity approaching O(n*m) where 'n' is the number of rows in the outer table and 'm' is the number of rows in the subquery.

CTEs and temporary tables, on the other hand, allow the database engine to materialize intermediate results, often within a temporary storage space, and then use those materialized results in subsequent operations. This significantly reduces the amount of scanning and processing needed. The primary difference lies in the execution plan. `NOT IN` typically leads to full table scans of the subquery on *each* evaluation of a row from the outer query, whereas CTEs and temp tables execute the subquery once and then utilize that resulting set.

Consider this in the context of a system I worked on several years back involving customer order details. We needed to identify customers who *had not* placed orders in a specific date range. The naive approach initially used looked like this:

```sql
-- Example 1: Inefficient NOT IN Usage
SELECT c.customer_id, c.customer_name
FROM customers c
WHERE c.customer_id NOT IN (
    SELECT o.customer_id
    FROM orders o
    WHERE o.order_date BETWEEN '2023-01-01' AND '2023-01-31'
);
```

This query, while conceptually straightforward, was horrendously slow against a table of a few million customer records and an order history exceeding tens of millions. The database essentially looped through each customer and then scanned the order table to see if the customer had an order within the specified dates. This demonstrated the poor performance characteristics of `NOT IN`.

The solution involved a temporary table to pre-select the customers who *had* placed orders in that range, followed by a left outer join and a filtering condition. This alternative drastically reduced query execution times:

```sql
-- Example 2: Efficient Temporary Table Approach
CREATE TEMP TABLE Temp_Ordered_Customers AS
SELECT DISTINCT o.customer_id
FROM orders o
WHERE o.order_date BETWEEN '2023-01-01' AND '2023-01-31';

SELECT c.customer_id, c.customer_name
FROM customers c
LEFT OUTER JOIN Temp_Ordered_Customers toc
ON c.customer_id = toc.customer_id
WHERE toc.customer_id IS NULL;

DROP TABLE Temp_Ordered_Customers;
```

Here, the filtering work is done in the temporary table creation once.  The subsequent selection and filtering are much quicker since the outer join operation on the result of the temporary table is more performant than repeatedly executing the subquery in example 1. Moreover, the `IS NULL` condition effectively isolates the customers not present in the orders result set. The temporary table effectively 'materializes' the subquery's result, allowing for better query optimization by the database engine.  The drop statement is used to clean up the table, adhering to good practice.

A CTE can achieve similar performance improvements, often with better readability since the intermediate results do not require explicit table creation and management. Here’s the equivalent example using a CTE:

```sql
-- Example 3: Efficient CTE Approach
WITH Ordered_Customers AS (
    SELECT DISTINCT o.customer_id
    FROM orders o
    WHERE o.order_date BETWEEN '2023-01-01' AND '2023-01-31'
)
SELECT c.customer_id, c.customer_name
FROM customers c
LEFT OUTER JOIN Ordered_Customers oc
ON c.customer_id = oc.customer_id
WHERE oc.customer_id IS NULL;
```

This example demonstrates the conciseness of CTEs. The `Ordered_Customers` CTE acts in much the same way as the temporary table. The database evaluates it, effectively materializing a result set, and uses that set in the outer query to identify customers without associated orders during the specified date range. This is functionally identical to the temporary table version but avoids the overhead of explicitly creating and dropping a temporary table.

While `NOT IN` may seem simpler on the surface, it often masks underlying inefficiencies. Furthermore, `NOT IN` with NULL values in the subquery can produce unexpected results because `NULL` compared with any value (including other nulls) is usually not TRUE, and so no rows will be returned (unless rows from the main table are also NULL). Temporary tables and CTEs offer more explicit control over the query execution plan and often enable better optimization. I consistently favor these latter constructs for exclusion tasks, especially in scenarios involving large datasets, because they provide more clarity in both operation and intention. It is also simpler to optimize if required by adding indexes to the intermediate or final tables used by the CTE.

For those who wish to deepen their understanding of database performance optimization, I recommend the following resources:
*  Database documentation for the specific RDBMS being used (e.g., Oracle, PostgreSQL, SQL Server). These typically have extensive sections on query optimization and execution plans.
* Books specializing in database performance tuning. These books offer comprehensive guidance on best practices and often feature real-world case studies.
* Online courses focused on database management and SQL querying techniques. These resources provide structured learning paths and hands-on exercises.
These learning options will prove invaluable to comprehend how different query expressions affect database performance. They also emphasize the importance of understanding the underlying database engine's behavior when deciding between approaches like `NOT IN`, temporary tables, and CTEs.
