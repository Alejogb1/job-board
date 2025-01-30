---
title: "How does MySQL's INFORMATION_SCHEMA.PROFILING table work?"
date: "2025-01-30"
id: "how-does-mysqls-informationschemaprofiling-table-work"
---
The `INFORMATION_SCHEMA.PROFILING` table in MySQL provides a detailed look into the execution times of SQL queries, but its utility hinges entirely on enabling server-side profiling, a feature that is disabled by default due to its performance overhead. I've leveraged this table extensively while optimizing database-heavy applications, and I've found its granular data invaluable for pinpointing performance bottlenecks, especially in complex stored procedures and multi-table joins.

The `PROFILING` table is not populated unless you explicitly activate profiling for your database session. This is done using the `SET profiling=1;` command. When enabled, MySQL records timing information for each statement executed within that session, storing this data in a set of memory buffers. Upon the termination of the session or the disabling of profiling via `SET profiling=0;`, the buffered data is then transferred to the `PROFILING` table, overwriting the previous session’s data. Consequently, the `PROFILING` table offers a transient, per-session view of query execution, rather than a historical record. If the session closes without disabling profiling, the profiling data will be lost, emphasizing that this feature is intended more for debugging and analysis than long-term data collection.

The structure of the `PROFILING` table is relatively straightforward, but interpreting the information requires understanding what each column represents: `QUERY_ID` is a unique identifier for each query within the profiling session; `SEQ` is the sequence number of the execution stage for a particular query; `STATE` describes the stage of the query execution; `DURATION` indicates the time spent in the particular state in seconds, typically measured in milliseconds; and `INFORMATION` provides some additional information about the execution stage such as the name of the function being called, for example. The most crucial column for performance analysis is `DURATION`, which allows you to identify time-consuming phases of query processing, like Sending data or System lock.

To illustrate its usage, consider a scenario where I was debugging a slow reporting query on an e-commerce database. The following code snippets demonstrate how to enable profiling, execute a query, retrieve profiling data and subsequently disable profiling:

**Example 1: Basic Query Profiling**

```sql
-- Enable profiling for the current session
SET profiling = 1;

-- Execute the problematic reporting query
SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    SUM(o.total_amount) as total_spent
FROM
    customers c
JOIN
    orders o ON c.customer_id = o.customer_id
WHERE
    o.order_date BETWEEN '2023-01-01' AND '2023-12-31'
GROUP BY
    c.customer_id
ORDER BY
    total_spent DESC
LIMIT 10;

-- Retrieve profiling results for the last query
SELECT
    query_id,
    seq,
    state,
    duration,
    information
FROM
    information_schema.profiling
WHERE
    query_id = (SELECT MAX(query_id) FROM information_schema.profiling);

-- Disable profiling
SET profiling = 0;
```

In this example, `SET profiling = 1` initiates profiling, followed by the execution of a complex `SELECT` query involving joins, aggregations, and ordering. The profiling data is retrieved using a query that filters the `PROFILING` table to show only the results of the most recently executed query by obtaining the `MAX(query_id)`. Finally, profiling is disabled with `SET profiling = 0;`.  The results from this query would reveal a step-by-step breakdown of the execution, potentially highlighting whether the slow performance is due to the join, the aggregation, or the ordering.  Without profiling, it's challenging to determine the bottleneck with such accuracy.

**Example 2: Profiling a Stored Procedure**

The utility of `PROFILING` extends beyond simple `SELECT` queries, as demonstrated in this example of analyzing a stored procedure:

```sql
-- Enable profiling for the current session
SET profiling = 1;

-- Assume a stored procedure `CalculateMonthlySales` exists
CALL CalculateMonthlySales('2023-05-01', '2023-05-31');

-- Retrieve profiling results for the last query
SELECT
    query_id,
    seq,
    state,
    duration,
    information
FROM
    information_schema.profiling
WHERE
    query_id = (SELECT MAX(query_id) FROM information_schema.profiling);

-- Disable profiling
SET profiling = 0;
```

This example calls a hypothetical stored procedure named `CalculateMonthlySales`. Analyzing the `PROFILING` table output here provides visibility into the internal workings of the procedure. You will find a breakdown of the times for each `SELECT`, `UPDATE`, or other SQL statement executed inside the procedure. This is particularly useful for pinpointing poorly performing sections within the procedure logic itself.  This helps in deciding where to optimize, such as adding indices to certain columns or re-writing parts of the procedure for performance.

**Example 3: Targeted Profiling and Filtering**

When working in sessions with multiple operations or a high volume of queries, it becomes useful to filter and refine the profiling information. The following provides an example of profiling a specific query and only getting the data for the exact query execution:

```sql
-- Enable profiling for the current session
SET profiling = 1;

-- Execute multiple queries to demonstrate the need to filter the data
SELECT * FROM products WHERE price > 100;
SELECT * FROM users WHERE registration_date > '2023-01-01';
SELECT * FROM orders WHERE order_date > '2023-06-01';

-- The problematic query we want to profile:
SELECT
    c.customer_id,
    c.first_name,
    c.last_name,
    SUM(o.total_amount) as total_spent
FROM
    customers c
JOIN
    orders o ON c.customer_id = o.customer_id
WHERE
    o.order_date BETWEEN '2023-01-01' AND '2023-12-31'
GROUP BY
    c.customer_id
ORDER BY
    total_spent DESC
LIMIT 10;

-- Retrieve profiling results for the specific query based on its query_id
SELECT
    query_id,
    seq,
    state,
    duration,
    information
FROM
    information_schema.profiling
WHERE
    query_id = LAST_INSERT_ID();


-- Disable profiling
SET profiling = 0;
```

This final example demonstrates a more advanced scenario. We execute a series of queries before executing the complex query we wish to profile. Rather than relying on the `MAX(query_id)` which gives the id of the most recently executed query, we utilize the `LAST_INSERT_ID()` function which retrieves the `query_id` of the last insert, which in our case is the last execution of the query being profiled. This method is more robust when multiple statements are executed within the same session and ensures that profiling data only refers to the problematic query. The `WHERE` clause in the profiling query filters the results to show only profiling data associated with the specific query executed previously. This approach is particularly helpful when multiple queries are executed in a session.

Several good resources exist for understanding MySQL query performance and utilizing profiling techniques effectively. The official MySQL documentation provides an in-depth explanation of the `INFORMATION_SCHEMA` tables, including `PROFILING`. In addition to the MySQL manuals, books on database design and optimization often contain practical examples and best practices for utilizing profiling data in real-world scenarios. Also, various community blogs and forums frequently share troubleshooting tips and advanced optimization techniques based on their hands-on experiences with tools like the `PROFILING` table.

In conclusion, the `INFORMATION_SCHEMA.PROFILING` table is a vital tool for SQL query analysis and performance tuning. However, it's essential to understand that it is a per-session, transient debugging tool and should not be considered a long-term performance monitoring solution. Enabling profiling adds overhead, and consequently, you should only do so when actively diagnosing performance problems. With a solid understanding of the table's structure and the various ways to filter and analyze its data, you can effectively uncover bottlenecks and make informed optimization decisions that considerably improve the performance of your MySQL applications.
