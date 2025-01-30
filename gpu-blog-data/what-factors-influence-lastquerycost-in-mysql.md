---
title: "What factors influence `last_query_cost` in MySQL?"
date: "2025-01-30"
id: "what-factors-influence-lastquerycost-in-mysql"
---
The `last_query_cost` in MySQL's `INFORMATION_SCHEMA.PROFILES` table represents the estimated cost of the most recently executed query, as determined by the query optimizer. Understanding what drives this cost figure is crucial for effectively tuning database performance. It's not a direct measurement of time, but rather a relative estimate based on numerous factors relating to data access and operations. I’ve spent considerable time analyzing query plans in demanding production environments, and observed how variations in these elements dramatically impact the `last_query_cost`.

Firstly, the single most significant influence on the reported cost is the chosen query execution plan. The optimizer’s job is to evaluate multiple possible ways to retrieve the data, estimating the cost of each, and selecting the lowest cost option. Factors contributing to plan choice include table statistics, indexing strategy, and the specific operations required by the query. When statistics are out-of-date, for example after large inserts or deletes, the optimizer can make incorrect assumptions, leading to inefficient plans and high `last_query_cost` values. I remember debugging a particularly sluggish reporting query that was using a full table scan instead of a keyed index lookup; simply updating table statistics with `ANALYZE TABLE` brought the cost down significantly and improved query execution time.

Secondly, data access patterns play a pivotal role. Full table scans are exceptionally expensive, while index lookups or range scans are generally much cheaper. The presence of appropriate indexes directly impacts this. A query with a `WHERE` clause on a non-indexed column will likely result in a costly full table scan. If a useful index exists, but the optimizer does not use it, other issues may be at play such as a data type mismatch, or the optimizer is incorrectly estimating the number of rows returned. For example, using `LIKE` with a wildcard at the start of the pattern can often render indexes unusable. I once encountered a situation where a seemingly simple lookup on a `varchar` column was performing poorly until we discovered the client was passing numerical strings, preventing the index from being used. We resolved it with type conversion in the query itself which encouraged the optimizer to choose the indexed path.

Thirdly, operations performed as part of the query contribute to the overall cost. Joins, sorts, and aggregations are all computationally intensive and can drastically increase cost. The type of join (e.g., nested loop, hash join), the join order, and the amount of data processed through these operations has a direct correlation to the reported cost. A query that joins multiple large tables without appropriate indexes, for instance, could have a very high cost. Aggregation involving large datasets with `GROUP BY` or `HAVING` also tends to be costly and can be an area to analyze for optimization.

Here are a few code examples to illustrate the impact of some of these factors:

**Example 1: Full Table Scan vs. Index Lookup**

This example shows the contrast between a query lacking an index and one utilizing a highly selective index.

```sql
-- No index on 'customer_name' -  Full table scan is expected
SET profiling=1;
SELECT * FROM customers WHERE customer_name LIKE '%smith%';
SHOW PROFILES; -- Observe high last_query_cost.

-- An index is assumed to exist on `customer_id` - likely an index lookup
SELECT * FROM customers WHERE customer_id = 12345;
SHOW PROFILES; -- Observe relatively low last_query_cost
```

**Commentary:**

The first query, searching using a `LIKE` pattern without any starting fixed characters for the non-indexed `customer_name` will likely result in a full table scan. The optimizer will have no efficient way to find the matching rows other than by examining every record. This will be reflected as a significantly higher `last_query_cost` in comparison to the second query which utilizes an index-based lookup by `customer_id`. Assuming an index exists on `customer_id`, the second query can quickly jump to the specific record, leading to a much lower estimated cost. This highlights the crucial role of indexes in avoiding costly table scans. Note that when benchmarking queries using `SHOW PROFILES`, you should perform them several times to be sure you're capturing their general execution cost after data has been loaded into cache.

**Example 2: Effect of Table Statistics**

This example demonstrates how stale table statistics can lead to a suboptimal query plan.

```sql
-- Assume the `orders` table has just had a large number of inserts.
SET profiling=1;

-- Initial query execution (before statistics are updated)
SELECT COUNT(*) FROM orders WHERE order_date > '2023-01-01';
SHOW PROFILES; -- Observe relatively high last_query_cost


-- Update statistics to reflect changes
ANALYZE TABLE orders;

-- Re-execute the same query
SET profiling=1;
SELECT COUNT(*) FROM orders WHERE order_date > '2023-01-01';
SHOW PROFILES;  -- Observe lower last_query_cost, if optimizer is then choosing a better plan.
```

**Commentary:**

Before updating the table statistics using `ANALYZE TABLE`, the optimizer might make incorrect assumptions about the distribution of data in the `orders` table, leading to a less efficient plan. This suboptimal plan would be reflected in a higher `last_query_cost`. After the `ANALYZE TABLE` statement, the optimizer has more accurate information, potentially resulting in a better plan, such as utilizing an index on `order_date`, resulting in a lower reported cost and likely faster execution. It is critical to have updated statistics especially for tables undergoing regular modifications.

**Example 3: Impact of Joins**

This example shows the effect of join order on the query cost.

```sql
-- Assumed indexes exist on foreign keys as well as `order_id` in each table.
SET profiling=1;

-- Join order one. Note that join order optimization is limited in MySQL.
SELECT * FROM customers c
JOIN orders o ON c.customer_id = o.customer_id
WHERE o.order_date > '2023-01-01';
SHOW PROFILES; -- Observe the cost for this specific plan.

-- Force a specific join order using STRAIGHT_JOIN
SET profiling=1;
SELECT * FROM orders o STRAIGHT_JOIN customers c ON c.customer_id = o.customer_id
WHERE o.order_date > '2023-01-01';
SHOW PROFILES; -- Observe the cost for this specific plan.
```

**Commentary:**

The default join order which the optimizer chooses may or may not be the optimal choice; in many cases it will be the correct option for the given query and tables. By forcing a different order with `STRAIGHT_JOIN` we may observe a different cost profile. With large tables it's often beneficial to start the join with the table which has a higher degree of selectivity from our WHERE clause. While MySQL's optimizer is adept at choosing efficient plans, situations arise where manual overrides may be needed, and profiling tools like this are invaluable in discovering those situations. The cost variation will reflect differences in how the join is performed and data access patterns based on the order of the tables being joined.

In summary, `last_query_cost` is a complex metric influenced by the query optimizer's chosen execution plan, data access patterns (particularly index usage), table statistics, and the types of operations performed. Understanding these factors is paramount in writing optimized queries.

For further reading, I recommend exploring resources detailing MySQL's query optimizer, how indexes work, and techniques for analyzing query execution plans using `EXPLAIN` statements. Publications focused on database performance tuning, especially those covering indexing strategies and schema design also prove beneficial. MySQL's own official documentation also contains detailed explanations on statistics, query optimization and index behavior. While there are numerous vendor-specific resources available, it's generally a good idea to build foundational knowledge based on the core fundamentals of relational database management.
