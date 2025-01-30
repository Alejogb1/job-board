---
title: "How does Oracle's query optimizer diagnose performance issues?"
date: "2025-01-30"
id: "how-does-oracles-query-optimizer-diagnose-performance-issues"
---
Oracle's query optimizer, at its core, relies on cost-based optimization. This means it doesn't merely execute queries based on the literal order provided; instead, it explores multiple execution plans, assigns a cost to each, and then chooses the plan with the lowest estimated cost. This cost is not wall-clock time; it is a theoretical representation of the resource consumption required to execute a plan. Understanding this principle is paramount to understanding how the optimizer diagnoses performance issues.

The process starts with parsing the SQL statement, then generating a parse tree. From this, the optimizer creates various execution plans. These plans represent different ways the database can access the necessary data: full table scans, index range scans, nested loop joins, hash joins, sort-merge joins, and so on. The generation of these options is where the optimizer’s diagnostic capabilities begin to surface. It takes into account several vital factors: table statistics, index statistics, system statistics, data distribution, and available resources. These statistics are not static; they are captured by the `DBMS_STATS` package and need to be updated frequently, especially after significant data changes. Stale statistics are a leading cause of poor execution plans. I've personally encountered several cases where a seemingly innocuous data load crippled query performance simply due to outdated statistics.

Once various plans are generated, the optimizer calculates the cost of each based on the gathered statistics. This cost model attempts to predict the I/O overhead, CPU usage, and memory consumption involved in each potential approach. The "best" plan, or the plan with the lowest estimated cost, is then selected and executed. This entire process is transparent to the end user, but we can examine it through various means.

Oracle provides several tools and techniques to diagnose performance problems originating from suboptimal plans. For instance, the `EXPLAIN PLAN` statement allows you to examine the chosen execution plan without actually running the query. This provides a valuable insight into the optimizer's decision-making process. The plan shows you the order of operations, the access methods employed, and whether indexes are being utilized efficiently. Combined with SQL tracing, specifically trace level 10046, the actual run-time statistics of the query can be examined against the predicted cost allowing for deeper analysis into variances.

Let's illustrate this with examples. Assume a scenario with a `CUSTOMERS` table containing customer details and an `ORDERS` table containing order information, where orders reference customers.

**Example 1: Missing Index**

Assume we have the following query designed to find all orders placed by a specific customer.

```sql
SELECT *
FROM ORDERS
WHERE customer_id = 12345;
```

If no index exists on the `customer_id` column of the `ORDERS` table, the optimizer will most likely choose a full table scan. Using `EXPLAIN PLAN`, I’d see something similar to:

```
Plan
--------------------------------------------------------------------------------
| Id | Operation         | Name    | Rows  | Bytes | Cost (%CPU)| Time     |
--------------------------------------------------------------------------------
|  0 | SELECT STATEMENT  |         |    10 |  1000 |   10 (0)| 00:00:01 |
|  1 |  TABLE ACCESS FULL| ORDERS  |    10 |  1000 |   10 (0)| 00:00:01 |
--------------------------------------------------------------------------------
```

This output indicates a full table scan on `ORDERS` (Operation: `TABLE ACCESS FULL`). The cost (`Cost`) is an arbitrary unit, but the full table scan indicates that every row of the table is read. Adding an index drastically improves the performance:

```sql
CREATE INDEX idx_orders_customer_id ON ORDERS(customer_id);
```

After updating statistics, the same query and `EXPLAIN PLAN` will now reveal a plan similar to this:

```
Plan
--------------------------------------------------------------------------------
| Id | Operation                   | Name                | Rows  | Bytes | Cost (%CPU)| Time     |
--------------------------------------------------------------------------------
|  0 | SELECT STATEMENT            |                     |     1 |   100 |    1 (0)| 00:00:01 |
|  1 |  TABLE ACCESS BY INDEX ROWID| ORDERS              |     1 |   100 |    1 (0)| 00:00:01 |
|  2 |   INDEX RANGE SCAN          | IDX_ORDERS_CUSTOMER_ID|     1 |       |    0 (0)| 00:00:01 |
--------------------------------------------------------------------------------
```

Here, the optimizer uses an `INDEX RANGE SCAN` to find the matching rows using the newly created index and then performs a `TABLE ACCESS BY INDEX ROWID` to retrieve the full data. The cost is much lower, representing a considerably more efficient approach. This clearly illustrates how missing indexes lead to full table scans and how the optimizer attempts to mitigate the cost with an alternative plan.

**Example 2: Incorrect Join Method**

Let’s consider a more complex scenario involving a join.

```sql
SELECT c.customer_name, o.order_date
FROM CUSTOMERS c
JOIN ORDERS o ON c.customer_id = o.customer_id
WHERE c.customer_region = 'NORTH';
```

Assume the `CUSTOMERS` table has many rows but the filter on `customer_region` reduces the result set drastically. If the optimizer incorrectly chooses a nested loop join, it may iterate through all records in the `ORDERS` table for each customer, resulting in poor performance.

```
Plan
-----------------------------------------------------------------------------------------------------
| Id | Operation                  | Name        | Rows    | Bytes     | Cost (%CPU) | Time     |
-----------------------------------------------------------------------------------------------------
|  0 | SELECT STATEMENT           |             |     10 |      500 |     5 (0)   | 00:00:01 |
|  1 |  NESTED LOOPS              |             |     10 |      500 |     5 (0)   | 00:00:01 |
|  2 |   TABLE ACCESS FULL        | CUSTOMERS   |      1 |       50 |     1 (0)   | 00:00:01 |
|  3 |   TABLE ACCESS BY INDEX ROWID| ORDERS      |     10 |      450 |     4 (0)   | 00:00:01 |
|  4 |     INDEX RANGE SCAN       | IDX_ORDERS_CUSTOMER_ID |    10 |       |     1 (0)   | 00:00:01 |
-----------------------------------------------------------------------------------------------------

```
Here we can see the use of a `NESTED LOOPS` join and a `TABLE ACCESS FULL` on the `CUSTOMERS` table. The `INDEX RANGE SCAN` used on the `ORDERS` table is good, but a hash join may be more appropriate here due to the potential large number of rows in the orders table. Providing hints is one way of influencing the optimizer, but usually statistics are the root problem. Updating statistics and the optimizer might produce the following execution plan:

```
Plan
--------------------------------------------------------------------------------------------------
| Id | Operation          | Name           | Rows | Bytes | Cost (%CPU) | Time     |
--------------------------------------------------------------------------------------------------
|  0 | SELECT STATEMENT   |                |    10 |   500 |    3 (0)   | 00:00:01 |
|  1 |  HASH JOIN         |                |    10 |   500 |    3 (0)   | 00:00:01 |
|  2 |   TABLE ACCESS FULL| CUSTOMERS      |    1  |    50 |    1 (0)   | 00:00:01 |
|  3 |   TABLE ACCESS FULL| ORDERS         |   10  |   450 |    2 (0)   | 00:00:01 |
--------------------------------------------------------------------------------------------------
```
In this version the optimizer has selected a `HASH JOIN` approach. It is important to note that just because a different plan has been selected this doesn't necessarily mean it is better, the underlying data should be analyzed carefully before selecting a hint driven approach to optimizing the plan. The optimizer uses statistics to evaluate cost, so providing hints should be the last option after understanding and addressing the underlying data distributions.

**Example 3: Skewed Data**

Consider the situation where the `ORDERS` table contains a disproportionate number of orders for a specific customer. If statistics are not up to date or histograms are not collected on the `customer_id` column, the optimizer may underestimate the number of rows produced for that customer. This could cause it to select an unsuitable plan, like a nested loop join, even if a hash join or sort-merge join is more efficient. Skewed data is another factor that the optimizer has to account for, and with missing or inaccurate statistics it will not do so.

```
CREATE TABLE orders (
  order_id NUMBER PRIMARY KEY,
  customer_id NUMBER,
  order_date DATE
);

INSERT INTO orders (order_id, customer_id, order_date)
  SELECT level, 1, SYSDATE - LEVEL/10
  FROM dual
  CONNECT BY level <= 9999;

INSERT INTO orders (order_id, customer_id, order_date)
  SELECT level + 10000, 2, SYSDATE - LEVEL/10
  FROM dual
  CONNECT BY level <= 10;

COMMIT;
```

Here, customer id 1 has significantly more entries in the `ORDERS` table. Without data distributions or up-to-date statistics, the optimizer may choose a plan that's inefficient for customer id 1 due to it underestimating the number of rows. Providing histograms on the `CUSTOMER_ID` column in this case would be critical in guiding the optimizer towards a better plan.

In summary, Oracle's query optimizer diagnoses performance issues through a comprehensive process. It considers statistics, available resources, and data distributions to calculate costs for various execution plans. By using tools such as `EXPLAIN PLAN` and SQL tracing, we can see the optimizer’s choices and use the insights to fine-tune database configuration, create indexes where needed, and ensure statistics are up-to-date. For further exploration of these topics, several resources are available online from Oracle directly that discuss cost-based optimization, plan stability, SQL tuning, and the `DBMS_STATS` package. Additionally, educational publications on advanced database performance provide detailed explanations of indexing strategies and join algorithms.
