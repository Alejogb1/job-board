---
title: "Why is MySQL performing a full table scan on my indexed columns?"
date: "2025-01-30"
id: "why-is-mysql-performing-a-full-table-scan"
---
MySQL's reliance on a full table scan despite the presence of indexes on relevant columns is a common performance bottleneck I've encountered throughout my years optimizing database systems.  The root cause often lies not in a defect within the MySQL engine itself, but rather in the interaction between the query, the data, and the optimizer's cost-based decisions.  Understanding this interaction requires a detailed examination of several factors.

**1.  Inefficient Query Structure:** The most frequent culprit is a poorly structured query that prevents the optimizer from effectively leveraging the existing indexes.  This can manifest in several ways.  Firstly, the presence of functions or calculations applied directly to indexed columns will often force a full table scan. The optimizer cannot directly utilize an index if the column value is being modified before comparison. Secondly, using wildcard characters (%) at the beginning of a `LIKE` clause prevents index usage.  Indexes are typically B-tree structures optimized for prefix searches; a leading wildcard necessitates a full scan to compare against each row. Thirdly, complex `WHERE` clauses containing multiple conditions joined with `OR` can lead to the optimizer choosing a full table scan, even when some individual conditions could benefit from indexing.  The optimizer's cost model might determine that the overhead of accessing multiple indexes is greater than the cost of a full table scan, particularly on smaller tables.  Finally,  `UNION ALL` without explicit optimization hints may lead to individual scans on each table even if indexes exist.

**2. Data Statistics and Optimizer Choices:** The MySQL query optimizer relies on table statistics to estimate the cost of different query execution plans.  Outdated or inaccurate statistics can lead it to misjudge the effectiveness of using an index. If the statistics are not up-to-date, the optimizer may not correctly assess the selectivity of the index, leading to an erroneous cost evaluation that favors a full table scan. This is especially pertinent with frequently updated tables.  The `ANALYZE TABLE` command should be regularly employed to ensure that the optimizer has current and accurate information.

**3.  Index Ineffectiveness:** While the presence of an index is a necessary condition, it is not sufficient. The index itself might be suboptimal for the specific query. Consider scenarios where the indexed column has a high cardinality (many distinct values) and the query's `WHERE` clause only filters a small subset of those values. In such cases, an index lookup might only eliminate a relatively small portion of the table, rendering it less efficient than a full table scan. Similarly, composite indexes (indexes on multiple columns) must be carefully designed. If the query doesn't use the leading columns of the composite index, the index will be ignored, resulting in a full table scan.

Let's illustrate these concepts with code examples.  Assume we have a table named `products` with columns `product_id` (INT, PRIMARY KEY), `name` (VARCHAR(255), indexed), and `price` (DECIMAL(10,2)).


**Example 1: Function on Indexed Column**

```sql
SELECT * FROM products WHERE LENGTH(name) > 10;
```

In this scenario, the `LENGTH()` function applied to the indexed `name` column prevents the optimizer from using the index. The function introduces a computation that needs to be performed on every row, making the index irrelevant.  The solution is to refactor the query if possible, or create a computed column and index that column instead.


**Example 2: Leading Wildcard in LIKE Clause**

```sql
SELECT * FROM products WHERE name LIKE '%example%';
```

The leading wildcard in this `LIKE` clause renders the index on the `name` column unusable. The database needs to check every row to see if "example" appears anywhere within the name. To optimize this, consider alternative approaches, such as using full-text indexing if appropriate, or refactoring the search logic if possible to start with a known prefix.


**Example 3:  Outdated Statistics**

```sql
SELECT * FROM products WHERE price > 100;
```

If the statistics for the `products` table are outdated, and a large number of rows have been added or updated with prices greater than 100 since the last `ANALYZE TABLE` command, the optimizer may underestimate the selectivity of the `price` column's index, and it will opt for a full table scan.  Running `ANALYZE TABLE products;` prior to running this query is crucial to ensure the optimizer makes informed decisions.


**Recommendations:**

To effectively troubleshoot this issue, systematically investigate the factors mentioned earlier. Begin by examining the query itself for any of the pitfalls described. Check the execution plan using tools provided by MySQL (e.g., `EXPLAIN`), which detail how the optimizer intends to execute the query. This execution plan reveals whether indexes are used and, if not, the reasons why.  Pay close attention to the `key` column in the `EXPLAIN` output; an empty value signifies a full table scan.

Analyze the table statistics using `SHOW TABLE STATUS` to ascertain their freshness. If they're outdated, update them with `ANALYZE TABLE`.  If the indexes themselves seem inefficient, consider redesigning them, possibly creating composite indexes or dropping and recreating them if necessary, ensuring they are aligned with common query patterns.  Finally, profile your queries and identify the most frequent and resource-intensive queries. Focus your optimization efforts on these critical queries. Through a methodical approach involving query review, statistic analysis, and index evaluation, you can identify the root cause and implement the necessary optimizations. Remember to systematically test performance improvements after each change.  These combined strategies, honed through years of practical experience, allow for a comprehensive approach to optimizing MySQL performance and avoiding full table scans.
