---
title: "Why are my Postgres queries faster without indexes?"
date: "2025-01-30"
id: "why-are-my-postgres-queries-faster-without-indexes"
---
PostgreSQL's query planner makes decisions based on statistical data; inaccurate or absent statistics can lead to suboptimal choices, sometimes even causing it to avoid otherwise beneficial indexes. I've encountered this specific problem multiple times over my years managing databases, typically when dealing with data sets undergoing significant changes.

The core issue isn't that indexes are inherently slow; instead, it’s that the query planner might misjudge their cost-effectiveness given the current data distribution. When a table is initially created or after substantial modifications, the planner relies on default statistics which are often insufficient for selecting the most efficient execution path. An index, designed to expedite lookups, only offers performance benefits when the planner correctly estimates its efficiency and chooses to use it. The planning stage includes calculating the cost for various operations, including table scans, index scans, and joins. If the statistics on which these calculations are based are outdated or missing, the planner can err.

Specifically, if a query planner sees that a significant percentage of rows would satisfy a query's `WHERE` clause, it might determine that a full table scan, which reads all rows sequentially, is faster than using an index scan, which involves random access across the table. For instance, if an index is on a column with a small cardinality (limited number of distinct values) and the `WHERE` clause is filtering by a frequently occurring value, using that index would necessitate retrieving many rows from the index followed by accessing the same rows in the table. This might introduce more overhead than reading the table sequentially. A critical factor is *selectivity*, which represents the proportion of rows expected to be returned by a given condition. An index is highly beneficial when the selectivity is low. However, if a query involves a low selectivity condition, the planner might decide against an index scan.

Let's illustrate with some examples, considering a `users` table with columns `id`, `name`, `email`, and `status`.

**Example 1: Low Cardinality Index**

Assume we have an index on the `status` column, which can have values ‘active’, ‘inactive’, or ‘pending’.

```sql
-- Index Creation
CREATE INDEX idx_users_status ON users (status);

-- Query 1 (Potentially slow if status 'active' is common):
EXPLAIN SELECT * FROM users WHERE status = 'active';
```

If ‘active’ constitutes a significant portion of the `users` table, say 80%, the planner may decide not to use `idx_users_status`. The `EXPLAIN` output might reveal a "Seq Scan," indicating a full table scan.  The sequential table scan reads all rows which in a situation with high cardinality of 'active' records, might be faster than a index lookup and then retrieval of every one of the required rows. This decision is not due to any problem with the index itself, but the planner's determination, based on insufficient statistics, that the overhead of using the index outweighs the benefit. This is a common cause of surprising index performance issues.

**Example 2: Outdated Statistics**

Let's say the `users` table initially contains a relatively small number of inactive users, and the statistics were generated before the table drastically changed.

```sql
-- Query 2 (Potentially fast initially with the index):
EXPLAIN SELECT * FROM users WHERE status = 'inactive';

-- Assume we run this after the table has grown with many inactive users
-- and statistics haven't been updated:
EXPLAIN SELECT * FROM users WHERE status = 'inactive';
```

The initial query might utilize the `idx_users_status` index efficiently. However, after a massive batch update that sets many users to ‘inactive’, but without a corresponding `ANALYZE` operation, the planner would still have the old statistics. It might incorrectly choose the same indexed query path, assuming a low selectivity of 'inactive' entries, which is not longer the case, which leads to a performance degradation. A table scan might actually be faster, but the old stats have not informed the query planner of the now low-selectivity issue.

**Example 3: Combination Index Inefficiency**

Consider a composite index on `(status, name)` and the following queries:

```sql
-- Composite Index Creation
CREATE INDEX idx_users_status_name ON users (status, name);

-- Query 3 (potentially slow due to a non-leading column):
EXPLAIN SELECT * FROM users WHERE name LIKE 'John%' ;

-- Query 4 (potentially faster if the planner selects the index appropriately)
EXPLAIN SELECT * FROM users WHERE status='active' AND name LIKE 'John%';

```

The query planner might not use `idx_users_status_name` efficiently for Query 3 since the leading column `status` isn't in the `WHERE` clause of the first query. It might again choose a table scan, even if it would be able to filter down records quicker by utilizing the `idx_users_status_name`.  Query 4, however, is likely to benefit from the composite index as it filters on the leading column, then the second column. This illustrates that even if an index exists, proper usage depends on the query structure. Proper ordering of columns in a composite index is crucial, based on query patterns. If the query is not matching the composite index's leading column, it may not be used, even if the index could be used to retrieve the rows quicker then a table scan.

To mitigate situations where indexes seem slower, the primary step is to ensure that your database has up-to-date statistics. In PostgreSQL, the `ANALYZE` command gathers these statistics, allowing the planner to create more accurate cost estimations. This should be a routine practice, particularly after large data modifications, schema changes, or additions. Furthermore, frequent `ANALYZE` on tables with frequently changing distributions is highly recommended. Another practice is to avoid selecting an unnecessarily high number of columns or using a `SELECT *` statement. Retrieving all columns from the table might lead to more data retrieval overhead than only selecting the columns needed. This can become an issue when there are many columns on the table.

Additionally, pay careful attention to your query patterns and ensure that existing indexes match. Using `EXPLAIN` to analyze the generated query plans is a necessity for every query, especially when performance issues are detected. This allows you to verify which type of scan operation is chosen by the planner and to determine if indices are being used. This will allow for fine-tuning of query structure to fit index patterns or, alternatively, creation of more fitting indexes. It is imperative to understand the underlying data and the specifics of how the query planner works before applying indexing solutions.

Finally, when faced with indexing problems it is necessary to have a good understanding of basic query processing theory, especially selectivities and cardinalities as well as basic index structures. Books such as "Database System Concepts" and "Designing Data-Intensive Applications" offer in-depth knowledge on this topic. Likewise, it is very useful to read and understand the official PostgreSQL documentation for the query planner and indexing sections. These resources will build a foundation to understanding query planning decisions which can be very specific. A strong theoretical basis allows for easier debugging of index issues.
