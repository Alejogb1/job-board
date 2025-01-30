---
title: "Why are single-table queries slow in MariaDB despite indexes?"
date: "2025-01-30"
id: "why-are-single-table-queries-slow-in-mariadb-despite"
---
Indexed single-table queries in MariaDB, despite theoretical efficiency, can exhibit sluggish performance due to several interconnected factors beyond simple index absence. I've observed this frequently across diverse projects, particularly when dealing with high-cardinality data and poorly understood query characteristics. The assumption that an index alone guarantees speed is often misleading; the optimizer's choices, the data's shape, and even server configuration all play significant roles.

The primary culprit is often the optimizer's suboptimal plan selection. MariaDB's query optimizer, while sophisticated, relies on statistical estimates of data distribution within tables. These statistics, derived from `ANALYZE TABLE` operations, can become outdated as data changes rapidly. When these estimations diverge significantly from reality, the optimizer may choose an inefficient access path. This is particularly prevalent in tables with skewed data distributions, where a large fraction of values might cluster in a small subset of the rows. A seemingly selective `WHERE` clause might then unintentionally trigger a full table scan if the optimizer believes the index would yield too few rows, based on stale statistics.

Further exacerbating this is the concept of index selectivity and cardinality. A highly selective index drastically reduces the number of rows the database needs to examine to satisfy a query. A low cardinality index, one containing many duplicate values, is far less effective, as the optimizer might find scanning the entire table faster than using the index. Even if an index is highly selective, the optimizer will also consider the cost of retrieving the remaining columns if those aren’t covered by the index. This phenomenon, termed ‘key lookup’ or ‘bookmark lookup,’ can drastically slow down queries if a significant portion of table rows need to be accessed based on indexed keys which then necessitate fetching the remaining columns from the base table's storage.

Additionally, the query structure itself can render indexes ineffective. Operations like using a function on the indexed column in the `WHERE` clause, or using a leading wildcard character in a `LIKE` expression (e.g., `LIKE '%value'`) prevent index usage. This results in full table scans, which scale linearly with table size, making performance suffer as the table grows. Implicit type conversions can also disrupt index use; comparing a string to an integer value using an index on the numeric column can cause the index to be bypassed, as the database engine must potentially convert all the values within the indexed column to strings to execute the comparison. Finally, hidden data conversions within complex calculations might create conditions that disallow the use of indexes, leading to a full table scan.

The server’s resource allocation can further restrict performance. If the server experiences resource contention, such as inadequate RAM allocated to the buffer pool, or high disk I/O wait times, even a well-optimized query using an index can struggle. Also, inadequate tuning of MariaDB server variables, such as those governing the query cache, can impair the benefits of index usage.

I’ll now illustrate with concrete code examples. Assume we have a `user_activity` table tracking user interactions with a website. It has columns such as `user_id` (indexed), `activity_type`, `activity_time`, and `details`.

**Example 1: Index Ignored due to Function Call**

```sql
-- This query might seem efficient due to the indexed user_id.
SELECT
    *
FROM
    user_activity
WHERE
    YEAR(activity_time) = 2023 AND user_id = 12345;
```

Here, the `YEAR()` function applied to the indexed column `activity_time` prevents the optimizer from using an index on `activity_time`. This results in a table scan. The optimizer cannot efficiently use the existing index since the transformation of each row value by the `YEAR` function means the index cannot be leveraged.

A better approach would involve restructuring the query to allow index usage:

```sql
-- Revised query to use index.
SELECT
    *
FROM
    user_activity
WHERE
    activity_time >= '2023-01-01 00:00:00' AND activity_time < '2024-01-01 00:00:00' AND user_id = 12345;
```

This revised query now allows the optimizer to directly use an index on `activity_time` for filtering the records by year. The `user_id` index would be subsequently used.

**Example 2: High Selectivity, but Inefficient due to Key Lookups**

```sql
-- High selectivity query with potential performance issues
SELECT
    activity_type,
    details
FROM
    user_activity
WHERE
    user_id = 98765;
```

While `user_id` is indexed, the query selects columns beyond those covered by the index. Assuming only the `user_id` column is indexed, retrieving `activity_type` and `details` requires the database to first locate the matching records through the index, and then perform a ‘key lookup’ for each of those rows to retrieve the remaining columns from the base table’s physical storage. If the percentage of rows matching `user_id=98765` is sufficiently high, a full table scan can actually be faster than a mix of index scan and a series of lookups. This scenario can result in a performance bottleneck for such requests on the table.

To mitigate this, an index encompassing all columns used by the query (`user_id`, `activity_type`, `details`) could be employed. Such an index can significantly reduce the overhead by covering the requested columns, thus avoiding the key lookups:

```sql
-- Example of using a composite index to cover the query
CREATE INDEX idx_user_activity_covered ON user_activity (user_id, activity_type, details);
```

**Example 3: Poor Cardinality Index Use**

```sql
-- Low Cardinality Index
SELECT
    *
FROM
    user_activity
WHERE
    activity_type = 'login';
```

If `activity_type` has limited unique values (e.g., 'login', 'logout', 'page_view'), the index on this column is of low cardinality. The optimizer might decide that using the index is not efficient because the index alone does not sufficiently reduce the result set. The optimizer would likely choose a full table scan instead due to the large number of rows that could potentially match this predicate.

In this case, if a query frequently searches by `activity_type` in conjunction with another field such as `user_id`, a composite index on both `activity_type` and `user_id` would likely provide better performance if both conditions were included in the query:

```sql
-- Example of using composite index
CREATE INDEX idx_activity_user ON user_activity (activity_type, user_id);

-- Modified query to leverage the composite index.
SELECT
    *
FROM
    user_activity
WHERE
    activity_type = 'login' AND user_id = 12345;
```

To further investigate and improve the performance of slow queries, several tools and techniques should be considered. The `EXPLAIN` statement in MariaDB provides detailed query execution plans and is invaluable in identifying performance bottlenecks. Analyzing the output of `EXPLAIN` reveals whether the query is using indexes effectively and identifies areas where optimizations are necessary. The `slow_query_log` can be enabled in the MariaDB server to identify specific queries that are taking a long time to execute. Tools like `mytop` allow for the real-time monitoring of database server performance and identify resource bottlenecks such as high I/O or resource contention that might be slowing down queries.

To maintain database performance, performing `ANALYZE TABLE` periodically, especially after large data updates, is crucial. This step ensures that the optimizer has up-to-date statistics, enabling more efficient query plans. Proper index design tailored to query patterns is critical for maximizing performance. This entails analyzing real-world query patterns instead of blindly adding indexes. Finally, server configuration tuning should be reviewed regularly to ensure optimal usage of allocated resources. Server variables which affect how quickly the cache can be accessed or whether queries will run in parallel are important to keep an eye on and can drastically increase the speed of seemingly slow queries. Proper resource allocation, monitoring and maintenance, along with careful query design will ensure that the database continues to meet application needs.
