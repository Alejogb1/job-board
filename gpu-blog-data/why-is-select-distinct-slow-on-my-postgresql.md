---
title: "Why is SELECT DISTINCT slow on my PostgreSQL table?"
date: "2025-01-30"
id: "why-is-select-distinct-slow-on-my-postgresql"
---
The performance bottleneck frequently encountered with `SELECT DISTINCT` queries on PostgreSQL stems from the underlying operation required to identify and eliminate duplicate rows: a sorting or hashing process applied to the result set. This process, while logically straightforward, can become computationally intensive, especially with large datasets, broad column selections, and the absence of suitable indexes.

To elaborate, when a `SELECT DISTINCT` clause is included, PostgreSQL essentially needs to compare every row returned by the initial query against every other row to determine uniqueness. There are two primary strategies the query planner will employ to achieve this: sorting or hashing. The choice of method is based on a variety of factors, including the estimated size of the result set, the available memory, and the complexity of the data types.

The sort-based approach involves sorting all the returned rows according to the selected columns. Once sorted, duplicate rows become adjacent, allowing for a relatively efficient scan to remove them. This approach has a time complexity of approximately O(N log N), where N is the number of rows. While this is generally acceptable for smaller tables, the sorting operation’s overhead becomes significant as N increases. Consider, further, that this sorting is typically conducted in memory or, failing that, on disk which adds additional I/O overhead.

Alternatively, the hash-based method uses a hash function to generate a numerical representation of each row based on selected column values. Rows with identical hashes are then presumed (with a small possibility of collision) to be duplicates. This method can often achieve a time complexity of approximately O(N), which is more favorable than sorting. The performance of hashing, however, degrades when the number of unique hashes approaches the available hash table size, resulting in excessive collisions and a slowdown. Hash calculations also require CPU time, which is non-trivial for complex data.

The efficiency of either method is dramatically impacted by the scope of the columns included in the `SELECT DISTINCT` clause. When numerous columns or large textual fields are involved, both sorting and hashing become more resource-intensive due to the larger data that needs to be manipulated. Furthermore, if the chosen column set is not well-indexed or not part of a composite index, the initial query might incur additional time searching for the result set. Let's examine code examples to highlight these concepts.

**Code Example 1: Wide Column Selection**

Consider a `users` table with columns like `id`, `username`, `email`, `first_name`, `last_name`, `address`, `registration_date`, and `last_login`. Executing the following:

```sql
SELECT DISTINCT
    username,
    email,
    first_name,
    last_name,
    address
FROM
    users;
```
This query selects five relatively large columns. Due to their size and the volume of the data, the database is forced to either sort or hash significant amounts of data, resulting in considerable performance degradation, even if there are few duplicates. The database engine needs to read, process, and compare the entirety of these selected fields for each row, making it costly in terms of computation and I/O operations. Without an index specifically covering these columns, the initial read of the table data is also slow.

**Code Example 2: Narrow Column Selection with Index**

Let’s modify the query to select a much narrower column, assuming we have an index created on the `username` column:

```sql
CREATE INDEX idx_users_username ON users (username);
SELECT DISTINCT username FROM users;
```
In this case, the database can leverage the `idx_users_username` index. Since the index contains a sorted list of usernames, PostgreSQL can efficiently scan the index data, eliminating duplicates directly during index scanning without accessing the full table or performing a large sort. This yields a substantial performance improvement compared to Example 1. The crucial point is not merely selecting fewer columns, but selecting indexed columns.

**Code Example 3: Query with Filter and Index**

This example demonstrates the combination of a filter and an index:

```sql
CREATE INDEX idx_users_username_regdate ON users (username, registration_date);

SELECT DISTINCT username
FROM users
WHERE registration_date > '2023-01-01';
```
Here, we've created a composite index on both `username` and `registration_date`. The database is able to filter the table based on `registration_date`, and then use the index for the `username` column in finding unique values. This can be a highly optimized scenario as it constrains the dataset processed by `DISTINCT` to only relevant rows, and can leverage index efficiency. However, it is critical the index ordering and filtering clause alignment are advantageous to the query execution plan. If the filter is highly selective, this can dramatically reduce the initial dataset size which reduces the amount of work required by `DISTINCT`.

In summary, a slow `SELECT DISTINCT` operation in PostgreSQL typically arises from a combination of factors, including broad column selections, a lack of appropriate indexing, and large table sizes. By understanding these underlying mechanisms and designing tables and queries with performance in mind, we can avoid these bottlenecks.

To mitigate these issues, consider the following steps. First, evaluate the actual needs for the columns included in your `DISTINCT` clause; selecting unnecessary columns introduces significant overhead. Second, analyze your query execution plans using `EXPLAIN` to understand whether a sorting operation or a hashing strategy is being employed and if indexes are being effectively used. Third, focus on appropriate indexing. Consider creating indexes on frequently used columns in your `SELECT DISTINCT` clauses, preferably with the order of columns reflecting query filters. Remember that a single-column index, while helpful, may not be sufficient, and a compound index might be more effective when you often filter on additional columns. Lastly, consider materialized views or pre-aggregated tables if you need the same distinct set frequently. The initial overhead of these data structures can pay off with optimized lookup performance for subsequent queries. Further reading on query optimization techniques and indexing strategies in PostgreSQL will provide even more detailed insights. In particular, investigate the effects of different indexing types such as B-trees and hash indexes, along with partial indexes that target only a portion of data relevant for the queries. Reviewing Postgres documentation on query planning and execution should give additional depth to these concepts and strategies.
