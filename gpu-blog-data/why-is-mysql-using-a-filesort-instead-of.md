---
title: "Why is MySQL using a filesort instead of an index for ORDER BY queries?"
date: "2025-01-30"
id: "why-is-mysql-using-a-filesort-instead-of"
---
A frequently encountered performance bottleneck in MySQL stems from its query optimizer choosing a filesort operation rather than leveraging an available index for `ORDER BY` clauses. This decision isn't arbitrary; it arises from specific circumstances surrounding index structure, query specifics, and data distribution. My experience optimizing countless SQL queries indicates that understanding these nuances is critical for ensuring efficient data retrieval.

The core issue lies in how MySQL evaluates the potential cost of different execution plans. When a query includes an `ORDER BY` clause, MySQL ideally seeks an index that can provide the requested sort order without needing additional sorting operations. An index inherently stores data in a particular order, based on the indexed columns. Therefore, if the index matches the `ORDER BY` clause *and* can also be used to access the rows identified in the `WHERE` clause (if present), it's the fastest path. However, several factors can make a filesort the more appealing, albeit less performant, option.

Firstly, the most direct cause is a lack of a suitable index. If no existing index has its leading columns matching the columns specified in the `ORDER BY` clause, a filesort is often inevitable. The optimizer needs to retrieve the rows from the underlying table (or via another index) and then perform a separate sorting operation. This sorting can happen in memory (if the result set is small enough) or involve temporary disk storage, impacting query speed significantly. An index on `(column_a, column_b)` can satisfy `ORDER BY column_a, column_b` or `ORDER BY column_a` effectively, but *not* `ORDER BY column_b`.

Secondly, the `SELECT` clause itself plays a crucial role. If the `SELECT` clause retrieves a large number of columns, and these columns are not part of the index used for ordering (meaning the index is not covering), MySQL often finds it more efficient to scan a smaller index using the `WHERE` clause if one is present, and then retrieve the remaining columns during a filesort, as opposed to using a larger, slower index for both filtering and ordering. The query planner aims to minimize disk I/O. The rationale here is that fetching all rows via a clustered index and then sorting can actually be faster than doing lookups in a non-covering index for a selection of rows, and then going back to clustered index to retrieve remaining columns. MySQL will choose the least overall work.

Thirdly, even with a suitable index, a filesort might be chosen due to a combination of `WHERE` clause filtering and the size of the result set. If the `WHERE` clause drastically reduces the number of rows, the cost of applying that filter may outweigh the benefit of using the ordering index early. MySQL might choose to retrieve the filtered set of records first and then sort, especially if the `WHERE` clause involves a range query with low selectivity. Furthermore, if using index forces retrieval of every row matching `WHERE`, a full table scan with file sort can be more efficient for low selectivity queries. The query optimizer uses statistics to estimate these costs.

Let's illustrate with a few practical scenarios:

**Example 1: Lack of Matching Index**

Consider a `users` table with columns `id`, `name`, `email`, and `creation_date`. If I were to run:

```sql
SELECT id, name, email FROM users ORDER BY name;
```

And no index existed on the `name` column, then a filesort is highly likely. MySQL needs to scan all rows, extract these three columns, and then order them by name using a sorting algorithm. To mitigate this:

```sql
CREATE INDEX idx_users_name ON users (name);
```

This index allows MySQL to read the records directly in the requested `name` order, eliminating the filesort. However, this does not make it a covering index; if the query had been `SELECT * FROM users ORDER BY name`, then still would need to go to the table to retrieve other columns. This index would be beneficial even if the query had a `WHERE` clause, because the query is more likely to use the index on `name` in that case for ordering even if the filtering is done using different column.

**Example 2: Non-Covering Index and Large Data Selection**

Assume a more complex query and an existing index:

```sql
SELECT id, name, email, creation_date, last_login FROM users WHERE creation_date > '2023-01-01' ORDER BY name;
```

Suppose we have an index on `(creation_date, name)`. While the `creation_date` part of the index is useful for filtering, the index cannot provide a direct ordering on `name` for the result set selected by `creation_date > '2023-01-01'`. If the selectivity of this `WHERE` condition is low enough (i.e., many rows match `creation_date > '2023-01-01'`) and the `SELECT` list includes several columns not present in the index, MySQL might opt for a full table scan filtered by the `creation_date` condition, followed by a filesort, instead of using the `(creation_date, name)` index and additional table lookups to retrieve rest of columns. The plan is influenced by cost estimations based on table statistics. A potential solution is a covering index:

```sql
CREATE INDEX idx_users_creation_name ON users (creation_date, name, id, email, last_login);
```

This covering index (now also including `id, email, last_login`), potentially allowing MySQL to avoid a second scan to retrieve other columns. When using a covering index like this, MySQL is able to fetch all the data only from the index, which is stored in a pre-sorted manner. It will also use a range scan over `creation_date` if it determines that it is beneficial based on the data statistics.

**Example 3: Range Query and File Sort**

Consider a query with a low selectivity range query and ordering on a different column.

```sql
SELECT id, name, email FROM users WHERE id > 100000 ORDER BY name;
```

Even with an index on `(id, name)`, if the majority of the rows satisfy `id > 100000`, the cost of using that index and then doing the table lookup might be higher than just doing a full table scan and sorting. In this case, MySQL might just retrieve all rows, then filter based on `id` and perform sorting. There isn't an easy fix using indexing alone in this scenario. Consider that for this example, adding covering index like in the previous example might not help due to the low selectivity in `id`. Often, this means restructuring the application to avoid this kind of query.

In conclusion, MySQL's preference for filesort over an index-based `ORDER BY` is rooted in cost analysis of various execution strategies. Understanding how indices are used, the impact of non-covering indices, and the interplay of the `WHERE` and `SELECT` clauses is crucial for effective performance optimization. To gain a deeper understanding, research MySQL documentation on the query optimizer, indexing best practices and `EXPLAIN` statement outputs. Also, explore articles on how table statistics influence query planning. Understanding these will improve your ability to troubleshoot and optimize these types of query bottlenecks.
