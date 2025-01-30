---
title: "Why is a DESC ordered query slow in MariaDB?"
date: "2025-01-30"
id: "why-is-a-desc-ordered-query-slow-in"
---
The performance degradation observed in `DESC` ordered queries in MariaDB, especially on large tables, often stems from the lack of an index that directly supports the reverse order. Standard B-tree indexes, the workhorses of most databases, are inherently ordered in ascending sequence. Therefore, when a `DESC` order is requested, the database generally has two options: either scan the entire table and then sort the resulting data, which can be very inefficient, or leverage an existing index, potentially in an indirect fashion which still carries overhead. Neither of these methods are as performant as a direct index lookup.

The core issue isn’t that MariaDB is inherently “bad” at reverse ordering. It is the fundamental architecture of B-tree indexes and the optimization strategies employed. The most performant way to satisfy an ordered query is by traversing a pre-sorted index. If the index is ordered in ascending fashion (as is the default), retrieval of results in descending order requires additional steps, typically a filesort operation. This involves extracting the data from the table based on the index, storing it in a temporary file (or memory, if feasible), and then sorting that result set before returning the data to the client. This sort process, if large, is inherently a resource intensive task, especially when the data set cannot be fully accommodated in RAM.

I encountered this exact problem while optimizing a reporting system for a large e-commerce platform. We had a `transactions` table containing millions of entries, and users frequently needed to retrieve transactions ordered by date in descending order for recent activity reports. The initial implementation used a naive query like `SELECT * FROM transactions ORDER BY transaction_date DESC`. Performance was abysmal; queries took several seconds, often timing out. The execution plan showed a full table scan followed by a filesort, confirming the absence of suitable index.

To illustrate, consider a simplified version of the problem. We can create a table called `events` with a timestamp and some sample data:

```sql
CREATE TABLE events (
  id INT AUTO_INCREMENT PRIMARY KEY,
  event_time TIMESTAMP,
  event_type VARCHAR(50)
);

INSERT INTO events (event_time, event_type) VALUES
  (NOW() - INTERVAL 1 HOUR, 'login'),
  (NOW() - INTERVAL 30 MINUTE, 'logout'),
  (NOW() - INTERVAL 10 MINUTE, 'purchase'),
  (NOW() - INTERVAL 5 MINUTE, 'review'),
  (NOW(), 'comment');
```

A query like:

```sql
SELECT * FROM events ORDER BY event_time DESC;
```

Would result in a filesort if there is no index on `event_time` or if there is only a ascending index on this column. The database would likely examine each row, construct a temporary set, then sort by the timestamp, and finally return the requested data.

The first approach to mitigate this involves adding an index on `event_time`, but a standard index still won't directly address the problem:

```sql
CREATE INDEX idx_event_time ON events (event_time);
```

This index, while helpful for filtering and ascending order lookups, will not directly facilitate descending order with optimal performance. The database might use it for retrieval, but then still needs to sort to satisfy the `DESC` requirement.

A better approach, where MariaDB version 10.3.0 and above is used, would be to leverage "descending indexes".  We can create a descending index like this:

```sql
CREATE INDEX idx_event_time_desc ON events (event_time DESC);
```

This index explicitly stores the `event_time` column in descending order, so a query like `SELECT * FROM events ORDER BY event_time DESC` can directly retrieve the rows from this index without the need for any sorting operation, dramatically improving performance. This improvement is especially noticeable with large amounts of data. The execution plan will show the index is used directly for the ordering, removing the filesort.

There are limitations, of course. Descending indexes consume additional disk space. However, in many read-heavy scenarios, this tradeoff is often worthwhile for the performance improvement it brings. It is important to note that descending indexes are not supported in all versions of MariaDB or MySQL, and the exact behavior and optimization possibilities can vary. Also, consider the indexing strategy regarding other clauses in the query like WHERE conditions. A proper compound index could provide even better performance.

Another less direct approach, when descending indexes are unavailable for example or if you must maintain older MariaDB versions,  can sometimes utilize a composite index in a clever way, particularly when the primary key or some other strictly increasing value is available. Consider a `products` table with an auto-incrementing `id` and a `creation_date` column:

```sql
CREATE TABLE products (
    id INT AUTO_INCREMENT PRIMARY KEY,
    product_name VARCHAR(255),
    creation_date TIMESTAMP
);

INSERT INTO products (product_name, creation_date) VALUES
('Laptop', '2023-01-15 10:00:00'),
('Keyboard', '2023-02-20 14:30:00'),
('Mouse', '2023-03-25 16:00:00'),
('Monitor', '2023-04-01 09:00:00');
```

If the primary key (`id`) is auto-incrementing and therefore correlated with insert time (newer records have higher IDs, in a majority of scenarios, even if deletion and insertion occurs in parallel), an index on a combination of columns can be used effectively to satisfy a descending order, albeit in an indirect way:

```sql
CREATE INDEX idx_creation_id_desc ON products (id DESC, creation_date);
```

Then, instead of directly ordering by date DESC we can use the auto-incrementing ID as a proxy like this:

```sql
SELECT * FROM products ORDER BY id DESC, creation_date DESC;
```

This strategy works on the assumption the auto-increment id is a good enough proxy. The result is still ordered by creation_date but it will probably leverage the new index and skip a file sort, improving performance. It is not as efficient as a direct index but it does offer an alternative in particular older environments. The primary sort is done by `id DESC` which can be quickly retrieved from the index. The secondary sort is then by `creation_date` but since the dataset will already be significantly reduced this operation is less expensive.

In conclusion, the perceived slowness of `DESC` ordered queries in MariaDB typically arises from the necessity to perform a filesort when no suitable descending index exists. Creating descending indexes is the preferred method of addressing the issue on MariaDB 10.3.0 or higher.  If direct descending indexes are not available, careful crafting of composite indexes that leverage other columns or a proxy column (if available) can improve performance significantly. It is critical to analyze the query execution plans and consider data characteristics to apply the correct optimization strategy. Finally, regular database performance monitoring and profiling is advised. The official MariaDB documentation is a key resource, especially the sections covering indexing and query optimization. Additionally, books focusing on MariaDB database administration and performance tuning can provide a more in depth knowledge.  Practical experience with specific databases and their internals, as well as experimentation, are the most invaluable tools for resolving performance bottlenecks.
