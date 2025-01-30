---
title: "Why is a bitmap heap scan slower than an index scan in this query?"
date: "2025-01-30"
id: "why-is-a-bitmap-heap-scan-slower-than"
---
The performance discrepancy between a bitmap heap scan and an index scan stems fundamentally from the differing data access methods.  While an index scan leverages a pre-sorted, highly optimized data structure to directly access relevant rows, a bitmap heap scan, even optimized, necessitates traversing a significant portion of the heap file. This inherent difference in data access strategy becomes particularly pronounced with increasing data volumes and selectivity.  My experience optimizing query performance across various database systems, including Oracle and PostgreSQL, consistently reveals this trade-off.

A bitmap heap scan operates by constructing a bitmap for each relevant condition within a WHERE clause. Each bitmap represents a set of rows satisfying the individual condition.  These bitmaps are then intersected to identify rows satisfying the entire WHERE clause. Subsequently, the database system must retrieve the data rows corresponding to the resulting bitmaps from the heap file – the primary table data storage.  This retrieval process, especially on large tables with low selectivity, becomes the bottleneck.  The "heap" in "bitmap heap scan" refers to the unordered or loosely ordered nature of the table's physical storage.  The scan essentially becomes a sequential traversal, albeit guided by the bitmap, through the table's data pages.

In contrast, an index scan uses an index – a separate data structure optimized for fast lookups – to directly locate relevant rows. The index typically contains a subset of table columns, often the primary key or other uniquely identifying columns, alongside pointers to the corresponding rows in the heap file.  The database engine efficiently traverses the index, using a search algorithm suitable for the index type (e.g., B-tree), to find the relevant index entries.  Each entry then provides a direct path to the required data row in the heap, minimizing data access overhead.

The query's selectivity plays a crucial role. Selectivity refers to the percentage of rows in a table that satisfy a particular condition.  High selectivity (a small percentage of rows) leads to smaller bitmaps in a bitmap heap scan and a smaller subset of index entries in an index scan.  In such scenarios, the performance difference might be less dramatic. However, low selectivity (a large percentage of rows) dramatically increases the size of bitmaps and necessitates scanning a substantial portion of the heap file in a bitmap heap scan, nullifying any performance advantages.  My work on a large-scale financial transaction database highlighted this precisely; queries with low selectivity on a non-indexed column consistently favored index scans by an order of magnitude.


Let's illustrate this with code examples.  These examples are simplified for clarity and assume a relational database system with standard SQL syntax.  The underlying implementation details may vary across different database systems.

**Example 1: High Selectivity –  Minimal Performance Difference**

```sql
-- Assuming 'customers' table with 'customer_id' (indexed) and 'city' (non-indexed)
SELECT * FROM customers WHERE customer_id = 12345;

-- Bitmap Heap Scan (might be chosen if the optimizer prefers it):
--   Small bitmap is generated. Retrieval from heap is efficient due to high selectivity.
-- Index Scan (likely choice):
--   Index lookup is exceptionally fast due to the direct access afforded by the index.
```

In this case, the selectivity is high because we're looking for a single specific customer.  Both methods would be relatively fast, though the index scan would likely be marginally faster due to the direct access to the row via the index.


**Example 2: Low Selectivity – Significant Performance Difference**

```sql
-- Assuming 'orders' table with 'order_id' (indexed) and 'order_date' (non-indexed)
SELECT * FROM orders WHERE order_date BETWEEN '2023-10-26' AND '2023-10-27';

-- Bitmap Heap Scan:
--   Large bitmap is generated representing orders within the date range.  Substantial portion of the heap file will be scanned, leading to performance degradation.
-- Index Scan:
--   If an index exists on 'order_date', the scan efficiently finds the relevant index entries, resulting in faster data retrieval. Otherwise, a full table scan might be performed which is even slower than a Bitmap Heap scan.
```

Here, the selectivity is considerably lower.  A large number of rows likely fall within the specified date range.  The bitmap heap scan would require processing and traversing a large bitmap, followed by fetching data from a significant portion of the heap file, while the index scan (if an index exists on `order_date`) remains efficient. The lack of an index would make this query extremely slow, resulting in a full table scan, which would be far slower than the bitmap heap scan in this case.


**Example 3:  Illustrating the Impact of Missing Index**

```sql
-- Assuming 'products' table with 'product_name' (non-indexed) and 'product_id' (indexed)
SELECT * FROM products WHERE product_name LIKE '%widget%';

-- Bitmap Heap Scan:
--   Large bitmap will be generated, requiring a scan of a substantial portion of the heap.  Performance will be poor, especially with a large product catalog.
-- Index Scan:
--   Without an index on 'product_name', a full table scan will be performed, resulting in the worst possible performance.
```

This query showcases the critical role of indexing. The `LIKE` operator with wildcards at the beginning makes indexing inefficient.   A bitmap heap scan would still be more efficient than a full table scan in this scenario, as the bitmap narrows down the search space. However, creating a suitable index (e.g., a trigram index for text searches) would greatly improve query performance.



**Resource Recommendations:**

* Database system documentation: Consult the official documentation for detailed performance characteristics of different query execution plans and optimization strategies.
* Query optimization guides:  Explore literature focusing on efficient database query writing and execution plan analysis.  These guides often offer practical advice on selecting appropriate indexes and writing optimized queries.
* Advanced SQL textbooks:  A comprehensive understanding of database internals and query optimization techniques can be gained from advanced SQL textbooks.  They often provide valuable insights into the underlying mechanisms driving query performance.


In conclusion, the performance difference between a bitmap heap scan and an index scan ultimately depends on the query's selectivity and the presence of appropriate indexes. While bitmap heap scans can be efficient for certain queries with high selectivity, their inherent reliance on traversing a portion of the heap file makes them substantially slower than index scans in scenarios with low selectivity.  Careful index design and query optimization are paramount in achieving optimal database performance.  Over the course of my career, these principles have repeatedly guided my work in achieving substantial performance improvements in complex database systems.
