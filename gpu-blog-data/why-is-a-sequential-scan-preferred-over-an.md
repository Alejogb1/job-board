---
title: "Why is a sequential scan preferred over an index scan with a more restrictive predicate?"
date: "2025-01-30"
id: "why-is-a-sequential-scan-preferred-over-an"
---
The efficacy of sequential versus index scans hinges critically on data distribution and the nature of the predicate.  My experience optimizing query performance across several large-scale data warehousing projects has consistently shown that while index scans are generally faster for point lookups and equality predicates, they can become significantly less efficient than sequential scans when dealing with highly selective predicates on non-indexed columns, or when the data distribution renders index traversal less advantageous than a linear read. This isn't a simple "always use one over the other" situation; it demands a careful analysis of the specifics.

**1.  Clear Explanation:**

An index scan leverages an index structure – typically a B-tree or similar – to quickly locate rows matching a given predicate.  The index contains a subset of the table's columns, allowing for direct access to the data pages containing the relevant rows without examining every row. This is highly efficient when the predicate directly targets an indexed column and involves equality or range comparisons.  However, the index scan's performance degrades when the predicate is complex, involving multiple columns, many conditions combined with logical AND/OR operators, or when the selectivity of the predicate is very high (i.e., it returns a very small subset of the table).  In such cases, the overhead of traversing the index and then fetching the corresponding rows from the table can outweigh the benefits.

A sequential scan, conversely, reads the data pages of the table linearly from the beginning to the end.  While seemingly inefficient, it becomes preferable under specific circumstances. Firstly, when the predicate involves columns not included in any index, an index scan is simply not feasible; a sequential scan is the only option.  Secondly, if the predicate is highly selective, resulting in a small fraction of rows being returned, a sequential scan might be faster. The reason is that the cost of navigating the index, potentially multiple index structures for complex predicates, could exceed the cost of reading a small portion of the table's data pages.  Thirdly, in some database systems, the architecture of data storage and page access might favor sequential reads, particularly in specific hardware configurations or when dealing with massive tables where the cost of random I/O during index traversal significantly increases response time. I've encountered such situations on enterprise-grade systems where optimized sequential access surpassed indexed lookups for unusually selective queries, despite seemingly defying conventional wisdom.


**2. Code Examples with Commentary:**

Let's illustrate with three examples using a simplified SQL syntax. Assume a table named `customer_orders` with columns `order_id` (INT, primary key, indexed), `customer_id` (INT, indexed), `order_date` (DATE), `total_amount` (DECIMAL), and `product_category` (VARCHAR).


**Example 1: Index Scan is Efficient**

```sql
SELECT * FROM customer_orders WHERE customer_id = 123;
```

This query benefits greatly from an index scan.  The `customer_id` column is indexed, and the predicate is a simple equality comparison. The database optimizer will almost certainly choose an index scan, rapidly locating the relevant rows using the `customer_id` index.


**Example 2: Sequential Scan Might Be Preferable**

```sql
SELECT * FROM customer_orders WHERE order_date BETWEEN '2024-01-01' AND '2024-01-31' AND total_amount > 1000 AND product_category = 'Electronics';
```

This query involves multiple predicates on non-indexed or less-frequently used columns (`total_amount`, `product_category`). Even if indices exist on individual columns, combining them in the predicate can lead to complex index lookups, potentially resulting in a large number of I/O operations.  A sequential scan, which directly reads a subset of the data pages representing orders in January 2024, might prove more efficient, especially if the conditions make the resulting dataset a small percentage of the total.  The optimizer will evaluate the cost of each approach based on statistics collected on the table and index structures.


**Example 3:  Sequential Scan is Necessary**

```sql
SELECT * FROM customer_orders WHERE DAYOFWEEK(order_date) = 1 AND total_amount > 500;
```

Here, `DAYOFWEEK` is a function applied to the `order_date` column.  Unless a functional index exists (a less common scenario), the database system cannot directly use an index to locate rows meeting this condition. A sequential scan, filtering rows based on the function's result, becomes the only viable option.


**3. Resource Recommendations:**

For a deeper understanding of query optimization and database internals, I recommend consulting comprehensive texts on database management systems.  Specifically, focus on chapters dealing with query processing, query optimization algorithms (including cost-based optimization), index structures, and storage management.  A thorough grounding in data structures and algorithms is also indispensable.  Moreover, practical experience working with query profiling tools and analyzing execution plans is crucial for making informed choices regarding sequential versus index scans in real-world scenarios.  Pay close attention to the optimizer's choices; its analysis frequently reveals the underlying performance bottlenecks. Finally, consult the documentation for your specific database system; each has its own peculiarities in query optimization.
