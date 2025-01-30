---
title: "How can I optimize a MySQL query on a table with 23 million rows?"
date: "2025-01-30"
id: "how-can-i-optimize-a-mysql-query-on"
---
Optimizing queries on large tables, such as the 23-million-row table I encountered during a recent project involving a global e-commerce platform, necessitates a systematic approach.  My experience highlights that focusing solely on the query itself is insufficient; a holistic understanding encompassing table structure, indexing strategies, and query execution plans is crucial.  Ignoring any one of these aspects can lead to suboptimal performance, regardless of how well-written the SQL statement might appear.

**1.  Understanding the Bottleneck:**

Before attempting any optimization, I meticulously analyzed the query's execution plan using `EXPLAIN`. This provided insights into the query's cost, the access methods employed (e.g., index scan, full table scan), and the number of rows examined.  A full table scan on a 23-million-row table is a clear indication of inefficiency.  The key lies in identifying whether the bottleneck stems from insufficient indexing, poorly structured queries, or a combination thereof.  In my experience, ineffective indexing is the most frequent culprit.

**2. Indexing Strategies:**

Appropriate indexing is paramount.  A poorly chosen index can, counterintuitively, degrade performance.  Consider the following:

* **Composite Indexes:** For queries involving multiple `WHERE` clauses, composite indexes – indexes spanning multiple columns – are often more efficient than individual indexes on each column. The order of columns within a composite index is critical; the leading columns should be those most frequently used in `WHERE` clauses.  Improper ordering can lead to an index scan covering many more rows than necessary.

* **Index Cardinality:**  High cardinality is desirable. An index with low cardinality (few unique values) provides minimal benefit, possibly resulting in a table scan being preferred by the query optimizer.  For example, an index on a boolean column would likely be less useful than an index on a column containing unique identifiers.

* **Prefix Indexes:** For text-based columns, prefix indexes can be very beneficial.  They index only the first `n` characters of a string, allowing for efficient searches on the leading portion of the string while reducing index size.  This is particularly useful for columns with long text values where full-text indexing isn't required.

* **Fulltext Indexes:** For complex text searches, leveraging MySQL's full-text indexing capabilities is essential.  This provides optimized searching capabilities for natural language queries.


**3. Code Examples and Commentary:**

Let's assume a table named `products` with columns `product_id` (INT, primary key), `category_id` (INT), `product_name` (VARCHAR(255)), `price` (DECIMAL), and `description` (TEXT).

**Example 1: Inefficient Query and Optimization:**

```sql
-- Inefficient Query: Full table scan likely
SELECT * FROM products WHERE category_id = 123 AND price > 100;
```

This query, without an appropriate index, will likely result in a full table scan.  The following optimized query addresses this:

```sql
-- Optimized Query: Utilizes composite index
CREATE INDEX idx_category_price ON products (category_id, price);
SELECT * FROM products WHERE category_id = 123 AND price > 100;
```

The composite index `idx_category_price` enables the query optimizer to efficiently locate the relevant rows without scanning the entire table. The order of `category_id` and `price` reflects typical query patterns where category filtering precedes price filtering.


**Example 2: Leveraging Fulltext Index:**

```sql
-- Inefficient search on description column
SELECT * FROM products WHERE description LIKE '%widget%';
```

This query is highly inefficient for large datasets. A full-text index dramatically improves performance:

```sql
-- Creating a Fulltext Index
ALTER TABLE products ADD FULLTEXT INDEX idx_description (description);

-- Optimized Query using MATCH AGAINST
SELECT * FROM products WHERE MATCH (description) AGAINST ('widget' IN BOOLEAN MODE);
```

This uses MySQL's full-text search capabilities, offering substantial performance gains compared to the `LIKE` operator.  The `BOOLEAN MODE` allows for more sophisticated search criteria.


**Example 3: Optimizing `ORDER BY` and `LIMIT` Clauses:**

```sql
-- Inefficient query with ORDER BY and LIMIT
SELECT product_name, price FROM products ORDER BY price DESC LIMIT 10;
```

While seemingly simple, this query can be slow without an index on the `price` column.  Adding an index significantly improves performance:

```sql
-- Optimized Query: Index on price column
CREATE INDEX idx_price ON products (price);
SELECT product_name, price FROM products ORDER BY price DESC LIMIT 10;
```

This uses the `idx_price` index to efficiently retrieve the top 10 products ordered by price.  The `LIMIT` clause further refines the result set, avoiding the need to sort the entire table.



**4.  Beyond Indexing:**

Effective indexing is crucial, but other optimization strategies also apply:

* **Query Rewriting:** Carefully examine the query's logic.  Sometimes, rewriting a query can dramatically improve its performance.  For instance, subqueries can sometimes be replaced with joins.

* **Database Normalization:** Ensure that the database schema is properly normalized to reduce data redundancy and improve data integrity. This impacts query performance indirectly.

* **Hardware Resources:** Assess whether the server’s hardware resources (CPU, RAM, disk I/O) are sufficient to handle the workload.  Increasing RAM or upgrading to faster storage can often yield considerable performance improvements.

* **Query Caching:** MySQL's query cache can improve performance for frequently executed queries.  However, its effectiveness can vary based on the query's complexity and data volatility.  Its use has declined with the introduction of InnoDB's buffer pool improvements.


**5. Resource Recommendations:**

For further exploration, I recommend consulting the official MySQL documentation, specifically the sections on query optimization and indexing. Additionally, books on database performance tuning and SQL optimization offer valuable insights and practical techniques.  Thorough study of query execution plans, obtained using `EXPLAIN`, is essential.  Finally, profiling tools can pinpoint specific performance bottlenecks within complex queries.  These resources, combined with a methodical approach to identifying and resolving performance issues, will significantly enhance query performance on large datasets.
