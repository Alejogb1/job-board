---
title: "Why are PostgreSQL query row estimations inaccurate?"
date: "2025-01-30"
id: "why-are-postgresql-query-row-estimations-inaccurate"
---
PostgreSQL's query planner relies heavily on statistics gathered about the data within tables.  The accuracy of its row estimations, therefore, hinges directly on the quality and currency of these statistics.  In my experience troubleshooting performance issues across numerous large-scale deployments,  I've observed that inaccurate estimations stem primarily from outdated or insufficient statistics, coupled with the inherent complexities of query optimization in a relational database management system.

**1. Explanation of Inaccurate Row Estimations:**

The query planner employs cost-based optimization. It estimates the cost of various execution plans based on the predicted number of rows processed at each step.  These row counts are derived from table statistics maintained by PostgreSQL.  These statistics include the number of rows in a table, the distribution of values in indexed columns (histograms), and the null fraction for each column.  When these statistics are stale or incomplete, the planner's estimations become unreliable.

Several factors contribute to the staleness or incompleteness of these statistics:

* **Infrequent `ANALYZE` calls:**  The `ANALYZE` command updates table statistics.  If `ANALYZE` is not executed frequently enough (especially after significant data modifications, such as large inserts, updates, or deletes), the statistics will become outdated, leading to inaccurate row estimations.  This is particularly problematic with frequently updated tables.

* **Insufficient sampling:** The `ANALYZE` command, by default, samples a portion of the table data. This sampling might not accurately represent the data distribution, particularly if the data is highly skewed or contains outliers.  A smaller sample size will likely lead to poorer estimations.

* **Complex queries:**  Queries involving joins, subqueries, and complex WHERE clauses can amplify the impact of even minor inaccuracies in individual table estimations. The planner combines estimations from different parts of the query, and errors can accumulate.

* **Data distribution changes:**  If the distribution of data within a table significantly changes over time (e.g., a sudden influx of data with different characteristics), the existing statistics quickly become irrelevant, even if recently updated.

* **Lack of indexes or inappropriate indexing:**  Missing indexes or indexes on the wrong columns significantly affect the planner's ability to accurately estimate the number of rows that will satisfy a `WHERE` clause.  A poor index choice can lead to a full table scan, resulting in a dramatic overestimation (or, less commonly, underestimation) of rows.

* **Limitations of statistical models:**  The statistical models used by the planner have inherent limitations.  They make assumptions about data distributions that might not always hold true in reality, especially with complex data distributions.



**2. Code Examples with Commentary:**

**Example 1: Impact of outdated statistics:**

```sql
-- Create a table and populate it.
CREATE TABLE my_table (id SERIAL PRIMARY KEY, value INTEGER);
INSERT INTO my_table (value) SELECT generate_series(1, 1000000);
ANALYZE my_table;

-- Query to estimate rows.
EXPLAIN SELECT * FROM my_table WHERE value > 900000;

-- Perform a large UPDATE, making statistics outdated.
UPDATE my_table SET value = value + 1000000 WHERE value > 500000;

-- Query again – note the inaccurate estimation.
EXPLAIN SELECT * FROM my_table WHERE value > 1900000;
```

*Commentary*: The first `EXPLAIN` will yield a reasonably accurate row estimate. However, the `UPDATE` significantly alters the data distribution without updating the statistics.  The second `EXPLAIN` will likely show a significantly inaccurate estimation because the planner is using obsolete statistics.  Running `ANALYZE my_table;` after the update would correct this.


**Example 2: Impact of insufficient sampling:**

```sql
-- Create a table with a skewed distribution.
CREATE TABLE skewed_table (id SERIAL PRIMARY KEY, value INTEGER);
INSERT INTO skewed_table (value) SELECT 1 FROM generate_series(1, 1000);
INSERT INTO skewed_table (value) SELECT 2 FROM generate_series(1, 10);

-- Analyze the table with a very small sample fraction (for demonstration).
ANALYZE skewed_table WITH (n_top=10);

-- Query that will likely be estimated inaccurately.
EXPLAIN SELECT * FROM skewed_table WHERE value = 2;
```

*Commentary*:  Using a `n_top` of 10 in `ANALYZE` forces a very small sample. This likely leads to an inaccurate estimation of the rows where `value = 2` because the sample might not include any rows with that value.  A larger sample or a full analysis (`ANALYZE skewed_table;`) would produce a more accurate estimate.


**Example 3: Impact of missing indexes:**

```sql
-- Create a table without an index.
CREATE TABLE unindexed_table (id SERIAL PRIMARY KEY, name VARCHAR(255));
INSERT INTO unindexed_table (name) SELECT 'name_' || generate_series(1, 1000000) FROM generate_series(1, 1000000);

-- Query that will perform a full table scan.
EXPLAIN SELECT * FROM unindexed_table WHERE name LIKE 'name_50000%';

-- Create an index and re-analyze.
CREATE INDEX name_idx ON unindexed_table (name);
ANALYZE unindexed_table;

-- Query again – note the improved estimation.
EXPLAIN SELECT * FROM unindexed_table WHERE name LIKE 'name_50000%';
```

*Commentary*: The first `EXPLAIN` will show a full table scan, leading to a potentially highly inaccurate (and overly pessimistic) estimation of the number of matching rows. Creating an index on the `name` column and re-analyzing the table allows the planner to use the index for the `LIKE` clause, resulting in a far more accurate estimate in the second `EXPLAIN`.


**3. Resource Recommendations:**

The official PostgreSQL documentation provides comprehensive details on query planning and statistics management.  Consulting advanced texts on database internals and performance optimization will offer further insights into the complexities of query optimization and statistical modeling.  Understanding histogram construction within PostgreSQL is crucial to grasp the limits of the planner's accuracy.  Finally, actively monitoring query execution plans using `EXPLAIN` and `EXPLAIN ANALYZE` is paramount for identifying and addressing inaccurate estimations in real-world scenarios.  By combining these resources and practical experience, one can significantly enhance their ability to diagnose and resolve issues related to PostgreSQL query performance and estimation accuracy.
