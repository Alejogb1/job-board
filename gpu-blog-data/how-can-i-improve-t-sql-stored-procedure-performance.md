---
title: "How can I improve T-SQL stored procedure performance when using UNION or OR clauses?"
date: "2025-01-30"
id: "how-can-i-improve-t-sql-stored-procedure-performance"
---
Improving the performance of T-SQL stored procedures employing `UNION` or `OR` clauses hinges critically on understanding the underlying execution plans generated by the SQL Server query optimizer.  My experience working on large-scale data warehousing projects has repeatedly demonstrated that seemingly innocuous uses of these constructs can lead to significant performance bottlenecks if not carefully managed.  The key is to force the optimizer to utilize efficient strategies, primarily through index optimization and query restructuring.


**1. Understanding the Performance Impact**

`UNION` and `OR` operations, while conceptually simple, introduce complexities for the optimizer.  A `UNION` combines the result sets of multiple `SELECT` statements, potentially requiring distinct operations and sorting. Similarly, `OR` conditions within `WHERE` clauses can lead to full table scans if indexes are not properly utilized, resulting in a significant increase in I/O operations.  In my experience optimizing legacy systems, inefficient use of `UNION ALL` (avoiding duplicate removal) instead of `UNION` and poorly indexed tables have been the leading causes of performance issues in these scenarios.  Therefore, a thorough examination of indexes and execution plans is paramount.


**2. Optimization Strategies**

Several strategies significantly improve performance:

* **Index Optimization:** This is the most impactful change.  Ensure that indexes exist on the columns involved in `WHERE` clauses, `JOIN` conditions, and the `ORDER BY` clause (if present) for each `SELECT` statement within a `UNION`.  For example, if a `WHERE` clause uses conditions like `columnA = 'valueX' OR columnB = 'valueY'`,  separate indexes on `columnA` and `columnB` will be more efficient than a composite index covering both.  Composite indexes should be strategically crafted;  I've seen numerous instances where poorly chosen composite indexes negatively impacted performance compared to multiple single-column indexes.

* **UNION ALL vs. UNION:** Always prefer `UNION ALL` unless duplicate removal is absolutely necessary.  `UNION` performs a distinct operation, adding significant overhead compared to `UNION ALL`. This often represents a simple but highly effective optimization.

* **Query Rewriting:**  In many cases, `OR` conditions can be rewritten using `UNION ALL` or `EXISTS` to improve performance.  This requires a deep understanding of the data and relationships, but it often results in much more efficient execution plans.  I found this to be particularly relevant when dealing with large datasets and complex conditions.

* **Using EXISTS instead of OR:** If you're checking for the existence of records in another table based on certain criteria using `OR`, rewriting the query using the `EXISTS` operator is highly recommended. `EXISTS` stops searching as soon as a matching row is found, whereas `OR` conditions might evaluate numerous rows even after a match is found, leading to unnecessary I/O operations.

* **Execution Plan Analysis:** Analyzing the query execution plan is crucial. This provides insight into the optimizer's chosen strategy, highlighting potential bottlenecks like table scans, inefficient joins, or unnecessary sorts.  Using the execution plan, one can identify areas for index optimization and query restructuring.  I’ve leveraged this extensively throughout my career, often finding that seemingly minor adjustments to the query based on plan analysis significantly improve performance.


**3. Code Examples and Commentary**

**Example 1: Inefficient UNION**

```sql
-- Inefficient UNION with potential full table scans
CREATE PROCEDURE dbo.InefficientUnion (@Value INT)
AS
BEGIN
    SELECT columnA, columnB
    FROM Table1
    WHERE columnA = @Value
    UNION
    SELECT columnC, columnD
    FROM Table2
    WHERE columnB = @Value;
END;
```

This procedure is inefficient due to potential full table scans in both `Table1` and `Table2` if no indexes are present on `columnA` and `columnB` respectively.

**Example 2: Optimized UNION ALL with indexes**

```sql
-- Optimized UNION ALL with indexes
CREATE PROCEDURE dbo.OptimizedUnionAll (@Value INT)
AS
BEGIN
    CREATE NONCLUSTERED INDEX IX_Table1_columnA ON Table1 (columnA);
    CREATE NONCLUSTERED INDEX IX_Table2_columnB ON Table2 (columnB);
    SELECT columnA, columnB
    FROM Table1
    WHERE columnA = @Value
    UNION ALL
    SELECT columnC, columnD
    FROM Table2
    WHERE columnB = @Value;
END;
```

This version replaces `UNION` with `UNION ALL` and adds indexes, significantly improving performance by avoiding unnecessary sorting and leveraging index seeks.


**Example 3: Rewriting OR with EXISTS**

```sql
--Inefficient use of OR
CREATE PROCEDURE dbo.InefficientOR (@Value INT)
AS
BEGIN
    SELECT *
    FROM Table3
    WHERE columnE = @Value OR columnF = @Value;
END;

-- Optimized using EXISTS
CREATE PROCEDURE dbo.OptimizedExists (@Value INT)
AS
BEGIN
    CREATE NONCLUSTERED INDEX IX_Table3_columnE ON Table3 (columnE);
    CREATE NONCLUSTERED INDEX IX_Table3_columnF ON Table3 (columnF);

    SELECT *
    FROM Table3 T3
    WHERE EXISTS (SELECT 1 FROM Table3 WHERE columnE = @Value AND T3.PrimaryKeyColumn = Table3.PrimaryKeyColumn)
      OR EXISTS (SELECT 1 FROM Table3 WHERE columnF = @Value AND T3.PrimaryKeyColumn = Table3.PrimaryKeyColumn);
END;
```

This example demonstrates replacing `OR` in the `WHERE` clause with the `EXISTS` operator.  While seeming more complex, it's frequently more efficient for large datasets as `EXISTS` can short-circuit its evaluation once a match is found, minimizing unnecessary processing. The inclusion of a primary key ensures that only the relevant rows are considered within the `EXISTS` subqueries.


**4. Resource Recommendations**

For in-depth understanding, I strongly suggest consulting the official SQL Server documentation regarding query optimization, indexing strategies, and execution plan analysis.  Furthermore, several books dedicated to T-SQL performance tuning provide invaluable insights and practical techniques.  Finally, reviewing SQL Server Profiler traces can reveal crucial performance bottlenecks often missed through casual observation.  Understanding these resources will significantly improve your ability to fine-tune T-SQL queries involving `UNION` or `OR` clauses.
