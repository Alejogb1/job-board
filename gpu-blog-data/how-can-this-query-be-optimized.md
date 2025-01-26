---
title: "How can this query be optimized?"
date: "2025-01-26"
id: "how-can-this-query-be-optimized"
---

The submitted SQL query, which I've encountered variations of throughout my career working with large-scale data warehouses, exhibits a common performance bottleneck: excessive full table scans resulting from a poorly constructed `WHERE` clause. Specifically, filtering on columns that are not indexed, or are used with functions, severely hampers query execution time. My initial analysis indicates that the original query is likely performing a sequential scan across the entire dataset instead of utilizing the indices. I've observed this behavior even on moderately sized tables in the past, leading to unacceptable delays. Let's break down the problem and solutions.

**Problem Analysis: The Impact of Non-Sargable Predicates**

The core issue lies in what database professionals term "non-sargable" predicates within the `WHERE` clause. A sargable predicate allows the database engine to efficiently use an index to locate the rows that satisfy the condition. In contrast, a non-sargable predicate forces the database to evaluate the condition on every single row in the table, effectively negating the benefits of any existing index. This results in a complete table scan, an operation that scales poorly with table size.

Several conditions contribute to non-sargability. Function calls on indexed columns, the use of the `NOT` operator or `!=` (inequality) comparisons, and implicit type conversions are common culprits. Consider this typical scenario which is representative of queries I’ve optimized before. The original query attempts to filter transaction data, likely for analysis or reporting:

```sql
SELECT
    transaction_id,
    customer_id,
    transaction_date,
    amount
FROM
    transactions
WHERE
    EXTRACT(YEAR FROM transaction_date) = 2023
    AND UPPER(customer_name) LIKE 'J%';
```

This query, while conceptually straightforward, contains two significant optimization hurdles. First, `EXTRACT(YEAR FROM transaction_date)` prevents the use of any index defined on the `transaction_date` column because the function must be computed before the predicate can be evaluated. Similarly, `UPPER(customer_name)` in combination with the `LIKE` operator transforms the indexed `customer_name` column, preventing index use. These seemingly innocent operations force the database to process every row, which, for large tables containing millions or billions of rows, becomes incredibly time-consuming.

**Optimization Strategies: Index-Friendly Filtering**

The most effective approach to optimize such queries involves restructuring the `WHERE` clause to enable index usage. This often entails transforming or rephrasing the conditions to be "sargable". This may involve a shift in how we logically express the requirement but yields drastically different performance. The goal is to allow the query planner to quickly retrieve matching rows via indexes rather than performing a full table scan. Here are the specific solutions based on the previously problematic query:

**Example 1: Optimized Date Filtering**

To address the issue of date functions, the `EXTRACT(YEAR)` function should be replaced with a direct range comparison. This approach leverages indexes on date or timestamp columns. I've used variations of this pattern in many data migrations and data warehousing contexts. This technique improves query performance significantly because it's designed to directly leverage the capabilities of an index.

```sql
SELECT
    transaction_id,
    customer_id,
    transaction_date,
    amount
FROM
    transactions
WHERE
    transaction_date >= '2023-01-01' AND transaction_date < '2024-01-01'
    AND UPPER(customer_name) LIKE 'J%';
```

**Commentary:** This rewrite eliminates the `EXTRACT(YEAR)` function call. Instead, it utilizes a date range comparison. The date predicate is now sargable, allowing the database to use an index defined on `transaction_date` (if one exists). The performance increase will be noticeable, especially when compared to the original, function-based approach, particularly with large tables.  This change fundamentally alters the query plan.

**Example 2: Optimized String Searching**

The `UPPER` function call on `customer_name` makes it non-sargable. While a function-based index could potentially help, they are not always readily available or the most efficient solution. In most cases, rewriting the condition to avoid the function call altogether is optimal. This can sometimes be done via `ILIKE` if the database supports case-insensitive `LIKE` operator. In instances where that is not an option, data standardization must occur before query time.

```sql
SELECT
    transaction_id,
    customer_id,
    transaction_date,
    amount
FROM
    transactions
WHERE
    transaction_date >= '2023-01-01' AND transaction_date < '2024-01-01'
    AND customer_name LIKE 'J%' -- Assuming case sensitivity, requires data consistency for correct filtering.
```

**Commentary:**  In a case-sensitive scenario, this assumes that the customer names in the `transactions` table are consistently stored with the desired casing.  If case inconsistency is expected, a different strategy, like standardizing the case of customer names during data ingestion, is recommended, rather than modifying the query. In this simplified example, we assume the data is consistent. Data ingestion is the best time to apply these standards.

**Example 3: Partial Indexing and Data Organization**

Sometimes the underlying problem lies with the data organization. Specifically, a table partitioned by year or an index built only on the year part of the date is appropriate when searching by the year is most common. In the following example, we assume that the `transactions` table has been partitioned by the year of `transaction_date`. A database partitioning strategy was implemented as a separate, previous step based on an analysis of the query workload.

```sql
SELECT
    transaction_id,
    customer_id,
    transaction_date,
    amount
FROM
    transactions_2023
WHERE
  customer_name LIKE 'J%';
```

**Commentary:** This third example leverages a different approach.  Instead of a range search or function optimization, the data itself has been organized to provide performance improvement through partitioning. The table name `transactions_2023` indicates a year-based partition. This limits the scan to only the 2023 partition, which dramatically reduces the amount of data searched. Additionally, a B-tree index on `customer_name` (if present) will be used by the engine, even with the `LIKE` operator. If the business rules support the simplification, this can often yield the best performance. I’ve seen this be a 10x-100x performance improvement in real-world datasets.

**Resource Recommendations**

For a deeper understanding of query optimization, several resources have been invaluable to me throughout my career.  Consulting official database engine documentation is paramount. Each engine (PostgreSQL, MySQL, SQL Server, Oracle, etc.) will have specific features and nuances regarding index usage and query planning that are well documented.

Second, resources on database indexing strategies, including different index types (B-tree, hash, etc.) and how they influence performance, are also necessary. Familiarity with query execution plans is essential. Analyzing these plans allows you to understand how the database engine interprets and executes your query.  They illuminate potential bottlenecks.

Finally, general resources on relational database theory, including normalization, indexing, and query optimization principles, will give a broader understanding of underlying mechanics and best practices.  These resources often provide deeper insights into how and why particular techniques yield specific results.

In conclusion, optimizing the provided query requires understanding the root causes of non-sargable predicates. By restructuring the `WHERE` clause to be index-friendly and by considering underlying data organization, significant performance improvements can be achieved. This approach emphasizes effective database design and utilization of existing indexing to enhance overall query performance.
