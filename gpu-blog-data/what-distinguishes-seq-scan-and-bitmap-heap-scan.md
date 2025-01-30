---
title: "What distinguishes Seq Scan and Bitmap Heap Scan in PostgreSQL?"
date: "2025-01-30"
id: "what-distinguishes-seq-scan-and-bitmap-heap-scan"
---
PostgreSQL employs various scan methods to retrieve data from tables, and the choice between a sequential scan (Seq Scan) and a Bitmap Heap Scan significantly impacts query performance. The fundamental distinction lies in how each accesses rows on disk, particularly when dealing with indexed columns and the associated trade-offs in I/O operations. I’ve personally witnessed queries go from minutes to milliseconds just by understanding when one scan type is favored over the other.

A Seq Scan is the most basic scan method. When a query necessitates a Seq Scan, the database reads each row of the table sequentially, from the beginning to the end, irrespective of any existing indexes. It doesn't leverage any indexing mechanisms. This behavior becomes extremely inefficient for large tables, especially when a query needs to filter based on specific conditions or retrieve only a small subset of the rows. Every row is examined even if the vast majority do not match the filtering criteria. The advantage of a Seq Scan is its simplicity; it has minimal overhead. If a table is small enough to reside entirely within the database's shared buffers, the scan can be surprisingly rapid. However, this is the exception rather than the rule in most real-world scenarios involving substantial data volume.

Conversely, a Bitmap Heap Scan is a more sophisticated approach that, when applicable, leverages one or more indexes to rapidly pinpoint the locations of relevant rows within a table, prior to physically fetching the row data. The process involves two key phases. First, a Bitmap Index Scan is performed on one or more indexes, creating a bitmap. A bitmap is a data structure where each bit position corresponds to a tuple identifier (TID) of a row in the table. If a bit is set to 1, it means a row associated with the matching TID meets the search criteria. Because bitmap representations compress efficiently, even for large datasets, the bitmap requires relatively little memory. Second, the bitmap identifies which rows are to be retrieved. A Bitmap Heap Scan then uses this bitmap to access those specific rows in a non-sequential manner, essentially hopping from one relevant row to another. This avoids the cost of reading every single page of a table like in the case of a Seq Scan.

The benefit of a Bitmap Heap Scan materializes when the query has multiple predicates on columns with associated indexes, or even just one selective predicate. PostgreSQL’s optimizer will often consider a Bitmap Index Scan if there are multiple matching indexes or if it deems the selectivity of a single index sufficient for a performance gain. The bitmap provides an efficient method for combining results of these index scans before touching the table itself. A bitmap operation like “AND” or “OR” can merge the results from various indexes, further optimizing the retrieval process. This becomes particularly crucial in situations where a large dataset would otherwise require a slow, sequential pass. It is worth noting that building the bitmap and using it to retrieve the corresponding row locations does add overhead compared to a plain Seq Scan, but these costs are often dwarfed by the I/O savings.

Here are three illustrative code examples to show these mechanisms in action:

**Example 1: Simple Sequential Scan**

```sql
CREATE TABLE example_table (id SERIAL PRIMARY KEY, value INT, text_data TEXT);

INSERT INTO example_table (value, text_data)
SELECT random() * 1000, 'some sample text'
FROM generate_series(1, 100000);

EXPLAIN ANALYZE SELECT * FROM example_table WHERE text_data LIKE '%sample%';
```

This code creates a basic table named `example_table` with three columns. It populates the table with 100,000 rows, each containing a randomly generated number, and some sample text. The `EXPLAIN ANALYZE` command displays the execution plan for the provided query. In this particular case, I expect PostgreSQL will perform a Seq Scan. There is no index on the `text_data` column, and the `LIKE` clause using the leading wildcard will almost certainly prevent index usage in this query. The query will read every row in the table to check if the text field contains the sequence `sample`. This will show a complete table scan, with all 100,000 rows being examined before any rows are retrieved. The execution plan generated will clearly indicate a "Seq Scan".

**Example 2: Bitmap Heap Scan with a Single Index**

```sql
CREATE INDEX idx_value ON example_table (value);

EXPLAIN ANALYZE SELECT * FROM example_table WHERE value > 900;
```

Building upon the previous table structure, this example creates an index on the `value` column, an integer column with random numbers. The `EXPLAIN ANALYZE` statement now asks for all rows where `value` is greater than 900. Because the query uses an indexed column with a suitable comparison, the optimizer should choose a Bitmap Heap Scan instead of a Seq Scan. The execution plan will show a "Bitmap Index Scan" operation to utilize the new index and retrieve the TIDs into a bitmap. Then, a "Bitmap Heap Scan" will use the bitmap to only access the relevant row pages from disk. This demonstrates how the index helps to avoid reading unnecessary rows. Given that the numbers are randomly generated between 0-1000, this should result in approximately 10% of the rows being returned and read. This performance will be significantly faster than the first example, which needed to check all 100,000 rows.

**Example 3: Bitmap Heap Scan with Multiple Indexes**

```sql
CREATE INDEX idx_text_data ON example_table (text_data);

EXPLAIN ANALYZE SELECT * FROM example_table WHERE value > 500 AND text_data LIKE '%text%';
```

This third example builds off the previous table, and adds a secondary index on the `text_data` column. The query now filters on both the integer column `value`, and also on the text column `text_data`. This query contains two predicates with indices. The optimizer should select a Bitmap Heap Scan. Internally a bitmap index scan will be done against both indexes, resulting in two bitmaps. Those bitmaps will then be AND'd together, giving a result bitmap indicating the rows to fetch. A Bitmap Heap Scan is then used to retrieve the rows represented in the combined bitmap. This illustrates the power of combining indexes to reduce the number of rows actually accessed, and avoid a full table scan. The resulting execution plan will show multiple “Bitmap Index Scan” actions and the associated "Bitmap Heap Scan". This approach will be more efficient than using either index individually followed by filtering within memory.

In summary, the difference between Seq Scan and Bitmap Heap Scan comes down to the method of data access. Seq Scan linearly reads through a table, whereas Bitmap Heap Scan uses indexes to efficiently target the required rows, particularly where multiple filter predicates are present, or where indexes are sufficiently selective on their own. Understanding these mechanisms and analyzing execution plans helps diagnose and resolve query performance bottlenecks. It can even be the difference between a usable and an unusable application.

For further exploration, I would recommend resources such as *PostgreSQL documentation* on query planning and execution, focusing on scan methods and index utilization. Additionally, exploring books and articles discussing database performance tuning will provide a deeper understanding of these concepts. Specific resources which provide detailed breakdowns of execution plans, and how to interpret them, can be very helpful, as these demonstrate the different choices the query planner can make, and the logic behind them. Finally, hands-on exercises involving profiling and analyzing queries using `EXPLAIN` can be invaluable for solidifying knowledge.
