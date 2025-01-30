---
title: "Why does PostgreSQL's EXPLAIN show a Seq Scan when using the GIN index?"
date: "2025-01-30"
id: "why-does-postgresqls-explain-show-a-seq-scan"
---
PostgreSQL's reliance on sequential scans ("Seq Scan") despite the presence of a GiST or GIN index often stems from a mismatch between the query's WHERE clause and the index's capabilities, specifically concerning the selectivity of the query's predicates.  In my experience optimizing database queries for large-scale geospatial applications, I've encountered this repeatedly.  The index might exist, but the optimizer may determine a sequential scan to be more efficient, counterintuitively. This usually arises from queries with low selectivity predicates or complex expressions that hinder index utilization.


**1.  Understanding Index Selectivity and Query Optimization:**

PostgreSQL's query planner employs cost-based optimization.  It estimates the cost of various execution plans, including those using indexes and those performing full table scans.  The cost calculation considers factors such as the number of rows in the table, the size of the index, and the estimated number of rows that will satisfy the WHERE clause – the selectivity.  A highly selective predicate, one that filters out a significant portion of the table, makes an index scan far more efficient than a sequential scan. Conversely, a low-selectivity predicate, one that only filters out a small number of rows, might result in the query planner preferring a sequential scan. This is because the overhead of traversing the index and then fetching the matching rows from the table might outweigh the cost of simply scanning the entire table.

GIN (Generalized Inverted Index) indexes are particularly susceptible to this issue. While highly effective for searching within arrays and full-text searches, their efficiency is contingent upon the specificity of the search criteria.  Broad or vaguely defined search terms often render the index less beneficial than a sequential scan.  The planner assesses the potential gain from using the index against the cost of accessing the index itself. If the expected reduction in rows examined is negligible compared to the index access overhead, it opts for a sequential scan.

This is further compounded by the presence of other operators or conditions in the WHERE clause.  If the GIN index is only used for a single predicate within a complex WHERE clause incorporating joins or other conditions, its effectiveness might be diminished. The planner calculates the overall cost based on all conditions, and the benefit provided by the index might be overshadowed.



**2. Code Examples and Commentary:**

Let's illustrate this with three examples, showcasing scenarios where a Seq Scan might appear despite a GIN index:

**Example 1: Low-Selectivity Search:**

```sql
-- Table with GIN index on the 'tags' array column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    tags TEXT[]
);

CREATE INDEX gin_index ON documents USING gin(tags);

-- Query with low selectivity (many documents have 'general' tag)
EXPLAIN SELECT * FROM documents WHERE 'general' = ANY(tags);
```

In this case, if many documents possess the 'general' tag, the selectivity of the predicate is low.  The query planner might estimate that retrieving all rows and filtering them is faster than using the GIN index. The `EXPLAIN` output will likely show a Seq Scan.  Increasing the specificity of the search (e.g., `'highly_specific_tag' = ANY(tags)`) would likely improve the chances of the GIN index being utilized.


**Example 2: Complex WHERE Clause:**

```sql
-- Assuming a table with GIN index on 'keywords'
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name TEXT,
    keywords TEXT[],
    price NUMERIC
);

CREATE INDEX gin_keywords ON products USING gin(keywords);

-- Complex WHERE clause reduces index effectiveness
EXPLAIN SELECT * FROM products
WHERE 'database' = ANY(keywords) AND price > 100;
```

Here, the GIN index on `keywords` might be less impactful due to the additional condition `price > 100`. The planner considers both conditions; if the `price > 100` significantly reduces the candidate rows, it might make a sequential scan more efficient. The additional predicate can change the cost estimation enough to override the advantages of an index lookup.


**Example 3:  Function Calls within the WHERE Clause:**

```sql
-- Table with GIN index on 'location' (assuming a custom type)
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    location GEOGRAPHY,
    description TEXT
);

CREATE INDEX gin_location ON events USING gin(location);

-- Function call within WHERE clause hinders index usage
EXPLAIN SELECT * FROM events WHERE ST_Contains(location, ST_GeomFromText('POINT(10 20)'));
```

PostgreSQL might not be able to directly utilize the GIN index `gin_location` if a function like `ST_Contains` or `ST_DWithin` is used within the WHERE clause.  The function call might prevent the optimizer from recognizing the applicability of the GIN index, leading to a sequential scan.  In such cases, creating a functional index might be necessary.  For instance, if the `ST_Contains` function was frequently used with this specific geometry, an index on `ST_Contains(location, ST_GeomFromText('POINT(10 20)'))` might improve performance, though there are potential limitations to functional indexing.


**3. Resource Recommendations:**

For a deeper understanding of PostgreSQL query planning and optimization, I recommend thoroughly reviewing the official PostgreSQL documentation, focusing on sections pertaining to query planning, indexes, and GiST/GIN index specifics.  Consult advanced SQL tutorials and books that deal with query optimization techniques and index usage.  Furthermore,  exploring the output of `EXPLAIN ANALYZE` provides valuable insights into the actual execution plan and costs, allowing for fine-tuning of queries based on real-world performance data.  Finally, studying the behavior of the query planner across varying data distributions and query complexities will build an intuitive understanding of its decision-making process.  Analyzing query execution plans using tools like pgAdmin's `EXPLAIN` functionality will allow for iterative adjustments to your queries and indexing strategies.
