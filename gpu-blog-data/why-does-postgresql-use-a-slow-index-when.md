---
title: "Why does PostgreSQL use a slow index when no data matches the query?"
date: "2025-01-30"
id: "why-does-postgresql-use-a-slow-index-when"
---
PostgreSQL's selection of an index, even when no matching rows exist, is fundamentally tied to its query planning process and cost-based optimizer.  My experience optimizing queries in large-scale PostgreSQL deployments has shown that this behavior, while seemingly inefficient, stems from the optimizer's inability to definitively determine index ineffectiveness *a priori*.  The cost model employed isn't designed to predict the absence of results; instead, it estimates the cost of *accessing* the index, comparing that to the cost of a sequential scan.  If the estimated cost of index access remains lower than a sequential scan, even with a zero-result prediction, the optimizer will favor the index.

This is because the optimizer operates on statistics gathered about the table and its indexes. These statistics, while generally accurate, are not perfect predictors.  They reflect the distribution of data at the time of the `ANALYZE` command execution.  If the query involves conditions that highly restrict the result set,  even to the point of zero results, and those conditions are not sufficiently represented in the statistics, the optimizer might still erroneously predict a lower cost for index access.  The consequence is the observed use of a seemingly slow index when, in reality, the absence of results is the underlying cause.

Let's examine this with code examples.  Assume we have a table named `products` with columns `product_id` (integer, primary key), `category_id` (integer), and `name` (text).  We've indexed `category_id`.

**Example 1:  Index Used Despite No Matches**

```sql
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    category_id INTEGER,
    name TEXT
);

CREATE INDEX idx_category ON products (category_id);

INSERT INTO products (category_id, name) VALUES (1, 'Product A'), (1, 'Product B');

-- Query that yields no results, but still uses the index
EXPLAIN ANALYZE SELECT * FROM products WHERE category_id = 10;
```

The `EXPLAIN ANALYZE` output will likely show the index `idx_category` being used, despite the fact that `category_id = 10` doesn't exist in the table.  The optimizer's cost estimation predicted that checking the index would be faster than a full table scan, even though this prediction proves incorrect in this specific case due to a lack of matching entries.  This is particularly true for larger tables where the cost of a full table scan significantly outweighs the cost of even an ineffective index lookup.  A smaller table might see a sequential scan instead, but the principle remains the same: a comparison based on estimated cost.

**Example 2:  Illustrating Statistics Impact**

```sql
--Adding many products in category 1
INSERT INTO products (category_id, name) SELECT 1, ('Product ' || generate_series(1, 10000)) ;

-- Rerun the query, potentially changing the outcome
EXPLAIN ANALYZE SELECT * FROM products WHERE category_id = 10;

ANALYZE products;  --Update statistics after significant data modification

EXPLAIN ANALYZE SELECT * FROM products WHERE category_id = 10;
```

In this scenario, after adding many products to category 1,  the query might still use the index initially.  However, running `ANALYZE products` updates the table statistics.  The subsequent `EXPLAIN ANALYZE` might show a shift towards a sequential scan, as the updated statistics now better reflect the data distribution, making a sequential scan more cost-effective according to the optimizer's calculations.  The absence of data for `category_id = 10` becomes more evidently costly within the updated cost model.

**Example 3:  Index Choice with Partial Index**

```sql
CREATE INDEX idx_category_partial ON products (category_id) WHERE category_id > 5;

EXPLAIN ANALYZE SELECT * FROM products WHERE category_id = 10;
EXPLAIN ANALYZE SELECT * FROM products WHERE category_id = 2;
```

Introducing a partial index demonstrates the optimizer's selectivity consideration.  The query `category_id = 10` will likely use `idx_category_partial` (or potentially still the full index, depending on statistic) whereas `category_id = 2` would likely result in a full scan or use of the full index, as the partial index is not applicable.  This highlights that even with a non-matching query, the optimizer analyzes the *applicability* of the index before assigning a cost. The cost evaluation factors in the index's potential for reducing the search space. Even a partial index that is *not* useful is assessed by the cost model.

In conclusion, the perception of PostgreSQL using a slow index for queries with no matching data stems from the cost-based optimizer's estimations.  These estimations, based on statistics that may not perfectly reflect the current data distribution or the specific query's selectivity, can lead to suboptimal choices.  Regularly updating statistics via `ANALYZE` and careful index design, including the consideration of partial indexes, are critical for mitigating this issue.  Furthermore, understanding the output of `EXPLAIN ANALYZE` is paramount for identifying and addressing such performance bottlenecks.

**Resource Recommendations:**

The official PostgreSQL documentation, particularly sections on query planning, statistics, and indexing.  Also consult advanced PostgreSQL books covering query optimization strategies and performance tuning techniques.  Focus on the internal workings of the cost-based optimizer and the impact of statistics on query plan generation.  Consider dedicated resources on index design methodologies for large datasets.  These resources will provide a far more in-depth technical understanding of the processes at play.
