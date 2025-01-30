---
title: "How does PostgreSQL's `EXPLAIN` perform without `ANALYZE`?"
date: "2025-01-30"
id: "how-does-postgresqls-explain-perform-without-analyze"
---
PostgreSQL's query planner heavily relies on statistical data about tables to generate efficient execution plans. When executing `EXPLAIN` without `ANALYZE`, you are primarily observing the planner's estimates based on the last statistics collected, not the actual runtime behavior. This distinction is critical for understanding why plans can sometimes appear suboptimal and why running `ANALYZE` is often a necessary prerequisite for accurate plan interpretation.

The planner, without current statistics, operates under assumptions based on the data present in `pg_statistic` catalog. These statistics include things like the number of rows, data distribution, and the frequency of specific values within columns. These figures are used to estimate the cost associated with various potential execution strategies like sequential scans, index scans, joins, and sorts. Without fresh data after table modifications, especially with significant data changes, the planner might misestimate the selectivity of certain predicates (WHERE clause conditions), the cardinality of intermediate results, or the cost of individual operators.

The core challenge with `EXPLAIN` alone is that it provides a *predicted* plan, not a realized one. It highlights how the planner *intends* to execute the query given what it knows about the data distribution. If `ANALYZE` has not been run recently, these assumptions might be far removed from reality, leading to an execution plan that, while logically correct, may not be the most performant. For instance, a large table might be estimated to have very few rows because statistics have not been refreshed after a substantial data import. This might cause the planner to choose a sequential scan, thinking the table is small and an index scan would be too costly, when an index scan would in reality be far faster.

To illustrate this, I'll provide three practical code examples, drawing from situations I've encountered in my professional database management experience.

**Example 1: Misestimated Cardinality**

Imagine a `users` table storing millions of records, and I'm trying to retrieve data based on a `registration_date` column with many records added since the last `ANALYZE`.

```sql
-- No ANALYZE has been performed recently
EXPLAIN
SELECT *
FROM users
WHERE registration_date > '2023-10-26';
```

The `EXPLAIN` output might show something like this, with cost estimates that don't necessarily align with the current state of the data:

```
 Seq Scan on users  (cost=0.00..22601.40 rows=581 width=233)
   Filter: (registration_date > '2023-10-26'::date)
```

Here, the planner estimates 581 rows matching the filter. This would be a good estimate if only few rows were added since the last statistics update. However, if thousands of rows with that date or later have been inserted, the planner will drastically underestimate the number of rows, and hence, the actual cost will be far higher. It likely will choose a full table scan. Performing `EXPLAIN ANALYZE` will show the true cost and number of rows being processed, which might reveal a performance bottleneck not apparent in this analysis.

**Example 2: Index Usage Misinterpretation**

Now, consider a `products` table with an index on a `category_id` column, and suppose that most rows now have the same category id.

```sql
-- No ANALYZE has been performed recently after a massive category change
EXPLAIN
SELECT *
FROM products
WHERE category_id = 12;
```

The `EXPLAIN` output might present an incorrect plan as the statistics on `category_id` aren't up to date:

```
 Index Scan using products_category_id_idx on products  (cost=0.42..3211.30 rows=156 width=102)
   Index Cond: (category_id = 12)
```

The planner estimates only 156 rows for category_id = 12, therefore it thinks an index scan is efficient. However, after the data change, a large portion of the products belong to the same category ID, and a sequential scan would be more efficient, since the planner would be reading most of the table anyway. `EXPLAIN ANALYZE` would help show a situation in which a full table scan is faster than using the index. Without `ANALYZE`, this can be a confusing choice.

**Example 3: Misestimated Join Costs**

Let’s examine a join scenario. Assume we have a `orders` table and an `order_items` table. Suppose data in both tables has significantly changed, but we haven't performed `ANALYZE` yet.

```sql
-- No ANALYZE has been performed recently
EXPLAIN
SELECT o.order_id, oi.product_id
FROM orders o
JOIN order_items oi ON o.order_id = oi.order_id
WHERE o.order_date > '2023-01-01';
```

The output might appear as:

```
  Hash Join  (cost=125.59..3877.82 rows=450 width=10)
    Hash Cond: (oi.order_id = o.order_id)
    ->  Seq Scan on order_items oi  (cost=0.00..2333.33 rows=10000 width=10)
    ->  Hash  (cost=125.56..125.56 rows=450 width=4)
          ->  Seq Scan on orders o  (cost=0.00..125.56 rows=450 width=4)
                Filter: (order_date > '2023-01-01'::date)
```

The planner here estimates only 450 rows will match the filter on `orders` and 10000 rows in the `order_items`. If the actual number of matching orders is much higher, the hash join may not be the best choice. It might, for example, be more efficient to use an index join. The problem is that the planner is making this decision based on outdated assumptions about the actual size of the data. With `ANALYZE`, these figures would be more accurate, potentially resulting in a completely different join strategy, possibly involving the index.

The absence of `ANALYZE` leads to a significant caveat: the planner might not choose the most efficient plan. While the logical result of the query remains the same, its performance can suffer considerably due to outdated statistical information. As an example, imagine the `products` table is sorted by category_id. If the planner thinks there are a few items of category id 12, it might choose to do an index scan. However, if in reality there are a massive amount of such items and the table is ordered by category_id, it would be more efficient to do a full table scan, simply filtering through the category_id and stopping when it finds an item with a different `category_id`.

Furthermore, the estimated costs, expressed as abstract units, can be misleading. These costs are used by the planner to compare various plans but do not reflect actual runtime in milliseconds, CPU usage, or I/O operations without the `ANALYZE` flag providing actual measurements.

To effectively use `EXPLAIN`, always ensure that the statistics are up-to-date. This can be achieved using the `ANALYZE` command on the specific table or using `ANALYZE VERBOSE` to get details on the collected statistics. You can use `VACUUM ANALYZE` for periodic maintenance.

For resources, I recommend starting with the official PostgreSQL documentation which provides a thorough guide to understanding the planner and analyzing query plans. You can then investigate books on database performance, which often detail optimization techniques applicable to PostgreSQL. Online communities also offer a multitude of practical advice and examples. The key is to not only understand the theoretical underpinnings but also to practice analyzing real-world query plans with the aid of accurate statistics.
