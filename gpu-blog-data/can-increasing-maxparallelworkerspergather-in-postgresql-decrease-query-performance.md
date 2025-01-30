---
title: "Can increasing max_parallel_workers_per_gather in PostgreSQL decrease query performance?"
date: "2025-01-30"
id: "can-increasing-maxparallelworkerspergather-in-postgresql-decrease-query-performance"
---
Increasing `max_parallel_workers_per_gather` in PostgreSQL can, counterintuitively, lead to decreased query performance under specific circumstances.  My experience optimizing large-scale data warehousing solutions has highlighted this nuanced behavior.  The key factor isn't simply the *number* of parallel workers, but rather the interplay between this setting, the query plan, the data distribution, and available system resources.  Simply throwing more workers at a problem doesn't guarantee improved execution time.

**1. Explanation:**

`max_parallel_workers_per_gather` controls the maximum number of worker processes PostgreSQL can utilize for parallel `Gather` operations.  `Gather` is a crucial part of parallel query execution, responsible for aggregating results from multiple worker processes back to the leader process.  Increasing this parameter allows for more concurrent aggregation, theoretically speeding up queries with substantial data parallelism.

However, this parallelism comes at a cost. Each worker process consumes system resources: CPU, memory, and I/O.  If the overhead of managing these workers (context switching, inter-process communication, data transfer) outweighs the benefits of parallel execution, increasing `max_parallel_workers_per_gather` will negatively impact performance. This is particularly true in scenarios with:

* **Limited system resources:** On systems with constrained CPU, memory, or I/O bandwidth, adding more parallel workers can lead to resource contention, slowing down the entire process.  The system might become bottlenecked, resulting in longer overall query execution times.  I've seen this firsthand in environments with shared resources where poorly planned parallel query execution caused a significant degradation in overall system performance.

* **Small data sets or queries with minimal parallelism:**  For queries operating on relatively small datasets, the overhead of parallel execution often exceeds any potential performance gains.  The cost of worker process creation and management becomes disproportionately large compared to the actual processing time.

* **Inefficient query plans:**  If the query planner generates a suboptimal plan, even with increased parallelism, the performance won't improve significantly.  In such cases, optimizing the query itself (through indexing, rewriting, or other techniques) is far more impactful than simply increasing the number of parallel workers.  I've encountered situations where poorly designed indexes nullified any benefits from parallel query execution.

* **Network Bottlenecks:** In distributed deployments, network latency between the leader and worker processes can severely limit the effectiveness of parallel `Gather` operations. Increasing `max_parallel_workers_per_gather` in these scenarios may exacerbate network congestion.


**2. Code Examples with Commentary:**

**Example 1: Inefficient Parallelism on a Small Dataset:**

```sql
-- Table with 1000 rows
CREATE TABLE small_table (id INT PRIMARY KEY, value TEXT);
INSERT INTO small_table SELECT generate_series(1, 1000), 'data';

-- Query that benefits little from parallelism
SET max_parallel_workers_per_gather = 4;
EXPLAIN ANALYZE SELECT count(*) FROM small_table;

SET max_parallel_workers_per_gather = 1;
EXPLAIN ANALYZE SELECT count(*) FROM small_table;
```

In this example, increasing `max_parallel_workers_per_gather` to 4 likely yields only a marginal improvement or even a slight performance degradation compared to using a single worker (1).  The overhead of parallel processing dominates the negligible benefit gained from dividing a small task among multiple processes.  The `EXPLAIN ANALYZE` output will highlight the actual execution times and the planner's cost estimations, showing the difference in performance.


**Example 2: Effective Parallelism on a Large Dataset with Suitable Indexing:**

```sql
-- Table with 1,000,000 rows and a suitable index
CREATE TABLE large_table (id INT PRIMARY KEY, category INT, value NUMERIC);
CREATE INDEX idx_large_table_category ON large_table (category);
INSERT INTO large_table SELECT generate_series(1, 1000000), (random() * 100)::INT, random();

-- Query benefiting from parallelism due to data distribution
SET max_parallel_workers_per_gather = 8;
EXPLAIN ANALYZE SELECT category, SUM(value) FROM large_table GROUP BY category;

SET max_parallel_workers_per_gather = 1;
EXPLAIN ANALYZE SELECT category, SUM(value) FROM large_table GROUP BY category;
```

Here, a large table with a suitable index on `category` allows for efficient parallel processing.  The `GROUP BY` clause naturally lends itself to parallel execution, and increasing `max_parallel_workers_per_gather` to 8 is likely to significantly improve performance.  Again, the `EXPLAIN ANALYZE` output provides concrete evidence of performance gains.  The index on `category` enables efficient distribution of the work among parallel workers, minimizing contention and maximizing throughput.

**Example 3:  Resource Contention Scenario:**

```sql
-- Simulates a resource-constrained environment (requires a system with limited resources for a visible impact)
-- This would involve running multiple parallel queries concurrently alongside the test query
-- and monitoring system metrics (CPU usage, memory, I/O).
-- For demonstration purposes, I will note the EXPECTED outcome.

SET max_parallel_workers_per_gather = 8; -- Initially a high value
EXPLAIN ANALYZE SELECT ...; -- A resource-intensive query

SET max_parallel_workers_per_gather = 2; -- Reducing the number of parallel workers
EXPLAIN ANALYZE SELECT ...; -- The same resource-intensive query
```

This example highlights resource contention. Running a resource-intensive query with a high `max_parallel_workers_per_gather` setting in a resource-constrained environment (as simulated above) would lead to resource contention. This contention will result in the query running slower compared to using a reduced number of workers, as shown in the expected results. System monitoring tools would reveal high CPU utilization, memory pressure, or I/O saturation during the higher-parallelism scenario.


**3. Resource Recommendations:**

Consult the official PostgreSQL documentation.  Pay close attention to the system monitoring tools available in your environment for detailed performance analysis.  Understand your hardware limitations.  Learn about query planning and optimization techniques.  Thoroughly analyze query execution plans using `EXPLAIN ANALYZE`.  Perform load testing and benchmarking to determine optimal values for `max_parallel_workers_per_gather` within your specific environment.  Prioritize the optimization of queries and indexes before solely relying on increasing parallel workers. This approach will yield superior and more predictable performance improvements.
