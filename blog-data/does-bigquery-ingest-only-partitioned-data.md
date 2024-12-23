---
title: "Does BigQuery ingest only partitioned data?"
date: "2024-12-23"
id: "does-bigquery-ingest-only-partitioned-data"
---

Alright, let's get into this. The question of whether BigQuery *only* ingests partitioned data is a common point of confusion, and it’s something I've seen trip up even seasoned data engineers. To cut to the chase: no, BigQuery does not *only* ingest partitioned data, but partitioning is a very strong recommendation for almost every use case involving sizeable datasets. Let me break down why.

In my experience, working with various data pipelines, I’ve encountered scenarios where initially we weren't partitioning our ingested data. We’d start with relatively small datasets, and everything seemed perfectly adequate. We relied on BigQuery’s powerful query engine to churn through the data. However, as the datasets grew, query performance started taking a noticeable hit. The cost also increased dramatically. It was a direct, visceral lesson on the importance of thoughtful partitioning strategies.

BigQuery can indeed ingest data that isn’t partitioned, and it does so routinely, particularly when loading data into tables created without specifying partitioning columns. You can load data directly from various sources – cloud storage, streaming inserts, or via copy jobs, for instance – without the necessity of pre-partitioning the source data. Think of it this way: BigQuery can handle any data you throw at it, with or without partitioning, but with significant performance and cost differences. Partitioning isn't about whether ingestion will succeed; it is purely about *how efficiently* BigQuery will process your data and *how much you will pay*.

The core reason partitioning is crucial is query optimization. When a table is partitioned, BigQuery can effectively scan only the relevant partitions based on the query filter criteria, rather than scanning the entire table. This drastically reduces the amount of data being processed, leading to significantly faster query execution and, crucially, lower costs.

Partitioning, at its essence, is about segmenting your data into smaller, more manageable units based on a specific column. This column could be a timestamp, date, or an integer range. BigQuery offers two primary partitioning types: date-time partitioning (usually by a `DATE`, `TIMESTAMP`, or `DATETIME` column) and integer-range partitioning (using an integer column). Choosing the right partitioning column depends entirely on your query patterns. What filters are you typically applying? What kind of slice of data are you usually requesting? Those are the key considerations.

Let's explore this with some code examples.

First, consider a scenario where we create a table without partitioning. This is absolutely allowed:

```sql
CREATE OR REPLACE TABLE `mydataset.non_partitioned_table` (
  transaction_id STRING,
  transaction_date DATE,
  customer_id STRING,
  amount NUMERIC
);

INSERT INTO `mydataset.non_partitioned_table`
  (transaction_id, transaction_date, customer_id, amount)
VALUES
  ('tx001', '2023-01-01', 'cust123', 10.50),
  ('tx002', '2023-01-05', 'cust456', 20.00),
  ('tx003', '2023-01-10', 'cust123', 15.75),
  ('tx004', '2023-02-15', 'cust789', 5.25),
  ('tx005', '2023-02-20', 'cust456', 30.00);
```

In this case, BigQuery will ingest all data without any partitioning. Querying this table requires a full scan unless specific filtering is done on non-partition keys.

Now, let’s contrast this with date partitioning:

```sql
CREATE OR REPLACE TABLE `mydataset.partitioned_table` (
  transaction_id STRING,
  transaction_date DATE,
  customer_id STRING,
  amount NUMERIC
)
PARTITION BY
  transaction_date;

INSERT INTO `mydataset.partitioned_table`
  (transaction_id, transaction_date, customer_id, amount)
VALUES
  ('tx001', '2023-01-01', 'cust123', 10.50),
  ('tx002', '2023-01-05', 'cust456', 20.00),
  ('tx003', '2023-01-10', 'cust123', 15.75),
  ('tx004', '2023-02-15', 'cust789', 5.25),
  ('tx005', '2023-02-20', 'cust456', 30.00);
```

Here, we’ve created the same schema, but this time we’ve specified `PARTITION BY transaction_date`. BigQuery will physically store data separately by date, making subsequent filtering on dates immensely faster. A query that filters on a specific date range, such as `WHERE transaction_date >= '2023-01-01' AND transaction_date <= '2023-01-31'`, will now scan only the partitions for January, rather than the entire table. This is a significant advantage.

Finally, let’s consider an example using integer range partitioning, which can be very useful when segmenting based on an identifier:

```sql
CREATE OR REPLACE TABLE `mydataset.integer_partitioned_table` (
  order_id INTEGER,
  order_date DATE,
  customer_id STRING,
  amount NUMERIC
)
PARTITION BY RANGE_BUCKET(order_id, GENERATE_ARRAY(1, 1000000, 10000));

INSERT INTO `mydataset.integer_partitioned_table`
  (order_id, order_date, customer_id, amount)
VALUES
  (1000, '2023-01-01', 'cust123', 10.50),
  (15000, '2023-01-05', 'cust456', 20.00),
  (30000, '2023-01-10', 'cust123', 15.75),
  (500000, '2023-02-15', 'cust789', 5.25),
  (900000, '2023-02-20', 'cust456', 30.00);
```

This example partitions data based on `order_id`, using buckets of size 10,000. So records with `order_id` between 1 and 10,000 will fall into a single partition, between 10,001 and 20,000 in the next, and so on. A query such as `WHERE order_id >= 10000 and order_id < 30000` will only scan the relevant partitions that contain that range of `order_id` which drastically improves query performance.

So, in summary, while BigQuery absolutely *can* ingest non-partitioned data, it's rarely optimal from a performance and cost perspective. Partitioning is not mandatory for data loading, but it is strongly recommended, almost a best practice, particularly for larger datasets and frequently queried tables. It's about being smart about how you organize data for efficient processing and cost management.

For a deep dive into this, I’d recommend checking out the official BigQuery documentation, specifically the sections on partitioning and clustering. "Designing Data-Intensive Applications" by Martin Kleppmann also provides foundational knowledge on data storage and retrieval that is highly relevant. And if you’re looking for something more academic, "Database System Concepts" by Abraham Silberschatz, Henry F. Korth, and S. Sudarshan offers a very detailed theoretical background that can be beneficial in understanding the underlying principles of partitioning and data storage. Understanding these principles and applying them to your BigQuery datasets will lead to significant improvements in your data processing workflows.
