---
title: "cassandra is much slower than mysql for simple operations?"
date: "2024-12-13"
id: "cassandra-is-much-slower-than-mysql-for-simple-operations"
---

 so I see your question and yeah I've been there man Cassandra versus MySQL for simple ops it's a classic head scratcher Right off the bat if you're seeing dramatically slower performance from Cassandra on what you consider simple operations it's usually not Cassandra just being slow but rather something about how you're using it or your data model that's causing problems Let's dive in because I've fought this dragon myself a few times

First thing first the key difference Cassandra is built for scale and availability at the cost of some single-node write speed MySQL on the other hand is generally optimized for single-server performance with the ability to scale using replication and sharding but it's a different paradigm entirely A typical OLTP relational database like MySQL works by writing to a single disk location typically with ACID guarantees it's designed for things like transactional consistency and single row lookups which it does amazingly well Cassandra on the other hand writes to multiple locations in a distributed system for fault tolerance meaning that every write goes to multiple nodes and then there is a time for the data to get in sync so the trade-off is the single write speed for the ability to scale out

My first encounter with this issue was maybe 8 years ago I was working at this startup that was trying to use Cassandra as a direct drop-in for their MySQL user database It was a disaster We had simple CRUD operations that were taking seconds in Cassandra when they were milliseconds in MySQL It took me a while to understand that we were basically using Cassandra like it was MySQL which is a no-go We had to rethink the data model and access patterns completely

Think about a basic user profile retrieval You might have something like this in MySQL with a query for a specific user based on a user ID

```sql
-- MySQL example
SELECT * FROM users WHERE user_id = 123;
```

Easy enough Now imagine trying to do the same thing naively in Cassandra with the same exact model it is possible but its not going to be performant The same thing in Cassandra might look like this

```cql
-- Bad Cassandra example
SELECT * FROM users WHERE user_id = 123; -- This will be slow
```

The problem here is that Cassandra is not optimized for querying by non-primary key columns If the user\_id is not part of the primary key then Cassandra has to scan all the partitions which is a full table scan not good. Cassandra is fast when you know the partition key which is used to determine which node has the data

So a better approach in Cassandra would be to organize your data around your access patterns If you frequently need to look up users by ID then that should be part of your primary key You'd likely create a table like this:

```cql
-- Good Cassandra example
CREATE TABLE users_by_id (
    user_id int,
    name text,
    email text,
    -- other user data
    PRIMARY KEY (user_id)
);

SELECT * FROM users_by_id WHERE user_id = 123; -- Now this is fast
```

This is crucial the primary key in Cassandra is composed of the partition key followed by clustering keys. the partition key determines which node stores the data while the clustering keys determine the order of the data within the partition. so the queries should have the partition key to avoid a full scan or a slow execution. if the user ID is the partition key then you can go straight to the node that has the data and retrieve it without having to scan or query multiple nodes. This is where people usually trip up when moving from relational databases to NoSQL solutions.

Another thing that got me back in the day was consistency levels Cassandra provides tunable consistency where you can trade off consistency for performance If you are writing with a consistency level of QUORUM then Cassandra will wait for the majority of replicas to acknowledge a write before considering the write successful. if your data is on 3 replicas then Cassandra will need 2 confirmations before being a success. If your consistency level is ONE then Cassandra will need one confirmation and your write will be faster but it's less consistent A naive implementation would try for the highest consistency possible at all times resulting in more latency

So if you are reading and seeing slow operations you should verify your consistency level for both reads and writes You can lower the consistency level for reads where you can tolerate slight staleness for example but it's a tradeoff. Here's an example of how to write with different consistency levels:

```cql
-- Consistency level example
INSERT INTO users_by_id (user_id, name, email) VALUES (456, 'John Doe', 'john@example.com') USING CONSISTENCY QUORUM;

INSERT INTO users_by_id (user_id, name, email) VALUES (789, 'Jane Smith', 'jane@example.com') USING CONSISTENCY ONE;
```

My own personal experience with this is that we had a write operation that had a really low write speed because the application was configured to do all the writes with QUORUM which was way too overkill We had about 5 replicas but we didn't need to read all the replicas every time for our use case once we switch the write consistency level to ONE we had a massive improvement on the write times.

We also had an issue with a clustering key we were using a timestamp as a clustering key which was creating very big partitions in a small amount of time. Cassandra is designed to handle wide tables but you have to be mindful of the partition size. we also had to rethink that and instead of the timestamp we switch to something else that resulted in less wide partitions.

It's important to understand how Cassandra handles mutations and how the data is stored if you're experiencing slowness. The data is stored in SSTables which are immutable files. When you write data Cassandra creates a new SSTable and periodically merges the data to optimize for read performance. If the compaction is not done properly it can affect read performance and you may have slowness. In our case at the beginning we had too many SSTables which resulted in slower read performance until we tuned the compaction settings.

There are some resources if you want to learn more. The book "Cassandra The Definitive Guide" by Eben Hewitt and Jeff Carpenter is a good start and the "Designing Data-Intensive Applications" by Martin Kleppmann is also a great resource for any database in general. There is also the documentation itself that has a lot of information.

Oh and here is my attempt at the joke you asked for: Why did the NoSQL database break up with the relational database? Because it felt like they had too many "tables" between them.

Anyways getting back on topic the truth is there isn't a simple answer to why your Cassandra setup is slower than your MySQL setup. It's a combination of data modeling your use case and configuration. If you're struggling with slow queries you must check your partition key your query patterns your consistency levels and your compaction settings. If you're coming from a SQL background you will need to unlearn some of those patterns it's all about access patterns and understanding how Cassandra is actually working under the hood. So good luck man let me know if you have more questions.
