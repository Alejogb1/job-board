---
title: "tombstone in cassandra database deletion?"
date: "2024-12-13"
id: "tombstone-in-cassandra-database-deletion"
---

 so tombstone deletion in Cassandra yeah I’ve been down that rabbit hole a few times let me break it down for you from my personal experience because this is not theoretical stuff this is blood sweat and tears kind of knowledge we're talking about and I'll throw some code examples your way too

First off tombstones are markers for deleted data In Cassandra when you delete a row or even just a single column Cassandra doesn’t actually erase the data right away Instead it inserts a tombstone This tombstone essentially says hey this data is marked for deletion It’s a way to keep things consistent across all the nodes in your cluster especially when those nodes might be out of sync at times think network partitions and the like

Now why not just instantly delete it well that’s where the distributed nature of Cassandra kicks in Because data is replicated across multiple nodes you need a way to ensure every node eventually agrees on what data is deleted If you instantly deleted data on one node but another node was offline and came back online later it would still have the old data So tombstones make sure every node gets the memo and can apply the deletion

Tombstones are written with a timestamp and a time to live or ttl These values are used to decide when the tombstone itself can be purged The ttl is configured at the table level and it dictates how long the tombstone will linger around for The longer the ttl the longer the tombstone sticks around and potentially the more performance issues you'll have if you aren't careful

So the tombstone process is usually the following deletion happens a tombstone is inserted then compaction happens it is a process where data files or sstables in Cassandra that are immutable files where the data is stored are merged together and during that process tombstones are also checked and then finally tombstones that expired are also removed from the database The tombstone removal is very important because the existence of many tombstones degrades performance significantly

I had one client a few years back their application was creating a ton of delete operations which they thought was totally fine they even were thinking of increasing the number of deletions but little did they know we started to see massive performance drops reads were slow writes were slow and the cluster was starting to melt down we initially thought it was a hardware issue but we were wrong what we had was excessive tombstones which were causing reads to scan an unnecessary amount of data files

 let's dig into some code first how we see tombstones in cassandra. The most common way to see a tombstone is to run a `SELECT` statement with the `WRITETIME` and `TTL` modifiers. Here’s an example:

```cql
SELECT WRITETIME(column1), TTL(column1) FROM your_keyspace.your_table WHERE primary_key_column = 'your_key';
```

This will show you the timestamp at which the data was last written for column1 and also if it was a tombstone it will also show the ttl of the tombstone

Next example how you would generally do deletions in Cassandra lets imagine a table called `users` with some columns such as user_id and name

```cql
DELETE FROM your_keyspace.users WHERE user_id = 'user123';
```
This is a very simple deletion statement but under the hood it creates a tombstone for the row where user_id is equal to user123.

One more example to see the problem of excessive tombstones and the reason why we need to delete tombstones I will use `nodetool cfstats` which is a command line tool to see the status of table on Cassandra

```bash
nodetool cfstats your_keyspace.your_table
```
this command will show you a lot of useful information but lets focus on these 2 values:

`SSTable count` : number of sstable files where data is stored.

`Tombstone drop count` the number of tombstones that got dropped due to the compaction process.

If you notice that the `SSTable count` is high and `Tombstone drop count` is low it can mean you have an excessive amount of tombstones in your system and your compaction strategy is not working as it should.

Now how do you manage these tombstones efficiently Well there are a few key strategies you need to understand first compaction strategies as we saw earlier with the tool.

Compaction is Cassandra’s mechanism for merging data files and reclaiming space and removing tombstones There are various compaction strategies available the most common ones are `SizeTieredCompactionStrategy` STCS and `LeveledCompactionStrategy` LCS each have different ways of merging sstables STCS merges data files of similar size together while LCS works at levels of files and merges files in the same level together. I am not going to go in details in each strategy here but is important to know that each strategy has some implications in tombstone removal. STCS can be more aggressive in tombstone removal but also can consume more resources during compaction while LCS can be slower in tombstone removal but less intensive in the overall resources

You also should think of the TTLs on your tables Make sure that the TTLs are set to something reasonable for your use case If the TTL is too high tombstones will linger around too long if they are too short you might end up with resurrections where data comes back after you delete them I know this sounds funny but it did happen to me so if your data doesn't disappear after you delete it it's not ghost data it's most likely a tombstone issue

Another common problem is anti-patterns like deleting lots of rows all at once this can create a lot of tombstones rapidly leading to performance degradation Instead of deleting lots of data you can also think of redesigning your data schema if you can to reduce deletions for example use different tables where you expire or drop instead of deleting large quantities of data

So what resources should you check on this topic

First I recommend reading the official Cassandra documentation on tombstones and compaction strategies. It's a bit dense but it's the bible for this topic. Second there is a good book called "Cassandra The Definitive Guide" by Eben Hewitt it’s a great resource to learn about all the internals of Cassandra including tombstone deletion management. Finally the DataStax website also has excellent blogs and white papers on this subject it's usually updated and very well written
Oh and don’t forget to properly monitor your Cassandra cluster metrics like tombstones per read and compaction stats this will help you identify issues early instead of having a cluster meltdown like I had.

So yeah Tombstones in Cassandra are a tricky beast to tame but it's possible with the correct knowledge and management of the different knobs and processes in Cassandra. Remember tombstone creation is necessary for Cassandra's eventual consistency model so the deletion of tombstones is a necessary evil in the process. Hope this helped and let me know if you have more questions I'll try to do my best to answer them.
