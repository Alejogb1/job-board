---
title: "create in memory table sql server performance?"
date: "2024-12-13"
id: "create-in-memory-table-sql-server-performance"
---

Okay so you’re asking about in-memory tables in SQL Server and performance right been there done that a few times lets get into it

First off let’s be real in-memory tables aren't some magical speed bullet they’re powerful sure but they require understanding to use properly and not turn into a mess that’s just slightly faster than a regular disk based table with a headache you know the kind

Okay so a little about my personal history with these things way back when when I was first dabbling with SQL Server on a real project for a e-commerce site I had this crazy idea to throw in-memory tables at a problem we had it was our transaction logging you know every order every click pretty granular stuff we were drowning in I/O so I saw the in-memory thing and thought that’s it! speed boost incoming! well I was naive you could say my mistake was going straight for the biggest hammer possible without actually understanding the nuances of the nail to be hammered.

The first thing I didn’t really get was the memory limits I naively assumed it would like use what it needs you know turns out that’s not how it works you need to size your in-memory table precisely based on your expected data and some headroom because it all lives in RAM if you over estimate your gonna waste some good precious RAM and if you underestimate its all gonna crash and burn with allocation errors not pretty I spent days with query analyzer and memory dump analysis figuring out what the hell was going on after those crashes. It was a harsh learning process but a valuable one.

The other big thing I messed up was how transactions work with in-memory tables and disk based tables they’re not like a drop in replacement there is a major difference with how data is persisted on disk with memory optimized tables also persistence is different. Disk based tables get everything logged and ACID compliant while memory optimized tables in SQL Server don’t actually do a hard disk write immediately each transaction in memory tables have an initial checkpoint and a separate log checkpoint for durability to disk. This is a speed boost but also another thing to think about with durability I had to restructure our logging procedure a few times because we were running into recovery issues it took a while to get it stable but we eventually got there.

Now lets talk code and get this problem broken down the first thing you need is that create table syntax it’s not the same

```sql
CREATE TABLE dbo.MyInMemoryTable (
  ID INT IDENTITY(1,1) NOT NULL PRIMARY KEY NONCLUSTERED HASH WITH (BUCKET_COUNT = 1024),
  Data VARCHAR(200) NOT NULL,
  CreatedDate DATETIME2 NOT NULL
)
WITH (MEMORY_OPTIMIZED = ON, DURABILITY = SCHEMA_ONLY)
```

See that memory optimized thing? and the durability option? those are crucial for in memory tables. Also you need a non clustered hash index instead of a clustered index on disk based table that's because hash indexes are more efficient for lookups in memory but they have different behaviors for ordering. the `BUCKET_COUNT` thing is important too size that right based on how many rows you expect. `SCHEMA_ONLY` here means the data isn't persisted between server restarts you will lose the data if you chose SCHEMA_ONLY. `SCHEMA_AND_DATA` means the data will persist between server restarts

Then lets look at some basic inserts nothing fancy here

```sql
INSERT INTO dbo.MyInMemoryTable (Data, CreatedDate)
VALUES
('Test Data 1', SYSDATETIME()),
('Test Data 2', SYSDATETIME()),
('Test Data 3', SYSDATETIME());
```

See that? that’s exactly the same as a normal table insert. But the behavior behind the scenes is totally different and can have performance implications.

Now for querying it’s not rocket science but you need to consider index usage. Hash indexes are great for single lookups but not for ranged queries. If you need to do ranged queries then you need a nonclustered index to help you do those queries more efficiently. You should consider creating a nonclustered index for specific ranged queries.

```sql
-- Example query using primary key index
SELECT * FROM dbo.MyInMemoryTable WHERE ID = 2

-- Example of what not to do a full table scan will be used if not indexed properly.
SELECT * FROM dbo.MyInMemoryTable WHERE CreatedDate > '2024-01-01'
```

Again nothing mind blowing but it’s important that you use the right indexes. you should use your indexes effectively because there is no going back when you decide to create a in-memory table you have to keep the index structure in memory also. Also you should take into account that memory tables have different locking mechanism.

Okay enough of my boring stories lets talk about some things that I recommend you take a look at if you want to go into the specifics you should start with the Microsoft documentation for in-memory tables that is a gold mine and the go to source. Then you should move to the book Inside SQL Server by Itzik Ben-Gan is a classic and it has a great section on in-memory OLTP. There are more advanced papers from Microsoft Research on the technical details of the memory optimization engine. I also remember this one guy I used to work with who was an SQL Server guru he once said "Indexes are like socks you need them but they are useless if they are not used properly" (haha) he was a real character.

Okay so what are some general recommendations from a fellow techie who has been in the trenches with in-memory tables in sql server

1.  **Plan Your Memory:** Seriously think about sizing your in-memory tables. Get your estimates right or you’ll end up wasting resources and that costs money in the long run.

2.  **Understand Durability:** Decide if you really need the `SCHEMA_AND_DATA` option you don't need persistence if the data is transient.

3.  **Index Correctly:** Hash indexes are not a silver bullet learn what they are good at and also learn when you need something else like a nonclustered index if you need ranged queries.

4.  **Testing Testing Testing:** Don’t go throwing these things into production blindly. Benchmark your code test it under realistic loads before you go live.

5. **Monitor:** Monitor your memory usage regularly. You will want to spot potential bottlenecks and see if the table is being sized properly.

In short in-memory tables in sql server are a powerful tool if you know how to use them properly they are not a drop in solution they require careful planning design testing and also careful monitoring. Use them wisely and they can boost your performance but abuse them and you will have headaches. Don't learn the hard way like I did learn from my mistakes.
