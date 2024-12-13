---
title: "lck_m_x lock sql server?"
date: "2024-12-13"
id: "lckmx-lock-sql-server"
---

Alright so you're hitting a classic deadlock issue in SQL Server huh I've been there trust me It's like banging your head against a wall trying to figure out why your queries are just hanging there doing absolutely nothing

Let's break this down I see you mentioned `lck_m_x` that's the specific wait type in SQL Server indicating a lock acquired in exclusive mode on a memory object Now memory objects can be all sorts of things but in most situations where you're having deadlock problems they're typically related to data access or rather the resources SQL Server uses to manage data access Think indexes tables and so on

Been there done that Had this happen back when I was working on a large data migration project years ago We were doing some heavy inserts and updates on a table with a bunch of indexes and I remember clearly the day everything just seemed to freeze up the application started timing out users complained like it was the end of the world And of course the error logs were just filled with deadlocks related to this `lck_m_x` thing So it's not just you trust me it's a fairly common issue especially in busy environments

Okay first things first understand this a deadlock occurs when two or more processes or threads each hold locks on resources that the others need This creates a circular dependency where neither process can proceed because each is waiting for the other to release a lock

To address this you need to analyze your execution plans and your code to identify which resources are being locked and in what order SQL Server will resolve deadlocks by choosing one process as the "victim" rolling it back and freeing up the resources so the other process can finish the victim process will be chosen based on it's cost and will be returned an error

Here are some things you can check out based on what i've experienced and what I would have done back then

**1 Check for excessive locking**

Sometimes the problem is not necessarily a deadlock but simply a lot of blocking You can start by investigating to identify long-running or poorly written queries that are holding locks for too long This can look like a deadlock but it might just be some bad code I once spend a whole week debugging what was essentially a cross join that someone forgot to filter out it was horrible

You can use the sysdmoswaitstats DMV to check for `lck_m_x` and other wait types It will help you identify if this is a recurring problem

```sql
SELECT
    wait_type,
    waiting_tasks_count,
    wait_time_ms,
    max_wait_time_ms
FROM
    sys.dm_os_wait_stats
WHERE
    wait_type LIKE 'LCK%'
ORDER BY
   wait_time_ms DESC;
```

This will give you a list of wait types ordered by the amount of time processes are waiting for those resources If `lck_m_x` is at the top or near the top you have a clear indication that this is a locking issue

**2 Investigate the deadlock graph**

SQL Server actually has a tool to help you diagnose deadlocks When a deadlock happens SQL Server can log a deadlock graph You need to enable trace flag 1204 or 1222

```sql
DBCC TRACEON(1222, -1);
```

The trace flag 1222 outputs more information than 1204 and is often preferred The output will look like an XML that looks like this

```xml
<deadlock>
  <victim-list>
    <victimProcess id="process6384b88"/>
  </victim-list>
  <process-list>
    <process id="process6384b88" taskpriority="0" logused="0" waitresource="OBJECT: 5:1138261763:0 "  spid="53"  transactioncount="2" lastbatchstarted="2024-07-28T16:32:46.807" lastbatchcompleted="2024-07-28T16:32:46.807" clientapp="Microsoft JDBC Driver for SQL Server" hostname="my-host" hostpid="1234" loginname="sa" isolationlevel="read committed (2)" xactid="208355" currentdb="5" lockMode="S" schedulerid="1" kpid="12345" status="suspended" spid="53" sbid="0" ecid="0" priority="0" trancount="2" lasttranstarted="2024-07-28T16:32:46.797" lasttrancompleted="2024-07-28T16:32:46.797" clientoption1="671088672" clientoption2="128050" />
    <process id="process64406c8" taskpriority="0" logused="0" waitresource="OBJECT: 5:1138261763:0 " spid="54"  transactioncount="2" lastbatchstarted="2024-07-28T16:32:46.810" lastbatchcompleted="2024-07-28T16:32:46.810" clientapp="Microsoft JDBC Driver for SQL Server" hostname="my-host" hostpid="1234" loginname="sa" isolationlevel="read committed (2)" xactid="208356" currentdb="5" lockMode="X" schedulerid="1" kpid="12346" status="suspended" spid="54" sbid="0" ecid="0" priority="0" trancount="2" lasttranstarted="2024-07-28T16:32:46.807" lasttrancompleted="2024-07-28T16:32:46.807" clientoption1="671088672" clientoption2="128050" />
  </process-list>
  <resource-list>
    <objectlock lockPartition="0" objid="1138261763" subresource="FULL" dbid="5" objectname="dbo.MyTable" id="lock4c123450" mode="X" associatedObjectId="1138261763">
      <owner-list>
        <owner id="process64406c8" mode="X"/>
      </owner-list>
      <waiter-list>
        <waiter id="process6384b88" mode="S" requestType="wait"/>
      </waiter-list>
    </objectlock>
     <objectlock lockPartition="0" objid="1138261763" subresource="FULL" dbid="5" objectname="dbo.MyTable" id="lock4c123450" mode="S" associatedObjectId="1138261763">
      <owner-list>
        <owner id="process6384b88" mode="S"/>
      </owner-list>
      <waiter-list>
         <waiter id="process64406c8" mode="X" requestType="wait"/>
      </waiter-list>
    </objectlock>
  </resource-list>
</deadlock>
```

This XML contains all the juicy details: the processes involved the locks they were trying to acquire and the order in which they were trying to acquire them The key here is to look at the resource list and the modes that each transaction had

The `<process>` tags show the spids of the processes involved in the deadlock The `<resource-list>` tags will give you information of what resources are being fought over

**3 Try using READ COMMITTED SNAPSHOT**

READ COMMITTED SNAPSHOT isolation level is a SQL Server feature that reduces locking by utilizing row versioning It's like having a consistent snapshot of the data while preventing reads from blocking writes or writes from blocking reads if you are using row versioning

Using `READ COMMITTED SNAPSHOT` you'll see there is a lower number of deadlocks that can happen In my experience this is a great approach for most applications It can lead to a lot better throughput overall with the trade-off of an increased amount of tempdb usage But in most situations that will not be an issue

```sql
ALTER DATABASE [YourDatabase] SET READ_COMMITTED_SNAPSHOT ON;
```

**4 Look at query code**

Sometimes the problem is just bad code like I said before look at your queries make sure they are hitting the required indexes If a query is doing a lot of full table scans it will acquire more locks and cause more issues If you are doing multiple update statements that do not use the primary key then the order in which they run will be indeterministic and will create more deadlocks

Here is an example of bad code that causes a lot of `lck_m_x`

```sql
UPDATE MyTable SET column1 = 'some value'
GO
UPDATE MyTable SET column2 = 'some value2'
GO
UPDATE MyTable SET column3 = 'some value3'
GO

```

Here is a better approach assuming that your table has a primary key named Id

```sql
UPDATE MyTable
SET column1 = 'some value',
    column2 = 'some value2',
    column3 = 'some value3'
WHERE Id = 1
GO

UPDATE MyTable
SET column1 = 'some value',
    column2 = 'some value2',
    column3 = 'some value3'
WHERE Id = 2
GO
```

The second version allows for SQL Server to use the primary key and apply the updates to each row which is much better than updating entire columns with a default value.

**5 Review transaction size and duration**

Try to keep your transactions as small as possible It's generally a good rule of thumb to wrap individual operations into separate transactions When you have a big transaction it can lead to a lot more locking and increase the chances of a deadlock

**6 Indexing strategies**

This is a deep rabbit hole I recommend reading the books SQL Server Execution Plans and SQL Server Query Performance by Grant Fritchey as both are fantastic resources for this

Indexes are really important and without them you are basically giving your SQL Server a headache It's like trying to find a specific book in a library without an index it's going to take forever and in the case of a SQL database that translates to a lot more locking and deadlocks. Sometimes too many indexes will also cause locking problems as the database needs to maintain all of them

**A little tip from the past**

I once had a deadlock issue that was incredibly complex it turned out the issue was a stored procedure called from different threads that used the same table but in reverse order it was a disaster to figure out but after a whole day of head scratching I finally managed to find it it was almost like my SQL server was playing a game of "who can hold on to the lock the longest" which isn't very fun for anyone involved It really felt like finding a needle in a haystack.

**Resources**

For a deeper dive I highly recommend reading "Inside Microsoft SQL Server 2008: T-SQL Querying" or any of the newer versions by Itzik Ben-Gan and also books by Kimberly Tripp they are great resources for understanding the internals of the SQL Server engine.

I hope this helps let me know if you have more questions.
