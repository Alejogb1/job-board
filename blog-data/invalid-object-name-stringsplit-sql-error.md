---
title: "invalid object name string_split sql error?"
date: "2024-12-13"
id: "invalid-object-name-stringsplit-sql-error"
---

Okay I see the "invalid object name string_split sql error" thing yeah I know that pain. Been there done that bought the t-shirt probably stained it with coffee while debugging this exact thing late one night. Let's get down to business.

So first off "invalid object name" generally means SQL Server can't find what you're asking for simple as that. And *string_split* its a table-valued function added way back in SQL Server 2016. Before that it was the wild west of user-defined function or CLR wizardry to get string splitting working efficiently no fun at all. If you are seeing this error its pretty likely the *string_split* function itself is not available in your SQL Server instance or the database you are in. I bet you are using some older version or maybe it is disabled somehow its happened to me.

Let me give you a bit of my experience. Back when SQL Server 2012 was king (yikes those were dark times) I had a project that needed to parse CSV data from a legacy system. Think thousands of rows comma separated values like a nightmare fuel scenario. I remember using some awful string manipulation with *CHARINDEX* and *SUBSTRING* in a nested loop it was slower than a sloth on tranquilizers. Then I discovered that CLR stuff and built a splitter function in C#. It was faster but holy moly the whole deployment process was like juggling chainsaws. It was so much easier when *string_split* came around.

Okay enough of my ancient history. Let's diagnose this issue. First things first confirm you actually are on SQL Server 2016 or a later version. You can run the following query to check that you probably know this already though:

```sql
SELECT SERVERPROPERTY('ProductVersion')
```

This gives you back SQL Server's full version string. If it’s anything lower than 13.0.1601.5 you are out of luck and I am very sorry. You would need a major update to your SQL Server.

Okay assuming your version is right maybe the database you are working in has a lower compatibility level. This determines what features a database supports. Check this run this:

```sql
SELECT compatibility_level FROM sys.databases WHERE name = 'your_database_name';
```

Replace *your_database_name* with the actual name of the database you are querying. If the compatibility level is below 130 you need to bump it up. Use this query to change the level but be very careful in production please:

```sql
ALTER DATABASE your_database_name
SET COMPATIBILITY_LEVEL = 130;
```

I had once a situation where some obscure setting had disabled system table functions. It happened only in my dev environment thankfully never again. I couldn't find it in any documentation for weeks until I found a hidden config setting. Okay that was my joke for today a hidden config setting in SQL Server is about as funny as a blank line of SQL code. Moving on.

Now even if you have the right version and compatibility level there are still some less common cases. Sometimes SQL Server is just SQL Server doing SQL Server things.

For example *string_split* is a table-valued function you can’t just call it like a scalar function. You need to use it in the FROM clause or a CROSS APPLY to use its output which is basically a table as well. I have seen users try to use it in WHERE or SELECT clauses and they are confused with the syntax. Let's look at a simple example of how to use this

```sql
SELECT value
FROM string_split('apple,banana,cherry', ',');
```

This returns a table with 3 rows one for apple one for banana and one for cherry. Simple clean. If you try and use this like this:

```sql
SELECT string_split('apple,banana,cherry', ',');
```
This will give you errors and it is not how it works it is not a scalar function. That is the important thing to remember.

Okay so this covers the most common scenarios. One last thing to check. If you have a very very particular setup and have custom security settings on system objects it's possible that access to *string_split* has been restricted somehow but that's a very rare edge case.

If you’re still running into problems it could be a really deep underlying system issue I hope you don't have to debug that. Consider restarting the SQL server service as a first aid step. If you are working on the cloud check the resource health of the server itself.

For further resources and a better understanding of how table valued functions work I really recommend reading the official Microsoft documentation on SQL Server. It is a dense read I know but it helps. Also I recommend *SQL Server 2019 Query Performance Tuning* by Grant Fritchey its an older book but the fundamentals are still valid if you can find it. He covers a lot of the quirks and odd behaviors of SQL Server and it is a great read. Finally *T-SQL Fundamentals* by Itzik Ben-Gan is an excellent book covering basically all things T-SQL I have learnt a lot from that book.

In short check your version check your compatibility level check how you are using the function then check if a system setting is acting weird and if you still have the error get some coffee and prepare for the deep dive. That's my usual process anyway. Good luck and happy coding I hope you find a solution to your problem.
