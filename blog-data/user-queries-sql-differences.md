---
title: "user queries sql differences?"
date: "2024-12-13"
id: "user-queries-sql-differences"
---

Okay so you’re asking about SQL differences between various database systems right been there done that it’s a whole thing

I mean at a high level SQL is SQL mostly yeah it’s a standard but boy oh boy the devil’s always in the details and when you get down into the nitty-gritty of actually implementing stuff on different database engines things start to look way more like a battlefield than a nice clean specification document I've spent countless sleepless nights debugging SQL queries that ran perfectly fine on my local dev environment using Postgres only to completely explode in production on Oracle or SQL Server or MySQL you name it.

First off let's talk about the basics differences in syntax can be a real pain things like string concatenation can trip you up some databases use `||` others use `+` and still others use functions like `CONCAT` it’s just messy and it's not like you can just copy paste stuff across all the time I remember one time I was porting a fairly complex application between Postgres and SQL Server I had about 200 places in the codebase where I needed to fix the string concatenation I actually ended up writing a little script just to do the find and replace because I was going nuts.

Another thing that always gets people is date and time functions each database has its own quirky way of handling dates so you need to be careful for example formatting dates can be an adventure you could be using `TO_CHAR` in Oracle or `strftime` in SQLite or `FORMAT` in SQL Server or a whole other set of functions it's not pretty

Then we get to data type handling differences even seemingly basic stuff like integer types and string types can differ in subtle ways that can mess up your queries there are cases where small int versus a normal int is used in databases differently that could affect your calculations

And then we have the more advanced features like window functions Common Table Expressions CTEs and stored procedures that's where the real fun begins not all databases support the same set of these advanced features or they might have slightly different implementations of them for example one database might have full support for recursive CTEs while another might have partial or no support whatsoever so it’s not just the basic stuff you always need to check on the specific implementation of each feature.

Let's get into some specifics with examples

Example 1 String Concatenation

Here's how you might do it in Postgres or MySQL

```sql
SELECT first_name || ' ' || last_name AS full_name
FROM users;

-- or using CONCAT in MySQL
SELECT CONCAT(first_name, ' ', last_name) AS full_name
FROM users;
```

But in SQL Server you might need this

```sql
SELECT first_name + ' ' + last_name AS full_name
FROM users;
```
Different symbols same concept but could completely break your query and create an error in another database system

Example 2 Date Formatting

Here’s an example in Postgres

```sql
SELECT order_date, TO_CHAR(order_date, 'YYYY-MM-DD') AS formatted_date
FROM orders;
```

And here's an example in SQL Server

```sql
SELECT order_date, FORMAT(order_date, 'yyyy-MM-dd') AS formatted_date
FROM orders;
```

Different syntax different functions same goal the point here is that all of those need to be checked carefully when porting

Example 3 Window Functions

A fairly common way of using window functions in postgres or in any database that supports this would be this

```sql
SELECT
  user_id,
  order_date,
  order_total,
  ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_date) AS order_number
FROM orders;
```

but in more obscure databases window functions could be implemented in a slightly different way or even be entirely absent and you might have to rewrite your logic

Performance implications are also something to think about. Even if two different SQL databases appear to execute the same SQL query one might be drastically faster than the other depending on the engine implementation database design indexes and more if you do not pay attention to this this can severely affect the performance of your application and its response time in production.

I had this one client with a massive database of customer transactions they were switching from MySQL to a custom version of PostgreSQL and every time I run a simple query the performance was completely different I spent a solid week just profiling the queries trying to understand the nuances of how each database optimizer works. It turned out that the way that PostgreSQL handles indexing was slightly different than MySQL that was an interesting learning experience.

And there's the whole realm of vendor-specific extensions which are always fun Oracle and SQL Server are notorious for adding their own proprietary SQL extensions that do not exist in other databases while sometimes these extensions can be useful they also make your code less portable if you plan on migrating to another database. That's how you get into proprietary vendor lock-in the trap many companies fall in at the end.
You know it's like trying to speak different dialects of the same language sometimes things will get lost in translation

(Okay that’s my only allowed joke for the whole response just to adhere to the rules sorry)

So how do you deal with all this craziness? Well firstly avoid writing SQL at all if possible use an ORM that abstracts away the specific database implementation which is not always a possibility though but it's still a good thing to strive for a lot of the work can be abstracted away from the developer. If you need to write raw SQL make sure to carefully check documentation for each database you work on.

Some resources you should be looking into are:
- SQL standard documentation ISO/IEC 9075 always a must the actual standard documentation is a good start if you want to avoid many problems at the get go
- specific database documentation the actual documentation of the database system you’re targeting Oracle documentation Microsoft documentation Postgres documentation MySQL documentation this is very important because they have specific features and functions.
- "SQL for Smarties Advanced SQL Programming" this is a very good book that goes into detail on many features of the language and advanced concepts
- "Understanding SQL Query Performance" a very good book for optimizing your queries across different databases

And finally testing testing testing always test your queries against each database you intend to use and not just basic tests but also stress tests and different load tests. I often use test containers to create a disposable database instance for testing and integration purposes. It is also useful to use database specific tools like query analyzers to check query performance before deployments.

In summary SQL is a standard but databases are anything but the same even though they use the same language. You have to be aware of the specific features nuances and quirks of each database you work on if you expect to write portable and efficient SQL. This is not a simple task and you should expect some problems along the way. Be patient. I hope this helps.
