---
title: "what is the difference between inner join and semi join?"
date: "2024-12-13"
id: "what-is-the-difference-between-inner-join-and-semi-join"
---

Okay so you're asking about inner joins and semi joins right Been there done that more times than I care to admit Let's break this down in a way that hopefully makes sense without getting bogged down in theory okay

First off understand that both inner joins and semi joins are ways to combine data from two tables or relations depending on your background but they behave very differently in what they return and how they return it which is kinda what we're getting at here

**Inner Join Basics**

Alright let's start with the classic inner join The goal here is to produce a result set where each row in the result represents a matching pair of rows from the two tables based on a given condition You're essentially creating a Cartesian product of the two sets and then filtering it down to only the matching rows So a full match of all columns from both the tables are what you get if you've never seen the Cartesian product of two sets just think of a multiplication table that's kinda what's going on at the basic level

Think of it this way if table A has customer data and table B has order data an inner join on the customer ID would give you a result with each row representing a customer-order pair only if that customer actually placed an order That's why they call it an *inner* join you're keeping only the records where there's a match *inside* both tables

Here's a basic SQL example let's say I had this issue in an old project I was working on on database that used PostgreSQL

```sql
SELECT
    customers.customer_id
    customers.name
    orders.order_id
    orders.order_date
FROM
    customers
INNER JOIN
    orders ON customers.customer_id = orders.customer_id;
```

See Pretty straightforward you get all the columns you requested from both tables only for customers who have placed orders if that customer doesn't have any order you simply won't see it in the result which was my big mistake in my first few years using database joins I though I got *all* the customers and all the orders at once which was simply not correct

So inner join keeps only records that have counterparts in the second table and return full row data from both tables

**Semi Join Basics**

Now here's where it gets different with semi joins a semi join is also used for checking for existence of matching rows but with a twist The goal of a semi join is only to return data from the *first* table and not the second one based on whether matches are found in the *second* table And here's the catch It doesn't care *how many* matches exist in the second table only that at *least one* match is present

So a semi join is basically an existence check It's like saying "give me all the customer data where there is at least one matching order" You are checking if a join *could* happen if you did an actual inner join on two table

Back to the same example but this time using semi join I wanted all the customers which had orders in my older project So I had to learn that the semi join does not return the order's data just the customer information

```sql
SELECT
    customer_id
    name
FROM
    customers
WHERE
    EXISTS (SELECT 1 FROM orders WHERE customers.customer_id = orders.customer_id);
```

Notice the structure we aren't actually joining in the traditional SQL sense We are using EXISTS and a subquery. This is semantically a semi join.

The semi join only checks whether the subquery exists which in essence is checking if there is at least one matching record in the orders table for a customer from customer table. And it only returns data from the first table

**Key Differences Summarized**

*   **Output**: Inner joins return columns from *both* tables semi joins return columns from only the *first* table.
*   **Cardinality**: Inner joins can return multiple rows per record from the first table if there are multiple matches in the second table. Semi joins return at most one row per record from the first table regardless of how many matching records exist in the second table
*   **Goal**: Inner join is for matching and combining data semi join is for existence checks
*   **Complexity**: Semi joins often appear a bit more complex at first with syntax like EXISTS or IN with subqueries which is probably what is making the question interesting

**When to use which one**

The choice really boils down to what you need

*   Use an inner join when you need a combined view of information from both tables I used inner joins when I needed full information on orders and customer data for example in reports I did to business partners to analyze trends.
*   Use a semi join when you just need to know which records in the first table have corresponding records in the second table without needing any data from the second table in the query result. I used them frequently in my past for simple data validation and filtering before doing other analysis

So the semi join is much faster since you aren't extracting data you don't need from the other table unlike the inner join that could waste a lot of time retrieving information that you don't need

And a little joke about databases: Why did the database break up with the SQL query? Because they couldn’t commit. Okay okay I had to get it out of my system back to serious stuff

**Performance Notes**

While both are join mechanisms their underlying implementations can differ quite a bit across different database engines For example if you use databases like PostgreSQL or MySQL the database optimizers are usually smart enough to choose the correct plan even if you wrote it in a less optimal way It might internally rewrite your inner join or semijoin to be much more efficient under the hood for instance by turning an inner join into a semi join when possible if the select only contains one table. It's always good to check your query plan to see what the database engine is actually doing if you care about performance of your queries as I did in the past.

**Other Forms of Semi Join**

You might also see semi joins represented using the `IN` operator with a subquery which is semantically equivalent to `EXISTS` This is just a syntax variation for the same principle which you will also come across. Here's an example of that so you see what I mean

```sql
SELECT
    customer_id
    name
FROM
    customers
WHERE
    customer_id IN (SELECT customer_id FROM orders);
```

This version will give you the same result as the previous semi join using EXISTS but some people just prefer to use `IN` for readability purposes even though behind the scene they are pretty much the same. In fact they are both semi-joins.

**Resources**

Now for more in-depth study I'd suggest looking into some database theory books not just SQL books you see out there on google The classic "Database System Concepts" by Silberschatz Korth and Sudarshan is a great reference for understanding the theoretical foundations of joins and query processing. You'll also find more real-world examples in books dedicated to specific databases like "PostgreSQL Up and Running" by Regina Obe and Leo Hsu or the "High Performance MySQL" book by Baron Schwartz Peter Zaitsev and Vadim Tkachenko.

Also check out papers on relational algebra and query optimization they will teach you the inner workings of join operators and how optimizers choose execution plans for them. Learning about the theoretical foundations of this is important if you are serious about mastering databases as I have been.

**Conclusion**

So inner join returns the matching rows from both tables based on join conditions and semi join checks for existence of matches from one table into another and returns results based on the first table alone. They solve different problems so knowing the difference is very important and will save you a lot of headaches debugging your queries in the future. Hope that helps and happy querying. Let me know if you still have questions
