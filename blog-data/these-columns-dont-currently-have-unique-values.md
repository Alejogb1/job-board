---
title: "these columns dont currently have unique values?"
date: "2024-12-13"
id: "these-columns-dont-currently-have-unique-values"
---

Okay so you're saying these columns don't have unique values right I've been there trust me This sounds like a data integrity headache a classic if you ask me I've wrestled with this exact beast more times than I care to admit and it's almost always a pain

So you've got columns where the values aren't unique That means you've probably got duplicates and that's not good for a lot of operations especially if you're trying to treat these columns as primary keys or foreign keys Or even if you're just trying to do some aggregation that needs distinct values for accuracy This isn't just bad for your data it's bad for your soul So first things first let's diagnose the situation and see what we're working with

Okay you need to find out which columns exactly have those duplicated values It's not always obvious even with a small dataset and a huge one well let’s not go there We can use a simple SQL query to check this I use SQL most of the times for these kind of tasks and you should too

```sql
SELECT column_name, COUNT(*)
FROM your_table
GROUP BY column_name
HAVING COUNT(*) > 1;
```

This query will give you a list of the columns where there are non unique values that is where the count of distinct values is greater than one you need to substitute your actual table name where I put `your_table` of course and check that `column_name` can be one or more column names depending on your case

That's a good starting point if your database supports it and you are working with one I use postgres usually It’s reliable and does the job with minimal fuss but other SQL options will work just fine The next step now that we know which columns are the troublemakers we have to figure out what to do with these duplicated values You can go many routes depending on the context of the situation and your requirements

Now there are many ways to handle these duplicates I've seen people take many approaches for example you can deduplicate by using distinct keys if you have a natural one you can also use other columns that give uniqueness or you can add a new column with a sequence id so you can enforce that as a unique constraint but that also depends if you want to maintain the original values or not

Let’s say you have a table with user information and you realize that the `email` column isn't unique You might have users with the same email which is not only bad but also kind of funny right So what I would do is either pick a unique email for that user or just drop the duplicate values but I need to check the business requirements before doing anything I learned that the hard way believe me

One way to deduplicate based on some values is to use row_number function from SQL it's very practical for handling those cases with duplicate rows lets see this code

```sql
WITH Ranked AS (
    SELECT *,
           ROW_NUMBER() OVER(PARTITION BY email ORDER BY created_at DESC) as rn
    FROM users
)
DELETE FROM Ranked
WHERE rn > 1;
```

In this code we are selecting all rows from the table users and partitioning them by email this means each email value will get a number from the function `row_number` it will start from 1 and increment for the rest of the duplicate emails and sort them by created at date in a descending way this means that the last inserted value will be the one with row number 1 and the rest are duplicates we're selecting all the values from `Ranked` alias where row number is greater than 1 and are deleting them to keep the distinct values by email and the latest entry for those users

That is one way of deduplicating rows in sql but if you do not want to delete any rows and instead want to keep all of them and still enforce a unique constraint on one of those columns you can add a new column with an id that you can then use as a primary key in your database and in your code

Adding a unique id is helpful for cases like where there are no other columns for a natural key or the columns that exist can have duplicate values and the requirements state that you have to keep all rows in your dataset

```sql
ALTER TABLE your_table ADD COLUMN id SERIAL PRIMARY KEY;
```

This query adds a new column named `id` which is of type `SERIAL` meaning it's an auto-incrementing integer and it also sets it as the primary key This means you can now use this `id` column to uniquely identify each row in your table and avoid any other problems with your other columns that have duplicated values and need to be joined by other tables

So that's it that's the gist of it deduplicating columns is a classic problem and there are plenty of resources out there if you're interested in learning more For a deep dive into SQL window functions the one I use to deduplicate data, I recommend "SQL Window Functions" by Markus Winand, it's not just a book it's a bible and for general database design and data modeling "Database Design for Mere Mortals" by Michael J Hernandez and Richard J. Owens it's a foundational resource every developer should have read

The approach to deduplication it's more of a case by case decision for what you are dealing with and the requirements needed for those cases but the principles are the same understand the problem diagnose the data and apply the proper deduplication method for your specific case Good luck
