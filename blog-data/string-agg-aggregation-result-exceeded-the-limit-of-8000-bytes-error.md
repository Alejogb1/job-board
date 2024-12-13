---
title: "string agg aggregation result exceeded the limit of 8000 bytes error?"
date: "2024-12-13"
id: "string-agg-aggregation-result-exceeded-the-limit-of-8000-bytes-error"
---

Alright so you're hitting the "string agg aggregation result exceeded the limit of 8000 bytes" error huh Classic Happens to the best of us Seriously I've seen it more times than I've had hot coffee and that's saying something

I get it you're probably trying to concatenate a bunch of strings into one big ol' string using string_agg or a similar aggregate function in your SQL database and bam it throws this error It's like you're trying to cram way too much data into a tiny suitcase and the database is just going "Nope Not gonna happen"

Okay so this isn't exactly a database problem per se its more about how databases are structured and what they allow to prevent resource hogging Believe it or not this 8000-byte limit isn't some arbitrary number some database admin pulled out of thin air It's often a safeguard to stop queries from going wild and eating up memory like a kid in a candy shop

When you use `string_agg` or similar functions the database needs to store this increasingly larger string in memory as it aggregates The 8000-byte limit is often a default buffer size to stop this from going completely out of hand Imagine what would happen if a poorly crafted query decided to aggregate the content of all the tables into one single string the database server would instantly become unresponsive

Now I'm not gonna lie this limitation can be a real pain in the behind especially when you're dealing with unpredictable data sizes

I remember one project back in 2012 or maybe it was 2013 I was working on a data migration tool and I needed to aggregate a bunch of text comments for analysis we were moving this old legacy system to this brand new shiny cloud platform The old system did not enforce any text length restrictions on comments the comments could be as short as two words or as long as a short story The idea was to consolidate and index these comments I wrote a query similar to something like this and boy oh boy did I regret it

```sql
SELECT
    user_id,
    string_agg(comment_text, '; ') AS aggregated_comments
FROM
    comments_table
GROUP BY
    user_id;
```

Simple enough right Yeah I thought so too But lo and behold I started getting the 8000-byte error right away Turns out some users wrote epic length responses to every tiny change that happened on the platform the string aggregation became a huge problem

So first thing you need to do is accept this problem and its limits and understand you have a data problem not a database problem

Alright enough rambling let's look at some solutions I've picked up over the years

**1 Reduce the data**

This is the most common and usually most effective solution You can start by limiting the number of strings you are aggregating in one group The most common way to do this is filtering the dataset.

For example instead of fetching all comments you can fetch the most recent comments or fetch only comments that contain certain keywords

```sql
SELECT
    user_id,
    string_agg(comment_text, '; ') AS aggregated_comments
FROM
    (SELECT user_id, comment_text
    FROM comments_table
    WHERE creation_date >= DATE_SUB(CURDATE(), INTERVAL 30 DAY)) AS subquery
GROUP BY
    user_id;
```

Here I just added a `WHERE` clause in a subquery to fetch only comments from the past 30 days This alone might fix your problem

**2  Use substrings**

If cutting down records isn't possible then you can work on the strings themselves If you absolutely need to keep all the data one option is to truncate each string to a fixed length before aggregation This isn't ideal if you need the whole text but can work for summaries or short snippets

```sql
SELECT
    user_id,
    string_agg(LEFT(comment_text, 200), '; ') AS aggregated_comments
FROM
    comments_table
GROUP BY
    user_id;

```

Notice the `LEFT(comment_text, 200)` function which will take the first 200 characters of each comment this might be sufficient for your needs I'm not making the rules here I'm just a simple programmer trying to get the job done.

**3  Split the data and then aggregate**

This one is a bit more complex but often the most reliable for large datasets and unlimited string lengths The idea is to break down your data into chunks before aggregation this is most useful when aggregating values over columns rather than rows.

It involves doing the aggregation in multiple steps then join the aggregated values by `user_id` or a common key It's like cooking a big meal you prepare every ingredient first then you cook every part of it and then combine it all into a single dish. It takes a little more effort but the end result is usually better

This often involves the use of CTEs (common table expressions) for modularity and code readability.

```sql
WITH PreAggregated AS (
    SELECT
        user_id,
        comment_text,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY some_order_column) AS rn
    FROM
        comments_table
),
ChunkedAggregated AS (
  SELECT
        user_id,
        string_agg(comment_text, '; ') AS chunk_comments,
        FLOOR((rn - 1) / 10 ) as chunk_id
    FROM PreAggregated
    GROUP BY user_id, FLOOR((rn - 1) / 10)
),
FinalAggregated AS (
    SELECT
         user_id,
        string_agg(chunk_comments, '|||') as final_comments
    FROM ChunkedAggregated
    GROUP BY user_id
)
SELECT * FROM FinalAggregated
```

So what's happening here?

First `PreAggregated` simply adds a row number to each comment we will use this to chunk the data

Then `ChunkedAggregated` groups the comments based on `user_id` and the `chunk_id` that we calculated by `FLOOR((rn - 1) / 10 )` In my example it will chunk every 10 comments if a user wrote 100 comments each user will have 10 rows of aggregated comments

Then `FinalAggregated` aggregates each of those chunks into a final result using `|||` as the separator.

You might have to do postprocessing with some external code to replace `|||` separator with whatever delimiter you require but the idea is to separate the huge aggregations to avoid the 8000-byte limit

This is just an example of course the `chunk size` and other details will vary based on your database and requirements In this specific example we are using a `ROW_NUMBER` function to provide an arbitrary order if the original query has an order then that column can be used instead for example `ORDER BY creation_date`

The key here is that we are not aggregating every single comment of a user into a single large string rather we are aggregating smaller chunks of strings at a time avoiding the 8000-byte limit

**Important notes on solutions**

These solutions may need adaptations depending on the specifics of your situation I'm not really sure what your data schema is or what is your exact needs You should always choose the method that best suits your situation based on the characteristics of your data

For instance in my old migration project I ended up using a combination of methods 2 and 3 the first approach was not feasible because we needed all the comments and just shortening it would render our analysis useless so I split the comments into chunks of 10 and truncated the length of the strings so it would fit into the limit

**Where to learn more?**

Look at academic papers or books that cover database system design especially those related to query optimization and memory management The "Database System Concepts" by Silberschatz Korth and Sudarshan is a good start for theoretical concepts or "Understanding SQL" by Martin Gruber for more practical approaches on SQL and database engines.

And also maybe consider having a more detailed discussion with a data specialist if you need to optimize complex queries The 8000-byte limit is there for a reason and sometimes it requires more than just a quick fix

Oh one last thing I hope you have a better day than the guy who designed SQL error messages that was a joke but not a funny joke.
