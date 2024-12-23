---
title: "multiple case when sql query?"
date: "2024-12-13"
id: "multiple-case-when-sql-query"
---

so you're hitting a classic sql headache a multiple case when scenario right I’ve been there trust me been burned by that specific syntax more times than I care to admit especially early in my database journey let's break it down think of it like a series of if else statements just sql style

 so first things first we need to understand the core idea behind multiple `CASE WHEN` I mean it's not that complicated but you know sometimes it’s the simple stuff that trips you up it's essentially a way to evaluate different conditions within a single query and return a different value based on each one think of it as a switch statement but in the database realm instead of looping through data we use it to classify data into buckets or to transform it on the fly it's powerful stuff when you need to create computed columns or categorized data without changing the underlying table itself

I remember this one time back in like 2012 I was working on this e-commerce platform project you know the classic stuff we had a product table and we needed to generate a report showing a product's price category like "cheap" "medium" "expensive" based on price I mean we could have done it with application code or a stored procedure later but I was kinda new to sql at the time I went straight for the `CASE WHEN` in one giant query we had like 6 price categories and boy it was ugly the query was a monster I'm talking like 30 lines of just nested `CASE WHEN` statements and yeah you guessed it debugging that was a nightmare a proper spaghetti code situation but it worked though it worked I learned that day proper indentation and thinking before you type goes a long way it really does

Anyways before I go into war stories further let me give you a basic example of how a multiple `CASE WHEN` looks like

```sql
SELECT
    product_name,
    price,
    CASE
        WHEN price < 20 THEN 'Cheap'
        WHEN price >= 20 AND price < 50 THEN 'Medium'
        WHEN price >= 50 THEN 'Expensive'
        ELSE 'Unknown'
    END AS price_category
FROM
    products;
```

See this this is the most basic example if price < 20 we tag it as cheap if it’s between 20 and 50 its medium and so on if no condition is met we give it unknown you can nest them like I did in that earlier e-commerce project but again not recommended it becomes a huge mess it makes your queries less readable and much harder to debug especially when you start introducing complex logic

So that was example one pretty straightforward I'd say Let’s go with something a bit more realistic I had this issue not long ago where I had a table containing user activity data we needed to find the user's current status based on their latest activity timestamp we had three main activity types "login" "logout" and "idle" our status calculation rules were simple if a user’s last event was login it’s considered "active" if logout "inactive" and for idle we did a little more complex evaluation we checked the idle timestamp against a configured timeout value if it was before that timeout we tagged them idle if they were within the timeout period it was deemed as “away” so a few more cases but you can already imagine the case when statement we used

```sql
SELECT
    user_id,
    MAX(activity_timestamp) AS last_activity,
    CASE
        WHEN MAX(activity_type) = 'login' THEN 'active'
        WHEN MAX(activity_type) = 'logout' THEN 'inactive'
        WHEN MAX(activity_type) = 'idle' AND MAX(activity_timestamp) < (CURRENT_TIMESTAMP - INTERVAL '15 minutes') THEN 'idle'
        WHEN MAX(activity_type) = 'idle' AND MAX(activity_timestamp) >= (CURRENT_TIMESTAMP - INTERVAL '15 minutes') THEN 'away'
        ELSE 'unknown'
    END AS user_status
FROM
    user_activities
GROUP BY
    user_id;
```

This example shows how `CASE WHEN` can be used with functions like max and logical operators and time manipulation functions too you will see that we use group by user id as we are performing aggregation on the user activity table we are getting the latest activity and making a status out of it This situation requires more thinking and if you are not used to it can get a little confusing but it's not so bad in reality You have to group by to do this kind of stuff you just have to remember it

The other key point in complex use cases is the order of the conditions in your `CASE WHEN` statement the order matters a lot the first matching condition will be selected and the rest will be ignored it's like going to the store first store you find your stuff you buy it and move on you don't have to keep on going to another stores if you found what you were looking for already This also gives a very useful feature where you can have a default value at the end. Think of it like the else part in an if else statement but sql style. If none of the conditions are met then that default value will be returned. So be very careful when you're dealing with overlapping conditions in the `CASE WHEN` statements

Now to the last example and a real world gotcha a bug I encountered I was working on a project that had geographical data and we needed to categorize areas into different region types like “urban” "suburban" “rural” and we also had to handle the “null” cases properly the initial implementation using `CASE WHEN` looked simple enough but it kept returning unexpected results when it came to null values I mean that should not happen right everything was in place but the conditions with null were not working as they should we were losing a lot of data due to that small detail we did not take into account. I mean this might be more like an sql thing than specific to case when but you should be aware of it

```sql
SELECT
    area_name,
    location_type,
    CASE
        WHEN location_type = 'urban' THEN 'Urban Area'
        WHEN location_type = 'suburban' THEN 'Suburban Area'
        WHEN location_type = 'rural' THEN 'Rural Area'
        WHEN location_type IS NULL THEN 'No Location Data'
        ELSE 'Unknown Area'
    END AS region_category
FROM
    geographic_areas;
```

So after hours of debugging i discovered that the problem was that comparing with null using just `=` does not work in SQL you have to use IS NULL or IS NOT NULL that's the sql convention when dealing with null values as null is not a value but an absence of a value yeah i know it's a bit weird but that's just how SQL works I mean you can check the SQL standard definition for further clarification but that's a pretty weird read if you are not into formal language specs if you're trying to make sure you don't do this again check out "Understanding SQL" by Martin Gruber or any other good SQL book you can find for good explanations

So let’s see that was three examples of different ways to use the multiple `CASE WHEN` statement we talked about simple examples that can be used for categorization, more complex examples using aggregated data with a more real world scenario and also a nasty gotcha when dealing with null values.

A quick tip i can give you is to keep your conditions simple and your queries readable it always saves you a lot of time later on I mean I've worked with massive SQL statements with multiple `CASE WHEN` statements that were a real headache to read and maintain but you don't want to repeat my mistakes right? Think about the logic you are trying to implement make sure you understand it and do not just go straight for the query editor and start coding I know it's tempting but take a breath first. It's like trying to assemble a computer without having a manual first you are going to be in trouble very quickly and spend more time debugging than doing something useful I am speaking from my own experience here

Anyway that’s the multiple `CASE WHEN` in sql in a nutshell you have to know it well when you are in the database world it’s one of the most frequently used clauses in many projects you will be using it a lot I mean we programmers like to put the logic into the database most of the time for performance reasons, less data transfer faster response times and more. And that’s why `CASE WHEN` is so important in the SQL world.

Hopefully it was clear enough If you need any further clarification you know where to find me. You can hit me with any questions later on. But just be sure that you try it yourself and not just copy paste my code it really helps to learn by doing. You know what they say right? If I had a nickel every time someone copied a code and did not understood it I'd have a truckload of money or maybe several I guess it does not work like that in reality right?
