---
title: "time series database vs relational database difference?"
date: "2024-12-13"
id: "time-series-database-vs-relational-database-difference"
---

 so this again eh time series data versus relational databases I’ve been down this road so many times feels like a deja vu but hey let's get into it It's not a new question by any means and its importance is understated by many newcomers

First off relational databases your MySQL PostgreSQL Sqlite the usual suspects they're built for data where relationships are the key thing think of your standard application data users products orders stuff that’s naturally related in structured ways relational is in the name really. Data is typically normalized you know breaking stuff into tables with foreign keys and all that jazz. It’s great for ensuring data integrity and consistency that's its bread and butter. Queries are often complex requiring multiple joins and conditions to get the right answer. This is what they are optimized for. You can do time based queries no problem but if all you are querying are time based this is like using a tank to crack an egg.

On the other hand time series databases that’s where things get interesting if your data has a time dimension its first class citizen then these things are for you. These are databases like InfluxDB Prometheus TimescaleDB specifically designed to store data points that are indexed by time with optional tags and fields. It is about time first then other data. Imagine a sensor monitoring temperature voltage you have a lot of data coming in all indexed by time the query patterns are usually around time ranges and data aggregation not complicated joins and relationships. 

I remember way back in 2010 I was working on a system to monitor network devices and I was initially using PostgreSQL yeah mistake number one. It worked but it was slow queries were killing the database specially when trying to aggregate data over time. I was basically using a car to do gardening. It was cumbersome and inefficient. My queries ended up in unreadable SQL madness with loads of group by and window functions it was a mess I had to switch it out for influxDB for my own sanity at least. It made life significantly better and I started to sleep better that week.

One key difference and the biggest performance driver comes from how data is stored internally. Relational databases store data row wise meaning all the attributes of a record are stored together. Time series databases on the other hand store data column wise storing all time values together all tag values together and all field values together. This approach is much more suited to time series queries which usually aggregate over one field and a time range this means less disk I/O and faster query performance. It’s like trying to find all the same color blocks in a random pile versus finding all colors on their own piles. Obvious difference right?

Another factor is data retention policies which in time series database are first class citizens. They’re built in mechanisms to automatically drop older data which is crucial for managing space when dealing with high velocity time based data. You don't have to script your own deletion scripts anymore. This is a critical feature for many use cases and relational databases simply are not built for this. They are usually used as an archive not a drop box.

Let me give you some code examples just to show the difference first a SQL example.

```sql
-- Example SQL query in a relational database
SELECT
    time_bucket('1 hour', timestamp) AS time_hour,
    AVG(temperature)
FROM
    sensor_data
WHERE
    timestamp >= NOW() - INTERVAL '1 day'
GROUP BY
    time_hour
ORDER BY
    time_hour;

```

This SQL query is simple enough to understand it tries to get an hourly average temperature of a sensor. Notice the use of `time_bucket` this is a extension in Postgres. Most other SQL based databases do not even have that function. This demonstrates the awkwardness of such operations in the relational databases that is not their focus. Now let’s look at what it will look like in InfluxDB for example.

```influxql
-- Example InfluxQL query in InfluxDB
SELECT mean(temperature)
FROM sensor_data
WHERE time >= now() - 1d
GROUP BY time(1h)
```
See how much easier and simpler to read this is and more importantly it is very efficient compared to SQL variant for its specific purpose. It reads like how you would describe the problem you are trying to solve it is more intuitive. It should be obvious which one is better for this type of problem.

And here's one more using PromQL from Prometheus just to give you another perspective of time series query language.

```promql
# Example PromQL query in Prometheus
avg_over_time(temperature[1h])
```

Notice how simple and readable the syntax is it reads almost like you would describe the query in plain english. It's designed for quick retrieval and aggregation of time based metrics.

I once spent an entire weekend debugging a stored procedure for a client trying to generate hourly reports for some IoT sensors it was a nightmare. I was literally using a Swiss Army knife to cut a bread and we all know what’s gonna happen with that. After switching to InfluxDB the whole report generation process went from minutes to seconds. And the codebase was much easier to maintain I mean I almost lost my mind with those complex joins and nested sub queries in SQL and it was not even that complex but it was for the relational db.

So when do you choose one over the other

Relational database makes sense if you have data that has complex relationships and you need ACID transactions think financial systems inventory management users products etc.

Time series databases on the other hand are ideal when you are dealing with time series data that does not need complex relational properties where your focus is on time based aggregation data monitoring sensor data stock market data server metrics etc. Also if you want to go further you can try column store databases like apache parquet which is like a time series database in a format. In that case you should be using a processing framework like spark or trino. Also if your data is really really large you can also consider something like Clickhouse which is a column oriented database.

If you want to dive deep into the theory and performance of database systems I recommend “Database System Concepts” by Silberschatz et al it’s a classic text for foundational understanding. For a good understanding of the time series specifically checkout the “Time Series Databases: New Ways to Store and Access Data” paper by Peter Bailis et al it's a really good read. I swear if I have to read one more paper about database theory I'm going to print them all out and build a fort out of them. Seriously though read those if you are serious about it.

And please stop using the wrong tool for the wrong job it makes all of our lives way harder than it should be. You will thank me later trust me.
