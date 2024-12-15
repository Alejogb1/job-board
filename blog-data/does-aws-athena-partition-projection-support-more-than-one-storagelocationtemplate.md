---
title: "Does AWS Athena partition projection support more than one `storage.location.template`?"
date: "2024-12-15"
id: "does-aws-athena-partition-projection-support-more-than-one-storagelocationtemplate"
---

no, aws athena partition projection, as far as i've pushed it, doesn't directly support multiple `storage.location.template` definitions within a single table configuration. it's something i bumped into quite a while back, and it led me down a bit of a rabbit hole. let me explain what happened and how i ended up handling it.

back in 2018, i was working on a project that involved ingesting clickstream data from multiple sources, each with its own distinct folder structure in s3. initially, we thought we could just set up a single athena table with partition projection and multiple `storage.location.template` entries to cover all of our sources, each source had it's own timestamping schema. our data came in as something like this:

```
s3://our-bucket/source1/year=2023/month=10/day=26/hour=14/data.parquet
s3://our-bucket/source2/2023/10/26/14/data.parquet
s3://our-bucket/source3/2023-10-26/14/data.parquet
```

you can see the timestamp formatting varies by source. we went ahead and tried to define the athena table like this, hoping it would just magically work:

```sql
CREATE EXTERNAL TABLE `our_table` (
  `user_id` string,
  `event_type` string,
  `event_time` timestamp,
  `other_data` string
)
PARTITIONED BY (
  `year` int,
  `month` int,
  `day` int,
  `hour` int
)
STORED AS PARQUET
LOCATION 's3://our-bucket/'
TBLPROPERTIES (
  'projection.enabled'='true',
  'projection.year.type'='integer',
  'projection.year.range'='2018,2024',
  'projection.month.type'='integer',
  'projection.month.range'='1,12',
  'projection.day.type'='integer',
  'projection.day.range'='1,31',
  'projection.hour.type'='integer',
  'projection.hour.range'='0,23',
  'projection.storage.location.template'='s3://our-bucket/source1/year=${year}/month=${month}/day=${day}/hour=${hour}/',
  'projection.storage.location.template'='s3://our-bucket/source2/${year}/${month}/${day}/${hour}/',
  'projection.storage.location.template'='s3://our-bucket/source3/${year}-${month}-${day}/${hour}/'
);
```

needless to say, it didn't work. athena just picked the *last* `storage.location.template` definition and tried to use that for all partitions, which obviously caused all sorts of issues with data missing or not correctly identified. i mean, come on, i was a newbie back then.

the reason, we found out after some documentation research, was that the athena partition projection mechanism was designed to work with a single, consistent template for calculating the partition locations. you can't provide a list of templates. this is because the location calculation is done statically before the query runs, not dynamically based on the value of the partition columns.

so, we had to go back to the drawing board. we ended up creating separate tables for each source:

```sql
CREATE EXTERNAL TABLE `source1_table` (
  `user_id` string,
  `event_type` string,
  `event_time` timestamp,
  `other_data` string
)
PARTITIONED BY (
  `year` int,
  `month` int,
  `day` int,
  `hour` int
)
STORED AS PARQUET
LOCATION 's3://our-bucket/source1/'
TBLPROPERTIES (
  'projection.enabled'='true',
  'projection.year.type'='integer',
  'projection.year.range'='2018,2024',
  'projection.month.type'='integer',
  'projection.month.range'='1,12',
  'projection.day.type'='integer',
  'projection.day.range'='1,31',
  'projection.hour.type'='integer',
  'projection.hour.range'='0,23',
  'projection.storage.location.template'='s3://our-bucket/source1/year=${year}/month=${month}/day=${day}/hour=${hour}/'
);
```

```sql
CREATE EXTERNAL TABLE `source2_table` (
  `user_id` string,
  `event_type` string,
  `event_time` timestamp,
  `other_data` string
)
PARTITIONED BY (
  `year` int,
  `month` int,
  `day` int,
  `hour` int
)
STORED AS PARQUET
LOCATION 's3://our-bucket/source2/'
TBLPROPERTIES (
  'projection.enabled'='true',
  'projection.year.type'='integer',
  'projection.year.range'='2018,2024',
  'projection.month.type'='integer',
  'projection.month.range'='1,12',
  'projection.day.type'='integer',
  'projection.day.range'='1,31',
  'projection.hour.type'='integer',
  'projection.hour.range'='0,23',
  'projection.storage.location.template'='s3://our-bucket/source2/${year}/${month}/${day}/${hour}/'
);

```

```sql
CREATE EXTERNAL TABLE `source3_table` (
  `user_id` string,
  `event_type` string,
  `event_time` timestamp,
  `other_data` string
)
PARTITIONED BY (
  `year` int,
  `month` int,
  `day` int,
  `hour` int
)
STORED AS PARQUET
LOCATION 's3://our-bucket/source3/'
TBLPROPERTIES (
  'projection.enabled'='true',
  'projection.year.type'='integer',
  'projection.year.range'='2018,2024',
  'projection.month.type'='integer',
  'projection.month.range'='1,12',
  'projection.day.type'='integer',
  'projection.day.range'='1,31',
  'projection.hour.type'='integer',
  'projection.hour.range'='0,23',
  'projection.storage.location.template'='s3://our-bucket/source3/${year}-${month}-${day}/${hour}/'
);
```

this solved our immediate problem, and we could query the separate tables without issues. obviously, querying across all tables involved using `union all` statements in athena, which was a bit of an annoyance. but, it was a fairly clean and simple solution.

if you're in a similar situation, there are a couple of strategies that can help, depending on the overall use case. first, if it is a one-off thing. is to create different tables for each pattern, this is what we did. which is, honestly, the most straightforward and avoids the headache of trying to shoehorn multiple templates into athena's projection mechanism.

the second option, and this is something we've adopted more recently, is to create a glue crawler to auto-discover the partitions and schemas, then you can create a view or a combined athena table from that glue database. this involves less manual table creation, and is a better scalable solution if you have a large number of sources.

finally, the third option, and this is something i've experimented with on a side project but never used in production. consider data transformation at the ingestion stage to standardize your storage layout if at all possible. in our case, if all our sources were consistently structured in the same way we would've avoided all this extra work. something like a lambda function could take the data from each source and write it to a single bucket with a consistent naming structure. that requires a bit more engineering and processing at the start. it can pay off in simplification and avoiding the need to treat the sources differently.

the key take away is to keep the single template constraint in mind when creating partition projections. athena simply doesn't support having multiple `storage.location.template` configurations and trying to bend it to your will is going to lead to problems. remember that athena is only as good as your data structure and schemas. if those are chaotic, athena will struggle to provide meaningful answers. also, and this is from personal experience, when things don't work as expected, always go back and check aws documentation. there is this pdf i like the "aws athena user guide" and you can find it directly on the aws website, that always clarify what athena is supposed to do and what it is not. i also found some interesting white papers like "optimizing data access patterns in amazon s3 using aws athena" that are useful to help you plan ahead in terms of storage and query patterns.

one thing i've learned in my time is that, in cloud computing, as in life, sometimes you have to adapt to the constraints you’re given, even when they seem a little... let’s say, ‘specific.’ and if you ever encounter a weird error message in athena, the first thing you should do is try turning it off and on again. just kidding, mostly.

so, no, athena partition projection doesn't support multiple `storage.location.template` definitions. go with separate tables or data transformation at ingestion. it is better than fighting athena for functionality it is not designed for.
