---
title: "How to Extract value of Tags from cloudTrail logs using Athena?"
date: "2024-12-14"
id: "how-to-extract-value-of-tags-from-cloudtrail-logs-using-athena"
---

alright, so you're looking to pull data out of cloudtrail logs using athena, specifically the values within those nested json tags, yeah? i've been down this road myself, and it can get a little tricky if you're not careful. i spent a solid week once, trying to figure out why my queries were only returning nulls. turns out, it was just a small oversight in my json parsing.

first things first, let's talk structure. cloudtrail logs are delivered in json format, and typically they are stored in s3. athena then reads these s3 objects like tables, allowing us to use standard sql queries. this is great, but the json structure adds a layer of complexity. the crucial thing to grasp is that the tags you are after, aren’t just lying flat in the document. they’re usually nested within some json object, like `requestparameters` or `resources` and then again within `tags`. so, we need to navigate through this json tree properly using athena's json functions.

now, let's get down to the code examples, shall we? i'm assuming you've already set up your athena table pointing to your cloudtrail logs. if not, there are plenty of tutorials that will walk you through that process. i personally learned it from the aws documentation, its not the easiest read, but is very complete. i would recommend "aws big data analytics" book by shiva reddy, it really helps you with all the athena nuances.

**example 1: extracting tags from `requestparameters`**

let's start with a common scenario: fetching tags associated with an ec2 resource using the `requestparameters`. here’s how i’d approach it:

```sql
select
    eventid,
    eventtime,
    useridentity.arn,
    item.key as tag_key,
    item.value as tag_value
from
    "your_cloudtrail_database"."your_cloudtrail_table"
cross join
    unnest(json_extract(requestparameters, '$.tagspecificationSet.items') ) as t(item)
where
    eventname = 'createtags'
limit 100;
```

here is what this query does. first `select` is to choose fields, then we select some base information like event id and the time, plus the user that triggered the event. then `unnest(json_extract(requestparameters, '$.tagspecificationSet.items')) as t(item)` this is where the magic happens.  `json_extract` takes the `requestparameters` json blob and uses json path `$.tagspecificationSet.items` to pick the items of the tag specifications. this will return a json array of objects. then `unnest` function explodes that array into table rows with the name `item`. then we pick out the key and value from the item alias. and finally the `where` clause filters the records to only include ‘createtags’ events, and `limit` is just for previewing some of the results.

**important note**: the exact json paths like `$.tagspecificationSet.items` will vary based on the cloudtrail event and which service is generating the log, so you’ll need to explore the json structure of your specific events to identify the paths. usually, the documentation of cloudtrail events is really helpful. i used the documentation of cloudtrail events quite a bit, and saved me tons of hours. its best not to rely on memory as they change constantly.

**example 2: pulling tags from `resources` array**

sometimes, the tags are found within the resources array. here is how to tackle that:

```sql
select
    eventid,
    eventtime,
    useridentity.arn,
    resource.resourcename,
    tag.key as tag_key,
    tag.value as tag_value
from
    "your_cloudtrail_database"."your_cloudtrail_table"
cross join
    unnest(resources) as t(resource)
cross join
    unnest(resource.tags) as t2(tag)
where
    eventname like '%tag%'
    and resource.type = 'ec2:instance'
limit 100;
```

this example is similar to the first one, but slightly more complicated. first, we're unnesting the top-level `resources` array into separate rows as resource. then from each `resource` row we are unnesting the `tags` array again into separate rows as `tag`. this creates a flat table where each row contains info from the event, one specific resource, and one specific tag linked to that resource. the `where` clause is filtering the events to include anything with 'tag' in the event name and only for ec2 instances. this should return tag information for actions performed on ec2 instances. and again the `limit` clause to preview results.

**example 3: dealing with tags as a single json string**

in some cases, the tags are not structured as nested objects but as a single json string inside the resources array. here is the solution for that case:

```sql
select
    eventid,
    eventtime,
    useridentity.arn,
    resource.resourcename,
    tag.key as tag_key,
    tag.value as tag_value
from
    "your_cloudtrail_database"."your_cloudtrail_table"
cross join
    unnest(resources) as t(resource)
cross join
    unnest(json_parse(resource.tags) ) as t2(tag)
where
    eventname like '%tag%'
    and resource.type = 'ec2:instance'
limit 100;
```

in this example, almost identical to the previous one. the big change is the `json_parse(resource.tags)` function that parses the tag string into a json object making it possible to extract the keys and values of the tags. sometimes the tags are stored as strings and not as objects. for those cases this snippet is the solution.

let me tell you a funny story about this. i had a coworker who spent a whole day troubleshooting this, turns out he was querying the production database, but he had filtered by the wrong environment… yeah, not my finest moment when i laughed at that. but yeah these little details can get us all, i’ve been there.

**some points to consider**:

*   **performance**: for very large datasets, athena queries can take time. partitioning your data in s3 by date is crucial for performance. this will limit the amount of data athena has to scan when the query is performed. i would advise using the aws cli to update the partitions. and there are some aws labs projects for creating lambdas that update partitions automatically, i advise looking at them, they are very useful, but don’t blindly copy the code, look into it deeply and understand what they do. the paper “amazon athena: scalable interactive query engine” is useful too, that covers many aspects of performance and cost optimization for athena, i strongly recommend to read that paper.
*   **cost optimization**: athena charges based on data scanned, not by execution time, so try to be as specific as possible with the queries. limit your `select` to only the columns you really need, and narrow down the search criteria with the `where` clause.
*   **error handling**: when using `json_extract`, be prepared for null values if the json paths don't exist or if the structure isn't what you expect. using `try(json_extract())` is a great way to gracefully handle these situations, and avoid crashing the query.
*   **data types**: remember athena automatically deduces the data types of your fields, sometimes, this automatic deduction isn't correct and we will need to tell athena what is the data type. in some cases you'll need to use `cast` or other type conversion functions to work with numbers and dates in a proper way.
*   **case sensitivity**: by default, the query search is case insensitive, and this may cause you some problems, specially if your tags have names with different case versions, if that's the case, remember to use the `lower` function to avoid any problems.
*   **be patient**: you’ll almost surely encounter issues when extracting json from complex structures. experiment, try out different approaches, and be sure to understand your data before writing the query.

that's basically it. pulling tag values from cloudtrail using athena is a multi step process, requiring understanding of the structure of cloudtrail logs, and some experience with athena json functions. but, once you have a solid grasp of how to use `json_extract`, `unnest`, and `json_parse`, you will be able to slice and dice through the cloudtrail data very easily. it takes some experience, for sure. but keep at it. you’ll get there. i know i did after that initial week of null values and a few head scratching nights.
