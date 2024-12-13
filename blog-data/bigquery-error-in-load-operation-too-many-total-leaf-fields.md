---
title: "bigquery error in load operation too many total leaf fields?"
date: "2024-12-13"
id: "bigquery-error-in-load-operation-too-many-total-leaf-fields"
---

Okay so you're getting that "BigQuery error in load operation too many total leaf fields" eh been there done that got the t-shirt and probably accidentally deleted a few staging tables along the way let me tell you

This error screams one thing you've got a monstrous schema trying to cram way too many columns into BigQuery it's like trying to fit a giraffe into a Mini Cooper you know its gonna end badly BigQuery has its limits especially when you're dealing with nested or repeated fields each of those contributes to the leaf field count and it adds up quick

Now you might think hey I've got a super wide table that's just how my data rolls but BigQuery isnt really built for that kind of super wide schema instead its designed to work really efficiently with normalized or semi-normalized data structures think more Star Schema less flat spreadsheet from hell

I've had this happen back in my early days we were trying to load some data from a legacy system it was basically a dump of everything a bunch of XML files with deeply nested repeated elements we decided lets just jam that directly into BigQuery you know for speed we got that error and many other errors right away Lesson learned the hard way

The issue here is a hard limit that BigQuery imposes on the complexity of the schema. It's not necessarily about the data's size or volume it's about how many individual *fields* you have at the very bottom of that schema tree including your deeply nested records This limit isn't about total data size its about schema complexity think of it like counting every single grain of sand not the weight of the whole beach

So what do we do about it First you need to assess your current schema use the BigQuery UI or CLI to get a good view of how many leaf fields you actually have maybe just even copy the schema to a text editor and do a count there its a bit of a hack but sometimes the simple things work you might be surprised how many fields you've got hiding in those nested structures

Then the real work begins schema optimization which is basically fancy talk for restructuring your data So heres a few things I've done that have worked for me over the years

**1 Flattening Nested Data:**

This is the first thing I usually try if you've got repeated or nested records try to break them down into separate tables this will drastically reduce the leaf field count. You'll likely end up with more tables but simpler schemas in each table and a lot less headaches in the long run

```sql
-- Example: Flattening a nested JSON structure

CREATE OR REPLACE TABLE `your_project.your_dataset.flat_table` AS
SELECT
  record.id AS id,
  element.value AS element_value
FROM
  `your_project.your_dataset.source_table` AS record,
  UNNEST(record.nested_array) AS element;

```
In this example I'm taking a record with a nested array and turning it into a table where each element in the nested array has a row related to that original record this is the basic principle of denormalizing or flattening data and it reduces the complexity of the original table that produced the error I would probably start here in most cases as it provides the most value for the effort put in

**2 Using STRUCT types sparingly:**

While `STRUCT` types are great for grouping related fields together they also contribute to the leaf field count so if you are nesting too deeply in your STRUCT consider again flattening the data into separate columns or tables sometimes its about the complexity of the structure not the amount of the data

```sql
-- Example: Exploding a STRUCT into Separate Columns
CREATE OR REPLACE TABLE `your_project.your_dataset.flat_struct_table` AS
SELECT
  struct_field.name as name,
  struct_field.age as age,
  other_field as other_field,
FROM
  `your_project.your_dataset.struct_table`;
```
This example takes a table that uses STRUCT which I've encountered a lot it just explodes those fields into separate columns this is pretty simple stuff right here so it's an easy way to solve problems quickly

**3 Staging the data**

 Sometimes you have no choice you just have to load the complex schema and deal with the consequences what you can do here is to load it into a temporary staging table that has the complex schema then do the data transforms with SQL to output into proper tables with less leaf fields this might involve multiple queries and steps but it is often a necessary thing to get the desired end results
```sql
-- Example: Loading into a staging table and transforming to target table

--Step 1: Create the staging table
CREATE OR REPLACE TABLE `your_project.your_dataset.staging_table` (
    nested_data STRING
  )

--Step 2: Load the data into the staging table with the nested data as STRING
--Load the nested schema data as string

--Step 3: Process and create the target table
CREATE OR REPLACE TABLE `your_project.your_dataset.target_table` AS
SELECT
    JSON_VALUE(nested_data,'$.id') as id,
    JSON_VALUE(nested_data,'$.name') as name
FROM
    `your_project.your_dataset.staging_table`
```
In this example we are using JSON functions to extract data from the nested structure loaded as string into the target table so that we do not have a schema that has too many leaf fields this process is a bit slower than loading data directly but it allows you to load data when you are stuck in a corner and still avoid the leaf field limit

So what about other stuff? If you want to really understand this stuff I would highly recommend "Data Modeling for Big Data" by Thomas Erl and "Designing Data-Intensive Applications" by Martin Kleppmann both are must reads in my book. These aren't just about BigQuery they are about the core principles of data modeling and they will help you avoid these kind of issues in the first place.

And yeah there's always that one time where you spend hours debugging a query only to realize you had an extra comma somewhere. Its not funny you just want to yell at the computer.

Remember sometimes less is more in data modeling and BigQuery rewards you for keeping things simple and structured. Good luck out there you'll get the hang of it
