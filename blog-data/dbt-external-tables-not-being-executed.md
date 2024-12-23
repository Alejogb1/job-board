---
title: "dbt external tables not being executed?"
date: "2024-12-13"
id: "dbt-external-tables-not-being-executed"
---

 so you're having trouble with dbt external tables not running that's a classic I’ve been there trust me It usually boils down to a few common culprits I've debugged this exact thing more times than I care to admit

First things first lets talk about the basics of how dbt handles external tables Generally dbt uses a `schema.yml` file to describe your external tables The important part is the `external` config section in that file This tells dbt the file format where the data lives its location and crucially how to access it

```yaml
version: 2

models:
  - name: my_external_table
    description: "My external table from S3"
    config:
      materialized: external
      external:
        location: "s3://my-bucket/my-data/"
        file_format: parquet
        partitions:
          - name: dt
            data_type: date
        options:
           compression: snappy
```

Make sure you have this set correctly especially the `location` and `file_format` are paramount Obviously `s3://my-bucket/my-data/` should be your actual s3 location and `parquet` or whatever `csv` you have should match the data type you are working with And `partitions` if relevant is mandatory otherwise it won't work

Now if it's not the `schema.yml` which is usually the problem lets go deeper Here’s a common mistake I've personally made more than once: not having the right permissions. dbt itself runs under a certain service account or role and it needs to be able to read the external data source. This means your cloud provider's IAM setup needs to allow dbt's user or role to access the S3 bucket or database table or Google Storage location. Double check that your service principal dbt uses has the required permissions to your external storage.

I remember back in 2019 working on a massive data migration project we were using Google Cloud Storage for our data lake and I spent two whole days because the service account we were using for dbt was missing `storage.object.get` permission on the bucket The error message was something cryptic like 'access denied' I mean who writes these error messages really right We wasted so much time because of that

 so you've checked the yaml file and the permissions are sorted so where else could things go wrong? Ah right another frequent offender the infamous "data not found" error it’s almost always a location mismatch. dbt looks at the location in the `schema.yml` and if that location is wrong in any way even if a little bit then it fails without telling you the exact place it fails This means the location in `schema.yml` *must exactly match* where your data is stored. And it needs to be the correct bucket region too. It's so trivial but I can’t tell you how many times I've made this error in particular

```sql
-- A sample dbt model file using the external table
select *
from {{ source('your_schema', 'my_external_table') }}
where dt >= '2023-01-01'
```

If you have the `source()` defined in `schema.yml` or just have it defined in the SQL file as a CTE make sure the schema location and table name are also accurate Another potential gotcha is the partitions if you have them dbt uses them to filter the data for you by running it as part of the query Make sure you understand how partitions are defined how they are stored and in which type format and check if they match what you set in the `schema.yml` If there's a discrepancy between your declared partitions and how the data is actually partitioned dbt will either fail or worse give you wrong data and you will be debugging that all day which i am sure we all don't want

Also remember external table configuration options differ slightly depending on your cloud provider Each cloud provider has slightly different options for external tables so always verify the official documentation for your cloud service.

Now for the fun bit this one is a bit obscure It’s all about schema evolution and file changes sometimes the external data schema can change but the definition in your `schema.yml` might not be updated This is very very common especially when you have a lot of files or data that goes into the external table. dbt doesnt have a way of automatically knowing when your schema changes so you might need to manually update the `schema.yml` file.

```yaml
version: 2

sources:
  - name: your_schema
    description: "External data sources"
    tables:
      - name: my_external_table
        description: "External table from S3"
        external:
          location: "s3://my-bucket/my-data/"
          file_format: parquet
          columns:
            - name: column1
              data_type: integer
            - name: column2
              data_type: string
            - name: dt
              data_type: date
          partitions:
            - name: dt
              data_type: date
          options:
            compression: snappy
```

So if your data had schema changes now you are forced to manually update your `schema.yml` file with the new columns that you have This is essential because dbt uses it to do schema validations so you dont end up reading the wrong type or getting wrong data.

Now before moving on to specific errors sometimes dbt fails because you did not trigger a full refresh of the external table. If your changes are recent and dbt has the old version cached then you need to run the model with the `--full-refresh` flag sometimes this is the answer to a lot of problems. So always remember this flag when making drastic changes.

And another thing that can break things is incremental builds with external tables. External tables are often not meant to be incrementally updated if you try to define your external tables as such you might get an error or not have the correct updates available So be careful when setting incremental updates with external tables.

If you are still stuck debugging here are some other general things to verify.

1.  **dbt Logs**: Check the dbt logs for detailed error messages. They often give you clues about why the external table creation is failing The verbose mode `-v` is your friend here as they might contain some very helpful and detailed error messages.
2.  **Cloud Provider Logs**: Look at your cloud provider's logs for the resources being used by dbt to find any access denied errors
3.  **Version Compatibility**: Confirm that dbt-adapter version for your data warehouse is compatible with dbt version you have.
4.  **File Formats:** Make sure the file format of your external table matches what you have specified in the `schema.yml` file as some cloud providers are not flexible about the file types.
5.  **Network Issues:** Check if there are any network issues preventing dbt from reaching your external storage.

I would recommend reading *The Definitive Guide to dbt* by Fishtown Analytics it is a great starting point to dive into details of external tables and how they are constructed and used. Another good resource is the official dbt documentation which explains in depth every functionality of dbt including external tables. If you still cannot figure out the problem after looking at these two then you might want to go through your cloud providers documentation on external tables such as AWS documentation for Athena and S3 or Google BigQuery documentation on external data sources. And if after all these the errors still persist maybe it's time for a new job but hopefully not I hope this helps.
