---
title: "How to use BigQueryInsertJobOperator with Export Configuration?"
date: "2024-12-16"
id: "how-to-use-bigqueryinsertjoboperator-with-export-configuration"
---

, let's talk about `BigQueryInsertJobOperator` with export configurations. I've seen my share of data pipelines over the years, and I recall a particularly messy project involving a daily data dump from our relational database into BigQuery, followed by an export to cloud storage for further processing by a downstream system. It was initially a fragile setup, mostly due to insufficient understanding of BigQuery's job operations and configuration options. The key, I found, wasn't just knowing the syntax, but understanding the underlying mechanisms and the available parameters.

The `BigQueryInsertJobOperator`, in essence, is a powerful tool in Apache Airflow for triggering BigQuery jobs. While it’s often used for executing queries, its versatility extends to data manipulation tasks like exports. However, unlike the simpler `BigQueryOperator`, it hands over more explicit control over the job configuration using the `configuration` parameter, which allows for more nuanced operations, such as specific export setups.

So, how do we actually use this operator for exporting data? We are not just running queries; we're orchestrating a specific kind of BigQuery job: an extract job. This means the `configuration` dictionary we pass to the operator needs to be meticulously crafted. It must specify the `extract` key, which contains the source table, destination URI(s), and formatting specifications. It's not a free-for-all; the configuration needs to be in line with BigQuery's job structures. Incorrect configurations lead to job failures that can be tricky to debug without a good grasp of the underlying json structure that makes up the job specification.

Let's look at a concrete example. Imagine you have a table called `analytics.user_activity` and you want to export it to a set of CSV files in Cloud Storage, partitioned by date. Here’s how you could do it within an airflow dag using `BigQueryInsertJobOperator`:

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from datetime import datetime

with DAG(
    dag_id='bigquery_export_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    export_task = BigQueryInsertJobOperator(
        task_id='export_user_activity',
        configuration={
            "extract": {
                "sourceTable": {
                    "projectId": "your-gcp-project-id",
                    "datasetId": "analytics",
                    "tableId": "user_activity"
                },
                 "destinationUris": [
                   f"gs://your-gcs-bucket/user_activity/date={{ ds }}/user_activity-*.csv"
                ],
                "destinationFormat": "CSV",
                "printHeader": True,
                "fieldDelimiter": ",",
                "compression": "NONE",
            }
        }
    )
```

In this example, pay close attention to how the `sourceTable` dictionary defines the source using `projectId`, `datasetId`, and `tableId`. The `destinationUris` list leverages the Jinja templating offered by Airflow to generate a unique path with the data's logical date (`ds`). This way, you can easily version and partition your exports, and more importantly, avoid overwrites if the pipeline runs multiple times. Note also that I specified CSV as the `destinationFormat`, indicated a header, a comma field delimiter and no compression.

Sometimes, you may need to extract the result of a query, rather than an entire table. This is where we use a view or a temporary table. Let's examine another example where a select query results in an export, again partitioned by date:

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryCreateEmptyTableOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryDeleteTableOperator
from datetime import datetime

with DAG(
    dag_id='bigquery_query_export_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    create_temp_table = BigQueryCreateEmptyTableOperator(
        task_id="create_temp_table",
        project_id="your-gcp-project-id",
        dataset_id="analytics",
        table_id="temp_user_activity_{{ ds_nodash }}",
        schema_fields=[
            {"name": "user_id", "type": "STRING", "mode": "REQUIRED"},
            {"name": "event_timestamp", "type": "TIMESTAMP", "mode": "REQUIRED"},
            {"name": "event_type", "type": "STRING", "mode": "NULLABLE"},
            {"name": "country", "type": "STRING", "mode": "NULLABLE"}
        ]
    )


    query_and_export_task = BigQueryInsertJobOperator(
        task_id='export_query_results',
        configuration={
             "query": {
                "query": f"""
                    CREATE OR REPLACE TABLE `your-gcp-project-id.analytics.temp_user_activity_{{{{ ds_nodash }}}}`
                    AS
                    SELECT user_id, event_timestamp, event_type, country
                    FROM `your-gcp-project-id.analytics.user_activity`
                    WHERE DATE(event_timestamp) = '{{{{ ds }}}}'
                """,
                "useLegacySql": False
             },
             "extract": {
                "sourceTable": {
                    "projectId": "your-gcp-project-id",
                    "datasetId": "analytics",
                    "tableId": f"temp_user_activity_{{ ds_nodash }}"
                },
                 "destinationUris": [
                   f"gs://your-gcs-bucket/user_activity_query/date={{ ds }}/user_activity-*.json"
                ],
                "destinationFormat": "NEWLINE_DELIMITED_JSON",
                "compression": "NONE",
            }
        },
        job_id=f"query_export_job_{{ ds_nodash }}"
    )


    delete_temp_table = BigQueryDeleteTableOperator(
        task_id="delete_temp_table",
         project_id="your-gcp-project-id",
        dataset_id="analytics",
        table_id="temp_user_activity_{{ ds_nodash }}"
    )

    create_temp_table >> query_and_export_task >> delete_temp_table
```

Here, I’ve included a dependency to create a temporary table, export it, then delete the temporary table after the extract. The query within the `query` configuration dictates what’s put in the temporary table. Notice that the table used within the `extract` configuration is a dynamic table generated from the daily execution of the DAG. This allows for different slices of data to be extracted and exported to GCS, allowing for a flexible data processing pipeline. This technique of creating temporary tables is useful when you need to export transformed data. Note also that the output here is in newline delimited json format which is useful for downstream processing in other systems like data warehouses.

Now, let’s consider a situation where your dataset is large and you need to use wildcard uris to create multiple smaller files. For instance you might have to create multiple files per date based on a wildcard.

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from datetime import datetime

with DAG(
    dag_id='bigquery_wildcard_export_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    export_task = BigQueryInsertJobOperator(
        task_id='export_user_activity',
        configuration={
            "extract": {
                "sourceTable": {
                    "projectId": "your-gcp-project-id",
                    "datasetId": "analytics",
                    "tableId": "user_activity"
                },
                 "destinationUris": [
                   f"gs://your-gcs-bucket/user_activity_wildcard/date={{ ds }}/user_activity-*.parquet"
                ],
                "destinationFormat": "PARQUET",
                "compression": "SNAPPY",
            }
        }
    )
```
In this example, I’ve moved to parquet format and specified the snappy compression algorithm. This provides an optimized way to extract data to be used by other analysis tools, such as pandas, spark, or dask. The `user_activity-*.parquet` allows BigQuery to automatically create multiple parquet files, partitioned as it sees fit based on the table structure.

These examples demonstrate that the `BigQueryInsertJobOperator` with export configuration is powerful and flexible. However, you need to carefully define the `configuration` parameter to match the exact behavior you’re looking for.

For diving deeper into these concepts, I would highly recommend consulting the official Google BigQuery documentation on job structures, particularly the section on Extract Jobs. Also, “BigQuery: The Definitive Guide” by Valliappa Lakshmanan and Jordan Tigani can be very useful for understanding the inner workings of BigQuery. Additionally, the Apache Airflow documentation on the `BigQueryInsertJobOperator` is invaluable for understanding operator specific configuration options. By combining knowledge from these sources, you should have a complete and practical understanding of how to effectively export BigQuery data. Remember, a thorough understanding of the configuration parameters is crucial to avoid common pitfalls and to maximize the utility of your data pipelines.
