---
title: "How do I use `BigQueryInsertJobOperator` with an Export Configuration?"
date: "2024-12-23"
id: "how-do-i-use-bigqueryinsertjoboperator-with-an-export-configuration"
---

Let's dive into this. I recall a project a few years back where we needed to regularly back up specific subsets of our BigQuery tables to cloud storage, often formatted differently than their native structure. This meant we couldn't just rely on the standard export functionality provided through the console or command-line tools. It highlighted the need to orchestrate these exports using Apache Airflow's `BigQueryInsertJobOperator`, with a crucial twist—leveraging the export configuration parameters. Let's break down how you can do this.

The `BigQueryInsertJobOperator`, at its core, is an Airflow operator that allows you to initiate any BigQuery job, be it a query, load, copy, or, as is pertinent here, an export. The "insert" part refers to the fact that it's inserting (submitting) a job to BigQuery's job queue. What’s particularly relevant for us is that this operator can take a `configuration` argument, which is a dictionary mirroring the json structure of a BigQuery job, as defined in the BigQuery API documentation. This configuration is what we'll manipulate to set up our export operation.

Typically, you might see the `BigQueryInsertJobOperator` used for simple queries. However, to trigger an export, we need to craft a different configuration. We need to specify the `extract` property within this configuration to indicate that this job isn't a query, but rather an export. This configuration will include the source table, the destination URI(s) in Google Cloud Storage, and the format and compression options that we want. It's important to understand the configuration options available. You can review these options directly in the Google Cloud BigQuery documentation focusing on "Extracting data" and "Job resources." Furthermore, the official Apache Airflow documentation provides very detailed guidance for using the `BigQueryInsertJobOperator`.

Here's a crucial point: the `destinationUris` key accepts a *list* of strings, not just a single string. This enables you to export to multiple files simultaneously, an essential feature if your dataset is particularly large. Additionally, specifying formats like CSV or JSON, as well as compression methods like GZIP or NONE, falls under these configuration options. Let's move into some concrete examples.

**Example 1: Basic CSV Export**

Let's say you have a table called `my_project.my_dataset.my_table` and you want to export it to a single CSV file in Google Cloud Storage. Here's how the `BigQueryInsertJobOperator` would look within your Airflow DAG:

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from datetime import datetime

with DAG(
    dag_id='bigquery_export_csv',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['bigquery', 'export']
) as dag:
    export_csv_task = BigQueryInsertJobOperator(
        task_id='export_table_csv',
        configuration={
            "extract": {
                "sourceTable": {
                    "projectId": "my_project",
                    "datasetId": "my_dataset",
                    "tableId": "my_table",
                },
                "destinationUris": ["gs://my_bucket/my_output.csv"],
                "destinationFormat": "CSV",
                "printHeader": True,
            }
        }
    )
```

In this example, we create a simple DAG with a single task. The `configuration` dictionary contains an `extract` property. Notice that the `sourceTable` is an object with the `projectId`, `datasetId`, and `tableId`. The `destinationUris` is a *list* containing one GCS URI. We also set the `destinationFormat` to `CSV` and ensure a header row is printed in the csv using `printHeader`.

**Example 2: Compressed JSON Export**

Now, let’s consider a scenario where we need JSON output, compressed, and written into multiple files. This is common when dealing with large datasets that would otherwise result in very large single output files.

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from datetime import datetime

with DAG(
    dag_id='bigquery_export_json',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['bigquery', 'export']
) as dag:
    export_json_task = BigQueryInsertJobOperator(
        task_id='export_table_json',
        configuration={
            "extract": {
                "sourceTable": {
                    "projectId": "my_project",
                    "datasetId": "my_dataset",
                    "tableId": "my_table",
                },
                "destinationUris": [
                    "gs://my_bucket/my_output_part-*.json.gz"
                 ],
                "destinationFormat": "JSON",
                "compression": "GZIP",
            }
        }
    )
```

Here, we've changed the `destinationFormat` to `JSON` and added `compression: "GZIP"`. More importantly, the `destinationUris` uses `my_output_part-*.json.gz` which is a wildcard pattern. BigQuery will use this pattern to create multiple file shards for the export. Note that the `*` will be replaced with numeric sequences.

**Example 3: Export with Field Delimiter and Print Headers**

For CSV exports, you might need specific delimiters, and want to make sure to include a header row in the output.

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from datetime import datetime

with DAG(
    dag_id='bigquery_export_csv_delimited',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['bigquery', 'export']
) as dag:
    export_csv_delimited_task = BigQueryInsertJobOperator(
        task_id='export_table_csv_delimited',
        configuration={
            "extract": {
                "sourceTable": {
                    "projectId": "my_project",
                    "datasetId": "my_dataset",
                    "tableId": "my_table",
                },
                "destinationUris": ["gs://my_bucket/my_output.csv"],
                "destinationFormat": "CSV",
                "printHeader": True,
                 "fieldDelimiter": "|"
            }
        }
    )
```
This example demonstrates how to specify a custom field delimiter for a CSV export, using the `fieldDelimiter` property. Remember, this is a common requirement for downstream systems that expect files with specific structures. Also note that the `printHeader` property was kept as `True` to include header row in exported file.

A common issue you may encounter is permission errors. Make absolutely certain that the service account running your Airflow worker has sufficient access to read from your BigQuery tables and write to your cloud storage bucket. Also, check any firewalls between your Airflow installation and Google Cloud Platform.

To understand the intricacies of the BigQuery API and the specific properties you can utilize, consult the official Google BigQuery documentation; primarily, the sections covering "Extracting data" and "Job resources." I also recommend reviewing the official Apache Airflow documentation for the `BigQueryInsertJobOperator`, which offers clear guidance on its proper usage and parameters. A highly recommended read to strengthen your fundamentals on BigQuery is "BigQuery: The Definitive Guide" by Valliappa Lakshmanan and Jordan Tigani.

In closing, utilizing the `BigQueryInsertJobOperator` in conjunction with an export configuration provides a powerful and flexible way to manage BigQuery data exports. Understanding the available parameters and crafting the appropriate `configuration` object are crucial steps for successfully automating your data pipelines. I trust these examples will give you a good start and help with the practical implementation you are facing. Let me know if more specific challenges arise.
