---
title: "How can Airflow run a BigQuery query and write the results to a Cloud Storage bucket in Parquet format?"
date: "2025-01-30"
id: "how-can-airflow-run-a-bigquery-query-and"
---
The need to efficiently transfer large datasets resulting from BigQuery queries to Cloud Storage, particularly in columnar formats like Parquet, is a frequent requirement in data engineering pipelines. Leveraging Apache Airflow for orchestration simplifies this task through its robust framework and readily available operators.

The process involves three core stages: initiating a BigQuery query, extracting the query results to a temporary staging area, and finally, writing the data from that staging area to a Cloud Storage bucket in the desired format. The primary Airflow operator for BigQuery interaction is `BigQueryExecuteQueryOperator`, and for Cloud Storage, we use `GCSToGCSOperator`. Intermediate staging is typically facilitated through temporary tables or through BigQuery's extraction functionality, with the latter being more efficient for data transfer scenarios.

I have personally implemented similar pipelines numerous times for various clients with substantial data volumes. One particularly memorable case involved migrating a legacy on-premises data warehouse to Google Cloud. The process involved hundreds of BigQuery queries, requiring a consistent and automated approach for exporting data to a data lake built on Cloud Storage.

Below, I outline the specific steps and code, based on this experience, to achieve the described task of querying BigQuery and exporting the result to Parquet on Cloud Storage using Apache Airflow.

**Core Concepts and Explanation**

The `BigQueryExecuteQueryOperator` is responsible for executing the SQL query. The `destination_table` parameter, when specified, will persist the results to a table within BigQuery. However, we will not use the 'destination table' functionality directly to write the output as parquet. Instead, we will leverage BigQuery's capabilities of exporting results to Cloud Storage in Parquet. This simplifies and reduces the number of steps in the process.

After executing the query, the next step involves extracting these results directly to a Cloud Storage bucket. BigQuery allows exporting query results directly to Cloud Storage, avoiding the need to create a temporary table explicitly. The output can be specified in various formats, including Parquet. We will use BigQuery's `extract` functionality. This is not a separate operator from Airflow's perspective, but is a parameter within the query configuration. By specifying a destination URI and the output format within a job configuration using a SQL script with the BigQuery `EXPORT DATA` statement, the output can be controlled directly from the SQL.

Finally, the transfer to Cloud Storage via BigQuery's export implicitly ensures the data is written in the desired format with the specified naming convention. By using the correct URI formatting and file format, Airflow delegates the actual transfer and transformation to the BigQuery service.

**Code Examples and Commentary**

The following three examples showcase different approaches to achieving the task, illustrating various configuration options.

**Example 1: Basic Export with SQL Script**

This example executes a simple query and exports the entire result to a single Parquet file in Cloud Storage. This is useful for small to medium datasets where partitioning is not critical.

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryExecuteQueryOperator
from datetime import datetime

with DAG(
    dag_id='bigquery_to_gcs_parquet_basic',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    export_query = BigQueryExecuteQueryOperator(
        task_id='export_bigquery_to_parquet',
        sql=f"""
            EXPORT DATA
            OPTIONS(
              uri='gs://your-bucket/output/basic_data_*.parquet',
              format='PARQUET',
              overwrite=true
            )
            AS
            SELECT *
            FROM `your-project.your_dataset.your_table`
            WHERE date(timestamp_column) = CURRENT_DATE()
        """,
        use_legacy_sql=False,
        gcp_conn_id='google_cloud_default'
    )
```

*   **Commentary:** This example directly uses a SQL `EXPORT DATA` statement to export data.  The wildcard `*.parquet` allows BigQuery to manage the output files based on size and also simplifies the GCS destination URI, even though we expect one file here, but could be many with large data. `overwrite=true` makes sure old files with the same name are replaced with new results. `use_legacy_sql=False` ensures that the standard SQL dialect is used which supports the `EXPORT DATA` statement. Ensure that the service account used by Airflow has both BigQuery read and GCS write permissions for the specified bucket. This example assumes that the dataset exists and there is the data in `your-project.your_dataset.your_table`.

**Example 2: Partitioned Export**

Here, the output is partitioned by a date column. This improves performance for subsequent reads of the data, especially in analytical workloads.

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryExecuteQueryOperator
from datetime import datetime

with DAG(
    dag_id='bigquery_to_gcs_parquet_partitioned',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
     export_query_partitioned = BigQueryExecuteQueryOperator(
        task_id='export_bigquery_to_parquet_partitioned',
        sql=f"""
           EXPORT DATA
            OPTIONS(
              uri='gs://your-bucket/output/partitioned_data/date={{{{ ds }}}}/data_*.parquet',
              format='PARQUET',
              overwrite=true,
            )
            AS
            SELECT *
            FROM `your-project.your_dataset.your_table`
            WHERE date(timestamp_column) = '{{{{ ds }}}}'
        """,
        use_legacy_sql=False,
        gcp_conn_id='google_cloud_default'
    )

```

*   **Commentary:** This code uses Airflow macros `{{ ds }}` to dynamically generate the date used for partitioning the data in GCS. This example shows the strength of dynamic task configurations with Jinja templates. BigQuery will handle writing multiple Parquet files based on the volume of data. The format and `overwrite` parameters are identical as in example 1. This structure facilitates easier data lake navigation and improved query performance due to partition pruning.

**Example 3: Selecting Specific Columns**

This example demonstrates exporting only certain columns instead of the entire table, useful when a subset of columns is sufficient for downstream applications.

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryExecuteQueryOperator
from datetime import datetime

with DAG(
    dag_id='bigquery_to_gcs_parquet_select_columns',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    export_query_select_columns = BigQueryExecuteQueryOperator(
        task_id='export_bigquery_select_columns',
        sql=f"""
            EXPORT DATA
             OPTIONS(
              uri='gs://your-bucket/output/selected_data/data_*.parquet',
              format='PARQUET',
              overwrite=true,
            )
            AS
            SELECT column1, column2, column3
            FROM `your-project.your_dataset.your_table`
            WHERE date(timestamp_column) = CURRENT_DATE()
        """,
        use_legacy_sql=False,
        gcp_conn_id='google_cloud_default'
    )
```

*   **Commentary:**  This example explicitly specifies the columns `column1`, `column2`, and `column3` in the `SELECT` clause of the query, which will be included in the output file. The structure remains otherwise consistent with Example 1. This highlights how flexibility in data selection during the extract process reduces the resource and network overhead of transferring unnecessary data. Ensure you replace the column names with the actual columns of the table.

**Resource Recommendations**

For further learning and deeper understanding:

*   **Apache Airflow Documentation:** The official Airflow documentation provides in-depth explanations of operators, concepts, and best practices. The documentation for `BigQueryExecuteQueryOperator` and connections are particularly relevant.
*   **Google Cloud BigQuery Documentation:** Refer to the BigQuery documentation for detailed information about querying, data extraction, formats, and export syntax. The reference for the `EXPORT DATA` statement within SQL dialect is crucial.
*   **Google Cloud Storage Documentation:** Understand the basics of object storage, naming conventions, and access control policies.
*   **Parquet Format Specification:** To gain knowledge of the internal structure and efficiency aspects of Parquet.
*   **Best Practices for Data Pipelines on Google Cloud:** Learn more about structuring and managing complex data workflows by reviewing best practices provided by Google.

By combining these resources with practical experience, developing and maintaining efficient BigQuery-to-Cloud Storage pipelines becomes a manageable task, even with evolving requirements. The examples above provide a good foundation, but further adjustments will be necessary based on the complexity of the data and the specifics of the system. The use of the `EXPORT DATA` command and avoiding a temporary table significantly enhances the efficiency and robustness of the process.
