---
title: "How can I check if a BigQuery table partition exists in Airflow?"
date: "2024-12-23"
id: "how-can-i-check-if-a-bigquery-table-partition-exists-in-airflow"
---

,  This is a situation I've definitely run into a few times, especially when orchestrating complex data pipelines with BigQuery and Airflow. Verifying the existence of a partition before attempting to process it is crucial for avoiding pipeline failures and unnecessary computation. It's about robustness, efficiency, and ensuring your DAGs don't trip over unexpected data states. There are a few ways to approach this in Airflow, and I'll break down the method I've found most reliable and then show you some concrete code examples.

The core idea revolves around using the `google.cloud.bigquery` Python client within your Airflow DAG. Instead of assuming a partition exists, we proactively query the BigQuery metadata to confirm its presence. This metadata is exposed through the `INFORMATION_SCHEMA.PARTITIONS` view within BigQuery. By constructing and executing a specific query against this view, we can determine if a particular partition, identified by its partition key and value, is present in a given table. This avoids any race conditions or errors arising from blindly trying to interact with a partition that might not exist yet.

Essentially, we're shifting the logic from 'assume and react' to 'query and act'. This approach increases the reliability of your pipeline significantly. Before I jump into code, I’ll mention a few good places to deepen your knowledge on this. For detailed information on BigQuery's `INFORMATION_SCHEMA` and partition management, I'd highly recommend reviewing Google's official documentation, specifically around `INFORMATION_SCHEMA.PARTITIONS` and partitioned tables. Also, "BigQuery: The Definitive Guide" by Valliappa Lakshmanan and Jordan Tigani will offer a deeper dive into the subject. For the Airflow side of things, the Apache Airflow documentation for the `google-cloud-bigquery` provider is invaluable.

Now let's look at some practical examples. In the following, assume we have a table `my_project.my_dataset.my_table` partitioned by the column `partition_date` (or some analogous date/timestamp-based column).

**Example 1: Checking for a Specific Date Partition**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
from datetime import datetime, timedelta

def check_partition_exists(partition_date, dataset_id, table_id, project_id):
    bq_hook = BigQueryHook(gcp_conn_id='google_cloud_default') #replace with your connection ID if needed
    sql = f"""
    SELECT
        partition_id
    FROM
      `{project_id}.{dataset_id}.INFORMATION_SCHEMA.PARTITIONS`
    WHERE
      table_name = '{table_id}'
      AND partition_id = '{partition_date}'
    """
    results = bq_hook.get_records(sql)
    if results:
        print(f"Partition {partition_date} exists in {dataset_id}.{table_id}")
        return True
    else:
        print(f"Partition {partition_date} does not exist in {dataset_id}.{table_id}")
        return False

def process_partition_if_exists(**context):
    partition_date = (context['execution_date'] - timedelta(days=1)).strftime('%Y%m%d')
    dataset_id = 'my_dataset'
    table_id = 'my_table'
    project_id = 'my_project'

    if check_partition_exists(partition_date, dataset_id, table_id, project_id):
        print(f"Processing partition {partition_date}")
        # Your logic to process the partition here
    else:
        print(f"Skipping partition {partition_date} as it does not exist")

with DAG(
    dag_id='bigquery_partition_check',
    start_date=datetime(2023, 1, 1),
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:
    check_and_process_task = PythonOperator(
        task_id='check_and_process_partition',
        python_callable=process_partition_if_exists
    )

```

This example creates an Airflow DAG that checks if a partition exists for the previous day. `check_partition_exists` function uses the `BigQueryHook` to run a query, and then `process_partition_if_exists` uses that boolean response.

**Example 2: Checking for Multiple Partitions**

Sometimes, you need to check for multiple partitions concurrently. You could run multiple `check_partition_exists` individually, but a more efficient approach is to modify the query.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
from datetime import datetime, timedelta

def check_multiple_partitions_exist(partition_dates, dataset_id, table_id, project_id):
    bq_hook = BigQueryHook(gcp_conn_id='google_cloud_default')

    formatted_dates = [f"'{date}'" for date in partition_dates]
    date_list_str = ",".join(formatted_dates)

    sql = f"""
    SELECT
      partition_id
    FROM
      `{project_id}.{dataset_id}.INFORMATION_SCHEMA.PARTITIONS`
    WHERE
      table_name = '{table_id}'
      AND partition_id IN ({date_list_str})
    """
    results = bq_hook.get_records(sql)
    existing_partitions = [row[0] for row in results]
    print(f"Existing partitions are {existing_partitions} of the requested {partition_dates}")
    return existing_partitions


def process_multiple_partitions(**context):
    today = context['execution_date']
    partition_dates = [(today - timedelta(days=i)).strftime('%Y%m%d') for i in range(3)] # checking for last three days
    dataset_id = 'my_dataset'
    table_id = 'my_table'
    project_id = 'my_project'

    existing_partitions = check_multiple_partitions_exist(partition_dates, dataset_id, table_id, project_id)
    for partition_date in existing_partitions:
        print(f"Processing partition {partition_date}")
        # Your logic to process the partition here

with DAG(
    dag_id='bigquery_multiple_partition_check',
    start_date=datetime(2023, 1, 1),
    schedule_interval=timedelta(days=1),
    catchup=False,
) as dag:
    check_and_process_task = PythonOperator(
        task_id='check_and_process_partitions',
        python_callable=process_multiple_partitions
    )
```

This example uses an `IN` clause in the SQL query to fetch all existing partitions from a list.

**Example 3: Handling Integer Range Partitions**

Now, let’s say you have a table partitioned by an integer range rather than a date. The principle is very similar; we adjust the partition filter condition in the query. This example assumes your integer range partition column is named `partition_integer`.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
from datetime import datetime, timedelta

def check_integer_partition_exists(partition_integer, dataset_id, table_id, project_id):
    bq_hook = BigQueryHook(gcp_conn_id='google_cloud_default')

    sql = f"""
        SELECT
            partition_id
        FROM
        `{project_id}.{dataset_id}.INFORMATION_SCHEMA.PARTITIONS`
        WHERE
            table_name = '{table_id}'
            AND partition_id = CAST({partition_integer} as STRING)
        """
    results = bq_hook.get_records(sql)
    if results:
        print(f"Integer partition {partition_integer} exists in {dataset_id}.{table_id}")
        return True
    else:
        print(f"Integer partition {partition_integer} does not exist in {dataset_id}.{table_id}")
        return False

def process_integer_partition(**context):
     partition_integer = 100 # example integer partition value
     dataset_id = 'my_dataset'
     table_id = 'my_table_integer_partitioned'
     project_id = 'my_project'

     if check_integer_partition_exists(partition_integer, dataset_id, table_id, project_id):
         print(f"Processing partition {partition_integer}")
         # Your logic to process the partition here
     else:
         print(f"Skipping partition {partition_integer} as it does not exist")


with DAG(
    dag_id='bigquery_int_partition_check',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None, # Run manually for demonstration purpose
    catchup=False,
) as dag:
    check_and_process_int_task = PythonOperator(
        task_id='check_and_process_int_partition',
        python_callable=process_integer_partition
    )
```

This example showcases how to correctly check integer-based partitions and the necessary casting. The key difference is casting the integer to a string within the SQL query, because the partition_id from `INFORMATION_SCHEMA.PARTITIONS` is always a string.

These examples showcase the core logic. Remember to replace placeholders such as `my_project`, `my_dataset`, `my_table`, and `google_cloud_default` with your actual values. Also, consider adding logging and error handling within your Airflow DAGs for production deployments. By using the `INFORMATION_SCHEMA.PARTITIONS` view directly, you gain much finer control over your pipelines and greatly reduce the risk of failed or skipped processing due to missing partitions. It adds some complexity but results in more stable and resilient data workflows. This approach, from my experience, has been reliable and effective in numerous real-world applications involving BigQuery and Airflow.
