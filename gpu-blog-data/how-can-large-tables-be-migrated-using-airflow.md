---
title: "How can large tables be migrated using Airflow?"
date: "2025-01-30"
id: "how-can-large-tables-be-migrated-using-airflow"
---
Migrating large tables presents a significant challenge in data engineering, primarily due to resource constraints and time limitations. I've encountered this problem numerous times, specifically during a recent cloud data warehouse migration project involving tables exceeding several billion rows.  The key is to avoid attempting to migrate these tables as a single monolithic operation, which can lead to timeouts, network bottlenecks, and potentially system instability. Instead, a strategy of incremental and parallel processing, orchestrated by Airflow, provides a viable solution.

Airflow's strength in this context lies in its ability to manage complex workflows, specifically through the use of Directed Acyclic Graphs (DAGs). Rather than manually kicking off numerous migration tasks, I utilize a DAG to logically represent the migration process. The primary breakdown involves segmenting the source table into manageable chunks, usually based on date or ID ranges, and then migrating those chunks independently and concurrently.

The general process comprises the following steps:
1. **Extraction:** A query is constructed to fetch a slice of the source table, based on defined parameters.
2. **Transformation (Optional):**  If needed, any data transformation logic can be applied to the extracted slice before loading.
3. **Loading:** The transformed data is loaded into the target table.
4. **Verification:** Data counts and potentially row samples are validated to ensure data integrity.
5. **Logging:** Results for each slice are logged to facilitate monitoring and issue resolution.

This workflow requires careful consideration of several factors: The size and characteristics of the data (partitioned or not, suitable range columns), target database load limits, and the availability of compute resources to accommodate concurrent operations. I've found that parameterizing the DAG allows for flexibility, enabling adjustments based on these factors without altering the core logic.

Here are some code examples using Python and Airflow that illustrate this:

**Example 1: Basic Table Slice Extraction and Load**

This example focuses on a straightforward scenario where a table is divided using a date column.  It assumes the presence of `source_hook` and `target_hook` which are properly configured database connections in the Airflow environment.

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.hooks.base_hook import BaseHook
import logging

def extract_and_load(start_date, end_date, source_conn_id, target_conn_id, source_table, target_table):
    """Extracts a date range, loads it to target, and logs results."""
    source_hook = BaseHook.get_hook(source_conn_id)
    target_hook = BaseHook.get_hook(target_conn_id)

    query = f"""
    SELECT * FROM {source_table}
    WHERE date_column >= '{start_date}' AND date_column < '{end_date}'
    """

    logging.info(f"Extracting data with query: {query}")
    records = source_hook.get_records(query)

    if records:
        logging.info(f"Number of records extracted: {len(records)}")
        target_hook.insert_rows(table=target_table, rows=records, target_fields=[col[0] for col in source_hook.get_table_schema(source_table)]) # Get Source Table column list
    else:
        logging.info("No records extracted for this slice.")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG('table_migration_by_date', default_args=default_args, schedule_interval=None, catchup=False) as dag:

    date_ranges = [
    {"start": "2022-01-01", "end": "2022-02-01"},
    {"start": "2022-02-01", "end": "2022-03-01"},
    {"start": "2022-03-01", "end": "2022-04-01"}
    ]

    for range_data in date_ranges:
        task_id = f"migrate_slice_{range_data['start']}"
        migrate_task = PythonOperator(
            task_id=task_id,
            python_callable=extract_and_load,
            op_kwargs={
                "start_date": range_data["start"],
                "end_date": range_data["end"],
                "source_conn_id": "source_db", #Replace with actual connection name
                "target_conn_id": "target_db", #Replace with actual connection name
                "source_table": "source_table_name", # Replace with source table name
                "target_table": "target_table_name", # Replace with target table name
            },
        )
```
**Commentary:** This DAG defines a Python operator that fetches and loads records based on a date range. The `date_ranges` list defines the slices, and a PythonOperator is dynamically created for each.  Note the use of `BaseHook`  to connect to data sources.  This simple structure can migrate all historical data when applied iteratively over a suitable date range sequence.

**Example 2:  Table Migration Using an ID Range**

When a date field is unavailable or does not provide sufficient selectivity, slicing by an ID range is a feasible alternative.  The method is similar to Example 1,  but the SQL query uses ID ranges. It assumes the existence of a numeric column called 'id'.

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.hooks.base_hook import BaseHook
import logging

def extract_and_load_by_id(start_id, end_id, source_conn_id, target_conn_id, source_table, target_table):
    """Extracts an ID range, loads it to target, and logs results."""
    source_hook = BaseHook.get_hook(source_conn_id)
    target_hook = BaseHook.get_hook(target_conn_id)

    query = f"""
    SELECT * FROM {source_table}
    WHERE id >= {start_id} AND id < {end_id}
    """
    logging.info(f"Extracting data with query: {query}")
    records = source_hook.get_records(query)

    if records:
       logging.info(f"Number of records extracted: {len(records)}")
       target_hook.insert_rows(table=target_table, rows=records, target_fields=[col[0] for col in source_hook.get_table_schema(source_table)]) # Get Source Table column list

    else:
        logging.info("No records extracted for this slice.")



default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG('table_migration_by_id', default_args=default_args, schedule_interval=None, catchup=False) as dag:

    id_ranges = [
       {"start": 1, "end": 100000},
       {"start": 100000, "end": 200000},
       {"start": 200000, "end": 300000}
    ]

    for range_data in id_ranges:
        task_id = f"migrate_slice_{range_data['start']}"
        migrate_task = PythonOperator(
            task_id=task_id,
            python_callable=extract_and_load_by_id,
            op_kwargs={
                "start_id": range_data["start"],
                "end_id": range_data["end"],
                "source_conn_id": "source_db", #Replace with actual connection name
                "target_conn_id": "target_db", #Replace with actual connection name
                "source_table": "source_table_name", # Replace with source table name
                "target_table": "target_table_name", # Replace with target table name
            },
        )
```

**Commentary:** This example demonstrates a similar approach but uses an ID column range instead of a date range. It showcases that the approach can be adapted to different partitioning schemes based on data availability. Both Examples 1 and 2 are examples of full migration.

**Example 3: Handling Data Transformation**

This example builds on the previous ones by adding a simple data transformation before loading.

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.hooks.base_hook import BaseHook
import logging
import pandas as pd

def extract_transform_and_load(start_date, end_date, source_conn_id, target_conn_id, source_table, target_table):
    """Extracts a slice, transforms data, loads it and logs results."""
    source_hook = BaseHook.get_hook(source_conn_id)
    target_hook = BaseHook.get_hook(target_conn_id)

    query = f"""
    SELECT * FROM {source_table}
    WHERE date_column >= '{start_date}' AND date_column < '{end_date}'
    """
    logging.info(f"Extracting data with query: {query}")
    records = source_hook.get_records(query)

    if records:
        logging.info(f"Number of records extracted: {len(records)}")
        # Example of transformation - Convert to Pandas DataFrame for manipulation
        df = pd.DataFrame.from_records(records, columns=[col[0] for col in source_hook.get_table_schema(source_table)])
        df['transformed_column'] = df['original_column'] * 2 #Example transformation. Needs replacement with actual transformation logic
        transformed_records= df.to_records(index=False)
        target_hook.insert_rows(table=target_table, rows=transformed_records, target_fields=df.columns.tolist())  #Using the pandas data frame column list for targets
    else:
        logging.info("No records extracted for this slice.")



default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG('table_migration_transform', default_args=default_args, schedule_interval=None, catchup=False) as dag:

    date_ranges = [
    {"start": "2022-01-01", "end": "2022-02-01"},
    {"start": "2022-02-01", "end": "2022-03-01"},
     {"start": "2022-03-01", "end": "2022-04-01"}
    ]

    for range_data in date_ranges:
        task_id = f"migrate_slice_{range_data['start']}"
        migrate_task = PythonOperator(
            task_id=task_id,
            python_callable=extract_transform_and_load,
            op_kwargs={
                "start_date": range_data["start"],
                "end_date": range_data["end"],
                "source_conn_id": "source_db", #Replace with actual connection name
                "target_conn_id": "target_db", #Replace with actual connection name
                "source_table": "source_table_name", # Replace with source table name
                "target_table": "target_table_name", # Replace with target table name
            },
        )

```
**Commentary:**  Here, the extracted records are first converted into a Pandas DataFrame, on which a simple transformation is performed before the data is written to the target database. This highlights the ability to integrate data transformations into the migration pipeline.  This simple transformation represents a starting point; more elaborate transformations, including joining with other tables, can be embedded within this architecture using the capabilities of Python’s data processing libraries.

For continued learning and to solidify the presented approach,  I suggest exploring the official Airflow documentation, which delves into more advanced features of DAG building, task scheduling, and dynamic task mapping. Additionally, researching best practices in database migration strategies and cloud data warehouse performance tuning can significantly improve the efficacy of migrations in real-world deployments. Studying the internal implementations of database hooks within Airflow will help in creating robust pipelines that are also resource efficient. Specifically, understanding how Airflow manages connections pools will allow for optimizing data transfer speeds.
