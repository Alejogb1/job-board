---
title: "How can Airflow handle multiple BigQuery to GCS output destinations using the bigquery_to_gcs operator?"
date: "2025-01-30"
id: "how-can-airflow-handle-multiple-bigquery-to-gcs"
---
BigQuery's `bigquery_to_gcs` operator, by its design, executes a single BigQuery export job to a defined Google Cloud Storage location. Managing multiple destinations using this operator directly requires orchestration strategies rather than a modification of the operator's fundamental functionality. I've faced this precise scenario on several projects when developing ETL pipelines for our analytics platform. Each project required different exports based on table content and varying destination file formats. My experience has taught me the crucial role of dynamic task generation and templating when dealing with such multi-destination requirements.

The central challenge is that `bigquery_to_gcs` is conceived for a one-to-one relationship between a query and a GCS export location. It doesn't inherently loop or have branching logic. Therefore, the architecture for multi-destination exports must be built on top of its single-purpose functionality, which is best approached by dynamically generating tasks within the Airflow DAG (Directed Acyclic Graph). To implement this, we leverage Python's flexibility and Jinja templating within the Airflow framework.

Essentially, the process revolves around programmatically defining the specific configurations for each export, iterating over them, and constructing corresponding tasks that utilize the `BigQueryToGCSOperator`. The key is to maintain a structured, declarative approach that is both readable and easy to modify. This process involves several distinct steps. First, we need to define the different configurations: each includes the BigQuery SQL query, the GCS output location, the file format, and other relevant export parameters. I typically store these configurations as a dictionary or JSON that can be easily parsed. Second, using the configuration, we dynamically construct the operator instances, setting properties with template variables. Finally, ensure the workflow is logically coherent, typically using task dependencies.

Here are three code examples illustrating this process:

**Example 1: Basic Dynamic Task Generation**

```python
from airflow import DAG
from airflow.providers.google.cloud.transfers.bigquery_to_gcs import BigQueryToGCSOperator
from datetime import datetime

EXPORT_CONFIGURATIONS = [
    {
        'query': 'SELECT * FROM `project.dataset.table1` WHERE date = "{{ ds }}"',
        'destination_uris': ['gs://my-bucket/data1/{{ ds }}/output*.csv'],
        'export_format': 'CSV',
    },
    {
        'query': 'SELECT id, value FROM `project.dataset.table2`',
        'destination_uris': ['gs://my-bucket/data2/output.json'],
        'export_format': 'JSON',
    },
    {
         'query': 'SELECT  region, COUNT(*) FROM `project.dataset.table3` GROUP BY region',
         'destination_uris': ['gs://my-bucket/data3/{{ ds_nodash }}/aggregated*.parquet'],
         'export_format': 'PARQUET',
         'print_header': False
    }
]


with DAG(
    dag_id="multiple_bigquery_to_gcs",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['bigquery', 'gcs', 'export'],
) as dag:

    for index, config in enumerate(EXPORT_CONFIGURATIONS):
       BigQueryToGCSOperator(
            task_id=f'export_bigquery_to_gcs_{index}',
            sql=config['query'],
            destination_uris=config['destination_uris'],
            export_format=config['export_format'],
            print_header=config.get('print_header', True),  # Default to True if not specified
       )
```

In this first example, I showcase the basic approach. The `EXPORT_CONFIGURATIONS` list holds dictionaries describing each export's query, destination URI, and file format. The DAG then iterates over this list, creating a separate `BigQueryToGCSOperator` instance for each configuration. This method works well for relatively static configurations. Note the use of Jinja templating `{{ ds }}` and `{{ ds_nodash }}`. These Airflow variables inject the DAG run's execution date, ensuring dynamic file naming. Each task is named using a simple index to maintain unique task identifiers. I have also implemented a check for `print_header` in order to demonstrate how to use default values and how to work with optional values. This ensures the user can control if they want to add the header of the table in the exported file.

**Example 2: Using a Task Mapping approach**

```python
from airflow import DAG
from airflow.providers.google.cloud.transfers.bigquery_to_gcs import BigQueryToGCSOperator
from datetime import datetime
from airflow.utils.task_group import TaskGroup


EXPORT_CONFIGURATIONS = [
    {
        'task_id_suffix': 'export_table1',
        'query': 'SELECT * FROM `project.dataset.table1` WHERE date = "{{ ds }}"',
        'destination_uris': ['gs://my-bucket/data1/{{ ds }}/output*.csv'],
        'export_format': 'CSV',
    },
    {
        'task_id_suffix': 'export_table2',
        'query': 'SELECT id, value FROM `project.dataset.table2`',
        'destination_uris': ['gs://my-bucket/data2/output.json'],
        'export_format': 'JSON',
    },
    {
         'task_id_suffix': 'export_table3',
         'query': 'SELECT  region, COUNT(*) FROM `project.dataset.table3` GROUP BY region',
         'destination_uris': ['gs://my-bucket/data3/{{ ds_nodash }}/aggregated*.parquet'],
         'export_format': 'PARQUET',
         'print_header': False
    }
]


with DAG(
    dag_id="mapped_multiple_bigquery_to_gcs",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['bigquery', 'gcs', 'export'],
) as dag:
    with TaskGroup("export_tasks") as export_tasks:
      BigQueryToGCSOperator.partial(
             project_id="project",
             gcp_conn_id = "google_cloud_default"
       ).expand(
            task_id = [ f'export_bigquery_to_gcs_{conf["task_id_suffix"]}' for conf in EXPORT_CONFIGURATIONS ],
            sql = [conf['query'] for conf in EXPORT_CONFIGURATIONS],
            destination_uris = [conf['destination_uris'] for conf in EXPORT_CONFIGURATIONS],
            export_format = [conf['export_format'] for conf in EXPORT_CONFIGURATIONS],
             print_header = [conf.get('print_header', True) for conf in EXPORT_CONFIGURATIONS]
        )
```

Example 2 demonstrates a more concise and modern approach using task mapping capabilities introduced in recent Airflow versions. Instead of manually looping, we use `BigQueryToGCSOperator.partial` which generates a function that is used to expand the task via the `.expand()` method. Note how the values are passed to the expand method as lists that are evaluated for the `EXPORT_CONFIGURATIONS` list. This results in tasks being generated implicitly.  This approach reduces code verbosity while accomplishing the same goal. Moreover, tasks created using the mapping method can be manipulated as a group. This will be useful when you need to set task dependencies. Also note the usage of `TaskGroup`, which is useful to organize tasks logically.

**Example 3: Templating from an External Configuration File**

```python
from airflow import DAG
from airflow.providers.google.cloud.transfers.bigquery_to_gcs import BigQueryToGCSOperator
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from datetime import datetime
import json
import os


CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'export_config.json')

def load_config():
  with open(CONFIG_FILE_PATH, 'r') as file:
      return json.load(file)

def create_export_tasks(**kwargs):
    ti = kwargs['ti']
    export_configurations = ti.xcom_pull(task_ids='load_config_task', key='return_value')
    
    with TaskGroup("export_tasks") as export_tasks:
        BigQueryToGCSOperator.partial(
               project_id="project",
               gcp_conn_id = "google_cloud_default"
          ).expand(
               task_id = [ f'export_bigquery_to_gcs_{conf["task_id_suffix"]}' for conf in export_configurations ],
               sql = [conf['query'] for conf in export_configurations],
               destination_uris = [conf['destination_uris'] for conf in export_configurations],
               export_format = [conf['export_format'] for conf in export_configurations],
               print_header = [conf.get('print_header', True) for conf in export_configurations]
           )

with DAG(
    dag_id="external_config_bigquery_to_gcs",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['bigquery', 'gcs', 'export', 'external-config'],
) as dag:
    load_config_task = PythonOperator(
        task_id='load_config_task',
        python_callable=load_config
    )

    create_tasks = PythonOperator(
         task_id="create_export_tasks",
         python_callable=create_export_tasks
        )

    load_config_task >> create_tasks
```
```json
// export_config.json file
[
  {
    "task_id_suffix": "export_table1",
    "query": "SELECT * FROM `project.dataset.table1` WHERE date = \"{{ ds }}\"",
    "destination_uris": ["gs://my-bucket/data1/{{ ds }}/output*.csv"],
    "export_format": "CSV"
  },
  {
    "task_id_suffix": "export_table2",
    "query": "SELECT id, value FROM `project.dataset.table2`",
    "destination_uris": ["gs://my-bucket/data2/output.json"],
    "export_format": "JSON"
  },
  {
    "task_id_suffix": "export_table3",
    "query": "SELECT  region, COUNT(*) FROM `project.dataset.table3` GROUP BY region",
    "destination_uris": ["gs://my-bucket/data3/{{ ds_nodash }}/aggregated*.parquet"],
    "export_format": "PARQUET",
    "print_header": false
  }
]
```
Example 3 tackles the scenario where configurations are externalized, typically from a JSON or YAML file. I've found this practice invaluable in production environments. This decoupling allows the export logic to be modified without requiring changes to the DAG itself. In this example, the `load_config_task` uses a python operator to read from the `export_config.json` file, and the configurations are extracted by accessing the return value through the `xcom_pull` mechanism. After loading the configurations, `create_export_tasks` builds the mapped tasks similarly to example 2. This approach also simplifies the process of maintaining a growing set of export operations.

To bolster your understanding and capabilities with this pattern, I recommend exploring the official Airflow documentation, particularly the sections covering templating, dynamic task generation, TaskGroups and the `expand()` function. Deep dives into Google Cloud's documentation for BigQuery and GCS will be beneficial, particularly the details on query syntax and GCS destination URIs. Also, exploring advanced features in Airflow such as task mapping and XComs will help in building more complex and maintainable DAGs. Consulting books or online resources concerning software architecture patterns for data pipelines can provide insights that can help guide your development approach.
