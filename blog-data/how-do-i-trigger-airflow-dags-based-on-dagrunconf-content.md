---
title: "How do I trigger Airflow DAGs based on dag_run.conf content?"
date: "2024-12-16"
id: "how-do-i-trigger-airflow-dags-based-on-dagrunconf-content"
---

Alright, let's dive into triggering Airflow DAGs using `dag_run.conf`. I’ve been working with Airflow for quite a while, and this is a scenario that comes up more frequently than you might think. The beauty of Airflow lies in its ability to be both rigorously scheduled and flexibly triggered based on external events or data, and manipulating `dag_run.conf` is a key part of that flexibility.

The short answer is: you leverage the Airflow API or CLI, providing the required `conf` parameters when creating a new `dag_run`. However, the "how" is far more nuanced, and I've seen teams stumble over some of the finer details. Let's break it down into a practical approach, along with some real-world examples and code snippets.

First, it's crucial to understand that `dag_run.conf` isn't a magic configuration that automatically triggers DAGs based on some internal polling mechanism. Instead, *you* initiate a `dag_run`, and the `conf` dictionary provides the context that the specific DAG instance uses during its execution. This context can be anything you need the DAG to know at runtime, such as file paths, API keys, or specific processing parameters. This is not to be confused with DAG-level configuration which is often passed in as Environment Variables or within the dag.py file itself.

Let's assume, for a moment, I had a data pipeline years ago that processed incoming CSV files. Each file required specific settings depending on its source, and we used the `dag_run.conf` to parameterize the data processing step. This isn't uncommon. We had to trigger each processing task by passing in unique `conf` values with each new file. The DAG itself remained constant; only the `conf` changed.

Here are three specific scenarios and how to handle them.

**Scenario 1: Triggering a DAG via the Airflow CLI with `conf`**

The simplest way to manually trigger a DAG with `conf` is using the Airflow CLI. The command takes a dictionary, which should be specified as a string. Note that `"` is not escaped inside the string and should be wrapped by a different character. The format for `conf` is a standard JSON string. It's a common trap to forget that and try a python dictionary, which of course fails.

```bash
airflow dags trigger my_data_pipeline \
  --conf '{"file_path": "/data/incoming/file1.csv", "source": "external_system_a", "process_date": "2023-11-20"}'
```

This command triggers the `my_data_pipeline` DAG and provides a JSON string as the `conf` payload. Inside your DAG definition, you'd access these parameters in your tasks using something like this:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


def process_data(**kwargs):
    conf = kwargs['dag_run'].conf
    file_path = conf.get('file_path')
    source = conf.get('source')
    process_date = conf.get('process_date')

    print(f"Processing file: {file_path} from {source} on {process_date}")
    # Your data processing logic would go here.


with DAG(
    dag_id='my_data_pipeline',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    process_data_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data
    )
```

The `kwargs['dag_run'].conf` will fetch your passed `conf` data and the task would process the file as defined in the passed `conf`. The important piece here is that the `conf` is a dictionary, even though it is passed as a string. It needs to be parsed correctly, or you get the dreaded type error and a failed dag run. Note that `dag_run.conf` can be `None` if no config is passed, which can be handled with a `conf.get('key')` as above.

**Scenario 2: Triggering a DAG Programmatically via the Airflow API**

For more complex scenarios, such as an event-driven pipeline, you might need to trigger the DAG via the Airflow API. Here is a python example, which leverages the `requests` library. I typically prefer using the client library but the `requests` library provides a simpler example.

```python
import requests
import json

AIRFLOW_URL = "http://your_airflow_webserver_address"
DAG_ID = 'my_data_pipeline'
AUTH = ('your_username', 'your_password')

def trigger_dag_with_conf(conf_data):
    url = f"{AIRFLOW_URL}/api/v1/dags/{DAG_ID}/dagRuns"
    headers = {
        'Content-Type': 'application/json',
    }
    data = {
        "conf": conf_data
    }
    response = requests.post(url, auth=AUTH, headers=headers, data=json.dumps(data))
    if response.status_code == 200:
        print("DAG triggered successfully with conf data.")
        return response.json()
    else:
        print(f"Failed to trigger DAG. Status code: {response.status_code}, Message: {response.text}")
        return None

if __name__ == '__main__':
    conf_params = {
        "file_path": "/data/incoming/file2.csv",
        "source": "external_system_b",
        "process_date": "2023-11-21"
    }
    response_data = trigger_dag_with_conf(conf_params)
    if response_data:
        print("Dag run ID:",response_data["dag_run_id"])
```

This snippet sends a `POST` request to the Airflow API, triggering the DAG with the `conf` data passed in the `json.dumps` format, and you can examine the response for dag run ID or error messages, if any. Again, it's crucial to pass a valid JSON string as the `conf`. If you're using the Airflow Python API, there are dedicated functions for initiating DAG runs with configuration, which might abstract away some of the JSON handling and provide greater ease of use, but they follow similar principles. This is very helpful to call from an external application in response to an external trigger.

**Scenario 3: Handling Complex Nested JSON Data in `conf`**

Often, the `conf` data isn’t as simple as the flat dictionary shown previously. It might be a deeply nested JSON structure which can contain lists and dictionaries, depending on the complexity of the use-case. I once dealt with a scenario where the `conf` contained a list of datasets to process, each with its own metadata.

Here is a simple example of how to handle this.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import json

def process_datasets(**kwargs):
    conf = kwargs['dag_run'].conf
    datasets = conf.get('datasets')
    if datasets:
      for dataset in datasets:
          dataset_name = dataset.get('name')
          dataset_path = dataset.get('path')
          dataset_format = dataset.get('format')
          print(f"Processing dataset {dataset_name} from {dataset_path} in format {dataset_format}")
          # Your logic to handle different datasets
    else:
       print ("No datasets found in conf.")


with DAG(
    dag_id='complex_data_pipeline',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    process_datasets_task = PythonOperator(
        task_id='process_datasets',
        python_callable=process_datasets
    )

if __name__ == '__main__':
    conf_params = {
        "datasets": [
            {
                "name": "dataset_a",
                "path": "/data/dataset_a.parquet",
                "format": "parquet"
            },
            {
                "name": "dataset_b",
                "path": "/data/dataset_b.csv",
                "format": "csv"
            }
        ]
    }
    print (f"Simulating conf: {json.dumps(conf_params)}")
    # Code to trigger using Airflow API or CLI similar to the above examples.
    # Here, instead of making an api call, simulating the conf processing
    process_datasets(dag_run = type('dag_run',(object,), {'conf':conf_params})())
```

This code snippet shows that the `conf` parameter is read in as a dictionary and can be handled as any other python dictionary, even if it is passed as a json string via the api or cli. The `process_datasets` task iterates over the list of datasets and extracts each parameter. When dealing with complex data structures, it's always good practice to add checks for missing keys, or unexpected data types to ensure the pipeline doesn’t fail due to invalid `conf`. This might be handled through a custom error handler, logging and a specific notification to the pipeline operator if the dag run fails.

To further refine your understanding, I would recommend referring to the official Apache Airflow documentation, as it's continuously updated and is the canonical source. Furthermore, the book "Data Pipelines with Apache Airflow" by Bas Harenslak and Julian de Ruiter offers in-depth insights into advanced Airflow concepts. A further detailed examination of the Airflow API documentation is also beneficial to see a list of all available endpoints and how to use them.

In summary, triggering Airflow DAGs with `dag_run.conf` is a powerful technique. The key is to ensure that the `conf` is passed as a correctly formed JSON string and to handle the parsed dictionary inside the DAG. Whether through the CLI or API, the same underlying principles apply. Be cautious with the data types being passed, as it is easy to inadvertently cause a data type mismatch and a failure of your DAG, and always log all the inputs when debugging a problem.
