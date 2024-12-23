---
title: "How do I trigger an airflow dag with a config which has a list as a value?"
date: "2024-12-23"
id: "how-do-i-trigger-an-airflow-dag-with-a-config-which-has-a-list-as-a-value"
---

Okay, let's break down how to trigger an airflow dag with a configuration that includes a list as a value. It's a common scenario, particularly when you need to parameterize your workflows based on a set of input data or processing targets. I've encountered this often enough across various data engineering pipelines and the challenges it introduces, and I'll walk you through the best practices I've found, illustrated by some concrete examples.

The core issue stems from how airflow handles DAG configurations, particularly when interacting with its API or when you’re invoking a DAG run via the command-line interface (cli). You're essentially passing a dictionary, and when that dictionary has a value that’s a list, it needs to be properly serialized and deserialized. Airflow internally uses json for these operations, meaning you need to ensure your list is json-serializable.

When we discuss ‘config’ in the context of an airflow dag, we're typically referring to the `conf` parameter available in several contexts: `dag_run.conf` within operators, when you’re running a dag via the api using the `create_dagrun` method or using the airflow cli’s `dags trigger` option. It’s how we effectively customize the execution of a specific run of your DAG with unique parameters. My experience has shown that properly preparing this config and understanding how airflow consumes it is the key to success.

The most direct method, and the one i prefer due to its predictability, is to construct the config in Python as a dictionary and then pass that dictionary, which includes your list, as the `conf` value when triggering the DAG. Airflow handles the necessary serialization automatically for you. Let’s look at a python example you could use outside of your dag definition, to understand how config needs to be formatted before we dive into a DAG example:

```python
import json
from airflow.api.client.local_client import Client

# Configure your connection
local_client = Client(
    api_base_url="http://localhost:8080/api/v1",
)

# Define your dag_id
dag_id = "your_dag_id"

# Define your configuration
config = {
    "some_key": "some_value",
    "my_list": ["item1", "item2", "item3"],
    "nested_dict": {"key1": "val1", "key2": 2}
}


# Trigger the dag run with the config
try:
  response = local_client.trigger_dag_run(
      dag_id=dag_id,
      conf=config
  )
  print(f"Dag {dag_id} triggered successfully. Run ID: {response.run_id}")

except Exception as e:
  print(f"Error triggering DAG: {e}")

```

In this snippet, `config` is our dictionary containing a list under the key `my_list`. When we use `local_client.trigger_dag_run()`, airflow serializes this dictionary to JSON behind the scenes. On the dag side of things, within your operators, this config can be accessed using `dag_run.conf`. Here’s how you could implement this inside an airflow dag:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


def process_config(**kwargs):
    dag_run = kwargs["dag_run"]
    if dag_run and dag_run.conf:
        config = dag_run.conf
        print(f"Received Config: {config}")
        my_list = config.get("my_list")
        if my_list:
          for item in my_list:
            print(f"Processing item: {item}")
        else:
            print("No list found in config")

        some_key_value = config.get("some_key")
        if some_key_value:
          print(f"some_key: {some_key_value}")
        else:
          print("no some_key in the config")

        nested_dict = config.get("nested_dict")
        if nested_dict:
          print(f"nested dict {nested_dict.get('key1')}")
        else:
           print("nested_dict not found in config")


    else:
        print("No config passed to the DAG Run.")


with DAG(
    dag_id="config_dag_with_list",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    process_config_task = PythonOperator(
        task_id="process_config_task",
        python_callable=process_config,
    )
```

This DAG example shows how to access and use the configuration passed during the trigger. The `process_config` function accesses `dag_run.conf` to retrieve the dictionary, and from there, you can extract the list using the associated key. If no config is passed or the key does not exist, the logic includes checks to handle those scenarios, which is very crucial in production environments. This demonstrates how the list is passed and consumed by your dag.

The final example, demonstrating the cli use case. This is helpful for quick tests and manual interventions.

```bash
airflow dags trigger config_dag_with_list -c '{"some_key": "cli_value", "my_list": ["cli_item1", "cli_item2", "cli_item3"], "nested_dict": {"key1": "cli_val1", "key2": 5}}'
```

Here, you are using the airflow cli to trigger a dag and you are providing a stringified json payload as the `conf` attribute, airflow parses this string into a dict. You could pass any valid json in this format. The key here is that your list, which is part of the json payload is treated correctly by airflow.

Regarding further learning, I’d recommend “Programming Apache Airflow” by J. Humble and K. S. Park. It’s a fantastic resource for understanding these intricacies of airflow. For more background on how API calls work, you can review the official airflow documentation and look at specific api calls, like the `trigger_dagrun` endpoint. This will help you when trying to build integrations with other tools. Another useful text would be the official python documentation around `json.dumps` and `json.loads`, since these functions are vital to how airflow handles configs under the hood.

When dealing with complex list operations, it’s often beneficial to decompose the logic within your operators, potentially leveraging other python libraries, ensuring your code remains manageable. These steps, along with testing strategies, provide a solid path to reliably trigger airflow dags with configurations that include lists. My own journey with these setups involved a series of trial and error, and I hope this detailed account will make your process more streamlined.
