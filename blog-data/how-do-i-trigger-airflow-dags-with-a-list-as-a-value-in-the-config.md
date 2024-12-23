---
title: "How do I trigger Airflow DAGs with a list as a value in the config?"
date: "2024-12-16"
id: "how-do-i-trigger-airflow-dags-with-a-list-as-a-value-in-the-config"
---

Alright,  I remember a particularly challenging deployment we had a few years back. We were dealing with a highly variable data ingest process, and the requirement was to trigger our Airflow dags with configurations driven by a list of values. It wasn't straightforward, and we had to iterate through a few solutions before settling on a reliable method. The core issue is that Airflow doesn't inherently support passing a list directly to the configuration like a simple string or number. We need to leverage its templating capabilities and, sometimes, custom operator logic to make this work smoothly.

Essentially, when you're talking about triggering a dag with a list as a configuration value, you're dealing with two main aspects: how to pass the list data to the dag at runtime and then how to effectively utilize that list within the dag's tasks. Airflow allows for passing parameters during a trigger event using the `conf` argument. This argument takes a dictionary, where the values are typically strings. Therefore, the first challenge is converting our list into a string format suitable for passing through this `conf` and then deserializing it back to a list in our dag.

Let's start with encoding the list to a string. The most effective way I’ve found is to use the `json.dumps` function from python. This reliably converts any python data structure into a json string. Then, in your dag’s Python code, you’d reverse that using `json.loads` to get the list back. This guarantees that the list will be properly represented during transfer and can be parsed back into a usable form.

Here’s the first code snippet illustrating this process. This simulates the external process that triggers the dag and includes the list of values as a dictionary in the `conf`:

```python
import json
from airflow.api.client.local_client import Client

# Simulating an external process that will trigger the dag
list_of_items = ["item_a", "item_b", "item_c"]
config_data = {"my_list": json.dumps(list_of_items)}

# Assuming you have an airflow client configured
client = Client(api_base_url='http://localhost:8080')  # replace with your api url
client.trigger_dag(dag_id="my_dag_with_list", conf=config_data)
print(f"Triggered dag with config: {config_data}")

```

In this snippet, we prepare the list to be sent and convert it to a json string using `json.dumps()`. We then bundle it into a dictionary. This is the structure Airflow expects for configuration at trigger time. I always recommend having logging statements when you're working with this level of data manipulation; it becomes crucial for debugging when issues arise.

Now, within the dag, we'll need to retrieve this value and deserialize it. This leads us to the next core part of the puzzle: accessing and using the list. Within your airflow dag, you should use jinja templating to get access to the `conf` dictionary that is passed to the dag. Within a task (especially a PythonOperator), you can access the `dag_run.conf` dictionary. This lets us get the encoded list. Here is a snippet showing a basic dag accessing the list in one task:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import json

def process_items(**kwargs):
    conf = kwargs['dag_run'].conf
    if conf and 'my_list' in conf:
        list_data = json.loads(conf['my_list'])
        for item in list_data:
            print(f"Processing item: {item}")
    else:
      print("No list found in configuration")


with DAG(
    dag_id="my_dag_with_list",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["example"],
) as dag:
    process_task = PythonOperator(
        task_id='process_my_list',
        python_callable=process_items,
    )
```

Here, in `process_items()`, we grab the `conf` dict from the task context via `kwargs['dag_run'].conf`, and, if our ‘my_list’ entry is present, we use `json.loads` to unpack it into a list. Then we loop over the items in the list. This example shows how you can access the list directly within a `PythonOperator`.

There are cases, however, where you need a more granular control over each element of the list, and one such strategy is to make use of the `TaskGroup` and `TaskFlow API` with dynamic task generation. In a scenario like this, it is very common for each task to consume one element in the list. The following demonstrates how you can dynamically generate tasks.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
import json
from airflow.decorators import task

@task
def process_item(item):
    print(f"Processing single item: {item}")

def create_tasks(**kwargs):
   conf = kwargs['dag_run'].conf
   if conf and 'my_list' in conf:
        list_data = json.loads(conf['my_list'])
        return list_data
   else:
        return []


with DAG(
    dag_id="my_dynamic_dag_with_list",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["example"],
) as dag:
    items = PythonOperator(
      task_id='fetch_items',
      python_callable=create_tasks
    )
    with TaskGroup('process_tasks') as process_tasks:
       #using taskflow to dynamically generate the tasks.
       tasks = items.map(process_item)


```

In this final example, we first create the `fetch_items` task using a `PythonOperator` to grab the list from the `conf` and return the list. This returned list is used as input to the `items` variable. Then we use `task.map` to create a task for each item in the list. This will create multiple instances of the `process_item` task. This is very helpful for scenarios in which you need to process elements in the list in parallel.

For those looking to delve deeper into these concepts, I highly recommend checking out “Programming Apache Airflow” by Bas P. Harenslak and “Data Pipelines with Apache Airflow” by Jesse Anderson, published by O'Reilly Media. Also, the official Apache Airflow documentation is indispensable, especially the sections on templating and the TaskFlow API. The documentation on configuration is equally important. Mastering these aspects is essential for efficiently working with Airflow.

My past experience shows that these techniques, while they might initially appear complex, provide a robust and flexible way to handle dynamic configurations within Airflow. The key is to understand how data flows into the dag from external triggers, how Airflow uses Jinja templates for variable access, and how the various operators and APIs can help manage data transformations within each individual task, or through dynamic task generation as in the last example.
