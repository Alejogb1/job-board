---
title: "How do I use Dynamic Task Mapping with Decorators in Airflow?"
date: "2024-12-16"
id: "how-do-i-use-dynamic-task-mapping-with-decorators-in-airflow"
---

Right then, let’s talk about dynamic task mapping with decorators in airflow; it’s a topic I've spent quite a few cycles on over the years. I remember a particularly challenging project where we were ingesting data from a constantly evolving set of external sources. Using standard, statically defined tasks just wasn't cutting it; we needed the workflow to adapt to the incoming data dynamically. That's when I really started to appreciate the power of task mapping combined with the convenience of decorators.

Now, for anyone unfamiliar, dynamic task mapping in airflow, especially when paired with decorators, allows you to create multiple instances of the same task based on a set of input values determined at runtime. This is a significant improvement over statically defining tasks, which become unwieldy and hard to maintain when dealing with variable inputs. Decorators, of course, simplify the syntax and make our dags more readable and concise.

Let's break down how this works, and then I’ll show you a few examples that reflect scenarios I’ve actually encountered. The key idea here is to use the `task.expand()` method within a decorated task definition. This method will, when executed during runtime, take an input parameter—typically a list or dictionary—and then create individual task instances for each of the provided elements or key-value pairs.

Before diving into the code, it’s worth recommending some background material. For a solid foundation on airflow itself, I suggest *“Airflow in Action”* by Ben Weber and Marc Lamberti. Also, for a deeper dive into the concepts of parallel processing and distributed systems, a good text is *“Designing Data-Intensive Applications”* by Martin Kleppmann. These resources should provide both the practical and theoretical contexts you need to fully grasp the intricacies of dynamic task mapping.

Okay, on to the first example. Imagine you're processing files from a varying number of cloud storage buckets. You don't know beforehand how many buckets there are, but at runtime you fetch this info via an api. Here's how you might implement that with dynamic task mapping and decorators:

```python
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from datetime import datetime

@dag(schedule=None, start_date=days_ago(1), catchup=False, tags=['example'])
def dynamic_bucket_processing():

    @task
    def get_bucket_names():
        # Imagine this pulls bucket names from some external source
        return ["bucket_a", "bucket_b", "bucket_c"]

    @task
    def process_bucket(bucket_name: str):
        print(f"Processing bucket: {bucket_name}")
        # Imagine actual processing logic here
        return bucket_name

    bucket_names = get_bucket_names()
    processed_buckets = process_bucket.expand(bucket_name=bucket_names)


dynamic_bucket_processing_dag = dynamic_bucket_processing()
```

In this first snippet, we have a `get_bucket_names` task that returns a list of bucket names. The `process_bucket` task is designed to process one bucket name, and instead of calling it directly, we use `.expand()` to pass the list `bucket_names`. Airflow then creates distinct `process_bucket` tasks for each item in the list, allowing parallel execution. This exemplifies the power of dynamic instantiation.

Next up, let's say you have to process json data, and you need to call a different downstream task based on a property within the individual data items. This is very common with heterogeneous data sources:

```python
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from datetime import datetime

@dag(schedule=None, start_date=days_ago(1), catchup=False, tags=['example'])
def dynamic_json_processing():

    @task
    def fetch_json_data():
        # Simulate fetching from an external API
        return [
            {"id": 1, "type": "A", "data": "data_for_a"},
            {"id": 2, "type": "B", "data": "data_for_b"},
            {"id": 3, "type": "A", "data": "data_for_c"},
            {"id": 4, "type": "C", "data": "data_for_d"}
            ]

    @task
    def process_type_a(item: dict):
        print(f"Processing item of type A: {item['id']}")
        # Logic specific for type A items.
        return item['id']

    @task
    def process_type_b(item: dict):
        print(f"Processing item of type B: {item['id']}")
        # Logic specific for type B items.
        return item['id']

    @task
    def process_type_c(item: dict):
        print(f"Processing item of type C: {item['id']}")
        # Logic specific for type C items.
        return item['id']

    json_data = fetch_json_data()
    type_a_tasks = process_type_a.expand(item=[item for item in json_data if item['type'] == "A"])
    type_b_tasks = process_type_b.expand(item=[item for item in json_data if item['type'] == "B"])
    type_c_tasks = process_type_c.expand(item=[item for item in json_data if item['type'] == "C"])

dynamic_json_processing_dag = dynamic_json_processing()
```

Here, we introduce some more sophisticated logic. The `fetch_json_data` task retrieves a list of dictionaries. Instead of expanding all items into a single task, we conditionally expand the data based on the 'type' field in each dictionary. This allows us to route different items to different task functions, again in a dynamically generated manner. The list comprehension provides the filter logic, showcasing how flexible this approach can be.

Finally, there’s a scenario I faced once where our configuration was stored as key/value pairs, each key representing a separate processing task. Here is an example of how this can be applied using dynamic task mapping:

```python
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from datetime import datetime

@dag(schedule=None, start_date=days_ago(1), catchup=False, tags=['example'])
def dynamic_config_processing():

    @task
    def get_config_data():
        # Simulate fetching from an external API
        return {
             "task_a" : {"param1": "val1a", "param2": "val2a"},
             "task_b" : {"param1": "val1b", "param2": "val2b"},
             "task_c" : {"param1": "val1c", "param2": "val2c"}
        }

    @task
    def process_config_entry(task_name: str, parameters: dict):
        print(f"Processing task {task_name} with parameters: {parameters}")
        # Actual processing logic using the given configuration.
        return task_name

    config = get_config_data()
    process_config_entry.expand(task_name=config.keys(), parameters=config.values())

dynamic_config_processing_dag = dynamic_config_processing()
```

This last example demonstrates how to expand using both the keys and values of a dictionary, creating instances of the same task with different parameters for each item in the configuration. `task_name` and `parameters` act as inputs, and airflow creates a separate process for each config key, providing both the task name and the parameters of this task instance. This is a really powerful technique for complex configuration-driven workflows.

A final consideration is how to handle complex output dependencies between these dynamically generated tasks. While this response doesn't have enough space to delve fully into that, it’s worth noting that airflow's built-in XComs (cross-communication) and the taskflow api provide all the necessary machinery to handle these dependencies. You can retrieve data from each of the generated tasks and use it as input to a new level of dynamic task mapping, if needed. It is crucial to design your DAGs with these dependencies in mind from the start to ensure a robust workflow.

So, there you have it: dynamic task mapping using decorators within airflow. These techniques, with their flexibility and power, are absolutely essential to handling real-world, complex data workflows. The three examples should provide a solid starting point, reflecting actual use cases, and the recommended readings will give you a deeper theoretical underpinning. Remember that like all powerful techniques, they require careful consideration and a good grasp of underlying concepts but can be well worth the effort.
