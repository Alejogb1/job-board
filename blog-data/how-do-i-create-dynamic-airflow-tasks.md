---
title: "How do I create dynamic Airflow tasks?"
date: "2024-12-16"
id: "how-do-i-create-dynamic-airflow-tasks"
---

, let's tackle dynamic task creation in Airflow. I’ve certainly spent my share of late nights navigating this particular area, and I’ve found a few approaches that consistently deliver the results you're probably after. Forget manually defining every single task – that’s a recipe for maintenance nightmares. The core issue is that Airflow DAGs, at their heart, are essentially static Python scripts. They’re parsed once at import time. So, the key is to leverage Python's dynamic capabilities *within* this static structure to generate task configurations at runtime.

I remember once dealing with a data ingestion pipeline where source data schemas were being updated daily. We started with hardcoded tasks and quickly found ourselves in a cycle of constant DAG edits, which as you can imagine, was not sustainable. We had to get dynamic and fast. So, first and foremost, let's be very clear: “dynamic” here doesn't mean the DAG definition itself changes constantly, because as mentioned, that's not how Airflow works. Instead, we are dynamically creating *task instances* within the static structure of a DAG. These task instances get created when the DAG is being parsed.

There are mainly three methods I've found particularly effective for handling this: task mapping with the `expand` method, task generation within a loop using functions or factory classes and finally, the use of XComs to pass runtime information. Let’s break these down with examples:

**1. Task Mapping with `expand`**

The most straightforward method for many use cases, especially when you have a known set of inputs that you want to process in parallel, is the `expand` method introduced in Airflow 2.0+. It allows you to dynamically generate task instances based on a list or dictionary. It works beautifully with the taskflow api and is, in my view, the preferred solution when it fits your use case.

Here's a snippet that illustrates this:

```python
from airflow import DAG
from airflow.decorators import task
from datetime import datetime

with DAG(
    dag_id="dynamic_mapping_example",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    @task
    def process_item(item_id):
        print(f"Processing item: {item_id}")
        # Placeholder for your specific processing logic
        return f"processed_{item_id}"

    item_ids = ["item_a", "item_b", "item_c"]

    processed_items = process_item.expand(item_id=item_ids)

    @task
    def finalize(processed_items):
        print(f"Final processing of: {processed_items}")

    finalize(processed_items)
```

In this example, `item_ids` is a list. The `process_item.expand(item_id=item_ids)` line dynamically creates three instances of the `process_item` task, each with a different value from the `item_ids` list. Notice the `expand()` method, this is what triggers dynamic mapping. The mapped task `processed_items` returns a list of xcom values from the mapped tasks. This list is passed as an argument to the final task.

This approach excels when you have a clear set of data points to process and want parallel execution for each of these. The generated task ids will be `process_item_1`, `process_item_2`, `process_item_3` within the airflow UI.

**2. Dynamic Task Generation within a Loop**

For more complex scenarios, sometimes you'll need greater control over task generation. This involves constructing task definitions within a Python loop or functions.

Let's look at a working example, this time using `BashOperator` to show how it can be achieved beyond the taskflow api:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime


def create_bash_task(task_id, command):
    return BashOperator(
        task_id=task_id, bash_command=command
    )

with DAG(
    dag_id="dynamic_loop_example",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    task_commands = {
        "task_a": "echo 'Executing task a'",
        "task_b": "echo 'Executing task b'",
        "task_c": "echo 'Executing task c'"
    }

    for task_id, command in task_commands.items():
         create_bash_task(task_id, command)
```

Here, we define a function, `create_bash_task`, that returns a `BashOperator` task instance with the provided id and bash command. We use this function within a loop to create tasks from a dictionary of task ids and their corresponding shell commands. This offers more flexibility when you need customized tasks with varying logic. The tasks will show in the airflow UI as `task_a`, `task_b`, `task_c`.

The logic to decide how many tasks or how they are to be parameterized can be easily extracted into a function which could do things like list databases, get file paths, or do anything that you require to determine your required task configuration.

**3. Using XComs for Runtime Task Configuration**

Sometimes the set of tasks or their parameters are dependent on the result of upstream tasks. For instance, one task may identify a set of files that downstream tasks need to process. In this scenario, you'll use XComs to pass runtime information from an upstream task to a downstream task that generates tasks.

Here is an example where an upstream task determines a list of parameters:

```python
from airflow import DAG
from airflow.decorators import task
from airflow.operators.python import PythonOperator
from datetime import datetime

def generate_parameters(**context):
    parameters = ["param1", "param2", "param3"]
    context['ti'].xcom_push(key='params', value=parameters)

def process_param(param):
        print(f"Processing parameter: {param}")

with DAG(
    dag_id="dynamic_xcom_example",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    generate_params = PythonOperator(
        task_id='generate_parameters',
        python_callable=generate_parameters
    )


    @task
    def process_parameters(params):
        for param in params:
           process_param.override(task_id=f"process_param_{param}")(param=param)


    process_parameters(generate_params.output)

```

In this instance, `generate_parameters` is a `PythonOperator` that generates a list of parameters and pushes them to XCom with the key `params`. The `process_parameters` task then uses the returned xcom value to loop through and create individual tasks. In effect this is a combination of the other methods.

While all these methods are powerful, some are more suitable than others depending on the use case. For instance, if you are dealing with a batch of identical tasks, the `expand()` method is ideal and should probably be your first choice. When the structure of the tasks varies, or you need more complex logic, the loop method is suitable. And if tasks need to be parameterized based on upstream tasks, then XComs are your best bet. Remember to use them all together as needed as well.

For further reading, I strongly suggest diving into the Airflow documentation, specifically the sections related to task mapping and XComs, you will find more advanced examples and techniques there. Also, consider looking at “Data Pipelines with Apache Airflow” by Bas P. Harenslak and Julian Rutger de Ruiter, this is a great book that dives deeper into the nuts and bolts of Airflow and data pipelines. Finally, keep exploring and experimenting. Each use case is unique, and the more you build, the better you will become at choosing the correct solution for the challenge at hand.
