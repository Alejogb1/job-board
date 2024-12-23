---
title: "Why is the Airflow on_success_callback not giving task information?"
date: "2024-12-23"
id: "why-is-the-airflow-onsuccesscallback-not-giving-task-information"
---

Okay, let's address this. It’s a scenario I've seen surface more than a few times, especially when teams start scaling their Airflow deployments. You’re expecting task context in your `on_success_callback`, but all you’re getting is a rather generic callback without the specific details you need. This isn’t a bug, per se, but rather a consequence of how Airflow structures its callbacks and the information it makes available in different contexts. Let me break down what’s happening and how we can effectively resolve it, drawing from past projects where we’ve battled through the same issue.

First, we need to clarify that the `on_success_callback` (and similarly, the `on_failure_callback` and others) at the DAG level are designed to trigger on the successful completion (or failure) of an *entire* DAG run, not a particular task within that run. The context provided to these DAG-level callbacks, therefore, isn't task-specific. The `ti` (TaskInstance) object you are typically expecting, which holds task details like execution_date, task_id, etc., isn't directly passed to these callbacks. The framework’s architecture aims to keep DAG-level callbacks lightweight and decoupled from the intricacies of specific task runs.

The default callback payload primarily provides information about the DAG run, such as the `dag_id`, `run_id`, start time, and execution date. This is useful if you're triggering alerts based on the overall success or failure of the pipeline. However, if you require information about the individual tasks within that DAG, you'll need to implement a different approach.

Historically, I remember one project where we had a complex ETL pipeline with hundreds of tasks. We initially attempted to rely on the DAG-level `on_success_callback` for detailed logging of task execution outcomes. What we found, of course, is that it provided very little granular information. We tried accessing the `ti` object, but to no avail; it’s simply not available at that stage. We realized we needed task-specific callbacks to get the task-level details we needed.

So, how do we get that juicy task information? The solution lies in leveraging callbacks at the task level within the DAG definition. Instead of using the DAG's `on_success_callback`, we define our callbacks directly on specific task definitions. These task-level callbacks *do* receive the `ti` object, granting us access to the specific task’s context.

Here’s how this works. We can pass functions or callables to the `on_success_callback` parameter of a `BaseOperator` or one of its subclasses like `PythonOperator`, `BashOperator`, etc. within a dag. This is where task context becomes available.

Let's illustrate with some code examples:

**Example 1: A simple PythonOperator with a task-level success callback.**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime

def task_success_callback_example(context):
    task_instance = context['ti']
    print(f"Task {task_instance.task_id} succeeded at {task_instance.end_date}, run id {task_instance.run_id}")

def some_function():
    print("This is an example task")

with DAG(
    dag_id='task_callback_example',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=['example'],
) as dag:

    task1 = PythonOperator(
        task_id='example_task',
        python_callable=some_function,
        on_success_callback=task_success_callback_example
    )
```

In this example, the `task_success_callback_example` function receives the entire context dictionary, which includes the `ti` object under the `'ti'` key. We extract the `TaskInstance` and log the task id, end date, and run id. This function will trigger only when `example_task` succeeds.

**Example 2: Accessing Task Instance details in a bash operator.**

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago

def bash_success_callback(context):
    ti = context['ti']
    print(f"Bash task {ti.task_id} completed. Log URL: {ti.log_url}")

with DAG(
    dag_id='bash_callback_example',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=['example'],
) as dag:

    bash_task = BashOperator(
        task_id='bash_example_task',
        bash_command='echo "Bash task executing"',
        on_success_callback=bash_success_callback
    )

```

Here, we’re doing the same thing, but this time within a BashOperator. We access the `ti` object inside the callback and use its attributes to print the task ID and the URL of the generated log for debugging purposes. You can see that you don't need to use the PythonOperator to have access to context.

**Example 3: Using a callback to update an external system.**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

from datetime import datetime
import requests

def update_api_with_task_success(context):
    ti = context['ti']
    api_url = "https://example.com/api/task_status" # Replace this with a suitable URL
    payload = {
        "task_id": ti.task_id,
        "execution_date": str(ti.execution_date),
        "status": "success"
    }
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status() # Raise an HTTPError for bad responses
        print(f"API updated for task: {ti.task_id}, Status Code: {response.status_code}")
    except requests.exceptions.RequestException as e:
         print(f"Failed to update API for {ti.task_id} due to error: {e}")


def some_other_function():
    print("This is some other task.")

with DAG(
    dag_id='api_callback_example',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=['example'],
) as dag:
    task2 = PythonOperator(
        task_id='example_task_with_api',
        python_callable=some_other_function,
        on_success_callback=update_api_with_task_success
    )

```
This example demonstrates a more practical usage of task callbacks, pushing success information to an external API upon the task’s completion, which could be part of a larger system monitoring solution. The `update_api_with_task_success` callback sends information about the finished task to the external service, including the execution date and the successful status. This can enable integration into an external monitoring system which may be beyond Airflow. We included basic exception handling here, as this is a real-world use case where network requests may fail.

These examples illustrate the critical difference: task-level callbacks provide specific task context via the `ti` object, while DAG-level callbacks provide general information about the DAG's overall execution.

To solidify your understanding, I'd recommend reviewing the Airflow documentation on operators, specifically focusing on the callback functionality for `BaseOperator` and its subclasses. The "Airflow's Context" section is especially pertinent. Additionally, a worthwhile reference is the "Programming Apache Airflow" book by Jarek Potiuk and Bartlomiej Gladysz. The book includes detailed explanations of many topics, including callbacks. Finally, diving into Airflow’s source code, specifically the `airflow.models.baseoperator` and `airflow.models.taskinstance` files, can give an even deeper understanding of how task context is managed within the framework itself.

In summary, If you want task-specific information in callbacks, don’t look at the DAG-level options but at task-level definitions. Use those `on_success_callback` (and counterparts) directly within task declarations to access the all-important `ti` object and its associated properties. This approach will grant you the granularity and control necessary for monitoring and managing complex Airflow workflows. Remember the separation of concerns between the DAG and its tasks, and use the appropriate callback for the level of information required. This will prevent many frustrating hours attempting to access missing data from the wrong context.
