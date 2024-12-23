---
title: "Why is Airflow's on_success_callback and other types of callbacks not giving proper information on the task level?"
date: "2024-12-23"
id: "why-is-airflows-onsuccesscallback-and-other-types-of-callbacks-not-giving-proper-information-on-the-task-level"
---

Alright, let’s tackle this. I’ve spent a fair bit of time wrestling with Airflow’s callback system, specifically the on_success_callback and its siblings. It's a frustration point for many, and for good reason – extracting granular, task-level information can feel like pulling teeth. The issue, at its core, isn't a flaw in Airflow's design, but rather a consequence of how callbacks are architected and the execution context they operate within.

From my experience, the initial assumption is often that callbacks will naturally receive detailed task-specific information, such as the task instance id, retry count, and specific error messages if a task fails. However, these callbacks are triggered at the *dag-run* level, not the task level directly. The callback is executed after the dag run has reached its final state – either success or failure – and thus has access primarily to the dag-run related context, not the individual task instance state. This means the information available is primarily about the overall execution of the dag itself rather than granular details of each task within that dag. This difference is quite subtle yet profoundly impacts how we approach the design of our callback functions.

Let's break down precisely why this occurs. Airflow, as a scheduler, manages workflows as directed acyclic graphs (dags). Each dag is executed through a dag-run, which is an instantiation of the dag. Within that dag-run, tasks are executed, retried, and eventually reach a terminal state. Callbacks such as `on_success_callback`, `on_failure_callback`, and `on_retry_callback` are inherently attached to the *dag-run*. When the dag-run transitions into success, failure, or retry state, these callbacks are triggered. They are provided a `context` dictionary which includes dag-run level data like the dag id, execution date, and logical date. The crucial detail often missed is that the context does not automatically propagate individual task instance details. This is not an oversight but a deliberate choice in how Airflow manages its resources and parallel task execution. Airflow is focusing on the overall status of the dag, not getting bogged down with individual task states while executing callbacks.

This design choice prevents callbacks from becoming a bottleneck, particularly in scenarios with many tasks. If each callback had to gather and process information about every task, performance could degrade quickly. Instead, the context provided is limited to the dag-run, allowing the callback execution to be lightweight and not directly tied to the performance of tasks within the dag. However, this means we need to take a slightly different path to gather task specific information, moving beyond the initial expectation that it will simply be part of the context dictionary.

Now, how do we actually get the task-level information we crave? The key is leveraging Airflow’s metadata database and the xcom mechanism. We can query the database using the Airflow API to retrieve information related to specific task instances within the current dag-run, or we can push and retrieve custom data using xcom.

Here’s an example demonstrating the database query approach using the built-in `airflow.models.taskinstance` class within a callback function:

```python
from airflow.models import DagRun, TaskInstance
from airflow.utils import timezone
from datetime import datetime
from airflow.utils.session import provide_session

@provide_session
def task_info_callback(context, session=None):
    dag_run_id = context['dag_run'].id
    dag_run = session.query(DagRun).filter(DagRun.id == dag_run_id).first()

    if dag_run:
        ti_query = session.query(TaskInstance).filter(
            TaskInstance.dag_id == dag_run.dag_id,
            TaskInstance.run_id == dag_run.run_id,
        )
        for ti in ti_query:
            print(f"Task ID: {ti.task_id}, State: {ti.state}, Start Time: {ti.start_date}, End Time: {ti.end_date}")


# Example usage in a DAG:
# with DAG(dag_id="my_dag", ... on_success_callback=task_info_callback)
```

In this example, we leverage the `provide_session` decorator to obtain an SQLAlchemy session connected to the Airflow metadata database. Within the callback, we use this session to obtain the dag-run and all related task instances, printing out key information about each. This pattern illustrates that we must explicitly query Airflow’s internals, rather than relying on the provided context directly, to get information about individual tasks.

A second approach involves using xcom to push necessary information from the task itself into a centralized location, making it accessible to the callback. This works well if the information is something dynamically generated as part of the task itself, such as error messages that don’t have a standard representation in Airflow’s core metadata. Here's an example:

```python
from airflow.decorators import dag, task
from datetime import datetime
from airflow.operators.python import PythonOperator
from airflow.models import XCom

def push_task_info(ti, **kwargs):
  # Simulate some task that might fail and get retried
  # For example, ti.task_id and ti.try_number
  ti.xcom_push(key="task_id", value=ti.task_id)
  ti.xcom_push(key="try_number", value=ti.try_number)

def xcom_callback(context):
  task_instance = context['ti']
  xcom_data = task_instance.xcom_pull(task_ids=task_instance.task_id, include_prior_dates=True)

  print(f"task_id: {xcom_data.get('task_id')}, try_number: {xcom_data.get('try_number')}")


# Example Usage in a DAG
@dag(start_date=datetime(2023, 1, 1), schedule=None, catchup=False, on_success_callback=xcom_callback)
def example_xcom_dag():
  task1 = PythonOperator(
    task_id='push_info',
    python_callable=push_task_info
  )


  task1

example_xcom_dag()

```

Here, the `push_task_info` task pushes the task id and try number to xcom. The `xcom_callback` function pulls the xcom and prints it. This shows how information can be exchanged between tasks and the callback function through xcom.

A third, perhaps less common, but potentially useful approach is to use the dag-level `on_failure_callback`, `on_retry_callback` etc. to implement custom logic that handles task level information. This method gives the programmer direct access to the `failed_task_instances` which is a list of task instances that failed within the dag run.

```python
from airflow.decorators import dag, task
from datetime import datetime
from airflow.operators.python import PythonOperator
from airflow.models import XCom
from typing import List
from airflow.models import TaskInstance


def failure_callback(context, failed_task_instances: List[TaskInstance]):
    print("Dag Failed.  The following tasks failed:")
    for ti in failed_task_instances:
        print(f"  Task ID: {ti.task_id}, Try Number: {ti.try_number}, State: {ti.state}, Log: {ti.log_url}")

# Example Usage in a DAG
@dag(start_date=datetime(2023, 1, 1), schedule=None, catchup=False, on_failure_callback=failure_callback)
def example_callback_dag():

    @task
    def fail():
        raise ValueError("This task is designed to fail")
    fail()

example_callback_dag()

```

This example illustrates how `on_failure_callback` provides access to `failed_task_instances`, which can be extremely useful for debugging and custom reporting.

These examples aren't exhaustive, but they highlight a crucial concept: Airflow's callbacks work at the dag-run level. If you need task-specific information within a callback, you have to make an explicit effort to gather it, either using the Airflow database or utilizing xcom to push information from the tasks themselves. This design might require additional effort but, in reality, allows for more performant and decoupled workflows. Understanding this distinction is essential to designing effective callback logic in Airflow. For a deeper dive, I highly recommend studying the official Airflow documentation, specifically the pages on dag execution, callbacks, and xcom as well as resources like “Data Pipelines with Apache Airflow” by Bas Geerdink and "Programming Apache Airflow" by Daniel Imberman which provide more nuanced perspectives on the complexities of Airflow. Remember, mastering Airflow requires understanding its core design principles, and callbacks are no exception.
