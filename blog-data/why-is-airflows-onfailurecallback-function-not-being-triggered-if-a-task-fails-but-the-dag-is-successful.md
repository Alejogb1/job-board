---
title: "Why is Airflow's `on_failure_callback` function not being triggered if a task fails but the DAG is successful?"
date: "2024-12-23"
id: "why-is-airflows-onfailurecallback-function-not-being-triggered-if-a-task-fails-but-the-dag-is-successful"
---

Okay, let's tackle this. I've seen this tripped up a fair number of folks over the years, and I've personally spent time debugging this exact scenario, so I understand the frustration it can cause. It's a nuanced issue stemming from how Airflow distinguishes between task failures and dag failures.

The core reason why your `on_failure_callback` isn't firing when a task fails, but the dag reports as successful, lies in Airflow’s internal logic regarding task states and dag run states. A task failing doesn't automatically equate to the dag failing. Airflow essentially sees tasks as individual units of work within a larger workflow. A dag run only fails if there's a failure that Airflow considers "terminal" for the *entire* workflow. This distinction is crucial to understand.

Consider this scenario: you have a dag with multiple tasks, some of which have retries enabled. If a task fails initially, it goes into a "failed" state, but because retries are configured, Airflow will try again. If the retries succeed, the task eventually becomes "success," even though it initially failed. The crucial part here is that the dag run itself continues onward. The `on_failure_callback` defined at the *dag* level is only triggered when there's a *dag* failure, not a transient *task* failure that is eventually resolved.

To illustrate, imagine a dag that loads data into a database. The first task might be to extract data from an api, and the second is to load it. Let's say the extract task fails twice before successfully fetching the data. The dag completes successfully overall, even with those initial failures in the extract task. That initial failure will not trigger the `on_failure_callback` of the DAG since the DAG run is a success.

Now, let's look at some specific examples to demonstrate this in action, including what *will* trigger the task level failure handling and how to work with it, and what triggers the dag level one.

**Example 1: Task-Level Failure Handling Using a Decorator**

To capture task-level failures, we use the `@task` decorator's `on_failure_callback` parameter:

```python
from airflow.decorators import dag, task
from datetime import datetime
from airflow.operators.python import PythonOperator


def my_failure_handler(context):
    print("Task failed! Context:", context)


@dag(
    dag_id="task_failure_example",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
)
def task_example_dag():
    @task(on_failure_callback=my_failure_handler, retries=0)  #no retries for clarity
    def failing_task():
      raise ValueError("This task is designed to fail.")

    PythonOperator(
        task_id="dummy_task",
        python_callable=lambda: print("This will run regardless of failing_task's fate")
    )

    failing_task()

task_example_dag()
```

In this example, `failing_task` has a specific `on_failure_callback`. Since it's designed to raise an exception, `my_failure_handler` will *always* be called immediately after the task fails. Note that even if you have `retries=2` here the `my_failure_handler` will be called each time the task fails but before retry is initiated. The dummy task will proceed normally as the DAG run has not failed.

**Example 2: Dag-Level Failure Handling**

Now, let’s see a dag failure example. The key to ensuring the dag failure handler is invoked is to let a task fail without retry *and* ensure that subsequent tasks are not designed to proceed regardless, so that the whole DAG run cannot complete normally.

```python
from airflow.decorators import dag, task
from datetime import datetime
from airflow.operators.python import PythonOperator


def dag_failure_handler(context):
    print("Dag failed! Context:", context)


@dag(
    dag_id="dag_failure_example",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    on_failure_callback=dag_failure_handler,
)
def dag_example_dag():
    @task(retries=0)
    def failing_task():
      raise ValueError("This task is designed to fail and halt the dag run.")

    PythonOperator(
        task_id="dummy_task",
        python_callable=lambda: print("This should NOT run as the dag failed.")
    )

    failing_task() >> "dummy_task"

dag_example_dag()
```

In this second example, we have the `on_failure_callback` defined at the dag level. When `failing_task` fails (and has no retry) and its subsequent task is not set to run "regardless" the entire dag run is marked as a failure. In such instances, `dag_failure_handler` is invoked, because now, the *dag* has failed, not just an individual task, stopping the DAG run execution from completing.

**Example 3: Using Task Groups to handle failures**

A third option that gives more fine-grained control is to create specific task groups and leverage that groups `on_failure_callback` to allow granular failure handling. This is particularly useful when only a portion of the DAG has experienced an issue.

```python
from airflow.decorators import dag, task
from datetime import datetime
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup


def group_failure_handler(context):
    print("Task group failed! Context:", context)


@dag(
    dag_id="task_group_failure_example",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
)
def task_group_example_dag():
    with TaskGroup("my_failing_tasks", on_failure_callback=group_failure_handler) as failing_tasks_group:
        @task(retries=0)
        def failing_task_in_group():
          raise ValueError("This task is designed to fail within a group.")

        @task
        def another_task_in_group():
          print("This task is NOT designed to fail.")


        failing_task_in_group() >> another_task_in_group()

    PythonOperator(
        task_id="dummy_task",
        python_callable=lambda: print("This will run even though the group failed")
    )

    failing_tasks_group >> "dummy_task"

task_group_example_dag()
```
Here, when any task inside the "my_failing_tasks" TaskGroup fails without a retry it will trigger the `group_failure_handler` callback. Since the dummy task is not inside the group and the group's failure does not stop the whole DAG run, the DAG will still complete successfully. This allows for isolated failure handling of a portion of the workflow, without affecting the broader DAG success or failure.

**Key Takeaways and Recommendations:**

*   **Task-Level Failures:** Use the `@task` decorator or the `BaseOperator`'s `on_failure_callback` for individual task failures. This callback is triggered after the task retries (if any) are exhausted.
*   **Dag-Level Failures:** The dag's `on_failure_callback` is triggered when the *entire dag run* fails to complete successfully. This typically means no retries are enabled, or tasks are designed to fail and block future steps.
*   **Task Groups:** Utilize Task Groups and their specific callback to isolate failures to a group of tasks. This avoids unnecessary DAG failures when only certain sections are having issues.
*   **Understand Task States:** Familiarize yourself with the various task states in Airflow (e.g., queued, running, success, failed, up_for_retry, etc.) to better grasp when callbacks are triggered.

For deeper understanding, I strongly recommend the following resources:

1.  **The official Apache Airflow documentation:** This is the go-to for detailed explanations of all features and concepts, including callbacks, task states, and DAG behavior. Pay close attention to the sections on the DAG and Task objects, as well as the explanation of task lifecycle.
2.  **"Data Pipelines with Apache Airflow" by Bas P. Harenslak:** This book offers a practical approach to building real-world data pipelines using Airflow. It explains concepts more clearly with use-case examples, particularly on advanced features like callbacks and failure handling.
3.  **The source code of Airflow itself:** If you’re comfortable with Python and want a more granular understanding, diving into the source code, specifically the `airflow.models.dag.DAG` and `airflow.models.taskinstance.TaskInstance` classes, will help clarify how these callbacks and task states are handled internally.

In my experience, it’s all about understanding that distinction between individual task failures and overall dag run failures. Knowing this, and using these callback mechanisms appropriately, you can build much more robust and reliable data pipelines. I hope these examples and resources provide some clarity on this commonly misunderstood aspect of Airflow.
