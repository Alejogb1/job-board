---
title: "Why isn't Airflow's `on_failure_callback` being triggered after task failures?"
date: "2024-12-23"
id: "why-isnt-airflows-onfailurecallback-being-triggered-after-task-failures"
---

Okay, let's tackle this. I’ve seen this tripping up a few teams, and it's almost always down to a subtle misunderstanding of how Airflow handles task states and the lifecycle of those callbacks. So, instead of diving headfirst into the code, I think it’s best we first explore the landscape of task states within Airflow.

Airflow task instances move through various states: `scheduled`, `queued`, `running`, `success`, `failed`, `skipped`, `up_for_retry`, and a few others. The key here is that the `on_failure_callback` specifically triggers when a task instance enters the `failed` state *after* all its retry attempts, if applicable, have been exhausted. It's not a catch-all for every error that might occur within a task's execution. This often leads to confusion because people assume that any task that doesn't complete successfully automatically triggers the `on_failure_callback`. It’s not that simple, unfortunately.

My own past experience with this comes from a project where we were orchestrating a complex ETL pipeline. I remember we had this beautifully designed dag, but then noticed that our notification system wasn’t working correctly. We’d had tasks silently fail. After some head-scratching, we realized the default retry policy was masking the issue, and that our callback function was essentially waiting for a failure that never *actually* registered as a terminal `failed` state, because tasks were always ending in `up_for_retry`.

Let's break down three common scenarios where your `on_failure_callback` might seem like it's not working.

**Scenario 1: The Default Retry Mechanism is Active:**

By default, Airflow tasks have a built-in retry policy. If a task fails during its initial run, it goes into an `up_for_retry` state. Airflow will then attempt to re-run the task after a specified delay, up to a `max_retries` threshold. The `on_failure_callback` is only triggered after *all* retries have failed and task enters the `failed` state.

Consider this example:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def failing_task():
    raise ValueError("Something went wrong!")

def failure_callback(context):
    print(f"Task failed! Task ID: {context['task_instance'].task_id}")

with DAG(
    dag_id='retry_dag',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args={
        'retries': 3, # default retry
        'on_failure_callback': failure_callback,
    }
) as dag:
    task = PythonOperator(
        task_id='failing_task',
        python_callable=failing_task,
    )

```

Here, `failing_task` will fail on its first run. It will not trigger the `failure_callback` immediately. Instead, Airflow will retry it three times before ultimately setting it to `failed`. Only *then* will your callback be invoked. If, for instance, you expect an immediate notification on the first error, this will be a source of your frustrations.

**Scenario 2: Exceptions are handled within the task and never reach Airflow’s exception handling:**

If you wrap your critical task code in a try/except block, and don't re-raise the exception, the task might finish with a successful state because you're explicitly catching the error, not letting it escape to Airflow to register a failure.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def try_except_task():
    try:
       raise ValueError("Inside try block error")
    except ValueError:
        print("Error handled within the task.")
        #Task exits without an error and Airflow will set it to "success"

def failure_callback(context):
    print(f"Task failed! Task ID: {context['task_instance'].task_id}")


with DAG(
    dag_id='try_except_dag',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args={
        'retries': 0,
         'on_failure_callback': failure_callback,
    }
) as dag:
    task = PythonOperator(
       task_id='try_except_task',
       python_callable=try_except_task
    )
```

In this example, even though an error occurs within `try_except_task`, the exception is caught within the function and the task execution is effectively "successful" from Airflow’s perspective. The `on_failure_callback` will never be triggered because the task does not end in a `failed` state.

**Scenario 3: Task failure occurs before the task itself starts:**

There are situations where a task may fail *before* your task function even begins to run. Examples include issues in dependency resolution, task serialization problems, or failure to pull a docker image (if you are using docker operator). These kinds of failures also do not trigger an `on_failure_callback`. The best way to troubleshoot these cases is checking the logs specifically related to the dag execution and see if any errors pop-up related to setup before entering task's execution.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime

def success_task():
   print("This task will never fail")

def failure_callback(context):
    print(f"Task failed! Task ID: {context['task_instance'].task_id}")

with DAG(
    dag_id='dependency_fail',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    default_args={
        'retries': 0,
         'on_failure_callback': failure_callback,
    }
) as dag:
    # dummy task that does not exist
    dependency_task = DummyOperator(task_id='dummy_task_not_there')
    success_task = PythonOperator(
       task_id='success_task',
       python_callable=success_task
    )
    success_task.set_upstream(dependency_task)
```
In this final example, the `success_task` will never run because it has an upstream dependency on `dummy_task_not_there` which will never execute, causing the dag to be in `failed` state immediately. The `on_failure_callback` configured in the dag will not fire.

**Recommendations and Troubleshooting:**

1.  **Understand Task Retries:** If you’re relying on `on_failure_callback` for immediate notifications, you might want to set `retries = 0`. While this works, be cautious because it sacrifices robustness of tasks which can be useful in production environment. Another strategy is to trigger a specific callback on `up_for_retry`.

2.  **Ensure Exceptions are Raised:** Inside your task, avoid catching exceptions unless you are taking a very specific action on them and then *re-raising* the exception so that Airflow understands that the task failed.

3.  **Check Logs Carefully:** Look at the task instance logs on Airflow UI. This is critical to understand where failure occured. Is it task function failing, or some error before execution.

4.  **Consider Dag Callbacks:** Dag-level `on_failure_callback` can be helpful if you need to react to entire dag's failure instead individual tasks.

5.  **Dive Deeper into Airflow Concepts:** I'd strongly suggest reading "Programming Apache Airflow" by Bas P. Harenslak and Julian J. Lange. This book delves deep into the lifecycle of a dag and task. Also check out the official Airflow documentation, particularly the sections on task execution, callbacks, and exception handling. In addition, the source code on github is also a good resource, particularly the `airflow.taskinstance.TaskInstance` class implementation, where task transitions are defined.
Remember that Airflow's execution model has its intricacies. Being patient, careful with the log analysis, and having a clear understanding of how tasks move through various states is the key to troubleshooting. This will resolve the majority of issues I have seen over time.
