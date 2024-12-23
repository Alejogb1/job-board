---
title: "Why is Airflow's `on_failure_callback` function not triggered if a task fails, but the DAG succeeds?"
date: "2024-12-23"
id: "why-is-airflows-onfailurecallback-function-not-triggered-if-a-task-fails-but-the-dag-succeeds"
---

Okay, let's tackle this peculiar behavior with Airflow's `on_failure_callback`. I remember facing this exact issue when we were migrating a hefty ETL pipeline to Airflow a few years back; it was baffling initially, as we expected every task failure to trigger the callback regardless of overall dag success. It took some thorough investigation to fully grasp the mechanics at play.

The core of the matter lies in the distinction between task states and dag states, and how Airflow orchestrates the execution flow. When a task within a dag encounters a failure, such as a python exception, the task's individual state is marked as "failed." This transition *does* indeed trigger the various task-level listeners like `on_failure_callback` if they're defined on that *specific task*. However, and this is the critical part, it doesn't necessarily translate into the entire dag failing.

A dag's success is judged primarily on its ability to reach its terminal state. If a dag is designed to allow certain tasks to fail without causing the entire pipeline to abort, it will still be marked as successful *as long as the downstream tasks which are not dependent on the failed task execute successfully*. This is by design, allowing for fault tolerance and handling transient failures without halting the entire process. In essence, a dag success means that the dag *workflow*, according to its dependencies, has been completed even if individual parts experienced problems.

It's helpful to visualize this as a relay race. If one runner falls (a task failure) but the next runner successfully picks up the baton and completes their leg, the team (the dag) still finishes the race. We might want to know that one runner fell (hence the need for task-level `on_failure_callback`), but it doesn't negate the overall result of the race.

Now, let’s illustrate this with examples. Suppose we have a simple dag with three tasks: `task_a`, `task_b`, and `task_c`. `task_b` is deliberately designed to fail, while `task_a` and `task_c` will always succeed. Here is how you’d typically set it up:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def task_a_func():
    print("Task A executed successfully")

def task_b_func():
    raise ValueError("This task is designed to fail!")

def task_c_func():
    print("Task C executed successfully")

def failure_callback_func(context):
    print(f"Task failed: {context['task_instance'].task_id}")

with DAG(
    dag_id="dag_with_task_failure",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:

    task_a = PythonOperator(
        task_id='task_a',
        python_callable=task_a_func,
    )

    task_b = PythonOperator(
        task_id='task_b',
        python_callable=task_b_func,
        on_failure_callback=failure_callback_func
    )

    task_c = PythonOperator(
        task_id='task_c',
        python_callable=task_c_func,
    )

    task_a >> task_b >> task_c
```

In this scenario, the `failure_callback_func` *will* execute when task_b fails, due to it being specified within the `task_b` operator. The DAG will *still* be marked as successful because, despite task_b failing, task_c is not dependent on its success. The workflow completes without any further impediments.

Let's explore a variation. Let's say we change the dag so that `task_c` now depends on the *successful* completion of `task_b`. In this case, the dag would indeed be marked as failed. Here’s that updated code:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.utils.trigger_rule import TriggerRule


def task_a_func():
    print("Task A executed successfully")


def task_b_func():
    raise ValueError("This task is designed to fail!")


def task_c_func():
    print("Task C executed successfully")


def failure_callback_func(context):
    print(f"Task failed: {context['task_instance'].task_id}")


with DAG(
    dag_id="dag_failure_on_dependent_task",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:

    task_a = PythonOperator(
        task_id="task_a",
        python_callable=task_a_func,
    )

    task_b = PythonOperator(
        task_id="task_b",
        python_callable=task_b_func,
        on_failure_callback=failure_callback_func,
    )

    task_c = PythonOperator(
        task_id="task_c",
        python_callable=task_c_func,
        trigger_rule=TriggerRule.ALL_SUCCESS,  # <--- This is the important change
    )

    task_a >> task_b >> task_c
```
The critical change is the `trigger_rule=TriggerRule.ALL_SUCCESS` parameter applied to `task_c`. This makes `task_c` require all its upstream tasks (in this case, only `task_b`) to be successful.  Because `task_b` fails, this now causes `task_c` to be skipped, and consequently, the overall DAG will be marked as failed. If you wanted to trigger a dag level on_failure callback, you would need to define it in the DAG constructor. The previous example, and its output, might confuse some at first.

To bring it home, let’s make it explicit: a DAG has `on_failure_callback` which executes when the dag as a whole is marked as failed (after being unsuccessful and retries exhausted). The *tasks* have `on_failure_callback` and this callback fires when the task itself has failed. The trigger is based on the *task*, not the *dag*.

Here is an example of dag-level on failure callback being fired. This will only be fired when the whole DAG has failed.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.utils.trigger_rule import TriggerRule


def task_a_func():
    print("Task A executed successfully")


def task_b_func():
    raise ValueError("This task is designed to fail!")


def task_c_func():
    print("Task C executed successfully")


def failure_callback_func(context):
    print(f"Task failed: {context['task_instance'].task_id}")

def dag_failure_callback_func(context):
    print(f"Dag Failed:{context['dag'].dag_id}")

with DAG(
    dag_id="dag_failure_on_dependent_task_with_dag_callback",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    on_failure_callback=dag_failure_callback_func
) as dag:

    task_a = PythonOperator(
        task_id="task_a",
        python_callable=task_a_func,
    )

    task_b = PythonOperator(
        task_id="task_b",
        python_callable=task_b_func,
        on_failure_callback=failure_callback_func,
    )

    task_c = PythonOperator(
        task_id="task_c",
        python_callable=task_c_func,
        trigger_rule=TriggerRule.ALL_SUCCESS,  # <--- This is the important change
    )

    task_a >> task_b >> task_c
```

To deepen your understanding of these behaviors, I highly recommend reviewing the official Apache Airflow documentation, particularly the sections on DAG and task states, callbacks, and trigger rules. Beyond that, the book "Data Pipelines with Apache Airflow" by Bas Penders is an excellent resource for understanding these nuances and building production-ready pipelines. Furthermore, papers such as "Airflow: A Workflow Management System for Data Engineering" offer insights into the architectural considerations that lead to such design choices, although they might focus more on the design principles rather than direct operational instructions. Finally, actively participating in the Airflow user community and exploring examples from other users is also very useful to get a broader perspective. Understanding the interplay between task and dag states is key to effective Airflow workflow management, and addressing this aspect early can save a significant amount of debugging effort down the line.
