---
title: "Why doesn't the Airflow DAG parameter `max_active_runs` limit the number of active runs?"
date: "2024-12-23"
id: "why-doesnt-the-airflow-dag-parameter-maxactiveruns-limit-the-number-of-active-runs"
---

Alright, let's tackle this one. It’s a classic head-scratcher that I've seen trip up many a data engineer, including myself, back in the days when I was knee-deep in Airflow v1.x configurations. The core issue isn't that `max_active_runs` is faulty, but rather it’s often misunderstood in its scope of influence. It doesn't directly limit the number of concurrently *running* dag instances, as one might initially expect. Instead, its role is focused on managing the *scheduled* dag runs that are eligible to be executed. Let me break this down with some technical nuance and then illustrate with some simplified Python examples.

`max_active_runs`, at its heart, is a parameter within the dag definition that limits the *number of scheduled dag runs that can exist in the 'running' or 'queued' state at any given time*. It doesn't police task-level concurrency; that's what other mechanisms within Airflow such as pool or executor configuration handle. Crucially, if more dag runs are triggered than permitted by `max_active_runs`, the scheduler will *not* immediately attempt to execute them. Instead, it'll hold them in the 'scheduled' state until slots become available, either through previous dag runs succeeding or failing. Think of it as a gatekeeper for running dag instances rather than an active limiter of running tasks within those instances.

This distinction is important because we're dealing with the orchestration of whole workflows rather than fine-grained task management directly with this parameter. There’s an interplay between the dag scheduler, its executor, and the configurations that determine *how* and *when* a task within a given dag run will execute. `max_active_runs` affects the “when,” ensuring we don’t flood the executor with more dag runs than it can reasonably handle. The executor's own configuration (e.g., number of worker processes for the Celery executor, or the concurrent slot limit for a Kubernetes executor) is what directly manages the parallelism of *tasks* across one or more dag runs.

Now, let's solidify that with some example code snippets. Consider this simplified dag definition:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

with DAG(
    dag_id='example_dag_max_runs',
    default_args=default_args,
    schedule_interval=timedelta(minutes=5),
    catchup=True,
    max_active_runs=2,
) as dag:
    task1 = BashOperator(
        task_id='print_date',
        bash_command='date',
    )
```

In this example, `max_active_runs` is set to 2. The dag is configured to run every 5 minutes. Let's assume that each run takes 2 minutes and that multiple dag runs become scheduled very quickly due to historical catchup enabled. What happens is the first two runs go into either the running or queued state. A third will be scheduled, but will remain as "scheduled" awaiting one of the earlier runs to complete. Once one of those finishes, the third will transition into a runnable state and the process repeats. The key here is not that the tasks within the runs are limited, but that only 2 dag runs can be active at a time. This example uses a single, simple task but the principle applies to complex dags too.

Now, let’s take a slightly more complex example that demonstrates this behavior under load:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import time

def my_long_task():
    time.sleep(120)  # Simulate a 2 minute task

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}

with DAG(
    dag_id='example_dag_max_runs_load',
    default_args=default_args,
    schedule_interval=timedelta(minutes=1),
    catchup=True,
    max_active_runs=3,
) as dag:
    task1 = PythonOperator(
        task_id='long_task',
        python_callable=my_long_task,
    )
```

Here, even though the dag is scheduled to run every minute and each run takes two minutes to complete, `max_active_runs` ensures a maximum of three *dag runs* are processing concurrently. If we had not set that, multiple dag runs would be attempted causing the scheduler and executor to potentially run into issues and certainly overload the systems. If your executor can handle more concurrent tasks, you might want to set `max_active_runs` higher and manage task-level concurrency via resource pool assignment, executor settings, or other task concurrency controls.

Finally, let’s consider an example where the task within the dag also has concurrency limits:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
}


with DAG(
    dag_id='example_dag_max_runs_with_pool',
    default_args=default_args,
    schedule_interval=timedelta(minutes=5),
    catchup=True,
    max_active_runs=3,
) as dag:
    task1 = BashOperator(
        task_id='task_using_pool',
        bash_command='sleep 30',
        pool='my_limited_pool' # Requires creating this pool in airflow
    )

    task2 = BashOperator(
        task_id = 'task_fails',
        bash_command='exit 1',
        trigger_rule=TriggerRule.ALL_DONE
    )
    task1 >> task2

```

In this third example, in addition to the dag being limited to three concurrent runs, `task1` is assigned to a named pool. Airflow’s pool mechanism further restricts the number of *concurrent task instances* of *any* dag using that pool. This interaction demonstrates how `max_active_runs` and pool settings manage concurrency at different layers, dag run-level and task level respectively, within the Airflow ecosystem. `trigger_rule=TriggerRule.ALL_DONE` makes `task2` run only once `task1` has completed or failed, regardless of the success of `task1`. In short, it allows us to schedule a task *after* `task1`. Pools, executors, resource configurations and `max_active_runs` interact, so you need a good mental model of how your workflow will be executed.

For deeper understanding, I recommend digging into the Apache Airflow documentation (certainly), but beyond that, consider these resources: “Programming Apache Airflow” by Bas P. Harenslak and "Data Pipelines with Apache Airflow" by Jesse Anderson are excellent practical guides. The official Airflow documentation's section on scheduling and concurrency is also invaluable. These go into much more detail than space permits here about the scheduler's inner workings, task assignment, and other concurrency considerations, which ultimately are needed to fully comprehend how to implement a proper solution for more intricate use cases. Understanding these layers is crucial for designing scalable and manageable data workflows with Airflow. It’s not just about the parameters themselves, but how they fit into the larger architectural puzzle.
