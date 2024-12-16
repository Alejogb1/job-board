---
title: "Why is my airflow DAG parameter max_active_runs not working?"
date: "2024-12-16"
id: "why-is-my-airflow-dag-parameter-maxactiveruns-not-working"
---

Okay, let’s tackle this one. I’ve seen this particular frustration pop up more times than I care to count, and it almost always boils down to understanding a few nuanced aspects of how Airflow schedules and limits task execution. It’s a common stumbling block, and not something instantly obvious from a quick read-through of the documentation, which, although thorough, can sometimes leave a gap when translating theory into actual operational scenarios.

The issue isn't that `max_active_runs` isn't *working*, per se, but rather it's often misunderstood how it interacts with other Airflow components, particularly the scheduler and the various types of task instances within a dag run. My experience, particularly a rather memorable incident while scaling a large data pipeline last year, has given me a very pragmatic understanding of this. In that particular case, we were seeing an overflow of dag runs consuming resources, even though we’d specified a `max_active_runs` value which we believed should be keeping things in check. We quickly learned it's far more complex than simply setting a single parameter and expecting everything to behave.

The core of the problem is this: `max_active_runs` controls the *maximum number of simultaneously running dag runs*, not the number of simultaneously running *tasks* across all dag runs. This distinction is absolutely critical. Airflow schedules individual dag runs, and this parameter restricts how many such runs can be in a ‘running’ state at any given moment. A 'running' state, in Airflow's parlance, involves active tasks within the run. However, tasks within a single dag run might all be pending or queued while awaiting resources or dependencies to be met. If those tasks are not actively in a state that is considered 'running', it is not counted against this limit.

To illustrate further, imagine you've got a DAG scheduled hourly, and you set `max_active_runs=2`. Ideally, you’d expect only two hourly runs to be executing at any given point. However, let’s say that, for example, each run kicks off a single task. That single task might take a very long time, or might depend on external systems, or could even be throttled by the executor. During the time that this task is pending, but not yet in a state recognized by the scheduler as ‘running’, Airflow might start another new dag run because, technically, it does not have two *active* runs. Airflow considers a dag run active when at least one of its tasks has entered an executing state, not when the dag run itself has been triggered. That is, 'queued' is not 'running' for this calculation.

Another common misstep occurs when using backfills or catchup scheduling. If these options are activated, and they aren’t managed carefully, you can easily exceed what you anticipated even with the limit in place. If a backfill occurs, Airflow might attempt to launch multiple dag runs simultaneously, all of which count against this parameter once their tasks start executing. This can create contention and overload, even when you believe you’ve got enough capacity. In our previous incident, this is precisely what happened when a late data stream backfill coincided with normal hourly runs, resulting in an overwhelming of our worker nodes.

To clarify, let's look at some working code examples illustrating how and why setting `max_active_runs` alone isn't always enough.

**Example 1: Basic DAG with Potential Overlap**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import time

def sleep_task():
    time.sleep(60)  # Simulate a task taking a while

with DAG(
    dag_id='basic_max_active_example',
    start_date=datetime(2024, 1, 1),
    schedule='@hourly',
    catchup=False,
    max_active_runs=2,
) as dag:
    task1 = PythonOperator(
        task_id='sleepy_task',
        python_callable=sleep_task
    )
```

In this basic DAG, if `sleep_task` is slow, new hourly runs might queue and trigger because the prior run isn't yet executing tasks. Even with `max_active_runs=2`, numerous scheduled dag runs might be waiting and then start executing in parallel if the resources become available at about the same time.

**Example 2: Controlling concurrency with resource pools**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
import time

def sleep_task():
    time.sleep(60)

with DAG(
    dag_id='resource_pool_example',
    start_date=datetime(2024, 1, 1),
    schedule='@hourly',
    catchup=False,
    max_active_runs=2,
) as dag:
    task1 = PythonOperator(
        task_id='sleepy_task_pool1',
        python_callable=sleep_task,
        pool='resource_pool_1'
    )

    task2 = PythonOperator(
        task_id='sleepy_task_pool2',
        python_callable=sleep_task,
        pool='resource_pool_2',
        trigger_rule=TriggerRule.ALL_DONE # Ensures task 2 runs after task 1 regardless of outcome
    )
```

In this example, we leverage resource pools. `max_active_runs` still limits concurrent DAG runs, but resource pools can help further restrict the concurrency of individual tasks across different dags or even within the same dag. This provides more granular control beyond the simple `max_active_runs` parameter. Create the pools via the Admin UI or by defining them in the config. We often use a combination of resource pools, task priority and queue assignments to ensure resources are appropriately scaled and tasks are executed in an organized and prioritized manner.

**Example 3: Combined strategy - max_active_runs and task concurrency control**

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='combined_control_example',
    start_date=datetime(2024, 1, 1),
    schedule='@hourly',
    catchup=False,
    max_active_runs=2,
    max_active_tasks_per_dagrun=1 # This parameter restricts task concurrency inside a specific dag run
) as dag:
    task1 = BashOperator(
        task_id='bash_task_1',
        bash_command='sleep 60'
    )
    task2 = BashOperator(
        task_id='bash_task_2',
        bash_command='sleep 30'
    )

```

In this final example, in addition to limiting dag runs to a maximum of 2 concurrently running instances, we added `max_active_tasks_per_dagrun`. This parameter limits the number of *tasks* that can be executed *concurrently* within a single dag run to 1. This allows you to prevent the tasks from within any single dag from saturating system resources and running tasks sequentially.

For a deeper dive, I strongly recommend looking at the official Airflow documentation on DAG scheduling and configuration, specifically the details regarding resource pools. "Programming Apache Airflow" by Bas Harenslak is a great hands-on guide that covers these concepts in a practical manner. Additionally, researching the concepts of concurrency and parallelism in distributed systems will shed light on some of the fundamental challenges that lead to these kinds of situations. Furthermore, reading research papers on workflow orchestration is also helpful. Papers from conferences like DEBS and ICDE often cover related concepts. Remember, troubleshooting such issues often requires a holistic approach and a careful review of not just the DAG configuration but the underlying infrastructure as well. Understanding these concepts deeply can help prevent these issues from reoccurring in the future. This is not an 'easy fix' scenario but rather one that requires a solid understanding of the framework.
