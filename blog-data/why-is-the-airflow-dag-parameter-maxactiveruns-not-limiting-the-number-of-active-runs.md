---
title: "Why is the Airflow DAG parameter `max_active_runs` not limiting the number of active runs?"
date: "2024-12-23"
id: "why-is-the-airflow-dag-parameter-maxactiveruns-not-limiting-the-number-of-active-runs"
---

Let's talk about `max_active_runs` in airflow. It’s a parameter that, on the surface, seems straightforward—limit the concurrent executions of a given dag. I've seen it trip up quite a few people, and I've had my own share of head-scratching moments with it. Based on my experience, its apparent simplicity conceals a few critical nuances that often lead to unexpected behavior, specifically, instances where it seems like it's simply ignored.

To tackle this, we need to understand that `max_active_runs` doesn’t directly constrain the *number* of running dag instances. Instead, it restricts the *number of scheduled runs* that can be actively executing. The distinction is important. Airflow manages dag runs by creating a 'dag run' object for each scheduled interval. It is these objects, rather than individual task executions, that are constrained. The scheduler essentially looks at the number of *unprocessed* dag run objects; only once those hit the specified `max_active_runs` limit will the scheduler stop *triggering new runs* of the dag. This doesn't immediately stop any already executing tasks within those running dag instances.

A key point is that `max_active_runs` impacts scheduling, not active task execution *per se*. If you have tasks with very long execution times, you might have many tasks running concurrently, even though the number of dag run instances respect the `max_active_runs` limitation. This is very crucial to grasp. The parameter doesn't limit the total number of concurrently running *tasks* across the dags; it caps *dag runs*.

Further complicating matters, the `max_active_runs` is evaluated by the scheduler on its interval of scanning the DAGs and isn’t enforced in real-time. It's a periodic check, typically every `min_file_process_interval` seconds (default is 300). So, there’s a chance that the scheduler, during one of its periodic scans, will trigger more runs than the configured `max_active_runs` due to variations in how fast dag runs complete. Consider a short peak load, for instance; even with `max_active_runs` set to 1, you can potentially see more than one dag run active within a short time window, if one run starts within that processing interval while an older run is still finalizing.

Let’s illustrate these points with some practical code examples. Imagine a simple DAG designed to mock the processing of log data:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import time

def simulate_processing():
    time.sleep(10)  # Simulate a 10-second processing time
    print("Processing complete.")


with DAG(
    dag_id="log_processing_dag",
    schedule="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
) as dag:
    process_logs = PythonOperator(
        task_id="process_logs",
        python_callable=simulate_processing,
    )

```

In this first example, I've set `max_active_runs=1`. The intent here is that only one dag run should be active at any time. However, if the processing time is long relative to the scheduler's process interval, especially if some tasks within a dag run complete faster than others, you could see two runs technically active at the same time. The newer run has been triggered and is 'active' (ie it has a status, such as 'running'), although, its tasks may still be in the queue. The important point here is the scheduler has not scheduled more than 1 run at once, but some of the running tasks may overlap. The limit is on the number of active dag *runs*, not the number of concurrent active *tasks*.

Now consider a scenario where the task processing time is far longer. Assume we modify the `simulate_processing` function:

```python
def simulate_processing():
    time.sleep(600) # Simulate a 10-minute processing time
    print("Processing complete.")
```

Using the same DAG configuration with the modified `simulate_processing` function, if your scheduler’s processing interval is shorter, like the default 300 seconds, you could easily have two runs seemingly active simultaneously. Because, despite `max_active_runs=1`, the scheduler periodically checks if any dag runs are queued or waiting to execute. If the previous run is taking longer than its process interval it can trigger another run, since the previous one has not yet been finalized to the scheduler.

Let’s add a second task to the same DAG for further illustration:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import time

def simulate_processing_1():
    time.sleep(30)
    print("First processing complete.")

def simulate_processing_2():
    time.sleep(10)
    print("Second processing complete.")

with DAG(
    dag_id="log_processing_dag_multi_task",
    schedule="@hourly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
) as dag:
    process_logs_1 = PythonOperator(
        task_id="process_logs_1",
        python_callable=simulate_processing_1,
    )

    process_logs_2 = PythonOperator(
      task_id="process_logs_2",
      python_callable=simulate_processing_2
    )
    process_logs_1 >> process_logs_2
```

Even with `max_active_runs=1`, it is still the dag *run* that is capped at one. The *tasks* within the dag instance will proceed according to dependencies and available resources. The limit applies to the number of dag instances the scheduler creates. This reinforces that tasks can still execute in parallel if resources permit, despite the configured `max_active_runs`. The constraint affects the creation of new dag runs, not the concurrency of tasks within them.

To better understand the behavior of the scheduler, and thus the impact of `max_active_runs`, I'd recommend consulting the official Apache Airflow documentation, specifically the sections dealing with the scheduler and dag run management. Additionally, a deep dive into the source code related to these components can offer more nuanced insights. For a strong foundation on distributed task processing, "Designing Data-Intensive Applications" by Martin Kleppmann provides invaluable background knowledge. Another useful resource is "Data Pipelines Pocket Reference" by James Densmore which explains key concepts clearly.

Finally, let me summarize. `max_active_runs` is not designed to restrict the overall number of concurrent tasks across all dag runs; it controls the number of active, scheduled dag *runs*. The scheduler does not enforce this parameter in real-time but at its periodic intervals. Tasks within a running dag instance can execute simultaneously based on task dependencies and resource availability. Misunderstanding this distinction leads to the common misconception that `max_active_runs` is being ignored. It's doing what it was designed to do—limiting scheduled dag runs, not necessarily concurrent task executions. Understanding these specifics is key for effective workflow orchestration with Airflow.
