---
title: "How do I trigger an Airflow DAG depending on dag_run.conf?"
date: "2024-12-23"
id: "how-do-i-trigger-an-airflow-dag-depending-on-dagrunconf"
---

Okay, let's delve into triggering airflow dags conditionally based on the `dag_run.conf`. This is a fairly common requirement, and I've certainly tackled it multiple times in my career. The beauty, and sometimes the challenge, lies in achieving this dynamically. It's not just about passing parameters; it's about making airflow understand those parameters *before* the dag actually starts running to decide *if* it should run.

The core principle here hinges on leveraging airflow's capabilities for external triggers. We typically don't directly modify the dag file for conditional execution at runtime, which is a bad practice. Instead, we set up our dag to evaluate conditions at the beginning, typically within the `start_date` context of the dag or within a very early operator, and gracefully exit if those conditions aren't met.

One of my earlier projects involved a data pipeline that ingested reports from various external sources. The pipeline was parameterized via `dag_run.conf` to specify the report source. We wanted to avoid running the pipeline if a report for a given source was not available or not ready for processing. We designed it to check this before any actual data processing began.

Here's how I’d generally approach this and what I’ve found to be most effective in different cases:

**Method 1: Using the `start_date` for Conditional Triggering**

This method primarily leverages airflow's dag initialization and, in my experience, is ideal for simple checks that don't require elaborate logic. The key is to make the dag start date dependent on whether the necessary conditions in `dag_run.conf` are satisfied. If not, the dag essentially will be considered 'not scheduled' for that run, so a no-op for that particular execution context. This is often helpful when you only want to run certain things weekly or monthly. We can use the `datetime` module and python logic to define this:

```python
from airflow import DAG
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator

def check_config_condition(**kwargs):
    conf = kwargs.get('dag_run').conf or {}
    target_report_type = conf.get('report_type')
    if target_report_type is None:
        return False  # Exit the DAG
    # Example: If 'report_type' is 'daily' we run it, otherwise we don't run it
    if target_report_type == 'daily':
        return True
    else:
        return False

with DAG(
    dag_id="conditional_start_date_dag",
    start_date=datetime.now() - timedelta(days=1) ,
    schedule=None,  # We don't want a schedule trigger for this example
    catchup=False,
    tags=['example'],
) as dag:
    
    check_start_condition = PythonOperator(
        task_id='check_start_condition',
        python_callable=check_config_condition
    )
    
    # dummy task to illustrate the DAG
    dummy_task = PythonOperator(
        task_id='dummy_task',
        python_callable=lambda : print("doing something useful!")
    )
    
    check_start_condition >> dummy_task
```

Here, the `check_config_condition` function reads the `report_type` from `dag_run.conf`. If the condition is not met, it returns `False`. A return of `False` results in skipping the rest of the DAG. If the condition is satisfied, we can proceed. Notice that I use `timedelta` to create an initial start date. This method is good for skipping entire dag runs, but might not be as flexible as needed if you need to have the DAG running for other purposes, too.

**Method 2: Early Exit Using a Conditional Branch**

This approach involves including a very early task in your dag that evaluates the `dag_run.conf`. Based on the evaluation, the dag either proceeds with its normal execution or performs a graceful exit. I find this to be more robust when you need to be more explicit about when the dag exits and might want to log why the dag did not execute. It doesn't modify the start date directly.

```python
from airflow import DAG
from airflow.utils.dates import days_ago
from datetime import datetime
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.utils.trigger_rule import TriggerRule

def check_config_condition(**kwargs):
    conf = kwargs.get('dag_run').conf or {}
    target_report_type = conf.get('report_type')
    if target_report_type is None:
        print(f"No report_type provided. Exiting DAG.")
        return 'exit_task'  # Direct the flow to exit task.
    if target_report_type == 'monthly':
        return 'data_processing_task'
    else:
        print(f"Report type is {target_report_type}. Exiting DAG.")
        return 'exit_task' # Direct the flow to exit task

with DAG(
    dag_id="conditional_branching_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=['example'],
) as dag:
    
    check_config = PythonOperator(
        task_id='check_config',
        python_callable=check_config_condition,
        do_xcom_push=True
    )
    
    exit_task = DummyOperator(
       task_id='exit_task',
       trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    data_processing_task = PythonOperator(
        task_id="data_processing_task",
        python_callable=lambda: print("Data processing starting")
    )
    
    check_config >> [exit_task, data_processing_task]
```

In this instance, the `check_config_condition` function uses XCom push to communicate which task to trigger, allowing for a conditional path. The logic directs the flow toward the data_processing_task if 'monthly' is specified, or toward the `exit_task` otherwise, using a dummy task that still signals an execution without doing any actual work. The trigger rule of `exit_task` ensures that we at least start and signal we ended, even if not all of the tasks completed. This is good for control over logging and signaling.

**Method 3: Utilizing a Sensor for External Conditions**

Finally, for more complex conditional checks that depend on external states, using a sensor operator is the most reliable choice. For instance, you might have a separate system that signals when data is ready via a specific file creation, a flag in a database, or a message in a queue. A sensor checks continuously and signals to start the pipeline once it receives a signal.

```python
from airflow import DAG
from airflow.utils.dates import days_ago
from datetime import datetime
from airflow.sensors.time_delta import TimeDeltaSensor
from airflow.operators.python import PythonOperator
from airflow.operators.dummy import DummyOperator

def file_ready_check(**kwargs):
    # Replace this with your custom check - Example
    conf = kwargs.get('dag_run').conf or {}
    target_file = conf.get('target_file')
    
    if target_file:
        return True # return true or logic of your choice
    else:
        return False
    
with DAG(
    dag_id="external_sensor_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=['example'],
) as dag:
    
    wait_for_file = TimeDeltaSensor(
        task_id='wait_for_file',
        delta=timedelta(minutes=5),
        mode="reschedule",
        poke_interval=30,
        timeout=timedelta(hours=1),
        python_callable=file_ready_check,
    )
    
    data_processing = PythonOperator(
        task_id="data_processing",
        python_callable=lambda: print("Data processing started")
    )

    wait_for_file >> data_processing
```

In this scenario, the `TimeDeltaSensor`, with a custom function `file_ready_check` handles the logic by executing the sensor every 30 seconds and timing out in an hour if the condition is not met. Once that function returns true, the DAG proceeds to the `data_processing` step. This is the most robust for external and complex dependencies.

**Key Considerations and Resources**

When working with conditional DAG triggers, it's essential to keep these points in mind:

*   **Error Handling:** Always handle cases where the `dag_run.conf` might be missing or invalid. Default values are essential.
*   **Logging:** Ensure you log why a dag is skipping execution. It simplifies debugging.
*   **Idempotency:** Design your DAGs so that if they run multiple times with the same `dag_run.conf`, they don't cause any unintended side effects.
*   **Monitoring:** Use airflow's monitoring tools to observe DAG runs and any issues with conditional execution.

For deeper insights, I recommend reviewing these resources:

*   **The official Apache Airflow documentation:** Specifically the sections on DAG runs, triggers, sensors, and XComs.
*   **"Data Pipelines with Apache Airflow" by Bas P. Harenslak and Julian Rutger de Ruiter:** Excellent for understanding airflow's concepts in depth.
*   **"Designing Data-Intensive Applications" by Martin Kleppmann:** While not airflow specific, provides a framework to reason about the complexities of data pipelines and distributed systems.
* **The source code for apache airflow:** It's really helpful to look at the internals of how these things work.

In closing, triggering airflow dags conditionally based on `dag_run.conf` is a powerful capability. By combining careful planning with the provided techniques, you can build robust, dynamic, and efficient data pipelines. The examples should provide you with a good starting point. If you have other questions or more specific scenarios, I’m always happy to help.
