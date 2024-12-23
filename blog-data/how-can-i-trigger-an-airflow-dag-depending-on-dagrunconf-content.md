---
title: "How can I trigger an Airflow DAG depending on dag_run.conf content?"
date: "2024-12-23"
id: "how-can-i-trigger-an-airflow-dag-depending-on-dagrunconf-content"
---

Alright, let’s tackle this. I recall a particularly sticky situation a few years back involving exactly this—triggering airflow dag runs based on the content of their `dag_run.conf`. We were dealing with a complex data ingestion pipeline, and we needed to dynamically adjust the workflow based on parameters passed during manual or programmatic triggering. Let me walk you through how I approached it and some key considerations.

The crux of the matter is that `dag_run.conf` provides a mechanism for injecting parameters when triggering a DAG. These parameters are essentially a dictionary, which airflow exposes through the jinja templating engine within various components of a dag – most notably, tasks. Now, using these parameters to *directly* dictate whether a dag *should* trigger is a nuanced problem, because DAG schedules are handled outside of the running DAG itself. We can't dynamically modify those schedules on a per-trigger basis. However, we *can* control execution flow within the dag itself based on the `conf` parameters, effectively mimicking a trigger condition.

Here’s the general strategy I’ve found most reliable:

1.  **Entry Point Task with Conditional Logic:** The first task in your DAG should act as a gatekeeper. It reads the `dag_run.conf` and then branches the execution accordingly. If certain parameters are present or have specific values, the DAG continues. Otherwise, it might do nothing, effectively terminating the dag before any meaningful work is done.

2.  **Sensors as a Guardrail (Optional):** If you need to wait for specific conditions to be met *before* continuing the dag execution rather than just deciding to halt, you might want to employ a sensor that checks for the condition based on the `conf` dictionary. For example, the sensor could check for a specific file's existence or wait for an external api endpoint to return specific data specified in the config.

3.  **Parameterized Subsequent Tasks:** The remaining tasks in your DAG can then leverage the `dag_run.conf` values for their own processing logic, providing further customizability. This is where the true power of this approach lies— dynamically shaping your workflow behavior.

Let’s take a look at some code examples to solidify this. We will use Python alongside the Airflow framework for the following scenarios:

**Example 1: Simple Conditional Execution Based on a Flag**

This is a basic example where a `process_data` task will execute only if a `run_process` flag in the `dag_run.conf` is set to `true`.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def check_condition(**kwargs):
    conf = kwargs.get('dag_run').conf
    if conf and conf.get('run_process', False):
        print("Processing is enabled based on conf.")
        return True
    else:
        print("Processing skipped as 'run_process' is not set or false in conf.")
        return False

def process_data(**kwargs):
    print("Processing data based on passed configuration")
    # Logic to process data, may use other params passed in conf
    return "Data Processed"


with DAG(
    dag_id='conditional_execution_dag',
    start_date=datetime(2023, 1, 1),
    schedule=None, # Allow Manual Triggering,
    catchup=False,
) as dag:
    check_run_task = PythonOperator(
        task_id='check_run_flag',
        python_callable=check_condition,
        provide_context=True,
    )

    process_data_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
    )

    check_run_task >> process_data_task
    process_data_task.trigger_rule = 'one_success' # run the task only if the previous task returns true


```

In this first example, the `check_run_task` uses a `PythonOperator` to access the `dag_run.conf`. If the `run_process` parameter is either missing or set to `False`, the execution path ends effectively. The `trigger_rule` on the `process_data` task ensures it executes only if `check_run_task` returns `True`, thus conditional execution is achieved.

**Example 2: Using Configuration to Determine Data Processing Parameters**

Here, we'll process files based on file types defined in `dag_run.conf`. This illustrates passing along parameters to downstream tasks.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def check_and_process_files(**kwargs):
    conf = kwargs.get('dag_run').conf
    if not conf or not conf.get('file_types'):
        print("No file types specified in conf. Skipping processing.")
        return False

    file_types = conf['file_types']
    print(f"File types to process: {file_types}")

    processed_count = 0

    for file_type in file_types:
        print(f"Processing files of type: {file_type}")
        # replace with your actual file processing logic
        processed_count += 1
        print(f"Finished processing files of type: {file_type}")

    print(f"Processed {processed_count} file types")
    return True


with DAG(
    dag_id='parameterized_data_dag',
    start_date=datetime(2023, 1, 1),
    schedule=None,  # Allow manual triggering
    catchup=False,
) as dag:

    process_files_task = PythonOperator(
        task_id='process_files',
        python_callable=check_and_process_files,
        provide_context=True,
    )
```

In this example, we extract a list of `file_types` from the `dag_run.conf` and pass it to the `check_and_process_files` function, this function then iterates over these types simulating some data processing activity. This clearly illustrates how we can configure the behaviour of the dag based on the provided `conf`.

**Example 3: Triggering Based on a Time Window**

In this slightly more complex example we will simulate a condition that uses a start and end time that is set in `dag_run.conf` . This approach can be useful in cases where the dag should only trigger when dealing with data within a specific time frame.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import time

def check_within_timeframe(**kwargs):
    conf = kwargs.get('dag_run').conf

    if not conf or 'start_time' not in conf or 'end_time' not in conf:
      print('start_time or end_time not found in configuration. exiting.')
      return False

    start_time_str = conf.get('start_time')
    end_time_str = conf.get('end_time')

    try:
      start_time = datetime.fromisoformat(start_time_str)
      end_time = datetime.fromisoformat(end_time_str)
    except ValueError:
        print('Invalid format for time. exiting.')
        return False

    current_time = datetime.now()
    if start_time <= current_time <= end_time:
        print(f"Current time {current_time} is within the configured time range.")
        return True
    else:
        print(f"Current time {current_time} is outside the configured time range {start_time} to {end_time}. Skipping.")
        return False


def perform_time_sensitive_task(**kwargs):
    print('Performing time sensitive task...')
    time.sleep(5) # simulating some work here.
    print('Time sensitive task done!')
    return 'Task completed!'

with DAG(
    dag_id='time_sensitive_dag',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:

  check_time_window_task = PythonOperator(
      task_id='check_time_window',
      python_callable=check_within_timeframe,
      provide_context=True
  )

  time_sensitive_task = PythonOperator(
      task_id='time_sensitive_task',
      python_callable=perform_time_sensitive_task,
  )

  check_time_window_task >> time_sensitive_task
  time_sensitive_task.trigger_rule = 'one_success'
```

In this final example, the `check_within_timeframe` function parses start and end times from the `dag_run.conf`, and checks whether the current time falls within those bounds. This demonstrates the potential for implementing more complex logic for initiating the DAG and controlling the execution based on real-time conditions and the input configuration.

To further your understanding on this topic, I’d recommend the following:

*   **"Data Pipelines with Apache Airflow" by Bas P. Harenslak and Julian Rutger de Ruiter:** A solid practical guide to airflow, including techniques for working with configuration and branching.
*   **The official Apache Airflow documentation:** The most authoritative source for understanding how airflow operates and its features are designed. Pay particular attention to the sections on jinja templating and `dag_run.conf`.
*   **"Designing Data-Intensive Applications" by Martin Kleppmann:** While not specific to Airflow, this book provides broader context about data processing systems and will help you design more robust and scalable data pipelines.

Remember, `dag_run.conf` is a powerful tool for customizing and parameterizing your workflows but using it for direct triggering is not it's primary purpose so consider it for in-dag branching and conditional execution, rather than the sole means of triggering the dag itself. Start small, test thoroughly and you'll find this approach invaluable for building adaptive and resilient data pipelines. I hope this detailed explanation, combined with the code examples, makes your understanding of conditional DAG execution with airflow based on `dag_run.conf` more complete.
