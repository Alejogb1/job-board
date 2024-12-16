---
title: "How do I implement Airflow's one_done trigger rule?"
date: "2024-12-16"
id: "how-do-i-implement-airflows-onedone-trigger-rule"
---

Alright, let's talk about Airflow's `one_done` trigger rule. It's one of those features that seems straightforward on the surface but can be surprisingly nuanced when you’re orchestrating complex workflows. I’ve encountered this quite a bit over the years, particularly when dealing with scenarios where a DAG’s downstream tasks shouldn’t proceed until at least one of their upstream tasks has successfully finished, but not necessarily *all* of them.

For context, recall the standard Airflow behavior. Without specifying a `trigger_rule`, the default is `all_success`. This means a task will only proceed if *all* of its upstream tasks have completed successfully. This works well for most linear workflows but falls short when we need more flexible dependency management. That’s where `one_done` comes in; it’s the antithesis of that 'all-or-nothing' approach. I've used it to handle situations involving redundant data pipelines where only one source needs to successfully feed a downstream processing task. Specifically, imagine we're pulling data from multiple APIs, each with a slightly different response structure but ultimately providing the same data. Instead of failing the entire pipeline when one API is down, we use `one_done` and continue processing the data from whatever sources have succeeded.

Now, getting into the nitty-gritty, the `one_done` trigger rule requires a bit of careful consideration regarding task design. It essentially implies that your downstream task must be able to handle various data formats and potential states stemming from one successful upstream task versus another. Think of it as handling several possible outputs from different upstream processes, each leading to the desired end state. The challenge here lies not so much in implementation itself, but rather in ensuring your downstream logic is robust and can gracefully accommodate the varying outcomes.

Let’s illustrate this with some concrete examples. I’ve tried to keep it succinct, focusing on the pertinent parts:

**Example 1: Simple API Data Collection**

Here's a basic scenario where we are pulling data from three different APIs, and the downstream `process_data` task executes once *any* of those APIs have responded successfully. Note that we aren't actually calling external APIs in this example; these are just placeholders.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime

def fetch_api1():
    return "Data from API 1"

def fetch_api2():
    return "Data from API 2"

def fetch_api3():
    # Simulate a failure
    raise Exception("API 3 is down.")


def process_data(**kwargs):
    ti = kwargs['ti']
    api1_data = ti.xcom_pull(task_ids='fetch_api1', dag_id=kwargs['dag_run'].dag_id)
    api2_data = ti.xcom_pull(task_ids='fetch_api2', dag_id=kwargs['dag_run'].dag_id)
    api3_data = ti.xcom_pull(task_ids='fetch_api3', dag_id=kwargs['dag_run'].dag_id, default=None)

    # Logic to handle potentially missing/failed data
    collected_data = [data for data in [api1_data, api2_data, api3_data] if data]
    print(f"Processing data from successful API pulls: {collected_data}")

with DAG(
    dag_id='one_done_example_1',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    fetch_api1_task = PythonOperator(
        task_id='fetch_api1',
        python_callable=fetch_api1
    )
    fetch_api2_task = PythonOperator(
        task_id='fetch_api2',
        python_callable=fetch_api2
    )
    fetch_api3_task = PythonOperator(
        task_id='fetch_api3',
        python_callable=fetch_api3
    )

    process_data_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data,
        trigger_rule=TriggerRule.ONE_DONE
    )

    [fetch_api1_task, fetch_api2_task, fetch_api3_task] >> process_data_task

```

In this example, even though `fetch_api3_task` fails, `process_data_task` still runs because at least one upstream task, either `fetch_api1_task` or `fetch_api2_task`, succeeded. The `process_data` task then intelligently handles that only a subset of the potential source data is available. Note the usage of `xcom_pull` with a `default` to ensure we don't crash when pulling the data from a failed task.

**Example 2: File Availability from Multiple Servers**

Let's consider a scenario where we receive files from multiple servers. Only one file needs to be present for downstream processing. This is another place `one_done` shines.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
import os

def check_file_server1():
    # Dummy file creation for demo, in real case check for file presence on remote server.
    if not os.path.exists("server1_file.txt"):
        with open("server1_file.txt", 'w') as f:
            f.write("Data from server 1.")
    return True

def check_file_server2():
    # Simulating that server 2 is unavailable.
    raise Exception("Server 2 is down.")

def check_file_server3():
    # Dummy file creation for demo.
    if not os.path.exists("server3_file.txt"):
        with open("server3_file.txt", 'w') as f:
            f.write("Data from server 3.")
    return True


def process_files(**kwargs):
    ti = kwargs['ti']
    server1_status = ti.xcom_pull(task_ids='check_file_server1', dag_id=kwargs['dag_run'].dag_id)
    server2_status = ti.xcom_pull(task_ids='check_file_server2', dag_id=kwargs['dag_run'].dag_id, default=None)
    server3_status = ti.xcom_pull(task_ids='check_file_server3', dag_id=kwargs['dag_run'].dag_id)

    available_files = []

    if server1_status:
        available_files.append("server1_file.txt")

    if server3_status:
        available_files.append("server3_file.txt")
    
    print(f"Processing files from available servers: {available_files}")

    # Process the available files, etc.

with DAG(
    dag_id='one_done_example_2',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    check_file_server1_task = PythonOperator(
        task_id='check_file_server1',
        python_callable=check_file_server1
    )
    check_file_server2_task = PythonOperator(
        task_id='check_file_server2',
        python_callable=check_file_server2
    )
    check_file_server3_task = PythonOperator(
        task_id='check_file_server3',
        python_callable=check_file_server3
    )

    process_files_task = PythonOperator(
        task_id='process_files',
        python_callable=process_files,
        trigger_rule=TriggerRule.ONE_DONE
    )

    [check_file_server1_task, check_file_server2_task, check_file_server3_task] >> process_files_task

```

Here, even though server 2 isn’t available, the `process_files_task` will still proceed as long as either server 1 or 3 have provided their files successfully. The task checks for the presence of files and processes them.

**Example 3: Prioritized Task Execution (with Caution)**

Finally, here's an example of using `one_done` in a kind of prioritized execution. It's crucial to use this with *extreme* caution because it can lead to potentially unexpected behavior if your "priority" logic isn't carefully thought through.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime

def process_priority_data():
    print("Processed data with priority logic.")

def process_standard_data():
    print("Processed standard data.")

def fail_priority_data():
    raise Exception("Priority failed")

with DAG(
    dag_id='one_done_example_3',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    priority_task = PythonOperator(
        task_id='process_priority',
        python_callable=process_priority_data
    )
    priority_fail_task = PythonOperator(
        task_id='fail_priority',
        python_callable=fail_priority_data
    )
    standard_task = PythonOperator(
        task_id='process_standard',
        python_callable=process_standard_data,
    )

    downstream_task = PythonOperator(
       task_id='downstream_task',
       python_callable= lambda: print('downstream task ran'),
       trigger_rule=TriggerRule.ONE_DONE
    )

    [priority_task,priority_fail_task] >> downstream_task
    standard_task >> downstream_task

```

In this case, we may prefer the 'priority_task' to be attempted before the 'standard_task', with both feeding a downstream task. If 'priority_task' succeeds, then 'downstream_task' executes before the standard task. If 'priority_task' fails however, 'standard_task' may still execute and trigger 'downstream_task'. This setup *should* lead to less computation overall. But this introduces additional complexity to the execution order and should only be done when truly necessary.

For deepening your understanding, I would highly recommend studying the 'Airflow documentation' directly, as it's consistently the most accurate and authoritative source. Also, the book "Data Pipelines with Apache Airflow" by Bas P. Harenslak and Julian Rutger is a fantastic resource that dives into practical applications of Airflow. You might find “Designing Data-Intensive Applications” by Martin Kleppmann useful for the broader architectural considerations around managing dependencies and resilience in distributed systems.

In closing, `one_done` is a valuable tool when used judiciously. It adds flexibility to your DAGs but requires careful planning and a robust design of your tasks to handle the potentially varying outcomes of upstream tasks. Always prioritize readability and robustness over cleverness when it comes to orchestrating your workflows.
