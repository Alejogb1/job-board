---
title: "How can I set custom trigger rules in Airflow?"
date: "2024-12-16"
id: "how-can-i-set-custom-trigger-rules-in-airflow"
---

Alright, let's tackle this. Custom trigger rules in airflow, it's a topic I’ve spent a fair amount of time with, especially back during my stint at 'DataStreams Inc'. We needed precise control over dag executions to avoid cascading failures and resource bottlenecks. What you're essentially aiming for is to move beyond the standard 'all_success' or 'all_failed' conditions. Thankfully, airflow offers several ways to achieve this granular level of control. It's not just about boolean operators anymore; you can implement quite sophisticated logic.

First off, it’s crucial to understand the underlying mechanics. Airflow, at its core, uses the `trigger_rule` parameter within its task definition. This parameter dictates when a task should move from a 'scheduled' state to 'running'. The standard options, like `all_success`, `all_failed`, `all_done`, `one_success`, `one_failed`, and `none_failed`, are fine for straightforward pipelines, but they quickly become insufficient for intricate workflow needs. You will see these in most introductory documentation and tutorials, but that’s rarely enough for complex production setups.

Now, let's talk about making things more interesting. Beyond these defaults, you can leverage python code within your dag definition to craft truly custom trigger rules. Specifically, you'll be using the `TriggerRule` enum in combination with the `depends_on_past` property, the `wait_for_downstream` parameter and other control flow mechanics within the dag definition. The key to unlocking true customization is using the `ShortCircuitOperator` in conjunction with conditional python statements. Let me lay out a few practical scenarios we had to solve, and how we approached them.

**Scenario 1: Conditional Task Execution Based on Upstream Task Outcomes**

Imagine a data pipeline where an initial task fetches data from an api. Success here is important. However, depending on the response code, different downstream processing needs to occur. We don't want to move to any subsequent processing if the initial data fetch fails, and even further, we might want to trigger specific cleaning processes based on response codes. Here's how we did it:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime

def fetch_data():
    # Simulate fetching data from API
    import random
    response_code = random.choice([200, 400, 500])
    print(f"Response code: {response_code}")
    return response_code

def process_data_success():
    print("Processing data - Success")


def process_data_failure_400():
    print("Processing data - Failure 400")


def process_data_failure_500():
    print("Processing data - Failure 500")

def check_response_code(**kwargs):
    ti = kwargs['ti']
    response_code = ti.xcom_pull(task_ids='fetch_api_data')
    if response_code == 200:
       return True
    return False

def check_response_code_400(**kwargs):
    ti = kwargs['ti']
    response_code = ti.xcom_pull(task_ids='fetch_api_data')
    if response_code == 400:
       return True
    return False

def check_response_code_500(**kwargs):
    ti = kwargs['ti']
    response_code = ti.xcom_pull(task_ids='fetch_api_data')
    if response_code == 500:
       return True
    return False


with DAG(
    dag_id="conditional_data_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    fetch_api_data = PythonOperator(
        task_id="fetch_api_data",
        python_callable=fetch_data,
        do_xcom_push=True,
    )

    check_response_ok = ShortCircuitOperator(
        task_id='check_response_ok',
        python_callable=check_response_code,
    )
    check_response_400 = ShortCircuitOperator(
        task_id='check_response_400',
        python_callable=check_response_code_400,
    )
    check_response_500 = ShortCircuitOperator(
        task_id='check_response_500',
        python_callable=check_response_code_500,
    )


    process_success = PythonOperator(
        task_id="process_success",
        python_callable=process_data_success,
       trigger_rule = TriggerRule.ONE_SUCCESS,

    )


    process_failure_400 = PythonOperator(
        task_id="process_failure_400",
        python_callable=process_data_failure_400,
        trigger_rule = TriggerRule.ONE_SUCCESS,
    )

    process_failure_500 = PythonOperator(
        task_id="process_failure_500",
        python_callable=process_data_failure_500,
        trigger_rule = TriggerRule.ONE_SUCCESS,
    )

    fetch_api_data >> [check_response_ok, check_response_400, check_response_500]
    check_response_ok >> process_success
    check_response_400 >> process_failure_400
    check_response_500 >> process_failure_500

```

Here, we’re using a `ShortCircuitOperator` to evaluate the response from `fetch_api_data`. It uses `xcom_pull` to access the previous task's returned value. Depending on the outcome, the short-circuiting will only allow one path to continue to the appropriate processing method by evaluating the result of the prior function.

**Scenario 2: Implementing a Timeout Mechanism Based on Downstream Tasks**

Another challenge we faced involved running data validation checks only if a specific downstream process completed within a specified time. Sometimes these processes would stall, and we needed a way to gracefully handle these situations. While airflow provides timeouts for individual operators it is harder to trigger a failure condition based on the *duration* of several tasks. The key again is the `ShortCircuitOperator`, which when used with downstream dependency allows for flexible logic:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import time

def simulate_long_task():
    # Simulate a task that might take long time.
    time.sleep(60)
    print("Long task completed")

def validate_data():
    print("Data validated.")

def check_downstream_time(**kwargs):
    ti = kwargs['ti']
    start_time = ti.task_instance.start_date
    now = datetime.now()
    delta = now-start_time
    if delta.total_seconds() < 65:
        return True
    return False


with DAG(
    dag_id="timeout_data_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    long_running_task = PythonOperator(
        task_id="long_running_task",
        python_callable=simulate_long_task,
    )


    validate_data_task = PythonOperator(
        task_id="validate_data_task",
        python_callable=validate_data,
        trigger_rule = TriggerRule.ONE_SUCCESS
    )
    check_time = ShortCircuitOperator(
        task_id='check_downstream_time',
        python_callable=check_downstream_time,
    )

    long_running_task >> check_time
    check_time >> validate_data_task
```

In this setup, `simulate_long_task` represents a task that might stall. We have the check_time task, which measures the time elapsed since the start of the `long_running_task`. It returns True if less than the threshold, ensuring that the validation only occurs if there’s an acceptable execution time.

**Scenario 3: Combining Multiple Upstream Statuses**

Sometimes, we needed complex scenarios like only executing a task if *at least* two of three upstream tasks succeeded, but not all of them. This becomes important when dealing with situations like data aggregation from different sources. Some sources might occasionally fail, and a complete failure is undesirable. The `depends_on_past`, `wait_for_downstream`, along with the `ShortCircuitOperator` provides a robust way to accomplish this:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
import random


def simulate_data_source():
    # Simulate data retrieval from different sources
    import random
    outcome = random.choice([True, False])
    print(f"Data source outcome: {outcome}")
    return outcome

def aggregate_data():
    print("Data aggregated")

def check_upstream_statuses(**kwargs):
    ti = kwargs['ti']
    task1_success = ti.xcom_pull(task_ids='data_source_1')
    task2_success = ti.xcom_pull(task_ids='data_source_2')
    task3_success = ti.xcom_pull(task_ids='data_source_3')

    successful_tasks = 0
    if task1_success:
        successful_tasks += 1
    if task2_success:
        successful_tasks += 1
    if task3_success:
        successful_tasks += 1

    return successful_tasks >= 2 and successful_tasks < 3
with DAG(
    dag_id="multiple_upstream_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    data_source_1 = PythonOperator(
        task_id="data_source_1",
        python_callable=simulate_data_source,
         do_xcom_push=True,
    )

    data_source_2 = PythonOperator(
        task_id="data_source_2",
        python_callable=simulate_data_source,
         do_xcom_push=True,
    )

    data_source_3 = PythonOperator(
        task_id="data_source_3",
        python_callable=simulate_data_source,
         do_xcom_push=True,
    )


    aggregate_data_task = PythonOperator(
        task_id="aggregate_data_task",
        python_callable=aggregate_data,
        trigger_rule = TriggerRule.ONE_SUCCESS
    )

    check_statuses = ShortCircuitOperator(
        task_id='check_upstream_statuses',
        python_callable=check_upstream_statuses,
    )

    [data_source_1, data_source_2, data_source_3] >> check_statuses
    check_statuses >> aggregate_data_task
```

Here, we retrieve the boolean outcomes from the `data_source_` tasks using `xcom_pull`. The logic in `check_upstream_statuses` counts the successful upstream tasks. Then, only if *at least two* of the three tasks succeeded will the execution continue.

**Recommendations**

If you're delving deep into custom trigger rules, I highly recommend these resources:

1.  **The Apache Airflow Documentation:** This should be your primary reference. Pay close attention to the section on task relationships and trigger rules. You'll find the foundational understanding needed for any customization.

2.  **"Data Pipelines with Apache Airflow" by Bas P. Harenslak and Julian Rutger de Ruiter:** This book provides practical insights and use cases. It doesn’t focus solely on trigger rules but covers various aspects of pipeline design that directly impact them.

3.  **"Programming Apache Airflow" by Jarek Potiuk, Bartłomiej Baczmański, and Marcin Zięba:** This book is more in-depth and provides a much more advanced technical analysis on topics such as XComs, and task interactions. The sections dealing with task states and dependencies are very helpful.

4.  **Source Code Exploration of the `airflow.utils.trigger_rule` module:** Understanding how these enums work at a lower level allows you to innovate beyond prescribed uses.

To summarize, creating truly bespoke trigger rules requires a solid understanding of airflow’s task lifecycle, XComs, and the `ShortCircuitOperator`. The `trigger_rule` parameter is not just a set of defaults but a powerful tool for nuanced workflow management. It's something that evolves with your pipeline needs, and by learning to implement your own logic, you achieve a whole new level of control and stability in your deployments.
