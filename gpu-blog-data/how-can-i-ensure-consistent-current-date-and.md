---
title: "How can I ensure consistent current date and time across Airflow loop iterations?"
date: "2025-01-30"
id: "how-can-i-ensure-consistent-current-date-and"
---
A common challenge in Airflow DAGs involving loops, whether dynamic task mapping or traditional Python looping, stems from the fact that execution context variables like `ds`, `ts`, and `execution_date` are frozen at the start of the DAG run. This can lead to all loop iterations unintentionally operating on the same initial timestamp rather than the logical execution time of each individual task instance within the loop.

The core issue is that Jinja templating within Airflow, which provides these context variables, evaluates them *before* the individual tasks are generated for each loop iteration. This behavior means that any attempt to directly access, for example, `{{ ds }}` or `{{ ts }}` inside a looped task will result in identical values across all tasks generated within that iteration, not the per-task runtime. I've encountered this frequently in workflows where I need to partition or process data by individual days within a multi-day range.

To achieve a dynamically updated current date and time for each iteration, the solution requires bypassing Airflow’s templating system and explicitly generating timestamps within the execution context of each task instance. This involves using Python’s date and time libraries and leveraging Airflow’s built-in constructs for task execution. The strategy relies on defining a function to compute timestamps based on each loop index or a related per-iteration logic, and then using this computed information within the task. This requires a shift from using Airflow's templated values to a programmatic approach within the task definition itself.

Here are three concrete code examples illustrating this principle, each progressively more complex, tailored to common situations I've faced:

**Example 1: Basic Sequential Iteration with Per-Task Date**

In this scenario, we want to execute a task a fixed number of times and have each task operate on a day sequentially following the DAG start date.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pendulum

def process_date(execution_date, iteration):
    target_date = execution_date + timedelta(days=iteration)
    print(f"Processing date: {target_date.strftime('%Y-%m-%d')}")
    # Perform actions with the target_date here

with DAG(
    dag_id='sequential_date_loop',
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
) as dag:
    for i in range(5):
        PythonOperator(
            task_id=f'process_task_{i}',
            python_callable=process_date,
            op_kwargs={'iteration': i},
        )
```

*   **Commentary:** This example demonstrates the fundamental approach. The core of the solution lies within the `process_date` function. Here, instead of relying on Jinja templates, `execution_date` is directly accessed as a Python datetime object. `timedelta` is used to incrementally add the loop index to the execution date, effectively creating sequential dates for each task. The key aspect is how the loop index `i`, derived from the external Python loop, parameterizes the individual tasks and generates distinct logical dates. The `print` statement shows the computed date, but you would typically integrate your logic with this generated `target_date`.

**Example 2: Dynamic Task Mapping with Different Start Dates**

Here, we use dynamic task mapping to process a set of varying date ranges. Each mapped task operates on a different start date.

```python
from airflow import DAG
from airflow.decorators import task
from datetime import datetime, timedelta
import pendulum

date_ranges = [
    {'start_date': pendulum.datetime(2023, 3, 1, tz="UTC"), 'days': 3},
    {'start_date': pendulum.datetime(2023, 3, 5, tz="UTC"), 'days': 2},
    {'start_date': pendulum.datetime(2023, 3, 10, tz="UTC"), 'days': 4}
]

@task
def process_dynamic_date(start_date, days, execution_date, task_id):
    start_date_val = datetime.combine(start_date.date(), datetime.min.time())
    for day in range(days):
       target_date = start_date_val + timedelta(days=day)
       print(f"Task {task_id}: Processing date: {target_date.strftime('%Y-%m-%d')}")

with DAG(
    dag_id='dynamic_date_loop',
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
) as dag:
    process_tasks = process_dynamic_date.expand(
        start_date=[item['start_date'] for item in date_ranges],
        days=[item['days'] for item in date_ranges]
    )
```

*   **Commentary:** In this instance, Airflow’s dynamic task mapping is leveraged. The `expand` function allows each `process_dynamic_date` task to take different start dates and numbers of days from the `date_ranges` list. The `process_dynamic_date` function now accepts `start_date` and `days` as function parameters, rather than the iteration counter, allowing for complete control over date ranges within the mapped tasks. It constructs the `target_date` for each day within the loop using the respective `start_date`, ensuring that each task operates on the dates associated with its configured parameters. It also illustrates passing of `task_id` as a parameter to allow for debugging. Notice that a `datetime` is formed using date part of `start_date` and time component of the minimum time which defaults to midnight.

**Example 3: Handling Task Failures and Incremental Processing**

This example demonstrates how to keep track of processing and handle failure scenarios in a date-iterated manner.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.state import State
from datetime import datetime, timedelta
import pendulum
import logging

logger = logging.getLogger(__name__)

def process_date_with_status(execution_date, iteration, ti):
    target_date = execution_date + timedelta(days=iteration)
    formatted_date = target_date.strftime('%Y-%m-%d')
    try:
        # Simulate some processing that could fail
        if iteration % 2 !=0: # Simulating a failure on every other iteration
            raise ValueError(f"Simulated error for date {formatted_date}")
        logger.info(f"Successfully processed {formatted_date}")
        ti.xcom_push(key=f'processed_date_{iteration}', value=formatted_date)
    except ValueError as e:
        logger.error(f"Failed to process {formatted_date}: {e}")
        ti.xcom_push(key=f'failed_date_{iteration}', value=formatted_date)
        raise # Re-raise the error for task failure

    return formatted_date # Return formatted_date so it shows in xcom

def check_processed_dates(**context):
    processed_dates = []
    failed_dates = []
    for i in range(5):
        processed_date = context['ti'].xcom_pull(task_ids=f'process_task_{i}', key=f'processed_date_{i}')
        failed_date = context['ti'].xcom_pull(task_ids=f'process_task_{i}', key=f'failed_date_{i}')
        if processed_date:
            processed_dates.append(processed_date)
        if failed_date:
            failed_dates.append(failed_date)

    logger.info(f"Processed dates: {processed_dates}")
    logger.info(f"Failed dates: {failed_dates}")

with DAG(
    dag_id='failure_handling_date_loop',
    start_date=pendulum.datetime(2023, 1, 1, tz="UTC"),
    schedule=None,
    catchup=False,
) as dag:
    for i in range(5):
       PythonOperator(
            task_id=f'process_task_{i}',
            python_callable=process_date_with_status,
            op_kwargs={'iteration': i},
            provide_context = True,
            retries = 1
        )

    check_results = PythonOperator(
        task_id='check_processed_dates',
        python_callable=check_processed_dates,
        provide_context = True
    )
```

*   **Commentary:** This example incorporates error handling and the use of XComs for tracking processing status. Within the `process_date_with_status` function, a try-except block is used to simulate failures. On error, the function uses XCom to log failed dates, while successful processing records the processed dates. These records are then pulled in the subsequent task using `XCom_pull`. The `check_processed_dates` function illustrates how to aggregate status across all loop iterations. Note the use of `provide_context` is needed to enable pushing and pulling `xcom` values. The returned `formatted_date` is pushed automatically via xcom mechanism. Additionally, note that `retries` are enabled within the task definition. This illustrates how task failure states during each iteration can be individually handled.

**Resource Recommendations**

For further exploration and deeper understanding, I recommend consulting the following resources. First, review the official Apache Airflow documentation thoroughly, especially the sections on templating, task execution, and dynamic task mapping. Second, familiarize yourself with the Python standard library's `datetime` module and how to manage date and time objects effectively. Third, understand Airflow's concepts of XComs and how they facilitate communication between tasks, especially when tracking state across looped operations. Finally, study examples of dynamic task generation and explore different parameterization patterns for creating efficient and robust pipelines when handling time series data. While these are not code examples, the core concepts of date and time manipulation, task execution contexts and state management are crucial for robust and accurate Airflow DAGs when dealing with loops.
