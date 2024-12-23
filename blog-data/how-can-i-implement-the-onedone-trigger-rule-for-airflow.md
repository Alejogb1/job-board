---
title: "How can I implement the `one_done` trigger rule for Airflow?"
date: "2024-12-23"
id: "how-can-i-implement-the-onedone-trigger-rule-for-airflow"
---

Alright, let's talk about `one_done` in Airflow. I remember a particular project involving a complex ETL pipeline, where we had a bunch of downstream tasks that absolutely *had* to wait for at least one of several upstream tasks to succeed before they could even begin. We explored several options, and that's when I really got to grips with the nuances of `one_done`. Forget about the more common `all_done` approach; sometimes, you just need that initial signal of forward progress.

The `one_done` trigger rule isn't explicitly a built-in rule that you can simply use as a parameter like `trigger_rule='all_done'`. Instead, you’ll be leveraging other features of Airflow, primarily involving the `TriggerRule.NONE_FAILED` rule and task dependencies that are carefully constructed. The key is to recognize that this is a *logical* pattern implemented via a combination of Airflow features rather than a direct feature of the library. It takes a bit of set up, but it’s a powerful tool to keep your pipelines lean and responsive.

The basic idea is to have a "trigger" task, which is itself dependent on all of your upstream tasks. But crucially, this task is configured with `trigger_rule='none_failed'`. This means it will run as long as *none* of its upstream tasks have failed. And yes, even if some of the upstream tasks are skipped, the trigger task will fire, as they haven't "failed." Downstream of that trigger task, your actual dependent tasks can be initiated. They will run only when the trigger task completes.

Here's a basic illustrative example using Airflow's Python operator, and keep in mind, this structure translates similarly to other operators:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime


def task_success():
    print("Task executed.")


with DAG(
    dag_id="one_done_example_basic",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    task_a = PythonOperator(task_id="task_a", python_callable=task_success)
    task_b = PythonOperator(task_id="task_b", python_callable=task_success)
    task_c = PythonOperator(task_id="task_c", python_callable=task_success)

    trigger_task = PythonOperator(
        task_id="trigger_task",
        python_callable=task_success,
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    task_d = PythonOperator(task_id="task_d", python_callable=task_success)

    [task_a, task_b, task_c] >> trigger_task >> task_d
```

In this snippet, `task_d` will run only once `trigger_task` completes, and `trigger_task` will complete once *any* of task_a, task_b, or task_c completes. Essentially, we are constructing the "one_done" behavior through logical dependencies and `TriggerRule.NONE_FAILED`. If, for example, `task_a` succeeds, the `trigger_task` will run regardless of the state of `task_b` or `task_c`, and then `task_d` will execute.

Let's consider a slightly more complex case that reflects a common real-world issue: handling different data sources. Let's say you're pulling data from three different APIs, and you don’t necessarily need *all* three to succeed to process the available data. Here's the updated code:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
import random


def task_api_call():
    if random.random() < 0.7: #Simulating occasional API failure.
       print("API call successful.")
    else:
      raise Exception("API call failed.")



def task_process_data():
  print("Processing data...")

with DAG(
    dag_id="one_done_example_api",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    api_call_1 = PythonOperator(task_id="api_call_1", python_callable=task_api_call)
    api_call_2 = PythonOperator(task_id="api_call_2", python_callable=task_api_call)
    api_call_3 = PythonOperator(task_id="api_call_3", python_callable=task_api_call)

    trigger_task = PythonOperator(
        task_id="trigger_task",
        python_callable=task_process_data,
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    process_data = PythonOperator(
        task_id="process_data",
        python_callable=task_process_data
    )


    [api_call_1, api_call_2, api_call_3] >> trigger_task >> process_data
```

In this example, the `trigger_task` (which runs `task_process_data`) will execute as long as none of the API calls fail. The downstream `process_data` will then kick off to process any available data. Even if only one of the three APIs was successful, `process_data` will still run, achieving the `one_done` logic. The slight change is to simulate failures, since we will get the `one_done` effect even if one succeeds. We also had `task_process_data` be a part of both `trigger_task` and `process_data`.

Lastly, consider a more advanced example where you need to execute a cleanup task only if something *does* go wrong in the upstream tasks and no data processing will happen:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
import random

def task_api_call():
    if random.random() < 0.8: #Simulating occasional API failure.
       print("API call successful.")
    else:
      raise Exception("API call failed.")

def task_cleanup():
  print("Running cleanup procedure...")

def task_process_data():
  print("Processing data...")

with DAG(
    dag_id="one_done_example_cleanup",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    api_call_1 = PythonOperator(task_id="api_call_1", python_callable=task_api_call)
    api_call_2 = PythonOperator(task_id="api_call_2", python_callable=task_api_call)
    api_call_3 = PythonOperator(task_id="api_call_3", python_callable=task_api_call)


    trigger_task = PythonOperator(
        task_id="trigger_task",
        python_callable=task_process_data,
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    process_data = PythonOperator(
        task_id="process_data",
        python_callable=task_process_data
    )

    cleanup_task = PythonOperator(
      task_id="cleanup_task",
      python_callable=task_cleanup,
      trigger_rule=TriggerRule.ALL_FAILED,
    )


    [api_call_1, api_call_2, api_call_3] >> trigger_task >> process_data
    [api_call_1, api_call_2, api_call_3] >> cleanup_task
```

In this snippet, the cleanup task is triggered if *all* upstream tasks (`api_call_1`, `api_call_2`, and `api_call_3`) fail. Otherwise the `trigger_task` will run as before with `NONE_FAILED`. The cleanup task will be triggered after the `trigger_task` has run, and the `trigger_task` runs if at least one of the API tasks complete.

For a deeper dive into task dependencies and trigger rules, the official Apache Airflow documentation, specifically the sections on "DAG Definition" and "Operators," is a great resource. You should also explore the concept of XComs if you plan to have data or metadata pass between these tasks. Furthermore, “Data Pipelines Pocket Reference” by James Densmore is an excellent resource that delves into concepts of dependency management and workflow orchestration using Airflow in a concise manner. Also consider checking out “Programming Apache Airflow” by Jarek Potiuk and Bartlomiej Balewski for detailed information on various aspects of Airflow, including a good explanation of trigger rules.

In summary, implementing the `one_done` trigger pattern in Airflow involves a little more than just setting a parameter; it requires an understanding of dependencies and trigger rules. By using `TriggerRule.NONE_FAILED` on a carefully constructed "trigger" task, you can achieve the desired behavior, ensuring your pipelines are resilient and efficient. This construct works very well in a production environment.
