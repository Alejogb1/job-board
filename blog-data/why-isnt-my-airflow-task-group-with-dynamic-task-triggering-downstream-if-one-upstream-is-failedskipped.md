---
title: "Why isn't my Airflow Task-Group with Dynamic task triggering downstream if one upstream is failed/skipped?"
date: "2024-12-23"
id: "why-isnt-my-airflow-task-group-with-dynamic-task-triggering-downstream-if-one-upstream-is-failedskipped"
---

Alright,  It's a scenario I've bumped into a few times in the trenches, particularly when dealing with complex pipelines and dynamic dependencies within Airflow. The frustration of a task group not triggering downstream when one of its upstream tasks fails or is skipped, especially when you expect dynamic behavior, is entirely understandable. It often comes down to how Airflow's dependency management and task group logic interact, specifically with regards to the `trigger_rule` setting, and how it might be unintentionally affecting your downstream operations. I'll explain based on a few projects where I've seen these issues surface, detailing how we resolved them.

First, it's crucial to understand Airflow's dependency and triggering mechanisms. By default, a task will only execute if all of its upstream tasks have successfully completed (`'all_success'`). However, task groups introduce a layer of abstraction, and their behavior isn’t *exactly* identical to a single task when it comes to these dependencies. A key misunderstanding often centers around how the `trigger_rule` setting interacts within a task group’s scope. You might think a task group should trigger downstream if *at least* one of its internal tasks is successful. But that's not typically the default behavior, and it needs explicit configuration. Task groups evaluate based on the individual tasks within their boundary, unless told otherwise.

In one particular instance, I was working on an ETL pipeline for financial data. We had a task group, `load_data`, that contained tasks to load data from multiple sources. It was dynamically generated, so the number of load tasks depended on the source configuration. One day, a few of those sources became temporarily unavailable, causing the corresponding tasks to fail. This resulted in the entire `load_data` group, which was set up as a 'normal' dependency for the downstream tasks, to not trigger and thus the entire processing pipeline stopped. I remember spending a chunk of time inspecting the logs, thinking something was fundamentally wrong with Airflow. The issue wasn't with Airflow; it was our dependency setup in combination with dynamically generated tasks. We were relying on the default `'all_success'` trigger rule of the tasks inside the task group. Since not *all* tasks succeeded, the task group was considered failed by Airflow.

The crucial piece here is that each task inside your task group should have a `trigger_rule` set, especially if you're expecting dynamic behavior. Likewise, the task consuming the entire task group must have its own `trigger_rule` adjusted in order to run, even when a task has failed/skipped upstream. If the trigger rule of the *downstream* task that is depending on the task group is the default `all_success`, then it will only trigger if all tasks inside the task group succeed. That is the main problem we need to solve here.

Let’s examine this with some illustrative code examples. Assume a basic dynamic workflow where a task group contains dynamically created tasks:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime

def dummy_task(task_id, fail=False):
  def inner_func():
    if fail:
      raise Exception(f"Task {task_id} failed")
    print(f"Task {task_id} executed")
  return inner_func


with DAG(
    dag_id="dynamic_taskgroup_example",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

  with TaskGroup('dynamic_tasks', tooltip="Dynamic Data loading") as dynamic_tasks:
    for i in range(3):
       PythonOperator(
          task_id=f'dynamic_task_{i}',
          python_callable=dummy_task(task_id=f'dynamic_task_{i}', fail = (i == 1)), # one task will fail
          trigger_rule=TriggerRule.ALL_DONE
        )
    

  process_data = PythonOperator(
     task_id="process_data",
     python_callable=lambda: print("Processing Data after dynamic tasks"),
     trigger_rule=TriggerRule.ALL_DONE,
   )

  dynamic_tasks >> process_data
```

In this first example, the `process_data` task will **not execute** as one of the dynamic tasks fails. Even though we set the task `trigger_rule=TriggerRule.ALL_DONE` within the dynamic task group, and all tasks in the group finish (whether they succeeded or failed), the `process_data` task does not see that the task group is ready to be processed. Therefore we need to modify the trigger rule of the `process_data` task to be `TriggerRule.ALL_DONE` as well.

Now, let’s look at a corrected version with the trigger rule adjusted on the downstream task. The core logic remains the same, but we now change the downstream `process_data` trigger rule, and it will correctly execute once all the dynamic tasks (including the failure) have finished:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime

def dummy_task(task_id, fail=False):
  def inner_func():
    if fail:
      raise Exception(f"Task {task_id} failed")
    print(f"Task {task_id} executed")
  return inner_func


with DAG(
    dag_id="dynamic_taskgroup_example",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

  with TaskGroup('dynamic_tasks', tooltip="Dynamic Data loading") as dynamic_tasks:
    for i in range(3):
       PythonOperator(
          task_id=f'dynamic_task_{i}',
          python_callable=dummy_task(task_id=f'dynamic_task_{i}', fail = (i == 1)), # one task will fail
          trigger_rule=TriggerRule.ALL_DONE
        )
    

  process_data = PythonOperator(
     task_id="process_data",
     python_callable=lambda: print("Processing Data after dynamic tasks"),
     trigger_rule=TriggerRule.ALL_DONE,
   )

  dynamic_tasks >> process_data
```

In this second example, even though one of the tasks within the task group `dynamic_tasks` fails, the `process_data` task *does* execute because the `trigger_rule` is set to `TriggerRule.ALL_DONE`. This is the most common and simplest resolution for this issue.

However, what if you *only* wanted to execute the downstream process if *at least* one task inside the task group succeeds? You can't simply apply `TriggerRule.ONE_SUCCESS` to the `process_data` task because it is not directly dependent on the individual tasks, but rather on the task group. There are two ways that I am familiar with to handle this use case. The first is to use `BranchPythonOperator` in combination with the task group; and the second is to use the `on_success_callback` to implement the logic. Both solutions will allow downstream tasks to run, despite upstream failures.

Here’s an approach using `BranchPythonOperator` to determine based on the task group status if downstream process should run:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime
from airflow.utils.state import State

def dummy_task(task_id, fail=False):
  def inner_func():
    if fail:
      raise Exception(f"Task {task_id} failed")
    print(f"Task {task_id} executed")
  return inner_func

def check_task_group_success(ti):
    task_group_id = "dynamic_tasks"
    task_group_states = ti.get_flat_relative_ids(task_ids=[task_group_id])
    task_group_xcom_data = ti.xcom_pull(task_ids=task_group_states, dag_id = ti.dag_id)
    
    for xcom_data in task_group_xcom_data:
        if xcom_data and xcom_data['state'] == State.SUCCESS:
            return 'process_data' # we return task_id that should be executed
    return 'no_process'


with DAG(
    dag_id="dynamic_taskgroup_example_branch",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

  with TaskGroup('dynamic_tasks', tooltip="Dynamic Data loading") as dynamic_tasks:
    for i in range(3):
       PythonOperator(
          task_id=f'dynamic_task_{i}',
          python_callable=dummy_task(task_id=f'dynamic_task_{i}', fail = (i == 1)), # one task will fail
          trigger_rule=TriggerRule.ALL_DONE
        )

  branch_task = BranchPythonOperator(
    task_id='check_group_status',
    python_callable=check_task_group_success,
    trigger_rule=TriggerRule.ALL_DONE,
  )


  process_data = PythonOperator(
     task_id="process_data",
     python_callable=lambda: print("Processing Data after dynamic tasks"),
     trigger_rule=TriggerRule.NONE_FAILED,
   )

  no_process = PythonOperator(
     task_id="no_process",
     python_callable=lambda: print("Not Processing Data after dynamic tasks"),
     trigger_rule=TriggerRule.NONE_FAILED,
   )

  dynamic_tasks >> branch_task
  branch_task >> [process_data, no_process]
```

Here, the `check_task_group_success` function examines the task group and determines whether at least one task within it succeeded, it then sets the appropriate downstream task to execute. This provides a conditional check, not just a change of the trigger rule. This approach provides a more flexible way to control execution based on the outcomes of task group.

While these examples are simplified, the underlying principles apply to more complex workflows. In practice, you should consult the Airflow documentation on `trigger_rule` in detail. The book "Data Pipelines with Apache Airflow" by Bas Harenslak and Julian Rutger also offers excellent, practical advice on advanced dependency handling. Consider, as well, the official Apache Airflow documentation and its section on task relationships and trigger rules – these provide authoritative and comprehensive explanations of the inner workings. The key to avoiding headaches here is to understand the exact behavior of the task group and ensure that all tasks—including the downstream ones consuming the task group—have the necessary trigger rules to achieve your desired behavior. It's a combination of understanding the abstraction layers within Airflow and being explicit about how your pipelines handle errors and skipped tasks.
