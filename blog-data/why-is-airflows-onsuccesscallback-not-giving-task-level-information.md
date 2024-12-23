---
title: "Why is Airflow's `on_success_callback` not giving task-level information?"
date: "2024-12-23"
id: "why-is-airflows-onsuccesscallback-not-giving-task-level-information"
---

Alright, let's dive into this. I've definitely spent some quality time debugging Airflow, and the `on_success_callback` quirk you’re asking about is a classic head-scratcher for many. From my past experience building large-scale data pipelines, I recall specifically an issue where I wanted to capture details like task instance execution time or the specific configuration used for a particular run within the `on_success_callback`. Frustratingly, I found myself getting the dag-level information, but not the task’s. This behavior is by design, and it's rooted in how Airflow structures its execution context and manages callbacks.

The core issue lies in the scope and timing of the callback execution. The `on_success_callback` and its counterpart, `on_failure_callback`, are designed to be invoked at the *dag* level, not at the individual *task* level. This means that the callback function receives context related to the entire dag run, rather than a specific task instance. This is why, when you try accessing something like `context['ti']`, which you'd typically use within a task to get task instance details, you often end up with a `None` value or information that doesn't correspond to the completed task.

The problem is this: callbacks operate post-execution of the specific task, outside of the task's execution context. This happens after the task status is already marked as a success (or failure), and the immediate execution environment is no longer active. The execution context available at the dag level is designed for operations that apply to the entire dag run, such as sending alerts based on the overall outcome of a workflow, logging summary information, or triggering downstream processes. They’re not for fine-grained task-specific operations.

So, how do we work around this and retrieve task-level details? Well, there are a few reliable ways, and I will show you three examples that I've used successfully in the past.

**Example 1: Using the `xcom_push` Mechanism**

One effective method is to use Airflow's built-in `xcom` mechanism to "push" relevant task information during the task's execution. XCom, or cross-communication, is an Airflow service that lets task instances exchange messages. Then, you can access this information within the `on_success_callback` by querying the xcom messages related to that dag run. Here is some code I often rely on:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable
from datetime import datetime

def my_task(ti, **kwargs):
    # Simulate some work
    task_execution_time = datetime.now()
    ti.xcom_push(key='task_time', value=str(task_execution_time))
    ti.xcom_push(key='task_config', value=Variable.get("my_config", default_var="default_config"))


def dag_success_callback(context):
    dag_run = context['dag_run']
    task_instances = dag_run.get_task_instances()
    for ti in task_instances:
        if ti.state == 'success':
          task_time = ti.xcom_pull(key='task_time', task_ids=ti.task_id)
          task_config = ti.xcom_pull(key='task_config', task_ids=ti.task_id)
          print(f"Task {ti.task_id} completed at: {task_time}, with config: {task_config}")

with DAG(
    dag_id='xcom_example_dag',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    on_success_callback=[dag_success_callback],
) as dag:

    task_one = PythonOperator(
        task_id='my_task_one',
        python_callable=my_task,
    )

    task_two = PythonOperator(
        task_id='my_task_two',
        python_callable=my_task,
    )
```

In this snippet, each task pushes its execution time and a variable to XCom. The `dag_success_callback` then iterates through all task instances, retrieves this information, and outputs it. This allows task-level context to be passed along without being tied to the direct task execution context of `on_success_callback`.

**Example 2: Utilizing Logging within the Task**

Another way to retain task-level information is by embedding logging directly into your tasks. The logs generated are then accessible and searchable through the Airflow web interface or external logging services. You can then parse these logs post-execution to gather what you need. Here’s how that might look:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import logging
from datetime import datetime

def my_logged_task(**kwargs):
    log = logging.getLogger(__name__)
    task_start = datetime.now()
    # Simulate work here
    task_end = datetime.now()
    duration = task_end - task_start
    log.info(f"Task {kwargs['ti'].task_id} started at {task_start}, took {duration} to complete.")

def dag_success_callback_logs(context):
  # This callback wouldn't process the log messages here directly
    print("Dag success callback triggered. Check logs for task details.")


with DAG(
    dag_id='log_example_dag',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
     on_success_callback=[dag_success_callback_logs],
) as dag:
    task_one = PythonOperator(
        task_id='my_log_task_one',
        python_callable=my_logged_task,
    )

    task_two = PythonOperator(
        task_id='my_log_task_two',
        python_callable=my_logged_task,
    )

```

In this example, the tasks directly log information about their execution. The `dag_success_callback_logs` here doesn't do the log parsing itself; instead, you'd examine the logs associated with the task instance separately using the web UI or external log tools. You can then build other custom scripts that read from the log service, filter on the completed task ID's, and extract the relevant data.

**Example 3: A Custom Task Listener**

For more intricate scenarios, creating a custom task listener class that hooks into Airflow's event system, can be beneficial. While this is more involved, it offers better real-time information and allows you to perform operations when a task reaches a certain state. Here's a skeletal example demonstrating the concept:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.listeners.base_listener import BaseListener
from airflow.models import TaskInstance, DagRun
from datetime import datetime

class CustomTaskListener(BaseListener):

    def on_task_instance_success(self, context):
      ti = context['ti']
      print(f"Custom Listener: Task {ti.task_id} succeeded at {datetime.now()}")
      # Here, you would potentially execute other logic such as publishing to a message queue.

def my_task_simple(**kwargs):
    # Simulate work
    pass

with DAG(
    dag_id='listener_example_dag',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    listeners=[CustomTaskListener()],
) as dag:
    task_one = PythonOperator(
        task_id='listener_task_one',
        python_callable=my_task_simple,
    )

    task_two = PythonOperator(
        task_id='listener_task_two',
        python_callable=my_task_simple,
    )
```

In this example, `CustomTaskListener` extends `BaseListener`, an Airflow class, and overrides the `on_task_instance_success` method to receive context at the time of task completion. This allows more granular control over what happens upon successful task execution. The listeners get triggered immediately after the task succeeds.

To dive deeper into these topics, I'd recommend examining Apache Airflow's official documentation, particularly the sections on XCom, logging, and listeners. For a broader understanding of distributed systems and workflow orchestration, consider reviewing "Designing Data-Intensive Applications" by Martin Kleppmann. Also, the book "Data Pipelines Pocket Reference" by James Densmore provides very practical, hands-on guidance. You will also find interesting information in the "Software Engineering at Google" book for best practices in pipeline development.

In conclusion, while Airflow's `on_success_callback` doesn’t inherently provide task-level information, there are multiple robust techniques to get what you need. Each of these solutions offers a different balance between ease of implementation and the level of control they afford. Understanding the core limitations and employing these workarounds will definitely improve your troubleshooting and enhance the visibility of your data pipelines.
