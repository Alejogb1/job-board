---
title: "How to notify on Airflow after two consecutive task failures?"
date: "2024-12-23"
id: "how-to-notify-on-airflow-after-two-consecutive-task-failures"
---

Alright, let’s tackle this. Notifying on two consecutive task failures in Apache Airflow is a common need, and frankly, one I've run into a fair few times during my career managing data pipelines. The default retry mechanisms can sometimes mask recurring issues, and a more targeted alert system is often necessary to really understand what’s going on. It's about moving beyond just handling a single failure to spotting patterns that might indicate something deeper. I’ll walk you through how I’ve approached this, providing some practical code examples and context.

The challenge here isn't about Airflow’s capacity for retries – it’s about *context*. Airflow, by default, retries tasks based on the `retries` and `retry_delay` parameters defined in your DAG. However, those retries often happen sequentially. If you want to be alerted specifically *after* a second *consecutive* failure, you need a bit more control and logic. The problem with a single failure alert is that it may not be critical, especially if configured to retry. We need to wait to see if a problem persists.

My first experience with this was a particularly stubborn data ingestion pipeline. Data was coming from an external API which could be unstable. Simple retry logic wasn't enough; it kept triggering alerts that were ultimately resolved by the retries, leading to alert fatigue. I realised we needed to be smarter and more selective with these alerts. What worked for us was a combination of custom callbacks and leveraging Airflow's XCom system for state tracking.

Let's break it down. At its core, our approach will revolve around these steps:

1.  **Tracking Failure State:** We need to remember if a task has failed in the previous run. We’ll utilize XComs to persist this information across task executions.
2.  **Conditional Alerting:** Within a callback function, we check if the task failed in the current execution, *and* if it failed in the previous run. If both are true, we trigger an alert.
3.  **State Reset:** If the task succeeds, we need to clear any failure flags in XCom. This keeps track of *consecutive* failures, not just any failure.

Here’s a python code snippet illustrating how this might be implemented as a custom callback for your DAG:

```python
from airflow.models import DAG, TaskInstance
from airflow.operators.bash import BashOperator
from airflow.utils.state import State
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow.utils.trigger_rule import TriggerRule

def failure_callback(context):
    task_instance: TaskInstance = context.get('task_instance')
    task_id = task_instance.task_id
    dag_id = task_instance.dag_id
    previous_failed = task_instance.xcom_pull(key=f"{task_id}_failed", task_ids=task_id)

    if task_instance.state == State.FAILED:
      if previous_failed:
          # Execute alerting logic here. For example:
          print(f"Alerting for task {task_id} in DAG {dag_id}. Consecutive failures detected!")
          # Replace print with actual notification e.g., send email or Slack message.
          # add an xcom value to skip alert logic further down the DAG.
          task_instance.xcom_push(key="alert_sent", value=True)
      else:
        task_instance.xcom_push(key=f"{task_id}_failed", value=True)
    elif task_instance.state == State.SUCCESS:
        task_instance.xcom_push(key=f"{task_id}_failed", value=False)


with DAG(
    dag_id='consecutive_failure_notification',
    start_date=days_ago(1),
    schedule=timedelta(minutes=1),
    catchup=False,
    default_args={'retries': 1, 'on_failure_callback': failure_callback}
) as dag:
    task1 = BashOperator(
        task_id='bash_task_1',
        bash_command='exit 1',  # Simulates a failure
        on_failure_callback=failure_callback
    )
    task2 = BashOperator(
        task_id='bash_task_2',
        bash_command='echo "this task succeeds"',
         on_failure_callback=failure_callback
    )
    task3 = BashOperator(
        task_id='bash_task_3',
        bash_command='exit 1',  # Simulates a failure
         on_failure_callback=failure_callback,
         trigger_rule=TriggerRule.ALL_DONE
    )

    task1 >> task2 >> task3
```

In this example, each `BashOperator` has a `failure_callback` which checks the previous run state. The `task1` and `task3` will fail by design, while `task2` will succeed. Upon a second consecutive failure of `task1` (assuming `catchup=False`), the callback will trigger the print statement, which should be replaced with your desired alerting mechanism. `Task 2` clears the previous failure, and `task3` will behave in the same way as `task1`. In a real world example, we would likely use a more reliable and robust alerting process.

To make this even more adaptable, consider adding a configuration parameter to specify how many consecutive failures to wait for before triggering the notification. Here’s how that would look:

```python
def configurable_failure_callback(consecutive_failures_threshold: int):
    def callback(context):
        task_instance: TaskInstance = context.get('task_instance')
        task_id = task_instance.task_id
        dag_id = task_instance.dag_id
        failure_count_key = f"{task_id}_failure_count"

        current_failure_count = task_instance.xcom_pull(key=failure_count_key, task_ids=task_id) or 0

        if task_instance.state == State.FAILED:
            current_failure_count += 1
            task_instance.xcom_push(key=failure_count_key, value=current_failure_count)

            if current_failure_count >= consecutive_failures_threshold:
              print(f"Alerting for task {task_id} in DAG {dag_id}. {consecutive_failures_threshold} consecutive failures detected!")
              # Add your alerting logic here
              task_instance.xcom_push(key="alert_sent", value=True)
        elif task_instance.state == State.SUCCESS:
            task_instance.xcom_push(key=failure_count_key, value=0)
            task_instance.xcom_push(key="alert_sent", value=False)

    return callback

with DAG(
    dag_id='configurable_consecutive_failure_notification',
    start_date=days_ago(1),
    schedule=timedelta(minutes=1),
    catchup=False,
    default_args={'retries': 1, 'on_failure_callback': configurable_failure_callback(consecutive_failures_threshold=2)}
) as dag:
  task1 = BashOperator(
      task_id='bash_task_1',
      bash_command='exit 1',
      on_failure_callback=configurable_failure_callback(consecutive_failures_threshold=2)
  )
  task2 = BashOperator(
        task_id='bash_task_2',
        bash_command='echo "this task succeeds"',
        on_failure_callback=configurable_failure_callback(consecutive_failures_threshold=2)
    )
  task3 = BashOperator(
      task_id='bash_task_3',
      bash_command='exit 1',
      on_failure_callback=configurable_failure_callback(consecutive_failures_threshold=2),
      trigger_rule=TriggerRule.ALL_DONE
  )
  task1 >> task2 >> task3
```

In this modified version, `configurable_failure_callback` is a factory function which takes `consecutive_failures_threshold` as parameter which allows for different thresholds of consecutive failures across all the DAG's tasks. This callback will maintain a counter of consecutive failures, and only send an alert if this threshold is reached.

Finally, while this approach works well within the context of a single DAG, there may be instances where multiple DAGs share similar requirements. A decorator is an elegant way to avoid copy-pasting these callback functions in multiple DAGs. It could look like this:

```python
from functools import wraps
from airflow.models import DAG, TaskInstance
from airflow.operators.bash import BashOperator
from airflow.utils.state import State
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow.utils.trigger_rule import TriggerRule

def with_consecutive_failure_notification(consecutive_failures_threshold: int):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def failure_callback(context):
              task_instance: TaskInstance = context.get('task_instance')
              task_id = task_instance.task_id
              dag_id = task_instance.dag_id
              failure_count_key = f"{task_id}_failure_count"

              current_failure_count = task_instance.xcom_pull(key=failure_count_key, task_ids=task_id) or 0

              if task_instance.state == State.FAILED:
                  current_failure_count += 1
                  task_instance.xcom_push(key=failure_count_key, value=current_failure_count)

                  if current_failure_count >= consecutive_failures_threshold:
                      print(f"Alerting for task {task_id} in DAG {dag_id}. {consecutive_failures_threshold} consecutive failures detected!")
                      # Add your alerting logic here
                      task_instance.xcom_push(key="alert_sent", value=True)

              elif task_instance.state == State.SUCCESS:
                  task_instance.xcom_push(key=failure_count_key, value=0)
                  task_instance.xcom_push(key="alert_sent", value=False)

            kwargs['on_failure_callback'] = failure_callback
            return func(*args, **kwargs)
        return wrapper
    return decorator


@with_consecutive_failure_notification(consecutive_failures_threshold=2)
def create_my_bash_task(task_id, command):
    return BashOperator(task_id=task_id, bash_command=command)


with DAG(
    dag_id='decorated_consecutive_failure_notification',
    start_date=days_ago(1),
    schedule=timedelta(minutes=1),
    catchup=False,
    default_args={'retries': 1}
) as dag:
  task1 = create_my_bash_task(task_id='bash_task_1', command='exit 1')
  task2 = create_my_bash_task(task_id='bash_task_2', command='echo "this task succeeds"')
  task3 = create_my_bash_task(task_id='bash_task_3', command='exit 1')
  task3.trigger_rule = TriggerRule.ALL_DONE

  task1 >> task2 >> task3
```

Here, the `@with_consecutive_failure_notification` decorator adds the necessary failure callback logic to any task it wraps, promoting code re-use. In this example the decorator is applied when we create `BashOperator` tasks, setting the `on_failure_callback` to our custom callback. This is often a cleaner method when you have many tasks requiring the same logic and encourages maintainability.

For further learning on this topic, I’d recommend studying:

*   **“Data Pipelines Pocket Reference” by James Densmore.** This book gives a comprehensive overview of building and maintaining reliable data pipelines, which includes sections on monitoring and alerting.
*   The official **Apache Airflow documentation**, specifically the sections on callbacks, XComs, and task lifecycle. It’s comprehensive and up-to-date.
*   Papers on **"Observability for data pipelines"**. This is a broad field, but focusing on publications that discuss failure detection and alerting will be useful. Specifically, research papers on building fault-tolerant distributed systems often discuss similar challenges and can provide valuable theoretical context.

These resources should give you a solid theoretical grounding and practical techniques. Remember to tailor your approach to the specific demands of your pipelines. Consecutive failure notifications are a vital part of proactive monitoring, ensuring you catch issues before they become significant disruptions. And this way, you move from reactive to proactive issue management. It's worked well for me, and I hope it does for you too.
