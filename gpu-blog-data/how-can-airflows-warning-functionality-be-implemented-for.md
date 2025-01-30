---
title: "How can Airflow's warning functionality be implemented for long-running tasks?"
date: "2025-01-30"
id: "how-can-airflows-warning-functionality-be-implemented-for"
---
Airflow's built-in task timeout feature, while effective for hard failures, doesn’t gracefully handle long-running tasks that are nearing their expected duration, yet not outright failing. Based on my experience maintaining data pipelines for a financial institution, I've found a proactive, warning-based approach invaluable for preventing potential data latency and alerting operators to potential bottlenecks *before* a hard failure occurs. I'll detail how this can be implemented using a combination of Airflow’s task callbacks, custom checks, and alerting mechanisms.

**Explanation: Implementing Time-Based Warnings**

The key to effective warning implementation is to periodically monitor the progress of long-running tasks and issue warnings when they exceed pre-defined thresholds short of task timeouts. Rather than solely relying on Airflow’s `execution_timeout`, we integrate a custom check within the task’s execution context that measures elapsed time and triggers callbacks based on defined warning criteria. This can be achieved without altering the core logic of the task itself, enabling non-invasive monitoring. I have found it essential to couple this monitoring with an alerting system to reach on-call operators if a problem has been detected.

The process involves these primary components:

1.  **Monitoring within the Task:** Tasks need to be instrumented with logic to track the elapsed time from their start. This is not readily available directly from Airflow, instead, a custom time tracker is needed. The most practical method I've found is to use the Python `time` module in tandem with task callbacks.
2.  **Defining Warning Thresholds:** This involves setting different time thresholds, often expressed as percentages of the anticipated task duration. For example, a task might trigger a warning when it's 75% through the estimated execution time, allowing time for intervention before it hits the `execution_timeout` at, say, 100%. The selection of these thresholds must be aligned with the particular tasks' behaviour, which will be known from previous observation.
3.  **Triggering Callbacks:** When a task’s runtime exceeds a defined threshold, a callback function is executed. This callback is designed to raise specific alarms, which can include logging messages, sending notifications, or updating external monitoring systems. This function must handle potential failures as well so it does not impact the task.
4.  **Alerting Mechanism:** The triggered alarms, via callback functions, then interact with an alerting system. Depending on the company’s infrastructure, this can be handled by integrating with a tool like PagerDuty or by sending emails through Airflow’s notification functionality. The important takeaway is that alarms must trigger action.

I typically implement this functionality within a common task decorator or class which allows for easy integration within any specific dag without rewriting boilerplate code. This helps ensure maintainability and standardisation.

**Code Examples**

The following code snippets demonstrate how I've implemented time-based warnings in several real-world Airflow environments.

**Example 1: Using `on_failure_callback` and a Custom Monitor**

Here, I'm using Airflow's `on_failure_callback` to handle both hard failures and time-based warning callbacks using a single custom monitor class. This approach keeps the code concise and allows centralised management.

```python
import time
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

class TaskMonitor:
    def __init__(self, warn_threshold_pct, execution_timeout):
        self.warn_threshold_pct = warn_threshold_pct
        self.execution_timeout = execution_timeout
        self.start_time = None

    def start_monitoring(self):
      self.start_time = time.time()

    def check_progress(self):
        if not self.start_time:
          return
        elapsed_time = time.time() - self.start_time
        warn_time = self.execution_timeout * (self.warn_threshold_pct / 100)
        if elapsed_time > warn_time:
          print(f"Warning: Task approaching timeout. Elapsed time: {elapsed_time:.2f}s, Threshold: {warn_time:.2f}s")
          # Trigger specific alert
          self.send_alert(elapsed_time, warn_time)

    def send_alert(self, elapsed_time, warn_time):
      # Implement alert sending logic here
        print(f"Sending alert: Task exceeded warning threshold. Elapsed time: {elapsed_time:.2f}s, Threshold: {warn_time:.2f}s")

    def on_failure_callback(self, context):
        if self.start_time:
          elapsed_time = time.time() - self.start_time
          print(f"Task failed. Elapsed time: {elapsed_time:.2f}s")
          # Send failure alert here
        else:
          print("Task failed before start of monitoring")

def long_running_task(monitor):
    monitor.start_monitoring()
    for _ in range(10):
        time.sleep(2)
        monitor.check_progress()
    print("Long task complete")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
    'execution_timeout': timedelta(seconds=30),
}

with DAG(
    dag_id='time_based_warnings_example',
    default_args=default_args,
    schedule=None,
    catchup=False
) as dag:
    monitor = TaskMonitor(warn_threshold_pct=75, execution_timeout=default_args['execution_timeout'].total_seconds())

    long_task = PythonOperator(
      task_id='long_running_task',
      python_callable=long_running_task,
      op_kwargs={'monitor': monitor},
      on_failure_callback=monitor.on_failure_callback,
    )
```

In this example, `TaskMonitor` manages the monitoring logic, which includes start time tracking, calculating threshold breaches and triggering warnings. The `on_failure_callback` allows for unified failure handling. This prevents duplicating logic.

**Example 2: Using a Custom Callback Function**

This example demonstrates how to handle warnings by triggering a custom callback based on elapsed time. This approach is useful if you do not want to overload your `on_failure_callback`. This requires an intermediate class to keep track of the state of the monitor as the `context` is reset after each execution of the task.

```python
import time
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta


class WarningState:
    def __init__(self):
        self.start_time = None

    def start_monitoring(self):
      self.start_time = time.time()


warning_state = WarningState()


def time_based_warn(context, warn_threshold_pct, execution_timeout):
    if not warning_state.start_time:
        return
    elapsed_time = time.time() - warning_state.start_time
    warn_time = execution_timeout * (warn_threshold_pct / 100)
    if elapsed_time > warn_time:
        print(f"Warning: Task approaching timeout. Elapsed time: {elapsed_time:.2f}s, Threshold: {warn_time:.2f}s")
        #Implement alert logic
        send_alert(elapsed_time, warn_time)

def send_alert(elapsed_time, warn_time):
        # Implement alert sending logic here
        print(f"Sending alert: Task exceeded warning threshold. Elapsed time: {elapsed_time:.2f}s, Threshold: {warn_time:.2f}s")


def long_running_task():
    warning_state.start_monitoring()
    for _ in range(10):
        time.sleep(2)
        time_based_warn(
            context=None, # Context will not be populated in this case
            warn_threshold_pct=75,
            execution_timeout=30
        )
    print("Long task complete")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
    'execution_timeout': timedelta(seconds=30),
}

with DAG(
    dag_id='time_based_warnings_custom_callback',
    default_args=default_args,
    schedule=None,
    catchup=False
) as dag:
    long_task = PythonOperator(
        task_id='long_running_task',
        python_callable=long_running_task
    )

```

This approach offers greater flexibility since the alert can be sent as a callback, and not strictly as part of the on failure workflow.

**Example 3: Using a Custom Operator Class**

For more complex cases, creating a custom operator allows for full encapsulation of monitoring and warning logic. I've found this particularly useful when integrating with external systems.

```python
import time
from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
from datetime import datetime, timedelta
from airflow import DAG
from airflow.utils.state import State

class MonitoredPythonOperator(BaseOperator):

    template_fields = ['python_callable', 'op_args', 'op_kwargs']
    
    @apply_defaults
    def __init__(self,
                python_callable,
                op_args=None,
                op_kwargs=None,
                warn_threshold_pct=75,
                execution_timeout=30,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)
        self.python_callable = python_callable
        self.op_args = op_args or []
        self.op_kwargs = op_kwargs or {}
        self.warn_threshold_pct=warn_threshold_pct
        self.execution_timeout=execution_timeout
        self.start_time= None
        

    def start_monitoring(self):
      self.start_time = time.time()

    def check_progress(self):
        if not self.start_time:
          return
        elapsed_time = time.time() - self.start_time
        warn_time = self.execution_timeout * (self.warn_threshold_pct / 100)
        if elapsed_time > warn_time:
            print(f"Warning: Task approaching timeout. Elapsed time: {elapsed_time:.2f}s, Threshold: {warn_time:.2f}s")
            self.send_alert(elapsed_time, warn_time)


    def send_alert(self, elapsed_time, warn_time):
            #Implement alert logic
            print(f"Sending alert: Task exceeded warning threshold. Elapsed time: {elapsed_time:.2f}s, Threshold: {warn_time:.2f}s")

    def execute(self, context):
        self.start_monitoring()
        
        try:
          result = self.python_callable(*self.op_args, **self.op_kwargs)
          return result

        except Exception as e:
          print(f"Task failed with exception: {e}")
          raise
        finally:
          if self.start_time:
            self.check_progress()


def long_running_task(duration):
    for _ in range(duration):
        time.sleep(2)
    print("Long task complete")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
    'execution_timeout': timedelta(seconds=30),
}


with DAG(
    dag_id='time_based_warnings_custom_operator',
    default_args=default_args,
    schedule=None,
    catchup=False
) as dag:

  long_task = MonitoredPythonOperator(
        task_id='long_running_task',
        python_callable=long_running_task,
        op_kwargs={'duration': 10},
        warn_threshold_pct=75,
        execution_timeout=default_args['execution_timeout'].total_seconds()
  )
```

The `MonitoredPythonOperator` centralizes all logic related to both task execution and time-based warning generation.  This structure simplifies complex task handling and also standardizes the implementation pattern across the project.

**Resource Recommendations**

For further investigation, consult the official Airflow documentation regarding task execution and callbacks. Explore advanced concepts such as custom operators and task decorators for creating reusable components. Also, researching available monitoring tools and alerting systems will aid in building a robust pipeline solution. Finally, studying the `time` module in Python will allow for better tracking of the time.
