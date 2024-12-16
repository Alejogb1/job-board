---
title: "Why isn't Airflow's on_failure_callback triggered after failures?"
date: "2024-12-16"
id: "why-isnt-airflows-onfailurecallback-triggered-after-failures"
---

Okay, let’s dive into why Airflow’s `on_failure_callback` sometimes seems to mysteriously ignore task failures. I've spent my fair share of late nights debugging this very issue, and it often boils down to a few common culprits. It’s certainly frustrating when you're relying on those callbacks for critical alerting or cleanup and they just don't fire. The problem is rarely a bug in Airflow itself; it's usually about nuanced understanding of how task states transition and when the callback logic is actually evaluated.

Specifically, the `on_failure_callback` in Airflow is designed to be triggered when a task transitions into a ‘failed’ state. Sounds straightforward, doesn't it? However, there are particular failure modes and configurations that can prevent this transition from occurring, causing the callback to be bypassed. Here's a breakdown of the key issues and how to approach them, drawing from my experiences in several production deployments.

Firstly, and this is often overlooked, task retries significantly impact the callback mechanism. When you configure a task with `retries`, a failure isn't considered terminal until all the retries are exhausted. Imagine a task encountering a transient network error. Airflow will attempt the task, and if it fails, it will enter a ‘up_for_retry’ state, not a ‘failed’ state. The callback will not trigger at this point. It’s crucial to grasp the distinction; the `on_failure_callback` is strictly tied to the 'failed' task state. Only when all retries fail, will the task transition to 'failed'. If, on retry, the task succeeds, it progresses as normal without ever triggering the `on_failure_callback`. This is expected behaviour, but it's not always intuitive when you're expecting immediate notification.

Secondly, task dependencies can sometimes mask failures. If a task, let's say `task_b`, depends on `task_a` and `task_a` never completes due to an external issue such as a broken connection to an API, `task_b` may remain in a 'scheduled' or 'up_for_reschedule' state. In this scenario, `task_a`’s failure might not be directly flagged as a 'failed' task state that would trigger `on_failure_callback`, especially if you do not have retry logic on that task or the failures are intermittent. The DAG will still reflect the task failure eventually, but the transition to 'failed' might not be immediate, particularly if the scheduler isn't actively checking for downstream dependencies.

Thirdly, and this is particularly problematic, are upstream failures with trigger rules that don't lead to a task entering a failed state. Tasks that depend on other tasks have 'trigger rules', like 'all_success', 'all_failed', 'one_success', 'dummy', etc. If your trigger rule is 'all_success' and an upstream task fails, then the downstream task might be skipped, or enter a 'skipped' state, but not always a ‘failed’ state, which again, will circumvent the `on_failure_callback` logic. Careful consideration of your trigger rule logic in relation to the callback is crucial.

Here are three code snippets illustrating potential scenarios and solutions:

**Snippet 1: Retries Masking Failures**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def failing_task():
    raise Exception("Simulated failure")

def failure_callback(context):
    print("Task failed! Triggering alert...")
    task_instance = context['ti']
    print(f"Task ID: {task_instance.task_id}")
    print(f"Execution Date: {task_instance.execution_date}")
    # Additional alerting logic here

with DAG(
    dag_id='retry_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    task_with_retry = PythonOperator(
        task_id='failing_task_retry',
        python_callable=failing_task,
        retries=3, # This is the key here
        on_failure_callback=failure_callback
    )
```

In this case, the `on_failure_callback` will *only* trigger after three failed attempts of `failing_task_retry`. If you expect immediate alerts on the first failure, you'll be disappointed. The solution is either to reduce the retries, or redesign the alert mechanism to operate on ‘up_for_retry’ task states, if you wish. This can be done using Airflow's event listeners.

**Snippet 2: Dependency Blocking Failure Propagation**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def dependent_task():
    print("Dependent task executed")

def external_task():
   raise Exception("External API failure")


def failure_callback_dependency(context):
    print("Task failed due to dependency issue!")
    task_instance = context['ti']
    print(f"Task ID: {task_instance.task_id}")
    print(f"Execution Date: {task_instance.execution_date}")
    # Alerting logic specific to dependency failures

with DAG(
    dag_id='dependency_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    external_failing_task = PythonOperator(
        task_id='external_task_failure',
        python_callable=external_task,
        #on_failure_callback=failure_callback_dependency
    )

    dependent_on_external_task = PythonOperator(
        task_id='dependent_task',
        python_callable=dependent_task,
        trigger_rule='all_success', # important to see behaviour
        on_failure_callback=failure_callback_dependency
    )

    external_failing_task >> dependent_on_external_task

```

Here, if `external_task_failure` fails, `dependent_task` will not transition to a 'failed' state because the trigger rule is 'all_success' , instead it will be ‘skipped’. Adding the `on_failure_callback` to `external_task_failure` (uncommenting it) will alert to the failure of that task, but not of the dependent task, as it's been skipped. For situations like this, consider restructuring your DAG, or adding a failure-detection mechanism to both upstream and downstream tasks. For the downstream, we may need `trigger_rule='all_done'` or `trigger_rule='one_failed'` to trigger the alert callback.

**Snippet 3: Using Trigger Rules**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime

def success_task():
    print("Success task executed")

def failure_task():
    raise Exception("Task always fails")

def failure_callback_trigger(context):
    print("Failure due to trigger rule failure")
    task_instance = context['ti']
    print(f"Task ID: {task_instance.task_id}")
    print(f"Execution Date: {task_instance.execution_date}")

with DAG(
    dag_id='trigger_rule_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    succeeding_task = PythonOperator(
        task_id='success_task',
        python_callable=success_task
    )

    failing_task = PythonOperator(
        task_id='failure_task',
        python_callable=failure_task,
    )

    dependent_task_trigger_all_success = PythonOperator(
        task_id='dependent_task_trigger_all_success',
        python_callable=lambda: print("trigger_all_success triggered"),
        trigger_rule=TriggerRule.ALL_SUCCESS,
        on_failure_callback=failure_callback_trigger
    )


    dependent_task_trigger_one_failed = PythonOperator(
        task_id='dependent_task_trigger_one_failed',
        python_callable=lambda: print("trigger_one_failed triggered"),
        trigger_rule=TriggerRule.ONE_FAILED,
        on_failure_callback=failure_callback_trigger

    )

    succeeding_task >> dependent_task_trigger_all_success
    failing_task >> dependent_task_trigger_one_failed
```

Here, if `succeeding_task` succeeds and `failure_task` fails, then `dependent_task_trigger_all_success` will not run, because the trigger rule is ‘all_success’. Additionally, `dependent_task_trigger_one_failed` will run, because its trigger rule is ‘one_failed’, therefore, its `on_failure_callback` will be triggered. This is why you need to choose the correct trigger rule for your workflow.

For further reading, I recommend exploring the official Airflow documentation deeply; particularly, the sections detailing task states, task dependencies, and trigger rules are essential. For a broader understanding of distributed systems and event-driven architectures, which are highly relevant to understanding Airflow's internals, the book "Designing Data-Intensive Applications" by Martin Kleppmann is invaluable. Additionally, I'd recommend going through the Apache Airflow code base if you are looking for very specific information on the callback implementation. Remember, the `on_failure_callback` is a potent tool, but it demands a clear understanding of how Airflow handles task states and the ripple effects of upstream dependencies. Careful design and testing are key to reliability.
