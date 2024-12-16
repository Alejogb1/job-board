---
title: "Why is Airflow `on_success_callback` not providing proper task info?"
date: "2024-12-16"
id: "why-is-airflow-onsuccesscallback-not-providing-proper-task-info"
---

Okay, let’s tackle this. I've seen this specific issue pop up a fair bit over the years, and it usually boils down to a couple of key areas in how Airflow handles callbacks, particularly `on_success_callback`. It's definitely not always obvious on the surface why you aren't getting the expected task information, so let's break it down.

First off, the core problem usually stems from the fact that `on_success_callback` (and similarly, `on_failure_callback`, or `on_retry_callback`) isn’t a magic portal directly to the completed task. Instead, it’s triggered *after* the task’s execution completes successfully, and it receives a specific context from the Airflow scheduler – the ‘context dictionary’. This context, while containing important information, doesn't always include *everything* you might need directly at that point, or in the format, you'd initially expect.

Specifically, what I’ve found happening in the field is that people expect the context variable passed to the callback function to contain, for instance, all the log details directly, or specific xcom values. It doesn’t necessarily work that way. The context contains references and metadata, not the raw results themselves. The task instance data is there, but not pre-processed.

Let's get into the typical issues and how to deal with them, illustrated with some hypothetical examples based on my past project experiences:

**Issue 1: Incorrect Accessing of Task Instance Information**

Often, the primary reason for not seeing proper task info is due to incorrect extraction from the context dictionary. The context gives you a reference to the *task instance* object. From this, you can get more information, but it's a multi-step process, and you need to know *what* you are looking for and *where* to look. The context directly contains an entry named `ti`, which represents the task instance object.

For example, a common pitfall is assuming `context['ti']['log_url']` will directly give you the log URL. Instead, you need to get the task instance and access its log URL method.

Here's an example of a callback that attempts to fetch log information incorrectly:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def incorrect_callback(context):
    log_url = context['ti']['log_url']  # This won't work as intended
    print(f"Log url: {log_url}") # incorrect

def some_task():
    print("Task executed successfully.")

with DAG(
    dag_id='incorrect_callback_example',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    task1 = PythonOperator(
        task_id='task1',
        python_callable=some_task,
        on_success_callback=incorrect_callback
    )
```

This code might execute without errors but the log url won't be correct, it might not return anything. The `log_url` property is not directly available, because the task instance object must be accessed correctly.

**Solution:** Access the log url via the method of the `ti` object.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def correct_callback(context):
    task_instance = context['ti']
    log_url = task_instance.log_url
    print(f"Log url: {log_url}")

def some_task():
    print("Task executed successfully.")

with DAG(
    dag_id='correct_callback_example',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    task1 = PythonOperator(
        task_id='task1',
        python_callable=some_task,
        on_success_callback=correct_callback
    )

```

As you can see, you have to access `log_url` from the task_instance, after grabbing it from `context['ti']`. Similarly, accessing other properties follows this pattern.

**Issue 2: Dealing with XCom Values**

Another common scenario is wanting to access XCom values produced by a task within the `on_success_callback`. XCom values are not automatically available in the callback context. XComs are meant to be pushed and pulled by tasks, not specifically passed to callbacks directly. You’ll need to use the task instance’s `xcom_pull` method to retrieve them.

Suppose your task pushes an xcom and you want to access it from the callback. A common mistake I’ve seen involves trying to access xcom values directly from the context, assuming they are automatically injected. They are not.

Here’s a non-working example illustrating the incorrect approach:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python import get_current_context
from datetime import datetime

def push_xcom_task(**kwargs):
    ti = kwargs['ti']
    ti.xcom_push(key='my_key', value='my_value')

def xcom_error_callback(context):
    xcom_value = context['ti']['xcoms']['my_key'] #This will result in an error
    print(f"XCom value: {xcom_value}")

with DAG(
    dag_id='xcom_error_example',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    task1 = PythonOperator(
        task_id='push_xcom_task',
        python_callable=push_xcom_task,
    )

    task2 = PythonOperator(
        task_id = 'callback_task',
        python_callable= lambda : None,
        on_success_callback=xcom_error_callback,
    )

    task1 >> task2
```

This code is likely to produce an error since you cannot directly access the xcoms as properties of the task instance.

**Solution:** Correct usage of `xcom_pull`:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python import get_current_context
from datetime import datetime

def push_xcom_task(**kwargs):
    ti = kwargs['ti']
    ti.xcom_push(key='my_key', value='my_value')

def xcom_correct_callback(context):
    task_instance = context['ti']
    xcom_value = task_instance.xcom_pull(task_ids='push_xcom_task', key='my_key')
    print(f"XCom value: {xcom_value}")


with DAG(
    dag_id='xcom_correct_example',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    task1 = PythonOperator(
        task_id='push_xcom_task',
        python_callable=push_xcom_task,
    )

    task2 = PythonOperator(
        task_id = 'callback_task',
        python_callable= lambda : None,
        on_success_callback=xcom_correct_callback,
    )

    task1 >> task2
```

Here, we correctly use `xcom_pull` on the task instance object (`task_instance`) to retrieve the `my_key` value from the task `push_xcom_task`. This ensures we are pulling the value correctly, after it was pushed by the previous task.

**Issue 3: Callback Execution Order and Context Inconsistencies**

Finally, remember that the callback happens *after* the main task execution. This can be important if you have complex interactions between tasks. The context within the callback reflects the state *at the end* of that task, not mid-execution. It is very important to understand that the task’s execution and the callback happen at different times, and that can influence the context.

Furthermore, if you use decorators to define tasks (such as `@task`), the context will be passed correctly, but understanding that callbacks are also tasks, is crucial to understanding how and when the code in the callback will execute.

**Key Resources:**

For further understanding and a deeper dive, I highly recommend referring to the official Airflow documentation, particularly the sections dealing with Task Instances, XComs, and callbacks. In addition, *Programming Apache Airflow* by Bas P. Harenslak and Julian Rutger is an excellent resource with practical information. The *Airflow's Task Instance* class documentation is especially helpful. You will find detailed information about the structure of the `ti` object, and how to interact with its methods.

In summary, when debugging issues with `on_success_callback` and task information, focus on properly understanding the structure of the context dictionary. Ensure you use methods like `ti.log_url` to get the right log info, and `ti.xcom_pull` to retrieve xcom values. Most importantly, remember that the context reflects the state of the task *after* it has completed successfully. By keeping these points in mind, you will be able to effectively use the `on_success_callback`, and more specifically the information within, in your workflow.
