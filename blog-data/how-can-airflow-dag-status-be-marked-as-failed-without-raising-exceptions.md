---
title: "How can Airflow DAG status be marked as failed without raising exceptions?"
date: "2024-12-23"
id: "how-can-airflow-dag-status-be-marked-as-failed-without-raising-exceptions"
---

Let’s tackle this thorny issue, something I’ve actually had to navigate a few times in previous roles managing complex data pipelines. The core challenge, as I see it, isn't just stopping a dag; it's doing so *cleanly*, without the cascade of exceptions that can muddy your logs and hinder debugging. We're aiming for a graceful failure, one where the dag’s status is set to 'failed' but without the dramatic theatre of unhandled exceptions bubbling up through the execution environment.

The default behaviour of Airflow, naturally, is to mark a dag as failed when an unhandled exception occurs within any of its tasks. This makes sense – it indicates something went fundamentally wrong. However, there are situations where we might want to explicitly control this outcome, perhaps due to a business logic rule being violated, a downstream dependency failing *without* throwing an error that airflow would recognize, or even simply because the task has reached a predetermined limit for allowed retries. We want to *intentionally* mark the dag as failed without relying on the error handling machinery doing it for us.

There are several methods to achieve this. The common thread through them all is that we utilize Airflow's capabilities to set the state of a task or the whole dag and we do it without triggering any exceptions that airflow would natively catch.

Firstly, let's look at using task callbacks. Airflow provides `on_failure_callback` functions which are triggered when a task instance fails. We can use this to change the dag state. The key here is to not raise an exception in the callback itself; instead, we programmatically mark the dag as failed. Below is an example showing this pattern.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.state import State
from airflow.models import DagRun
from datetime import datetime
from airflow.utils.session import provide_session


def fail_dag_callback(context):
    dag_run = context.get('dag_run')
    if dag_run:
        session = context.get('session')
        dag_run.state = State.FAILED
        session.merge(dag_run)
        session.commit()

def failing_task_function():
    #Simulating a failure condition that doesn't cause an exception
    if True:
      return False #this condition should indicate dag failure
    # ... other code here that might normally throw an exception
    return True

with DAG(
    dag_id="fail_dag_no_exception",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    failing_task = PythonOperator(
        task_id="simulate_failure",
        python_callable=failing_task_function,
        on_failure_callback=fail_dag_callback
    )
```

Here’s the breakdown: The `failing_task_function` simulates a failure by returning `False`. It could be checking the results of an API call or some other condition. However, it *does not* raise an exception. It's essential to understand that returning a value itself does not mark a task as failed in Airflow; it simply passes that return value to the next task. The magic happens in `fail_dag_callback`. This function is executed *only when* the task has failed (as detected by the `on_failure_callback` trigger). Inside this callback we pull the relevant dag_run object, modify its state to 'failed' and commit it to the database. Crucially, we do not throw any exceptions within `fail_dag_callback`; instead, we use the Airflow models directly to manipulate state which avoids the exception-triggered failure flow.

Another approach involves directly manipulating the `DagRun` state from within a task itself. This is useful if the failure condition is directly detectable inside the task, and you want to fail the entire dag *immediately* without waiting for the standard on_failure callback. Note that this requires a bit of manual management of the session and needs to be done carefully to avoid database deadlocks. This direct method might be preferrable to avoid complex callbacks and when the failure condition is localized and does not warrant a postmortem analysis within the callback.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.state import State
from airflow.models import DagRun
from datetime import datetime
from airflow.utils.session import provide_session

@provide_session
def fail_dag_direct(session=None, **context):
    dag_run = context.get('dag_run')
    if dag_run:
        dag_run.state = State.FAILED
        session.merge(dag_run)
        session.commit()


def failing_task_direct():
    if True:
      fail_dag_direct()
    # ... other processing logic
    return True


with DAG(
    dag_id="fail_dag_direct_manipulation",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    failing_task = PythonOperator(
        task_id="simulate_failure",
        python_callable=failing_task_direct
    )
```

In this example `fail_dag_direct` is called directly within the `failing_task_direct` task. The `@provide_session` decorator provides us with a SQLAlchemy session object which we need to update the state directly and it is crucial to `session.commit` the changes. Similar to the first example, `failing_task_direct` does *not* throw an exception it only changes the dag state.

Finally, a slightly more nuanced approach, particularly when dealing with complex dependencies or when a series of tasks need to be evaluated to assess a failure, is to use custom xcom pushes. The dag would push a specific failure indicator, and a dedicated final task would evaluate this indicator and, if necessary, explicitly mark the dag as failed. This is useful when the failure condition is not isolated to a single task but spread across several points in the pipeline.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.python import ShortCircuitOperator
from airflow.utils.state import State
from airflow.models import DagRun
from datetime import datetime
from airflow.utils.session import provide_session


def check_condition(ti, **context):
    condition_1 = ti.xcom_pull(key='condition_1', task_ids='task_1')
    condition_2 = ti.xcom_pull(key='condition_2', task_ids='task_2')
    if condition_1 is False or condition_2 is False:
         return False
    return True

def set_condition_1(ti):
    ti.xcom_push(key='condition_1', value=False)

def set_condition_2(ti):
    ti.xcom_push(key='condition_2', value=True)

@provide_session
def fail_dag_from_xcom(session=None, **context):
    dag_run = context.get('dag_run')
    if dag_run:
        dag_run.state = State.FAILED
        session.merge(dag_run)
        session.commit()


with DAG(
    dag_id="fail_dag_xcom_logic",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    task_1 = PythonOperator(
        task_id="task_1",
        python_callable=set_condition_1
    )
    task_2 = PythonOperator(
        task_id="task_2",
        python_callable=set_condition_2
    )
    check_task = ShortCircuitOperator(
        task_id="check_failure",
        python_callable=check_condition
    )
    fail_task = PythonOperator(
        task_id = 'fail_dag',
        python_callable = fail_dag_from_xcom
    )

    task_1 >> task_2 >> check_task >> fail_task
```

Here, `task_1` pushes 'condition_1' with a false value into xcom, and `task_2` pushes 'condition_2' with a true value. `check_condition` then combines these conditions. The `ShortCircuitOperator` prevents `fail_task` from running if the condition is True. Finally, `fail_task` will execute only if check_condition returns false. The logic ensures `fail_dag_from_xcom` which will fail the dag, runs only when required.

These examples all illustrate different facets of the same fundamental principle: to fail a dag gracefully, we must bypass Airflow’s built-in exception handling and explicitly set the dag state using the underlying model.

For a deeper understanding, I recommend exploring the following resources: The official Apache Airflow documentation, especially the sections on task callbacks, xcoms, and the underlying model, specifically the `DagRun` object. The "Programming Apache Airflow" book by J. Humphrey, or the "Data Pipelines with Apache Airflow" book by B. C. DeHaan are good starting points for getting into specifics of dag and task management. Exploring the Apache Airflow codebase is the final frontier to truly mastering its mechanisms, especially the `airflow.utils.state` and `airflow.models` modules.

Remember, handling dag failures with precision requires a nuanced understanding of Airflow’s inner workings. The techniques outlined here have proven invaluable to me, and I trust that they will be helpful to you too.
