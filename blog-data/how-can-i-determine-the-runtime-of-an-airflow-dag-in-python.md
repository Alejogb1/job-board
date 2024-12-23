---
title: "How can I determine the runtime of an Airflow DAG in Python?"
date: "2024-12-23"
id: "how-can-i-determine-the-runtime-of-an-airflow-dag-in-python"
---

Let's dive into the specifics of gauging the runtime of Airflow dags, something I've had to get intimately familiar with over the years, especially during my time managing a large-scale data pipeline for a fin-tech firm. We weren't just launching a few tasks; we're talking about hundreds of dependencies, multiple daily runs, and the constant need to optimize for performance. It’s a crucial consideration, not just for scheduling efficiency but also for spotting potential bottlenecks and capacity planning.

Firstly, understand that 'runtime' in the context of an Airflow dag isn’t a singular, static figure. It's actually a combination of various durations: the time the scheduler takes to parse the dag, the time it takes for individual tasks to execute, and then the overall wall-clock time for an entire dag run. To properly analyze performance, we really need to delve into each of these elements individually.

So, how can we get our hands on this information? Airflow provides a few different avenues, each with varying levels of granularity. Let’s start with the most common and accessible: task execution times.

Airflow logs everything, but you can also query the metadatabase directly. The most straightforward way to get task-level runtimes is via the `TaskInstance` model in the `airflow.models` module. When a task runs, a corresponding `TaskInstance` object is created which stores crucial runtime information such as the `start_date`, `end_date`, and `duration`. These are precisely what you need to determine the actual duration of a specific task.

Here's a basic Python snippet demonstrating how you could extract these times:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.state import State
from airflow.utils import timezone
from datetime import datetime, timedelta
from airflow.models import DagRun, TaskInstance
from sqlalchemy import and_
from airflow.configuration import conf
from sqlalchemy import create_engine
import os

def fetch_task_runtimes(dag_id, execution_date):
    db_uri = conf.get('database', 'sql_alchemy_conn')
    engine = create_engine(db_uri)

    session = engine.connect()

    query_result = session.query(TaskInstance).filter(
        and_(
            TaskInstance.dag_id == dag_id,
            TaskInstance.execution_date == execution_date,
            TaskInstance.state == State.SUCCESS
        )
    ).all()
    
    task_runtimes = {}
    for ti in query_result:
      if ti.end_date and ti.start_date:
          task_runtimes[ti.task_id] = (ti.end_date - ti.start_date).total_seconds()
      else:
          task_runtimes[ti.task_id] = None # or some suitable default, if the task did not finish or has incomplete timestamps.

    session.close()
    engine.dispose()
    return task_runtimes


with DAG(
    dag_id='task_runtime_example',
    start_date=timezone.datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    def simple_task():
        import time
        time.sleep(2)
        return "Done!"

    task_1 = PythonOperator(
        task_id='simple_python_task',
        python_callable=simple_task,
    )

    if __name__ == '__main__':
        now = timezone.datetime.now()
        dag.run(execution_date=now)
        runtimes = fetch_task_runtimes(dag_id="task_runtime_example", execution_date=now)
        print(runtimes)

```

In this snippet, we retrieve `TaskInstance` objects for a given `dag_id` and `execution_date` that finished successfully. For each retrieved task, we calculate the duration by subtracting `start_date` from `end_date` and converting the `timedelta` object into total seconds. This is a valuable baseline for analyzing individual task performance, enabling you to quickly pinpoint the slower components of your dag. This will yield a dictionary where keys are task ids and values are runtime in seconds, or none.

Next, let's consider the overall DAG runtime. This is calculated by measuring the time from when the DAG's scheduler process starts until all the tasks complete or the dag run state becomes a terminal state. Again, the `DagRun` model stores this information, specifically through `start_date` and `end_date`. The code is very similar to our previous example:

```python
from airflow import DAG
from airflow.utils.state import State
from airflow.utils import timezone
from datetime import datetime, timedelta
from airflow.models import DagRun, TaskInstance
from sqlalchemy import and_
from airflow.configuration import conf
from sqlalchemy import create_engine
import os
from airflow.operators.python import PythonOperator


def fetch_dag_runtime(dag_id, execution_date):
    db_uri = conf.get('database', 'sql_alchemy_conn')
    engine = create_engine(db_uri)
    session = engine.connect()
    dag_run = session.query(DagRun).filter(
        and_(
            DagRun.dag_id == dag_id,
            DagRun.execution_date == execution_date,
            DagRun.state == State.SUCCESS
        )
    ).first()
    session.close()
    engine.dispose()
    if dag_run and dag_run.end_date and dag_run.start_date:
        return (dag_run.end_date - dag_run.start_date).total_seconds()
    else:
        return None


with DAG(
    dag_id='dag_runtime_example',
    start_date=timezone.datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:

    def simple_task():
        import time
        time.sleep(2)
        return "Done!"

    task_1 = PythonOperator(
        task_id='simple_python_task',
        python_callable=simple_task,
    )
    task_2 = PythonOperator(
        task_id='another_simple_python_task',
        python_callable=simple_task,
    )
    
    task_1 >> task_2

    if __name__ == '__main__':
        now = timezone.datetime.now()
        dag.run(execution_date=now)
        runtime = fetch_dag_runtime(dag_id="dag_runtime_example", execution_date=now)
        print(f"Total DAG runtime: {runtime} seconds")
```

This code retrieves the corresponding `DagRun` for a specified `dag_id` and `execution_date`. If such a run exists and has both a `start_date` and `end_date`, it calculates the total runtime by subtracting the former from the latter, again returning the result in seconds. If not, it will return None.

Finally, for a comprehensive view, you might want to leverage Airflow's UI and historical data views. While this doesn't directly involve code, these views provide excellent visualizations and aggregate data that can be very beneficial. Specifically, the 'Graph View' for a specific Dag Run in the UI shows the execution time of each task directly overlaid on the graph itself, and the 'Runs' view presents summarized data for the dag including overall runtime.

Beyond the provided snippets, for a more in-depth understanding, I recommend diving into the following resources:

1. **"Programming Apache Airflow" by J. D’Arcy:** This book provides a comprehensive overview of Airflow’s architecture and concepts, including detailed explanations of the metadata store and the scheduler, which are fundamental to understanding runtime.

2. **The official Apache Airflow documentation:** You'll find detailed descriptions of the models (such as `TaskInstance` and `DagRun`), operators, and core concepts. Pay particular attention to sections on monitoring and metadata, as these are critical for analyzing runtimes.

3. **The SQLAlchemy documentation:** Since Airflow uses SQLAlchemy for database interactions, understanding SQLAlchemy is paramount for writing efficient queries against the metadata database. This is relevant when attempting advanced analysis and customized data pulls.

In summary, understanding and determining the runtime of an airflow dag requires an approach that accounts for both task-level execution time and overall dag run duration. These are accessible through the Airflow API, the database, and visualizations provided by the interface. By methodically extracting and analyzing these metrics, we can better optimize airflow pipelines for scalability and efficiency.
