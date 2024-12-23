---
title: "How can an Airflow DAG be scheduled to run immediately following the completion of a previous DAG?"
date: "2024-12-23"
id: "how-can-an-airflow-dag-be-scheduled-to-run-immediately-following-the-completion-of-a-previous-dag"
---

Okay, let's talk about chaining airflow dags. I've seen this requirement surface more times than I can count, typically in situations where data pipelines have a strong dependency on sequential processing. It's not uncommon to have a ‘data ingestion’ dag that feeds into a downstream ‘data transformation’ dag, for instance. Orchestrating this in a reliable and maintainable way is key. There are a few approaches, each with its own nuances and best-use cases, but let’s dive into a couple of common methods and illustrate with some examples.

The core problem, really, is ensuring one dag starts only after another has successfully completed. Simply relying on a fixed schedule for the second dag is brittle because varying runtimes of the first dag can cause overlaps or missed dependencies. We need a more dynamic approach.

One effective method uses the `TriggerDagRunOperator`. This operator, when placed at the end of the first dag, explicitly initiates the second dag upon successful completion. It's a rather straightforward way to implement direct dependencies. However, it does have limitations, specifically around how it handles failures in the downstream dag – which I’ll touch on shortly.

Let’s imagine I have worked on a pipeline. I had a dag named `dag_ingest` which reads data from an api and dumps it into a data lake. I had another dag called `dag_process` which takes the output of `dag_ingest` and performs further processing. Let's see how the `TriggerDagRunOperator` helped bridge these two dags:

```python
# dag_ingest.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.dagrun import TriggerDagRunOperator
from datetime import datetime

def ingest_data():
    # Code to simulate fetching and writing data
    print("Simulating data ingestion...")
    return

with DAG(
    dag_id="dag_ingest",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=["example"],
) as dag:
    ingest_task = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

    trigger_process_dag = TriggerDagRunOperator(
        task_id="trigger_dag_process",
        trigger_dag_id="dag_process",  # this refers to the dag_id of the DAG to trigger
    )

    ingest_task >> trigger_process_dag
```

```python
# dag_process.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def process_data():
  # Code to simulate further processing
  print("Simulating data processing...")

with DAG(
    dag_id="dag_process",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=["example"],
) as dag:
    process_task = PythonOperator(
        task_id="process_data",
        python_callable=process_data
    )
```

In this example, after `ingest_task` successfully completes, the `trigger_process_dag` task will launch an instance of `dag_process`. Notice the `schedule=None` in both dags, indicating these are intended to be triggered, rather than relying on a cron schedule.

Now, while this approach is relatively simple to implement and can be effective, it does present a slight difficulty in handling cascading failures. Specifically, if `dag_process` fails after being triggered, it doesn't automatically flag `dag_ingest` as failed in a direct, tightly coupled sense. This is an area to consider in robust production systems. Furthermore, this does not directly show the dependency of the dags in the airflow ui.

A more sophisticated and robust method involves using Airflow’s `ExternalTaskSensor`. Unlike the `TriggerDagRunOperator`, which directly *launches* another dag, the `ExternalTaskSensor` *waits* for a task in a different dag to reach a specific state (usually success). This has the benefit of creating a clearer dependency relationship, and is more resilient to various scenarios such as failed downstream dags. If the upstream dag fails, the sensor in downstream dag will also fail, indicating that upstream dag needs to be fixed or re-ran first. Let’s adapt our previous example:

```python
# dag_ingest_sensor.py (modified ingest dag)
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def ingest_data():
    # Code to simulate fetching and writing data
    print("Simulating data ingestion...")
    return

with DAG(
    dag_id="dag_ingest_sensor",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=["example"],
) as dag:
    ingest_task = PythonOperator(
        task_id="ingest_data",
        python_callable=ingest_data,
    )

```

```python
# dag_process_sensor.py (modified process dag)
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime

def process_data():
  # Code to simulate further processing
  print("Simulating data processing...")

with DAG(
    dag_id="dag_process_sensor",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=["example"],
) as dag:

    wait_for_ingest = ExternalTaskSensor(
        task_id="wait_for_ingest_dag",
        external_dag_id="dag_ingest_sensor", # dag id of dag to wait on
        external_task_id="ingest_data", # task id of task to wait for
        poke_interval=60, # check every 60 seconds
        timeout=60 * 60 * 2, # wait for 2 hours maximum
    )

    process_task = PythonOperator(
        task_id="process_data",
        python_callable=process_data
    )

    wait_for_ingest >> process_task
```

In the `dag_process_sensor` dag, the `ExternalTaskSensor` is configured to monitor the `ingest_data` task within the `dag_ingest_sensor` dag. Only once the specified task is marked as successful will the downstream `process_task` in `dag_process_sensor` begin. This method creates a dependency without directly triggering the downstream dag but rather waits for it to be completed. This is useful especially in a situation where the upstream dag might need to run more than once. For example, if the `dag_ingest_sensor` failed for some reason and re-ran, the `dag_process_sensor` will wait for the second successful completion of the `ingest_data` task.

Another method that can be used, is using an Airflow Variable to create conditional DAG runs. While this isn't a direct dag-to-dag dependency, it's valuable in scenarios that require more control over sequencing.

```python
# first_dag_var.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime

def update_variable():
    # set a variable
    Variable.set("var_trigger_second_dag", "ready")
    print("Variable set to trigger second dag.")


with DAG(
    dag_id="first_dag_var",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=["example"],
) as dag:
    set_variable_task = PythonOperator(
        task_id="set_variable",
        python_callable=update_variable,
    )
```

```python
# second_dag_var.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.base import BaseSensorOperator
from airflow.utils.state import State
from airflow.models import Variable
from datetime import datetime

class VariableSensor(BaseSensorOperator):
    def __init__(self, variable_name, expected_value, **kwargs):
        super().__init__(**kwargs)
        self.variable_name = variable_name
        self.expected_value = expected_value

    def poke(self, context):
        var = Variable.get(self.variable_name, default_var=None)
        if var == self.expected_value:
            return True
        return False

def process_data():
    print("Data processing started after sensing variable")

with DAG(
    dag_id="second_dag_var",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=["example"],
) as dag:

    wait_for_variable_task = VariableSensor(
        task_id="wait_for_variable",
        variable_name="var_trigger_second_dag",
        expected_value="ready",
        poke_interval=60,
    )

    process_task = PythonOperator(
        task_id="process_data",
        python_callable=process_data,
    )

    wait_for_variable_task >> process_task
```

Here, once `first_dag_var` completes, it sets an Airflow variable. The `second_dag_var` utilizes a custom `VariableSensor` that waits for this variable to match a specific value, which will then proceed with further processing in that dag.

In choosing the correct method, consider factors such as the desired level of coupling between dags, how you want to handle failure scenarios, and your team's familiarity with the concepts. I've found the `ExternalTaskSensor` is generally a safer, more resilient choice for most production scenarios, but simpler pipelines may benefit from the directness of the `TriggerDagRunOperator`. Regardless of your choice, it's always good practice to meticulously log all these operations to ensure observability.

For further learning, I recommend “Data Pipelines with Apache Airflow” by Bas P. Harenslak and Julian Rutger de Ruiter. It's a deep dive into airflow concepts and provides excellent insight into best practices. Also, you could review the apache airflow documentation in general, specifically the documentation on `TriggerDagRunOperator` and `ExternalTaskSensor`, it's thorough and provides concrete examples. Finally, the airflow examples repository on github contains implementations of most common operators which might also provide helpful insight. Remember, the best strategy will be tailored to your specific needs, but these methods should give you a robust toolkit for sequencing your Airflow dags effectively.
