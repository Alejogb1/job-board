---
title: "How can I detect a file's existence and return False using FileSensor?"
date: "2024-12-23"
id: "how-can-i-detect-a-files-existence-and-return-false-using-filesensor"
---

Let's dive right into this— file existence checks can seem straightforward but often hide subtle complexities, especially within the context of orchestration frameworks like apache airflow, which utilizes the `filesensor`. The core problem you're facing, as i understand it, is that you want a sensor that actively monitors for the *absence* of a file and signals `false` when that file is still not present or has disappeared. Typical `filesensor` behavior, as many newcomers to airflow discover, is designed to be the inverse; it patiently waits until a file materializes before proceeding. The standard `filesensor` yields `true` when the file exists.

In a past project, i distinctly recall having to deal with a similar scenario. We were processing a series of files dropped by an external system into a shared location, but we needed an alert whenever a *scheduled* file failed to appear. The default `filesensor` simply wasn’t appropriate for that use case. Instead of passively waiting for a file to exist, we needed a sensor to actively verify its non-existence, and to trigger downstream tasks only when the file was still not there after a certain deadline.

The key realization here is that directly inverting the sensor's logic within its core functionality is typically not the intended use of airflow operators. Rather, airflow prefers a declarative, state-based approach rather than an imperative, directly control-oriented one. Hence, we must think about *how* to use existing building blocks to achieve this rather than trying to fundamentally alter them.

The method i've found to be most reliable involves leveraging the `filesensor`'s underlying mechanism but employing a wrapper function or a custom sensor that implements the desired behavior. We can use `filesensor` to test for file presence, but then we wrap this in an operation where the sensor would return `true` if file not present, `false` when it becomes present.
Let’s explore a few examples, starting with the simplest approach:

**Example 1: Using a PythonOperator and `os.path.exists` within a `filesensor` wrapper.**

This snippet uses the `pythonoperator` combined with `os.path.exists` to check the file's status *prior* to invoking the sensor. The logic is then inverted before sending the boolean result to airflow.

```python
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.filesystem import FileSensor
from datetime import datetime
import os

def check_file_absence(**kwargs):
    file_path = kwargs['file_path']
    if not os.path.exists(file_path):
       return True  # File not present
    else:
        return False # File is present

with DAG(
    dag_id='file_absence_check',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    # Dummy file creation for testing
    create_file_task = PythonOperator(
        task_id='create_dummy_file',
        python_callable=lambda: open('/tmp/test_file.txt', 'w').close() #create a dummy file
    )

    check_absence = PythonOperator(
        task_id='check_file_absence',
        python_callable=check_file_absence,
        op_kwargs={'file_path': '/tmp/test_file.txt'}
    )

    file_sensor = FileSensor(
        task_id='file_sensor_wait',
        filepath='/tmp/test_file.txt', #wait for the file to appear
        poke_interval=5,
    )
    
    create_file_task >> check_absence >> file_sensor

```

Here, the `check_absence` task uses `os.path.exists` to determine if the file exists. It returns `True` if the file is *absent* and `False` if present, before the `FileSensor` proceeds to check for existence. However, this will still not *directly* give you a sensor that waits for non-existence. Instead, this example provides a workaround, with an operator before a normal `FileSensor`.
The critical part is the logic within `check_file_absence`. This method serves as a gatekeeper, checking the presence or absence of file before letting control be delegated to the traditional `filesensor` that waits for it to show up.

**Example 2: Extending the `filesensor` to achieve this behavior**

This involves creating a subclass of the `filesensor` that alters the `poke` function behavior. It's more involved but directly provides the needed sensor type.

```python
from airflow.sensors.filesystem import FileSensor
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowSensorTimeout, AirflowSensorError
from datetime import datetime
from airflow.models import DAG
import os

class FileAbsenceSensor(FileSensor):
    @apply_defaults
    def __init__(self, *args, **kwargs):
        super(FileAbsenceSensor, self).__init__(*args, **kwargs)

    def poke(self, context):
      if not os.path.exists(self.filepath):
           return True
      else:
         return False

with DAG(
    dag_id='file_absence_sensor_custom',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    # Dummy file creation for testing
    create_file_task = PythonOperator(
        task_id='create_dummy_file',
        python_callable=lambda: open('/tmp/test_file_custom.txt', 'w').close() #create a dummy file
    )
    
    
    absence_sensor = FileAbsenceSensor(
        task_id='wait_for_file_absence',
        filepath='/tmp/test_file_custom.txt',
        poke_interval=5,
        timeout=60 # set a timeout, if needed
    )
    
    create_file_task >> absence_sensor
```

Here, `FileAbsenceSensor` extends the existing `FileSensor`. The `poke` method is overwritten to return `true` when the file is absent, `false` otherwise. This creates a custom sensor that directly meets the specification of the question.

**Example 3: Using `FileSensor` in combination with an `XCom` and `ShortCircuitOperator`**

This approach leverages the `filesensor` to *detect* existence, and then uses an xcom and a short-circuit operator to create the needed flow. This is arguably a more robust and idiomatic way to handle such conditional branching in airflow.

```python
from airflow.models import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.operators.short_circuit_operator import ShortCircuitOperator
from datetime import datetime
import os


def check_file_status(**kwargs):
    file_path = kwargs['file_path']
    return not os.path.exists(file_path) #return inverse of file exists
    

with DAG(
    dag_id='file_sensor_with_xcom',
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False
) as dag:

    # Dummy file creation for testing
    create_file_task = PythonOperator(
        task_id='create_dummy_file',
        python_callable=lambda: open('/tmp/test_file_xcom.txt', 'w').close() #create a dummy file
    )

    file_sensor = FileSensor(
        task_id='file_existence_sensor',
        filepath='/tmp/test_file_xcom.txt',
        poke_interval=5,
        mode='reschedule',  # Use reschedule mode for efficiency
        timeout=60,
    )

    check_absence_task = PythonOperator(
        task_id='check_file_absence_with_xcom',
        python_callable=check_file_status,
        op_kwargs={'file_path': '/tmp/test_file_xcom.txt'},
    )


    short_circuit = ShortCircuitOperator(
        task_id='short_circuit_if_file_absent',
        python_callable=lambda ti: ti.xcom_pull(task_ids='check_file_absence_with_xcom', key='return_value')
    )
    
    trigger_downstream_dag = TriggerDagRunOperator(
      task_id = 'trigger_downstream_dag',
      trigger_dag_id = 'example_dag'
    )

    create_file_task >> file_sensor >> check_absence_task >> short_circuit >> trigger_downstream_dag
```

Here we use a conventional `filesensor`, which waits for the file to appear, followed by a `PythonOperator` which returns `true` only when the file is not present using the `check_file_status` function and pushing that value to an `xcom`. A `ShortCircuitOperator` pulls the value and uses it for a conditional trigger. This setup uses the `filesensor` as it's intended while achieving the effect of checking for file absence with the short circuit operator.

For deeper insights into airflow's operator and sensor mechanisms, I’d recommend consulting the official Apache Airflow documentation. Specifically, the sections on `sensors`, `baseoperator`, and how `xcoms` function are essential for understanding these implementations. “Programming Apache Airflow” by Bas Harenslak and Julian de Ruiter is also an excellent resource for a comprehensive understanding of the platform’s architecture and features.

In essence, detecting file *absence* with airflow requires a slight shift in perspective. Rather than directly inverting a `filesensor`, we construct workflows that use `filesensor` capabilities, combined with conditional logic, to effectively check for non-existence. Each of these methods provides a viable solution, depending on the level of complexity you are comfortable with and the specific context of your workflow. My recommendation is to opt for the extended `filesensor` or the `filesensor` and `ShortCircuitOperator` implementation, as they directly address the problem statement without sacrificing airflow best practices and readability.
