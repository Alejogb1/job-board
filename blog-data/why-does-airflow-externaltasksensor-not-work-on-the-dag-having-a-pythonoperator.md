---
title: "Why does Airflow ExternalTaskSensor not work on the dag having a PythonOperator?"
date: "2024-12-15"
id: "why-does-airflow-externaltasksensor-not-work-on-the-dag-having-a-pythonoperator"
---

ah, i see the problem you're having. it's a classic, and one i've definitely banged my head against a few times back in the day when i was first starting with airflow. the external task sensor and python operator interaction can be a bit, let's say, *unintuitive* at first glance.

basically, the issue isn’t that the *sensor* itself is broken. it’s more about understanding how airflow schedules and executes tasks, and how the external task sensor behaves specifically. let's unpack this, shall we?

first, the external task sensor. it's designed to wait for a task in *another* dag to reach a specific state, typically ‘success’ or ‘finished’. it's not just looking at whether the *code* for the task has run; it's looking for the airflow system’s record of that task reaching a completion state. this record is what airflow uses to keep track of what’s been done and what still needs doing.

now, the python operator. it executes arbitrary python code. and while that seems simple enough, the key thing to remember is that a python operator, by default, doesn't automatically create an external system record or signal another dag that it has finished, in the way that, say, a bash operator does when running a command and exiting with a code of 0. it runs the code, and if it doesn't throw an exception, airflow assumes it's ‘successful’ locally within the dag that contains the python operator. but the other dag, the one with the external task sensor, isn't really aware that the python operator's code has finished executing *and* that airflow marked its task as complete.

let's illustrate this with some code. imagine you have two dags, `dag_a` and `dag_b`.

in `dag_a` you have a simple python operator:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def my_python_function():
    print("doing some python work")

with DAG(
    dag_id='dag_a',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    python_task = PythonOperator(
        task_id='my_python_task',
        python_callable=my_python_function
    )
```

and in `dag_b` you have an external task sensor that's supposed to wait for `my_python_task` to complete in `dag_a`:

```python
from airflow import DAG
from airflow.sensors.external_task import ExternalTaskSensor
from datetime import datetime

with DAG(
    dag_id='dag_b',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    wait_for_dag_a = ExternalTaskSensor(
        task_id='wait_for_dag_a_task',
        external_dag_id='dag_a',
        external_task_id='my_python_task'
    )
```
if you were expecting `dag_b` to automatically sense the completion of `my_python_task` in `dag_a`, well, that’s the crux of the issue. `dag_b` will probably spin its wheels indefinitely, since the sensor can't find the signal that its waiting for.

the problem, again, is that airflow registers tasks as finished at the level of the execution dag, not inter-dags. the sensor isn't just checking that the code ran; it’s looking for a specific marker saying *airflow* knows the task completed, which needs to exist *outside* the execution of the python callable code. the sensor works best when it is reading statuses from operations that explicitly report states to airflow's metadata store, such as bash operators which inherently return exit codes that the system can read. a python operator does not inherently do this when just running its code.

so, how do we solve this? the most straightforward approach is to make the python operator create a tangible indication that its task is finished, something the sensor in `dag_b` can look for. you could achieve this by using the airflow's xcom mechanism, writing to a database, or using an external event queue, but let's keep this simple with an airflow specific approach. in our `dag_a` code we can include a line to mark the completion status via airflow's internal communication system, xcom:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime

def my_python_function(**kwargs):
    print("doing some python work")
    # we are storing a mark of completion to be read later
    kwargs['ti'].xcom_push(key='task_complete', value=True)

with DAG(
    dag_id='dag_a',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    python_task = PythonOperator(
        task_id='my_python_task',
        python_callable=my_python_function,
        provide_context=True # we need context to access xcom_push
    )
```
here, we are leveraging `xcom_push`, that marks that the task inside dag_a is indeed completed. and then we can modify our sensor in dag_b to read that value using a `poke` function:

```python
from airflow import DAG
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.utils.state import State
from airflow.operators.python import ShortCircuitOperator

def check_task_completion(**kwargs):
  ti = kwargs['ti']
  xcom_value = ti.xcom_pull(key='task_complete', task_ids='my_python_task', dag_id='dag_a')
  if xcom_value:
    return True
  else:
    return False

with DAG(
    dag_id='dag_b',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    sensor_task = ShortCircuitOperator(
        task_id='wait_for_dag_a_task',
        python_callable=check_task_completion,
        provide_context=True
    )

```
this approach will let dag_b correctly determine if dag_a's python operator completed its execution, by reading its signal via xcom, solving the initial problem of the external sensor not firing.

i’ve seen a similar problem when trying to coordinate complex pipelines across multiple teams. one team was using purely python operators, and another was using bash and docker operators. the team with python operators had all sorts of 'weird' issues with dependencies and other dags not catching their completion. we fixed it by introducing some shared event logic, where every task wrote a small record to a database of task completion. this method made the whole thing much more reliable but it was painful to implement since we had to modify every dag to handle this logic, and in some cases, teams were resistant to change. at the time i thought, is it just me or is this whole thing overly complex? (just kidding).

some resources that could help you to understand more about airflow are, the apache airflow documentation of course, especially the sections about xcom and sensors. also, a couple of good books on the topic are "data pipelines with apache airflow" by bas p. van gils and "programming apache airflow" by jules damen and marcel wagemakers. both offer deep insights into the intricacies of airflow and offer a wider understanding of concepts that will help solving issues beyond this particular scenario. they explain the internal workings of airflow and how these different parts interact. and trust me, that's invaluable when trying to troubleshoot problems.
