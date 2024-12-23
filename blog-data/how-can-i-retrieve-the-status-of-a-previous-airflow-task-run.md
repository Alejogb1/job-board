---
title: "How can I retrieve the status of a previous Airflow task run?"
date: "2024-12-23"
id: "how-can-i-retrieve-the-status-of-a-previous-airflow-task-run"
---

Alright, let's tackle this. Retrieving the status of a previous Airflow task run is a bread-and-butter operation, something I've done countless times while debugging pipelines. It might seem straightforward at first glance, but there are nuances that can trip you up if you're not careful. I've seen many teams struggle with this, often reinventing the wheel instead of leveraging Airflow's provided tools. Let's break it down systematically.

First, we need to understand the underlying data structure. Airflow stores its metadata—including task instance states—in its backend database. Depending on your setup, this could be postgres, mysql, or something else. Accessing this directly is generally *not* recommended for routine operations; instead, we use Airflow's API or command-line tools for a consistent and safe interface. My experience has taught me that direct database queries, while sometimes tempting for a "quick fix," can lead to maintenance nightmares and can break things when Airflow's schema changes during upgrades.

Now, the specific approach you'll use often depends on *where* you’re trying to retrieve this status. Are you doing it within another task in your DAG, or are you accessing it from outside the airflow environment, like from a different script or service? The methods will differ, and choosing the right one is crucial for performance and maintainability.

Let's assume you're doing it within another task, which is the most common case. You'll want to leverage Airflow's xcom mechanism and the airflow API. I've found this to be the most robust approach for inter-task communication. XComs (cross-communication) allow tasks to pass small amounts of data between each other and are ideal for passing something like a task status.

Here's a concrete scenario I faced on a past project. I was building a data pipeline that required a conditional branch based on the status of a task from a previous DAG run. I couldn’t assume the previous DAG was always successful. We needed a mechanism to check that status programmatically. Here's how I accomplished it using python, within an operator context, using the airflow api:

```python
from airflow.models import DagRun
from airflow.utils import timezone
from airflow.operators.python import PythonOperator
from airflow import DAG

def get_previous_task_status(dag_id, task_id, **context):
    """Retrieves the status of a specified task in the previous DAG run."""
    # Find the last successful DAG run
    last_dag_run = DagRun.find(dag_id=dag_id,
                              state="success",
                              execution_date=context["dag_run"].execution_date - timezone.timedelta(days=1)) # assumes daily runs

    if not last_dag_run:
       print(f"No successful DAG run found for {dag_id} prior to {context['dag_run'].execution_date}")
       return None

    if len(last_dag_run)>1:
        print(f"Multiple successful DAG runs found for {dag_id} prior to {context['dag_run'].execution_date}. Using the most recent: {last_dag_run[-1].execution_date}")
        last_dag_run = last_dag_run[-1]
    else:
      last_dag_run = last_dag_run[0]



    ti = last_dag_run.get_task_instance(task_id=task_id)

    if ti:
        print(f"Task Instance found: {ti}")
        print(f"Task {task_id} status on {last_dag_run.execution_date}: {ti.state}")
        return ti.state
    else:
        print(f"Task instance {task_id} not found in previous dag run {last_dag_run.execution_date}")
        return None



with DAG(
    dag_id='status_check_dag',
    schedule=None,
    start_date=timezone.datetime(2023, 1, 1),
    catchup=False
) as dag:
    check_status = PythonOperator(
        task_id='check_previous_status',
        python_callable=get_previous_task_status,
        op_kwargs={'dag_id': 'previous_dag_id', 'task_id': 'target_task_id'} # Replace 'previous_dag_id' and 'target_task_id'
    )

```

This example demonstrates how to access a specific task instance's status. Crucially, it first fetches the *previous* successful dag run. This is essential if you want consistency and are relying on the output or effect of a previous run. It handles cases where no previous success exists and picks the most recent dag run if multiple exist. Remember to replace `'previous_dag_id'` and `'target_task_id'` with your specific DAG and task IDs. You should also update the `timezone.timedelta(days=1)` if your dag isn't running daily.

Now, let’s say you need this information outside of an airflow task context. In that case, accessing the Airflow API directly using the provided REST endpoint would be more appropriate. You might be building an external monitoring service, for example. Using a basic python requests library will suffice, but you will need to enable authorization for this api endpoint via the webserver configuration to secure the endpoint. Here is a sample snippet for that. It is crucial to protect this endpoint with authentication, lest you expose sensitive information.

```python
import requests
import json

def get_airflow_task_status_external(dag_id, task_id, execution_date, airflow_api_url, auth_tuple=None):

  url = f"{airflow_api_url}/dags/{dag_id}/dagRuns/{execution_date}/taskInstances/{task_id}"
  headers = {'Content-Type': 'application/json'}


  try:
        response = requests.get(url, headers=headers, auth=auth_tuple)
        response.raise_for_status() # Raise an exception for bad status codes
        data = response.json()
        print(f"Task Status Data: {data}")
        if 'state' in data:
          return data['state']
        else:
          print("No task state found in response")
          return None

  except requests.exceptions.RequestException as e:
       print(f"Error fetching status: {e}")
       return None
  except json.JSONDecodeError as e:
        print(f"Error decoding json: {e}")
        return None

# Example usage with authentication (replace with your actual values)
# the auth tuple can be basic auth e.g. ('your_username','your_password') or token auth
# ensure you have configured authentication on the airflow webserver

if __name__ == '__main__':
  airflow_api_url = "http://localhost:8080/api/v1" # update with your airflow instance
  dag_id = "example_bash_operator"
  task_id = "run_this"
  execution_date = "2024-04-20T12:00:00+00:00"  #  format as YYYY-MM-DDTHH:MM:SS+00:00
  auth = ('airflow','airflow')  #Example authentication tuple, update this.


  status = get_airflow_task_status_external(dag_id, task_id, execution_date, airflow_api_url, auth_tuple = auth)

  if status:
    print(f"Retrieved task status : {status}")
  else:
    print ("Could not retrieve status.")

```

This snippet shows how to fetch the task status via Airflow's REST API. You will need to provide the dag_id, task_id, and the *specific* execution_date. The execution date requires precision. In addition, remember to include the necessary authentication details. Handling errors is important, which this code takes into consideration, but you may need to handle errors specific to your setup.

Finally, what if you need to retrieve the status of multiple tasks, or a complete historical view for auditing? While the previous examples work for specific task statuses, for larger datasets, querying Airflow's metadata directly with the help of the airflow CLI is a more efficient approach. This allows you to formulate more complex queries. Here's how you would do that using the command-line interface combined with a bit of Python to process the results:

```python
import subprocess
import json
from datetime import datetime

def get_historical_task_statuses(dag_id, task_ids, start_date, end_date):
    """Retrieves the status of specified tasks within a time range."""

    start_date_str = start_date.isoformat()
    end_date_str = end_date.isoformat()
    task_str = ",".join(task_ids)

    cmd = [
        "airflow",
        "tasks",
        "list",
        dag_id,
        "--start-date",
        start_date_str,
        "--end-date",
        end_date_str,
         "--task-regex",
         f"^{task_str}$",
        "--output",
        "json",
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Error executing airflow cli: {stderr.decode()}")
        return None

    try:
      task_data = json.loads(stdout.decode())
      return task_data
    except json.JSONDecodeError as e:
      print(f"Error decoding json from airflow cli: {e}")
      return None
    except Exception as e:
       print(f"An unexpected error occurred {e}")
       return None


if __name__ == "__main__":
    dag_id = "example_bash_operator"
    task_ids = ['run_this', 'also_run_this'] # list of task ids
    start_date = datetime(2024,4,15)  # your start date
    end_date = datetime(2024, 4, 20)  # your end date

    task_statuses = get_historical_task_statuses(dag_id, task_ids, start_date, end_date)
    if task_statuses:
        for task_instance in task_statuses:
          print(f"Task id: {task_instance['task_id']}, status: {task_instance['state']}, run id: {task_instance['run_id']}, execution date {task_instance['execution_date']}")
    else:
        print("Could not retrieve historical statuses")
```
This python function utilizes `subprocess` to interact with the airflow cli, requesting a historical view of the task statuses for the tasks specified, for the given date range. This is an efficient way to perform bulk retrievals. You must have the `airflow` cli in your environment's PATH for this to function properly. The output is then parsed from the json output returned by the CLI.

For further understanding, I highly recommend diving into the Airflow documentation's sections on the REST API, XComs, and the command-line interface. Reading through the source code of the `airflow.models.dagrun` and `airflow.models.taskinstance` classes can also provide deeper insight. Additionally, *“Programming Apache Airflow”* by J.J. Berenguer is a very helpful book, as is the official Apache Airflow documentation itself.

In summary, retrieving a previous task run's status in Airflow is a common requirement that can be approached in several ways, each with its own pros and cons. For most within-DAG operations, using xcom with the airflow api is a good default. External services can access the REST API, and complex data fetches should utilise the command-line interface combined with python. Choose the tool that best suits your particular use case and be aware of best practices to avoid pitfalls down the line.
