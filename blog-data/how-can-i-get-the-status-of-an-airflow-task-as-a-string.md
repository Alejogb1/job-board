---
title: "How can I get the status of an airflow task as a string?"
date: "2024-12-23"
id: "how-can-i-get-the-status-of-an-airflow-task-as-a-string"
---

, so grabbing the status of an airflow task, eh? It’s something I’ve had to tackle multiple times, and it definitely has its nuances. Early on in my career, I remember struggling with this when we were migrating a rather sizable legacy system to an airflow-centric setup. We needed to programmatically monitor tasks, report on their success or failure, and this task status thing became surprisingly crucial. It wasn't as simple as just grabbing a single value; you had to consider various contexts and conditions.

The core issue, as you've likely found, is that airflow's task status isn't directly exposed as a simple string by default. It's usually represented by an enum within the airflow metadata database, typically postgres, and then manipulated programmatically in airflow's codebase. So, what we need to do is bridge that gap, pulling information from airflow's internal structures and transforming it into a human-readable string, usable in logs, alerts, or other reporting systems.

To get this done properly, we’ll be primarily interacting with the airflow dagrun object, specifically its `task_instances` attribute. Each `task_instance` represents a run of a single task within a specific dag run, and these hold the key to the status. The status attribute for each task instance is an enum, which isn't directly useful, therefore, we must convert them to string.

Here's a Python function illustrating this process:

```python
from airflow.models import DagRun, TaskInstance
from airflow.utils.state import State

def get_task_status_string(dag_id: str, run_id: str, task_id: str) -> str:
    """
    Retrieves the status of a specific Airflow task as a string.

    Args:
        dag_id: The id of the DAG.
        run_id: The id of the DagRun.
        task_id: The id of the task.

    Returns:
        A string representing the task status. Returns "UNKNOWN" if not found.
    """
    try:
      dag_run = DagRun.find(dag_id=dag_id, run_id=run_id)[0]
      if not dag_run:
         return "UNKNOWN DAG RUN"

      for task_instance in dag_run.get_task_instances():
        if task_instance.task_id == task_id:
          return task_instance.current_state()
      return "TASK NOT FOUND"
    except Exception as e:
      return f"ERROR: {e}"
```

Let's break down what’s happening here. The function takes the dag id, run id, and task id as arguments. First, it attempts to find the specific dag run. Following this, we loop through each `task_instance` within that dag run. When we find the correct task id, we call `task_instance.current_state()` this returns the state of the task, and this function already returns the state as a human-readable string, such as 'success', 'running' etc. If no task matches the given `task_id`, the function will return `TASK NOT FOUND`. Finally, we include a `try-except` block to handle potential errors and log them if necessary for debugging.

Now, this function covers the most common use case, but what if you need to get status information for all the tasks in a DAG run? This is common for monitoring dashboards or aggregate reporting. Here's an adjusted code snippet to handle that:

```python
from airflow.models import DagRun
from airflow.utils.state import State

def get_all_task_statuses(dag_id: str, run_id: str) -> dict:
    """
    Retrieves the status of all tasks in a given DAG run.

    Args:
        dag_id: The id of the DAG.
        run_id: The id of the DagRun.

    Returns:
        A dictionary where keys are task ids and values are status strings.
    """
    status_dict = {}
    try:
        dag_run = DagRun.find(dag_id=dag_id, run_id=run_id)[0]
        if not dag_run:
            return {"error": "DAG RUN NOT FOUND"}
        for task_instance in dag_run.get_task_instances():
            status_dict[task_instance.task_id] = task_instance.current_state()
    except Exception as e:
        status_dict["error"] = f"An error occurred {e}"
    return status_dict
```

In this version, the function gathers statuses for every task instance and puts them into a dictionary. The dictionary’s keys are task ids, and the values are their corresponding status strings. If any problems occur, instead of returning 'UNKNOWN', the errors are captured in the `status_dict` as an error field with the exception as the value.

It is also essential to understand that a task status is not an immutable, it can change across runs, and a task may run multiple times in a single dagrun. A task can be retried or re-scheduled and for any specific run, the `current_state` function will return the latest state based on the execution information. Therefore, depending on what you’re aiming for you might need a function to get the status for a specific task try. Let’s say, you want to debug a specific task and want to see only the logs from a single attempt and the status associated with it, here's a function that would help in that case:

```python
from airflow.models import DagRun, TaskInstance
from airflow.utils.state import State

def get_task_attempt_status_string(dag_id: str, run_id: str, task_id: str, try_number: int) -> str:
    """
    Retrieves the status of a specific Airflow task attempt as a string.

    Args:
        dag_id: The id of the DAG.
        run_id: The id of the DagRun.
        task_id: The id of the task.
        try_number: The number of the task try.

    Returns:
        A string representing the task status for specific try. Returns "UNKNOWN" if not found.
    """
    try:
      dag_run = DagRun.find(dag_id=dag_id, run_id=run_id)[0]
      if not dag_run:
         return "UNKNOWN DAG RUN"

      for task_instance in dag_run.get_task_instances():
        if task_instance.task_id == task_id and task_instance.try_number == try_number:
          return task_instance.current_state()
      return "TASK TRY NOT FOUND"
    except Exception as e:
      return f"ERROR: {e}"
```

This function is similar to the first example, however, in this version we are filtering the task instances based on both the `task_id` and the specific `try_number`. If the task instance with the matching id and try number is found then the function will return the state of that specific task try.

When working with airflow status information, it's also critical to consider where and how you'll use the results of functions like these. Direct database access outside of the airflow scheduler/webserver process is not recommended and should be done with caution, prefer using airflow APIs whenever possible. Also, if these functions will be used frequently, caching strategies should be implemented to reduce load on the airflow meta store.

For delving deeper into airflow's internals, I’d recommend several key resources. The core source code is, of course, invaluable. You can find it on GitHub under the apache/airflow repository. For a structured look into airflow’s data model, I would suggest consulting the documentation's section on the database schema; this provides detailed insight into the structure of tables like `dag_run`, and `task_instance`. Additionally, for an excellent understanding of dag processing and task scheduling, the book “Data Pipelines with Apache Airflow” by Bas Harenslak and Julian Rutger also gives a strong overview. Lastly, to fully grasp how airflow manages task states internally, the `airflow.utils.state` module is worth examining.

In summary, while airflow doesn’t immediately give you task status as a string, using task instances and state enums provides a straightforward method to retrieve this information programmatically. Just be mindful of where your queries originate and use caching to avoid performance hits, and you will be in good shape.
