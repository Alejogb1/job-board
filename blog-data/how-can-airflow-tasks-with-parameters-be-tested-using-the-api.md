---
title: "How can Airflow tasks with parameters be tested using the API?"
date: "2024-12-16"
id: "how-can-airflow-tasks-with-parameters-be-tested-using-the-api"
---

Let’s tackle this head-on. I’ve spent more than a few cycles grappling with testing parameterized Airflow tasks, and it’s a nuanced area. The direct answer is: you can't directly test an individual task *execution* with specific parameters via the Airflow API the way you might naively expect, but you can simulate similar scenarios and verify your task logic through a combination of direct testing of the underlying task functions and indirectly through dag runs. This requires a shift in perspective on what "testing a task" really means in an Airflow context. Let me explain with some examples and lessons I’ve learned the hard way over the years, specifically related to testing parameterized task logic.

Firstly, it's vital to understand the separation between the task definition and the task *execution*. Airflow's api mostly concerns itself with the orchestration aspects, like scheduling, triggering, and monitoring dags and task runs. When you parameterize a task within a dag, you’re primarily setting up how the task will receive its dynamic context *during runtime*. This context, usually derived from macros, XCom values, or custom logic, isn’t directly something you can manipulate via the api to influence the *execution* of that specific task within the dag, especially outside of a Dag Run's context.

Think of it like this: your task definition specifies a blueprint, while the parameters are the ingredients at runtime. The api allows you to initiate the blueprint (the dag run), but not to individually alter the specific recipe at the point where a particular task executes, at least in any direct way. It's about testing your recipes properly and the interactions between components.

What we *can* do is test the *task function* separately with various parameters. This requires you to architect your airflow tasks such that the bulk of the core functionality exists in a separable, testable function or class method. I’ve seen this principle ignored time and time again, resulting in tangled, impossible-to-test task logic.

Let’s look at an example. Assume you have an airflow task that transforms data based on a parameter:

```python
# task_logic.py - The core functionality
def transform_data(input_data, transformation_type):
  """Performs a data transformation based on the type specified.

  Args:
      input_data: The data to transform.
      transformation_type: The type of transformation to apply.

  Returns:
      The transformed data.
  """
  if transformation_type == "uppercase":
    return [item.upper() for item in input_data]
  elif transformation_type == "lowercase":
    return [item.lower() for item in input_data]
  else:
      raise ValueError(f"Unsupported transformation type: {transformation_type}")

# airflow_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from task_logic import transform_data

with DAG(
  dag_id="parameterized_transformation_dag",
  start_date=datetime(2023, 1, 1),
  schedule=None,
  catchup=False
) as dag:

  transform_task = PythonOperator(
      task_id="transform_data_task",
      python_callable=transform_data,
      op_kwargs={"input_data": ["apple", "banana", "cherry"],
                 "transformation_type": "{{ dag_run.conf.get('transformation_type', 'uppercase') }}"},
  )
```

Here, `transform_data` contains the actual logic, taking input data and transformation type. It's separated from the airflow task. This makes it directly testable without any airflow machinery. We can thoroughly test it using a unit test setup.

```python
# test_task_logic.py
import unittest
from task_logic import transform_data

class TestTransformData(unittest.TestCase):

    def test_transform_data_uppercase(self):
        data = ["apple", "banana", "cherry"]
        transformed_data = transform_data(data, "uppercase")
        self.assertEqual(transformed_data, ["APPLE", "BANANA", "CHERRY"])

    def test_transform_data_lowercase(self):
        data = ["APPLE", "BANANA", "CHERRY"]
        transformed_data = transform_data(data, "lowercase")
        self.assertEqual(transformed_data, ["apple", "banana", "cherry"])

    def test_transform_data_invalid_type(self):
      data = ["apple", "banana", "cherry"]
      with self.assertRaises(ValueError):
        transform_data(data, "invalid_type")

if __name__ == '__main__':
    unittest.main()
```

This approach tests the task’s behavior exhaustively with different parameters. You use `unittest` framework (or `pytest`) and focus on what matters—the core logic within `transform_data`. This unit-level approach is critical; you can quickly confirm functionality without needing to spin up airflow.

Now, to approximate an “api test” of a task, you can trigger a dag run with specific configurations. You are not testing an isolated task run, but verifying the task works as expected in the context of a dag run given a set of provided parameters. This verifies the context resolution within airflow works correctly and that the entire system from dag run initiation to task completion functions properly.

Here's how you can trigger a dag run with configuration through the airflow api, and indirectly verify that the parameter passing works correctly:

```python
# test_dag_api.py
import requests
import json

AIRFLOW_API_URL = "http://localhost:8080/api/v1"
DAG_ID = "parameterized_transformation_dag"
AUTH = ("airflow", "airflow") # Use secure auth credentials in production

def trigger_dag_with_params(transformation_type):
  payload = {
      "conf": {
          "transformation_type": transformation_type
        }
    }
  response = requests.post(
        f"{AIRFLOW_API_URL}/dags/{DAG_ID}/dagRuns",
        json=payload,
        auth=AUTH
  )
  response.raise_for_status()
  return response.json()["dag_run_id"]

def check_dag_run_status(dag_run_id):
    response = requests.get(
        f"{AIRFLOW_API_URL}/dags/{DAG_ID}/dagRuns/{dag_run_id}",
         auth=AUTH
        )
    response.raise_for_status()
    run_data = response.json()

    state = run_data["state"]

    if state in ["success", "failed"]:
      return state
    else:
      return None

def verify_dag_execution(transformation_type):
  dag_run_id = trigger_dag_with_params(transformation_type)
  status = None
  while status not in ["success", "failed"]:
    status = check_dag_run_status(dag_run_id)

    if status is None:
      import time
      time.sleep(1) # add some polling interval
    else:
       break;

  if status == "success":
    print (f"Dag run for transformation type: {transformation_type} completed successfully.")
  else:
    raise Exception(f"Dag run for transformation type: {transformation_type} failed")


if __name__ == '__main__':
  verify_dag_execution("uppercase")
  verify_dag_execution("lowercase")
  try:
    verify_dag_execution("invalid_type") # This should result in failure due to the code logic we have implemented
  except Exception as e:
      print(f"Dag run with invalid parameter failed as expected: {e}")

```

In `test_dag_api.py`, I use `requests` to hit the Airflow API and trigger runs with varying `transformation_type` values. This will indirectly test our python operator that executes the transform_data function. The primary aim of the API test here is to make sure the overall dag run works correctly based on the configuration.

Keep in mind that the best practice is to combine this api-based testing with your regular unit-tests. Unit tests verify the core functionality, while these integration-style tests with the API verify the dag orchestration and parameter passing. You’ll need to set up authentication correctly for the api calls; the example uses basic auth with the default "airflow" credentials. Remember to handle api authentication appropriately based on your setup.

There are several excellent resources for learning more about proper testing strategies and airflow architecture. I'd strongly recommend "Testing in Python" by Daniel Roy Greenfeld, which, while not specific to Airflow, covers essential testing practices that are applicable. For a more in-depth dive into Airflow, check out "Data Pipelines with Apache Airflow" by Bas P. Geerdink.

In summary, directly testing parameterised task *executions* via the Airflow API isn’t feasible or necessary. Instead, shift your focus to unit testing the core function logic and integrate with API calls to ensure dag runs work correctly with provided config parameters. This separation is crucial for building a robust, maintainable, and testable Airflow environment, something I wish I had understood more fully early on.
