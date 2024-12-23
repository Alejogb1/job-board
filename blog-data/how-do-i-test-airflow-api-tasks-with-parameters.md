---
title: "How do I test Airflow API tasks with parameters?"
date: "2024-12-23"
id: "how-do-i-test-airflow-api-tasks-with-parameters"
---

Alright,  Testing Airflow API tasks, particularly when they involve parameters, can initially feel a bit like navigating a maze. I've definitely spent my share of late nights debugging these scenarios. The good news is, with the right approach, it's entirely manageable and, dare I say, even elegant. The trick lies in understanding how Airflow manages parameters and how best to simulate their behavior in a testing environment.

The core challenge stems from the fact that API tasks within Airflow often rely on a context populated during runtime. Parameters that define the behavior of these tasks are frequently injected through this context. So, directly instantiating your task within a unit test often leads to a 'missing context' error or similar frustrations. We need to bypass this by creating a controllable testing context.

When I first encountered this, on a project migrating a series of legacy data pipelines to Airflow, the initial inclination was to try and force the full Airflow context in our tests. That turned into an incredibly brittle solution, highly dependent on the specific Airflow configuration of that environment, which as we all know, fluctuates. The breakthrough came when we realized that we should be focusing on testing the *logic* within our API tasks, not the Airflow framework itself.

The standard practice is to decouple the core logic that interacts with an external API from the Airflow task definition. You achieve this by encapsulating that API call logic into a separate, callable function. This gives you several advantages:

1.  **Testability:** The core API call function becomes trivially testable since it no longer relies on the Airflow context. You can pass in your parameters directly and observe the outputs.
2.  **Reusability:** The function is now independent and can potentially be used in other contexts.
3.  **Clarity:** Your Airflow task remains a lightweight wrapper that calls the core function, significantly improving readability and maintainability.

Let me illustrate with some code examples, remembering this is based on actual scenarios I've worked through:

**Example 1: Defining the core API call function**

Let's say you're interacting with a hypothetical API that fetches data based on a date range and a client id.

```python
# api_caller.py
import requests
import json

def fetch_api_data(start_date, end_date, client_id, api_url):
  """Fetches data from a hypothetical API."""
  headers = {'Content-Type': 'application/json'}
  payload = {
      'start_date': start_date,
      'end_date': end_date,
      'client_id': client_id
  }
  response = requests.post(api_url, headers=headers, data=json.dumps(payload))
  response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
  return response.json()
```

This `fetch_api_data` function takes parameters directly, makes the API call, and returns the data. Crucially, it doesn't know or care that it might be running within an Airflow task.

**Example 2: The Airflow task definition**

Now, let's see how this function integrates within an Airflow task definition using a python callable operator.

```python
# dags/my_api_dag.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from api_caller import fetch_api_data

def api_task_wrapper(**kwargs):
  """Wrapper function to pass task parameters to the fetch_api_data function."""
  ti = kwargs['ti']
  api_url = kwargs['dag_run'].conf.get('api_url') # Retrieve from DAG configuration
  start_date = ti.xcom_pull(task_ids='extract_start_date', key='start_date')
  end_date = ti.xcom_pull(task_ids='extract_end_date', key='end_date')
  client_id = kwargs['dag_run'].conf.get('client_id') # Retrieve from DAG configuration

  data = fetch_api_data(start_date, end_date, client_id, api_url)
  ti.xcom_push(key='api_response', value=data)


with DAG(
    dag_id='api_parameter_example',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:

  extract_start_date = PythonOperator(
        task_id = 'extract_start_date',
        python_callable=lambda **kwargs:  kwargs['ti'].xcom_push(key='start_date', value='2024-01-01'),
        provide_context = True
    )
  extract_end_date = PythonOperator(
        task_id = 'extract_end_date',
        python_callable=lambda **kwargs:  kwargs['ti'].xcom_push(key='end_date', value='2024-01-07'),
        provide_context = True
    )
  api_task = PythonOperator(
    task_id = 'fetch_api_data',
    python_callable=api_task_wrapper,
    provide_context = True
  )

  extract_start_date >> extract_end_date >> api_task
```

Notice the use of `ti.xcom_pull` and `dag_run.conf.get`. These are typical methods for passing information between tasks and retrieving configurations, and we're still wrapping our core function so that it doesn't need to know about this airflow specific context. We define the parameters and then forward them to the core `fetch_api_data` function.

**Example 3: The Unit Test**

Here's how you would unit test the core `fetch_api_data` function:

```python
# tests/test_api_caller.py
import unittest
from unittest.mock import patch
import json
from api_caller import fetch_api_data
import requests

class TestApiCaller(unittest.TestCase):
  @patch('requests.post')
  def test_fetch_api_data_success(self, mock_post):
      mock_response = unittest.mock.Mock()
      mock_response.json.return_value = {'status': 'success', 'data': [{'item1': 'value1'}, {'item2': 'value2'}]}
      mock_response.raise_for_status.return_value = None
      mock_post.return_value = mock_response

      start_date = '2024-01-01'
      end_date = '2024-01-07'
      client_id = 'test_client'
      api_url = "https://testapi.com/data"
      result = fetch_api_data(start_date, end_date, client_id, api_url)
      self.assertEqual(result['status'], 'success')
      self.assertEqual(len(result['data']), 2)

      mock_post.assert_called_once()
      call_arg = mock_post.call_args.kwargs["data"]
      call_arg_dict = json.loads(call_arg)
      self.assertEqual(call_arg_dict["start_date"], start_date)
      self.assertEqual(call_arg_dict["end_date"], end_date)
      self.assertEqual(call_arg_dict["client_id"], client_id)

  @patch('requests.post')
  def test_fetch_api_data_failure(self, mock_post):
        mock_response = unittest.mock.Mock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Test error")
        mock_post.return_value = mock_response
        with self.assertRaises(requests.exceptions.HTTPError):
          fetch_api_data("2024-01-01", "2024-01-07", "test_client", "https://testapi.com/data")
```

Here, we are using `unittest` and `unittest.mock` to mock the http request and simulate various API response scenarios. We can ensure the function correctly handles different responses without ever running an Airflow dag. We assert not only the final result, but also verify that the correct parameters were used in the underlying http call.

For deeper dive into the principles of testability, I would strongly recommend reading "Working Effectively with Legacy Code" by Michael Feathers. Although the book focuses on legacy code, the principles of decoupling logic, dependency injection, and test-driven development are universally applicable. For a thorough understanding of API testing, explore resources like “API Design Patterns” by JJ Geewax. This book can enhance your ability to develop robust and testable API client integrations.

The key takeaway is: don't try to test the entire Airflow context in your unit tests. Instead, isolate the logic that directly interacts with external APIs, make it parameter-driven, and then thoroughly test that isolated logic. Your airflow tasks are then simple wrappers. This methodology makes it infinitely easier to verify your API tasks with parameters in a robust, maintainable, and efficient way.
