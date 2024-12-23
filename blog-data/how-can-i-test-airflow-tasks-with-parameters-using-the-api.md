---
title: "How can I test Airflow tasks with parameters using the API?"
date: "2024-12-23"
id: "how-can-i-test-airflow-tasks-with-parameters-using-the-api"
---

Right then, let's tackle this. Testing Airflow tasks, particularly those that rely on parameters, via the api—it’s a crucial but often overlooked aspect of robust dag development. I’ve spent more than a few late nights debugging unexpected behaviors in pipelines simply because the parameter handling wasn't thoroughly validated. I've learned the hard way, so let's try to avoid that for you.

Essentially, what we're aiming for is a systematic way to ensure that our tasks behave as expected across the spectrum of potential input values. Simply running the entire dag and visually inspecting the output isn't sufficient, especially when things get complex. We need to isolate tasks and simulate real-world conditions with specific parameters. The Airflow rest api provides a solid mechanism for doing just that, although it’s not always straightforward to piece together how to leverage it effectively.

The core concept hinges on triggering dag runs with specific *configuration overrides*, rather than just blindly launching a dag. These configuration overrides are what allow us to feed in the parameters directly to the task instance. You interact with the api, sending a payload that includes a `conf` object. This object acts as a dictionary of sorts, allowing you to pass in parameters that your dag uses via the `dag_run.conf` dictionary. It’s critical to remember that the `dag_run.conf` parameters are treated as strings, so your task code needs to handle appropriate conversions to other datatypes, if needed.

Let's walk through a practical scenario. Imagine a simplified version of something I actually worked on: a dag that processes data from different data sources. The data source is determined by a configuration parameter. We have a task, let's call it `process_data`, which expects a parameter named `source_id`.

First, we need to be able to trigger a dag run via the api. Assume our dag’s id is `my_data_pipeline`. Here is how you might make an api call with Python using the `requests` library. I’m using basic authentication, but your situation could involve tokens or other authentication.

```python
import requests
import json

def trigger_dag_with_params(dag_id, params, airflow_url, auth_tuple):
    url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"
    headers = {'Content-Type': 'application/json'}
    payload = json.dumps({'conf': params})

    try:
        response = requests.post(url, headers=headers, data=payload, auth=auth_tuple)
        response.raise_for_status()  # Raise an exception for bad status codes
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error triggering dag: {e}")
        return None


if __name__ == '__main__':
    airflow_url = 'http://localhost:8080' # Replace with your actual airflow url
    auth_tuple = ('airflow', 'airflow') # Replace with your credentials
    dag_id = 'my_data_pipeline'
    params = {'source_id': '12345'}
    result = trigger_dag_with_params(dag_id, params, airflow_url, auth_tuple)
    if result:
        print(f"Successfully triggered dag. Dag run id: {result['dag_run_id']}")
    else:
        print("Failed to trigger the dag.")
```

This snippet will fire off the `my_data_pipeline` dag, and crucially, it will pass the parameter `source_id` with the value of '12345' via the `conf` dictionary. Now, let's move on to how we’d access that inside the dag. I've seen new developers often confuse how to access these parameters, and they sometimes resort to overly complicated methods.

Here's a simplified dag that accesses and utilizes that parameter:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def process_data_task(dag_run_conf, **kwargs):
    source_id = dag_run_conf.get('source_id')
    if source_id:
        print(f"Processing data from source ID: {source_id}")
        # In a real scenario, you would use this to select/filter your processing logic
        # or target specific files / tables for this given id
    else:
        print("No source id provided in dag run configuration.")
        #Handle the case of missing configuration
    return f"Data processing complete for source_id : {source_id}"

with DAG(
    dag_id='my_data_pipeline',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    process_task = PythonOperator(
        task_id='process_data',
        python_callable=process_data_task,
        op_kwargs={'dag_run_conf': '{{ dag_run.conf }}'} # Pass dag_run.conf as an op_kwarg
    )
```

Note that I am explicitly retrieving `dag_run.conf` and passing it as an argument to the `process_data_task` by using `op_kwargs`. The `process_data_task` can then extract its `source_id` argument. This provides direct access to the conf dictionary supplied through the api. We are getting the conf dictionary through the context (via jinja templates `{{ dag_run.conf }}`) and passing it as argument to the python function. This is a common pattern that I use in many production dags.

Now, let's get into the heart of the testing problem. If you are testing with varied inputs, you don’t want to trigger a new DAG run every single time, or have to keep doing manual requests. You want to integrate these kinds of api calls into your testing framework. A simple unit test might look like this using Python's unittest library.

```python
import unittest
from unittest.mock import patch, Mock
import json

class TestDagParameters(unittest.TestCase):

    @patch('requests.post')
    def test_process_data_task_with_valid_source(self, mock_post):
      # Simulate a successful api call
        mock_response = Mock()
        mock_response.raise_for_status.return_value=None
        mock_response.json.return_value = {'dag_run_id': 'test_run_1'}
        mock_post.return_value = mock_response


        airflow_url = 'http://localhost:8080' # Replace with your actual airflow url
        auth_tuple = ('airflow', 'airflow') # Replace with your credentials
        dag_id = 'my_data_pipeline'
        params = {'source_id': '98765'}

        result = trigger_dag_with_params(dag_id, params, airflow_url, auth_tuple)

        self.assertIsNotNone(result)
        self.assertEqual(result['dag_run_id'], 'test_run_1')
        # Validate the api call was made with expected params
        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        self.assertEqual(call_args[0], f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns")
        self.assertEqual(call_kwargs['headers'], {'Content-Type': 'application/json'})
        self.assertEqual(json.loads(call_kwargs['data']) , {'conf': params})

    @patch('requests.post')
    def test_process_data_task_without_source(self, mock_post):
        # Simulate a successful api call
        mock_response = Mock()
        mock_response.raise_for_status.return_value=None
        mock_response.json.return_value = {'dag_run_id': 'test_run_2'}
        mock_post.return_value = mock_response
        airflow_url = 'http://localhost:8080' # Replace with your actual airflow url
        auth_tuple = ('airflow', 'airflow') # Replace with your credentials
        dag_id = 'my_data_pipeline'
        params = {}  # No source_id

        result = trigger_dag_with_params(dag_id, params, airflow_url, auth_tuple)

        self.assertIsNotNone(result)
        self.assertEqual(result['dag_run_id'], 'test_run_2')

        mock_post.assert_called_once()
        call_args, call_kwargs = mock_post.call_args
        self.assertEqual(call_args[0], f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns")
        self.assertEqual(call_kwargs['headers'], {'Content-Type': 'application/json'})
        self.assertEqual(json.loads(call_kwargs['data']), {'conf': params})


if __name__ == '__main__':
    unittest.main()

```

Here, I am using `unittest` along with `mock` to effectively test the api calling logic, as well as the parameter passing within the dag. It's important to note, this unit test only tests the *triggering* of a dag run. It doesn't directly evaluate the task outputs themselves which would be more complex to implement and outside the scope of parameter testing. In practice, you should use the api to trigger and then check the log output or results in an xcom for end to end verification.

For further study, I recommend examining the official Airflow documentation, specifically around the rest api and how `dag_run.conf` interacts with task instances. Additionally, “Designing Data-Intensive Applications” by Martin Kleppmann provides a great foundation on designing systems that include data pipelines. Another good resource is “Fluent Python” by Luciano Ramalho for anyone wanting to improve their Python skills which will in turn help greatly with dag development. Finally, the book “Testing Python” by Daniel Roy Greenfeld will help understand more robust testing techniques that you can use with Airflow.

Testing with parameters via the api is a critical component of ensuring your airflow dags behave as intended. By using a structured approach and testing different parameter scenarios we can catch issues early, and build more robust and dependable pipelines. Remember, consistent testing isn’t a luxury, it is a core component of any reliable data infrastructure.
