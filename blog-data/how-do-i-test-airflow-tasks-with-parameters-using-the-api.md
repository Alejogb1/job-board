---
title: "How do I test Airflow tasks with parameters using the API?"
date: "2024-12-16"
id: "how-do-i-test-airflow-tasks-with-parameters-using-the-api"
---

,  I've seen a fair share of Airflow setups in my time, and testing tasks, particularly when they involve parameters passed in via the API, is something that crops up regularly. It's a crucial step for reliable pipelines, and a lack of proper testing often leads to brittle and error-prone deployments. We're not just trying to make things work; we’re trying to make things work predictably and robustly.

The core issue stems from the fact that Airflow tasks are typically orchestrated via DAG runs, which are themselves triggered through the scheduler or the API. When parameters are introduced, they become part of this entire lifecycle, and testing needs to account for this entire flow – the task itself and its interaction with the passed-in parameters.

I recall a particular project where we were ingesting a large number of files based on dynamically generated file paths, which were passed as a parameter to our tasks. We initially fell into the trap of just testing the task code directly, bypassing the API and its associated parameter passing. This led to a lot of surprises in production, because the task would execute *differently* when launched through the API than it did in our isolated tests. We ended up needing to refactor everything to ensure we tested the whole pipeline, including parameter propagation.

So, how do you actually do it? Well, there are several approaches, but here's my preferred way, based on that experience, and it involves a combination of unit testing the core task logic and integration testing the task invocation via the API.

**Step 1: Unit Testing the Core Task Logic**

First and foremost, the actual function or operator within your task should be unit tested. This means we should be able to call the underlying logic with a range of possible input parameters and confirm it does what we expect. We bypass Airflow entirely for this part.

Let's say we have a PythonOperator which uses a function like this:

```python
def process_data(file_path, output_location):
    # Assume this function reads data from file_path, processes it, and saves it to output_location
    # This is a placeholder for actual data processing
    with open(file_path, 'r') as f:
      data = f.read()
    with open(output_location, 'w') as out_f:
        out_f.write(f"Processed: {data}")
    return output_location
```

We might have an airflow task like so:

```python
from airflow.operators.python import PythonOperator

def create_processing_task(file_param_name="file_path", out_param_name = "output_loc"):
  return PythonOperator(
      task_id='process_file',
      python_callable=process_data,
      op_kwargs={
          'file_path': '{{ dag_run.conf["' + file_param_name + '"] }}',
          'output_location': '{{ dag_run.conf["' + out_param_name + '"] }}',
      }
  )
```
Here, `file_param_name` and `out_param_name` would be strings representing the keys used to pass data via the API.

Now, a corresponding unit test might look like this:

```python
import unittest
import tempfile
from your_module import process_data

class TestProcessData(unittest.TestCase):
    def test_process_data_with_valid_input(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_input_file:
          temp_input_file.write("sample data")
          temp_input_path = temp_input_file.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_output_file:
          temp_output_path = temp_output_file.name

        result = process_data(temp_input_path, temp_output_path)
        self.assertEqual(result, temp_output_path)

        with open(temp_output_path, 'r') as f:
            output_content = f.read()
            self.assertEqual(output_content, "Processed: sample data")
    
        import os
        os.remove(temp_input_path)
        os.remove(temp_output_path)


    def test_process_data_with_empty_input(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_input_file:
            temp_input_path = temp_input_file.name
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_output_file:
            temp_output_path = temp_output_file.name
        
        result = process_data(temp_input_path,temp_output_path)
        self.assertEqual(result, temp_output_path)

        with open(temp_output_path, 'r') as f:
            output_content = f.read()
            self.assertEqual(output_content, "Processed: ")
        
        import os
        os.remove(temp_input_path)
        os.remove(temp_output_path)

if __name__ == '__main__':
    unittest.main()
```

The important part here is that we directly call `process_data` and verify its behavior, including edge cases, without involving Airflow. This makes tests significantly faster and helps pinpoint issues in the core logic rather than in the orchestration.

**Step 2: Integration Testing via the Airflow API**

Once you're confident in the core logic, you need to test how the task behaves when triggered through the API with parameters. This simulates how your tasks will run in the actual production environment. To do this, we will utilize the Airflow REST API, which can be used to trigger DAG runs and pass parameters through the `conf` dictionary.

Here's a conceptual python snippet using `requests` which shows how this can work.

```python
import requests
import json

def test_api_task_with_parameters(dag_id, file_path, output_location, api_endpoint="http://localhost:8080/api/v1/dags/{}/dagRuns"):
    url = api_endpoint.format(dag_id)
    headers = {'Content-type': 'application/json'}
    data = json.dumps({
        'conf': {
            'file_path': file_path,
            'output_loc': output_location,
        }
    })

    response = requests.post(url, data=data, headers=headers)

    if response.status_code == 200:
        dag_run_id = response.json()['dag_run_id']
        print(f"DAG run {dag_run_id} started with file_path:{file_path} and output_loc:{output_location}")
        # Poll the API to check task status if you want to do further validations.
        return dag_run_id
    else:
      print(f"Failed to trigger dag run. Code:{response.status_code} and text {response.text}")
      return None
```

In a test suite, you could call this function after making sure an appropriate DAG is active:

```python
class TestAirflowApiTask(unittest.TestCase):
    def test_parameterized_task_via_api(self):
         with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_input_file:
            temp_input_file.write("sample data via api")
            temp_input_path = temp_input_file.name

         with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_output_file:
            temp_output_path = temp_output_file.name
            
         dag_run_id = test_api_task_with_parameters("my_sample_dag", temp_input_path, temp_output_path)
         self.assertIsNotNone(dag_run_id)
         
         #Poll for status to be succesful then assert things about the result of the run. This needs more robust error handling and retries.
         #This is not fully fleshed out due to the scope of this example.

         import os
         os.remove(temp_input_path)
         os.remove(temp_output_path)
```

This test directly interacts with the API, passing in file paths as configuration. You would then, in a more fleshed out test, poll for success, examine logs, or potentially read from a datastore to verify the task's behaviour in the context of Airflow. This process ensures your task behaves as expected when invoked correctly with user supplied arguments.

**Recommended Resources**

For a deeper dive, I highly recommend:

1.  **"Effective Testing with RSpec 3" by Myron Marston and Ian Dees:** Although primarily focused on Ruby, the principles of testing structure and approach are universally applicable to software testing, and it provides some useful insights.
2.  **"Testing Python" by David Sale:** A guide focused on Python, covering specific strategies and tooling relevant to the language, which would be beneficial when dealing with Airflow in Python.
3.  **The official Airflow documentation:** In particular, sections on the REST API, DAG structure, and operator concepts, are invaluable for understanding the system itself. There are a number of well established examples on the Apache Airflow project pages.

In conclusion, testing Airflow tasks with API parameters is a two-fold process. Firstly, ensure that the core logic is unit tested thoroughly, bypassing Airflow and testing the functionality directly. Secondly, use the Airflow API to integrate test the execution of your task with different parameters passed in through the `conf` dictionary, ensuring that the entire pipeline behaves as intended. By following these steps and being diligent about test coverage, you can build more reliable and robust Airflow deployments.
