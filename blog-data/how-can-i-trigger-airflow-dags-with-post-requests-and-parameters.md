---
title: "How can I trigger Airflow DAGs with post requests and parameters?"
date: "2024-12-16"
id: "how-can-i-trigger-airflow-dags-with-post-requests-and-parameters"
---

 I've personally seen this exact need arise several times over the past decade working with various orchestration platforms, particularly when integrating data pipelines with external services. The challenge of triggering Airflow dags via post requests and, more importantly, passing parameters, is a common hurdle that requires a bit of setup but pays off significantly in flexibility.

The core of the solution lies in leveraging Airflow’s REST api, specifically the 'trigger_dag' endpoint. This isn’t something you’d typically do directly in a browser but instead through another application or a custom script. We’ll need to form a structured http post request to this endpoint, and here's where the parameters come in. These parameters are essentially metadata you attach to the dag run that your dag can subsequently use. Think of it as passing initial configuration to your workflow as it kicks off.

First and foremost, let’s acknowledge the security aspect. The default Airflow webserver doesn't have authentication enabled by default, which is fine for development but absolutely crucial to address for anything remotely production-related. I highly recommend researching and implementing a suitable authentication method. Airflow supports various options, including Kerberos, LDAP, and OAuth. For a deep dive into best security practices, I point you towards the “Designing Data Intensive Applications” by Martin Kleppmann, particularly the chapters on security and distributed systems. This isn't specific to Airflow, but its security concepts apply universally. Also, the official Airflow documentation itself has comprehensive information on authentication.

Once security is addressed, consider how you'll structure your post request payload. Airflow expects parameters to be passed within the request body as json data, nested under a ‘conf’ key. This json data will be converted into a python dictionary available within your dag’s code. You can use python libraries like 'requests' to construct and dispatch the http post request. Let me illustrate with a basic python example:

```python
import requests
import json

def trigger_airflow_dag(dag_id, params, airflow_url, auth):
    """Triggers an airflow dag with the specified parameters."""

    url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"

    payload = {
        "conf": params
    }
    headers = {
         "Content-Type": "application/json",
    }

    try:
        response = requests.post(url,
                            headers=headers,
                            data=json.dumps(payload),
                            auth=auth)  # assuming a Basic auth for simplicity
        response.raise_for_status()  # raise exception for http errors

        run_id = response.json()['dag_run_id']
        print(f"Dag run {run_id} triggered successfully")
        return run_id
    except requests.exceptions.RequestException as e:
        print(f"Error triggering dag: {e}")
        return None

if __name__ == "__main__":
    dag_to_trigger = "my_example_dag"
    my_parameters = {
        "start_date": "2024-01-01",
        "data_source": "s3://my-bucket/data.csv"
    }
    airflow_base_url = "http://localhost:8080" # replace with your airflow url
    airflow_auth = ("admin", "admin") # replace with your actual auth credentials

    trigger_airflow_dag(dag_to_trigger, my_parameters, airflow_base_url, airflow_auth)
```

In this example, replace `my_example_dag` and `http://localhost:8080` with the appropriate values for your setup. I've used basic authentication here; it is extremely important to use proper authentication mechanisms for production.

Now, let's look at the Airflow dag python code side, demonstrating how to access the parameters we sent through the post request.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def process_data(**kwargs):
    """Retrieves parameters and prints them."""
    params = kwargs['dag_run'].conf
    print(f"Received parameters: {params}")

    start_date = params.get('start_date')
    data_source = params.get('data_source')

    print(f"Processing data from: {data_source}, starting on {start_date}")
    # here you would actually process the data
    return None


with DAG(
    dag_id='my_example_dag',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:

    process_task = PythonOperator(
        task_id="process_data_task",
        python_callable=process_data,
    )

```

The `kwargs['dag_run'].conf` snippet within your python operator retrieves the parameters passed via the post request. The example shows extracting a ‘start_date’ and ‘data_source’ for processing; you'll obviously adapt this to whatever parameters your dag needs. The `kwargs` parameter is an important aspect of using custom parameters passed to your DAG run. It’s a dictionary that contains context variables, including information about the current dag run, such as `dag_run.conf`, which contains the custom configuration.

A common pitfall I’ve encountered is error handling during the http request. Always use `response.raise_for_status()` to detect http error codes early; otherwise, your script might proceed without actually triggering the dag. The `requests` library is robust, but you still must account for network issues, authentication problems, or api changes. Furthermore, consider implementing proper logging within your client script and your dag code for diagnostics.

Let’s go a bit further, imagining a scenario where your parameters involve more complex json structures. In such cases, it is important to test extensively. Here’s another practical example:

```python
import requests
import json
from datetime import datetime

def trigger_airflow_complex_dag(dag_id, params, airflow_url, auth):
    """Triggers a dag with a more complex parameter payload"""

    url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"

    payload = {
        "conf": params
    }
    headers = {
         "Content-Type": "application/json",
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), auth=auth)
        response.raise_for_status()
        run_id = response.json()['dag_run_id']
        print(f"Dag run {run_id} triggered successfully with complex parameters")
        return run_id
    except requests.exceptions.RequestException as e:
        print(f"Error triggering dag: {e}")
        return None

if __name__ == '__main__':
    complex_params = {
        "report_date": "2024-03-01",
        "filters":{
            "region": "EU",
            "product_type": ["Laptop", "Tablet"]
        },
        "options":{
            "include_archived": False,
            "output_format":"csv"
        }
    }

    dag_id_to_trigger = "my_complex_dag"
    airflow_base_url = "http://localhost:8080"
    airflow_auth = ("admin", "admin")

    trigger_airflow_complex_dag(dag_id_to_trigger, complex_params, airflow_base_url, airflow_auth)

```

This example illustrates triggering a dag with parameters structured as nested dictionaries. Within your Airflow dag, you would access these parameters exactly as shown in the earlier example using `kwargs['dag_run'].conf`, and subsequently drill down into your dictionary to access the individual parameters.

In terms of further reading, "Python Cookbook" by David Beazley and Brian Jones is beneficial for general python programming and data structures. The sections on data serialization are very valuable. Also, reviewing the official python requests documentation will help with error handling and other more advanced features.

Remember, constructing these http requests correctly and handling parameters robustly is crucial for building flexible and dynamically triggered Airflow pipelines. Don’t skip on the necessary security measures; always assume that unauthenticated apis will be exploited. Start with smaller example dags first and gradually add more complexity as you become more comfortable with this method.
