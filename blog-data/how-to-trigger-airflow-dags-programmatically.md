---
title: "How to trigger Airflow DAGs programmatically?"
date: "2024-12-16"
id: "how-to-trigger-airflow-dags-programmatically"
---

Alright, let's tackle this. Programmatically triggering Airflow DAGs is a common requirement, and over the years I've bumped into more than a few scenarios where relying solely on the scheduler isn’t quite enough. I remember a particularly tricky project where we needed to kick off a complex ETL pipeline based on real-time events coming from an external system. The scheduler alone couldn't handle that level of dynamism, necessitating a programmatic approach. There’s a spectrum of ways to achieve this, each with its own advantages and trade-offs. Let’s break it down.

The core concept revolves around the Airflow REST API. Airflow exposes various endpoints that allow us to interact with it programmatically, and one such endpoint is specifically designed to trigger DAG runs. This is the primary method for remote execution. We typically achieve this through scripting languages like Python, using libraries that can handle HTTP requests. Python's `requests` library is a solid workhorse here, but I have also used other alternatives depending on the specific system involved.

First, let's examine a basic Python example that illustrates how to trigger a DAG run using the Airflow API. This assumes you have your Airflow instance running and you know your DAG's `dag_id`.

```python
import requests
import json

def trigger_dag(dag_id, airflow_url, auth_token=None, conf=None):
    """
    Triggers a DAG run using the Airflow API.

    Args:
        dag_id (str): The ID of the DAG to trigger.
        airflow_url (str): The base URL of the Airflow instance.
        auth_token (str, optional): Authentication token for the API. Defaults to None.
        conf (dict, optional): Configuration parameters for the DAG run. Defaults to None.
    """
    url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"
    headers = {'Content-Type': 'application/json'}
    if auth_token:
         headers['Authorization'] = f'Bearer {auth_token}'

    payload = {}
    if conf:
        payload['conf'] = conf

    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"Successfully triggered DAG: {dag_id}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error triggering DAG {dag_id}: {e}")
        return None


if __name__ == '__main__':
    dag_to_trigger = "my_example_dag"  # Replace with your DAG ID
    airflow_base_url = "http://localhost:8080" # Replace with your airflow url
    api_auth_token = "your_api_token"  # Replace with your API token if auth is enabled
    config_params = {"param1": "value1", "param2": 123} # Example config parameters

    trigger_dag(dag_to_trigger, airflow_base_url, api_auth_token, config_params)
    #trigger_dag(dag_to_trigger, airflow_base_url) # call without token and config
```

This snippet demonstrates the core mechanics. The `trigger_dag` function constructs the necessary URL, including the `dag_id`, and sends a POST request to initiate a new DAG run. The `auth_token` parameter manages authentication, usually needed in a production environment. Furthermore, the `conf` parameter shows how to pass a custom configuration to your DAG run, enabling you to alter the behavior dynamically based on the context.

Moving beyond the basics, you might encounter situations where you have external services that need to trigger DAGs. This is where specialized tools or libraries come into play. Another common scenario is when you need to integrate message queues like RabbitMQ or Kafka. Here, an application will consume messages, process data, and then trigger specific DAGs based on the content of the message.

Here's a slightly more advanced Python example using Celery, a task queue system, which effectively acts as a middleman for this kind of integration. Note, this requires an Airflow provider for Celery (you'll have to install it, and your celery setup, but this is a general implementation):

```python
from celery import Celery
import requests
import json
import os

CELERY_BROKER = os.environ.get("CELERY_BROKER", "redis://localhost:6379/0") #Set your broker
CELERY_RESULT_BACKEND = os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0") #Set your result backend

app = Celery('tasks', broker=CELERY_BROKER, backend=CELERY_RESULT_BACKEND)

@app.task
def trigger_dag_celery(dag_id, airflow_url, auth_token=None, conf=None):
    """Triggers a DAG run through the celery task queue."""
    url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"
    headers = {'Content-Type': 'application/json'}
    if auth_token:
         headers['Authorization'] = f'Bearer {auth_token}'

    payload = {}
    if conf:
        payload['conf'] = conf
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        print(f"Successfully triggered DAG: {dag_id}")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error triggering DAG {dag_id}: {e}")
        return None


if __name__ == '__main__':
    dag_to_trigger = "my_example_dag"
    airflow_base_url = "http://localhost:8080"
    api_auth_token = "your_api_token"
    config_params = {"celery_param": "celery_value"}

    trigger_dag_celery.delay(dag_to_trigger, airflow_base_url, api_auth_token, config_params)

```

This script defines a Celery task, `trigger_dag_celery`, that uses the same logic as our first example. However, instead of executing the DAG trigger directly, it submits the request to the Celery queue, which will then execute asynchronously. This method adds decoupling, improving resilience, and scalability, especially in environments with fluctuating loads. The beauty of celery or similar systems is that you don't need to be in the same script, this could be a completely different system entirely just sending data and triggering via the broker.

Finally, another approach involves using Airflow’s `trigger_dag` operator *within* an existing DAG. This creates a chain of DAGs, where one DAG can trigger another upon successful completion or based on specific conditions. It allows the orchestration logic to be encapsulated within your Airflow environment. For example, a daily process DAG might finish its process and then trigger a reporting DAG for the day.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime

def print_trigger_message(**kwargs):
    print("Starting DAG run")
    return 'Print_trigger_message executed'

with DAG(
    dag_id='parent_dag',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:
    start_task = PythonOperator(
        task_id='start_task',
        python_callable=print_trigger_message
    )

    trigger_child_dag = TriggerDagRunOperator(
        task_id="trigger_child_dag",
        trigger_dag_id="child_dag",
        conf={"message": "Hello from parent DAG"},
        reset_dag_run=True
    )

    start_task >> trigger_child_dag

with DAG(
    dag_id='child_dag',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as child_dag:
    print_child_message = PythonOperator(
        task_id='print_child_message',
        python_callable=lambda **kwargs: print(f"Message from parent DAG: {kwargs['dag_run'].conf.get('message', 'No message provided')}")
    )
```

In this example, the `parent_dag` has a `TriggerDagRunOperator` that is configured to initiate `child_dag` upon successful completion of the start_task. The `conf` parameter allows you to pass data between DAGs. In my experience, this approach works best when you’re aiming for a workflow where the dependencies between DAGs are well-defined and fit into your existing Airflow structure. This pattern is also really helpful when working with more complicated or interdependent DAGs.

For further depth, I recommend "Data Pipelines with Apache Airflow" by Bas Harenslak and Julian Rutger, this book provides a very hands-on approach to Airflow concepts including triggering DAGs. Also, the official Apache Airflow documentation is a great resource, particularly the section on the REST API. For deeper knowledge on REST API design, and patterns consider reading "RESTful Web Services" by Leonard Richardson and Sam Ruby. You can also find numerous research papers online, specifically on distributed task queues such as Celery or similar.

Programmatically triggering DAGs involves understanding the exposed API, selecting the right approach for the specific requirements, and handling authentication and configuration appropriately. Whether you need a simple script or a more integrated solution involving queueing systems or interdependent DAGs, Airflow provides all the necessary tools. I've found it's all about finding the balance that matches both your needs and your existing system.
