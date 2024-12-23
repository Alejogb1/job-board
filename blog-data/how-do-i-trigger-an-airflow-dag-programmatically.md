---
title: "How do I trigger an Airflow DAG programmatically?"
date: "2024-12-23"
id: "how-do-i-trigger-an-airflow-dag-programmatically"
---

Let's unpack how to kick off an airflow dag outside its scheduled cadence; a topic I've encountered more than a few times in various production environments. The need to trigger dags programmatically arises quite often - maybe it's a critical data refresh, an emergency patch, or perhaps a user-initiated process that needs airflow's orchestration. Thankfully, airflow offers several solid methods to achieve this, each with nuances that make them suitable for different situations.

Over the years, I've seen teams stumble by relying solely on the scheduler, leading to delays and unnecessary complexity. For instance, in a previous role managing a large ETL pipeline, a significant bug fix required us to reprocess a substantial chunk of data. Relying on the scheduled runs wouldn't have been fast enough, so we had to implement a programmatic trigger mechanism. From that point forward, we've always leveraged a mix of scheduled and programmatically triggered runs based on the use case.

At its core, the programmatic triggering of dags essentially involves interacting with airflow’s api, which can be done in a variety of ways. One common approach, and arguably the most direct, is through the airflow cli. Another useful method, particularly within more complex systems or when integrated into web applications, involves making rest api calls to the airflow server. Lastly, and sometimes the most convenient for developers, you can programmatically trigger dags from within other python code, especially within airflow itself. Let's examine each of these approaches with specific code examples.

**Method 1: Using the Airflow CLI**

The airflow cli is a powerful tool for a lot more than just starting dags, but it's perfectly suitable for this task. The command you'll be most interested in is `airflow dags trigger`. This method is generally my first go-to for straightforward, ad-hoc triggering, particularly when manually initiating a dag or when it needs to be launched from a deployment script.

Consider this basic scenario. Suppose you have a dag named `data_ingestion_pipeline`. From your terminal or a script with airflow's cli available, you could execute the following command:

```bash
airflow dags trigger data_ingestion_pipeline
```

This simple command triggers a run of `data_ingestion_pipeline` using the latest dag configuration. It does not affect already scheduled runs, ensuring a clean execution.

However, sometimes you want to pass configuration parameters to your dag, perhaps to specify a different data set. The cli allows you to pass those too, using the `-c` flag.

```bash
airflow dags trigger data_ingestion_pipeline -c '{"start_date": "2023-10-26", "end_date": "2023-10-30"}'
```

Here, we are specifying that a dag run should use a specific start and end date which might control the portion of data the pipeline processes. These configurations can be accessed within your dag using the `dag_run.conf` dictionary.

The cli is beneficial because it is simple and doesn't require complex setup beyond having airflow installed and configured.

**Method 2: Using the Airflow Rest API**

For more sophisticated interactions, like when a web application or an external service needs to trigger dags, using the airflow rest api is typically the best route. This method requires you to authenticate with airflow and then send http requests to specific endpoints. It’s more involved than using the cli, but its flexibility and versatility make it extremely important.

Let’s assume you have set up airflow’s api with the necessary authentication mechanism (e.g., basic auth or api key). Here’s how you might use python’s `requests` library to trigger a dag named `ml_training_pipeline`:

```python
import requests
import json

airflow_url = "http://your_airflow_host:8080"  # Replace with your actual airflow url
dag_id = "ml_training_pipeline"
auth = ('airflow', 'your_airflow_password')  # Replace with your airflow username and password (or api key)

headers = {
    "Content-Type": "application/json"
}

data = {
    "conf": {
        "model_type": "transformer",
        "learning_rate": 0.001
    }
}

url = f"{airflow_url}/api/v1/dags/{dag_id}/dagRuns"

try:
    response = requests.post(url, auth=auth, headers=headers, data=json.dumps(data))
    response.raise_for_status() # Raises an exception for non-200 status codes
    print(f"Successfully triggered dag {dag_id}. Response: {response.json()}")
except requests.exceptions.RequestException as e:
    print(f"Error triggering dag {dag_id}: {e}")
```

In this python script, a post request is sent to the `/api/v1/dags/{dag_id}/dagRuns` endpoint, including configuration data as a json string. The `response.raise_for_status()` method verifies a successful request; otherwise, the try-catch block manages possible errors.

Using the api in this fashion allows for fine-grained control and is ideal for embedding dag triggers within other applications. I've found it particularly useful for self-service portals and real-time data analysis systems where a human-initiated process directly calls airflow.

**Method 3: Triggering from within an Airflow DAG or Python Code**

Finally, you can trigger dags from within other dags or regular python code running in the airflow environment. This is extremely powerful for creating dependent workflows or implementing complex triggering logic. Airflow provides the `trigger_dag` operator, which is the most direct method for this type of interaction.

Here’s how it might look within a python dag definition:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.dagrun import TriggerDagRunOperator
from datetime import datetime

def log_start(**context):
    print(f"Triggering dag at {datetime.now()} with context:{context}")

with DAG(
    dag_id='master_pipeline',
    start_date=datetime(2023, 1, 1),
    schedule=None,  # Make it manual only to demonstrate
    catchup=False,
    tags=['example'],
) as dag:
    log_start_task = PythonOperator(
        task_id="log_start_task",
        python_callable=log_start
    )

    trigger_data_refresh = TriggerDagRunOperator(
        task_id="trigger_data_refresh_dag",
        trigger_dag_id="data_refresh_pipeline",
        reset_dag_run=True, # Resets all previous attempts of the dag run if it was triggered previously
        conf = {"reprocess_all": True},
    )

    log_start_task >> trigger_data_refresh

```

In this example, the `master_pipeline` dag contains a `TriggerDagRunOperator` task that will programmatically trigger the `data_refresh_pipeline` dag. I've added an initial python task for logging purposes so you can easily see the trigger occur. The configuration data `{"reprocess_all": True}` is passed along to the triggered dag, showcasing how you can control the behaviour of the child dag from the parent. The `reset_dag_run` setting ensures a fresh start each time the trigger is used. This is an incredibly useful approach for managing dependency relationships between different pipelines.

**Recommendation**

For further understanding of airflow's internal workings, particularly regarding the api and configuration management, I highly recommend exploring the official airflow documentation. The "Apache Airflow documentation" directly available on the Apache Airflow project website, is an essential resource that covers everything from basic setup to advanced configuration options. I would also suggest diving into "Programming Apache Airflow" by J.T. Cashion, which offers a deep dive into the concepts and patterns for building robust airflow workflows, which can enhance the way you use these triggering mechanisms in practice.

In conclusion, programmatic dag triggering in airflow provides the flexibility necessary to address a broad range of needs. It allows you to bypass scheduled runs, integrate into other systems, and create complex workflows within workflows. The key is to understand the available methods and apply them appropriately based on your requirements. Whether you need the simplicity of the cli, the power of the rest api, or the integration capabilities of operators, airflow's programmatic triggering options provide a robust framework for managing your data pipelines.
