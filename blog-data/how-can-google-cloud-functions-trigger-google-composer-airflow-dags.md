---
title: "How can Google Cloud Functions trigger Google Composer (Airflow) DAGs?"
date: "2024-12-23"
id: "how-can-google-cloud-functions-trigger-google-composer-airflow-dags"
---

Okay, let's tackle this. I’ve seen this particular challenge come up quite a few times in my career, and it's a common need when you start stitching together serverless functions and orchestrating workflows. The interaction between Google Cloud Functions and Composer (which essentially is managed Apache Airflow) is a classic example of combining event-driven architectures with scheduled task management. It's not a straightforward integration, but a few well-established patterns make it quite manageable.

The key challenge, as i’ve found, is that Cloud Functions are designed to react to events, while Airflow DAGs are typically triggered on a schedule or through manual intervention. Thus, you need a bridge between the asynchronous and the scheduled worlds. One approach which tends to be the most effective is to use Cloud Functions to programmatically trigger a DAG run using the Airflow API exposed by Cloud Composer. This is my preferred way to go because it offers fine-grained control and avoids less efficient polling solutions.

The core idea involves the Cloud Function making an HTTP request to the Airflow API. Cloud Composer exposes a REST API that includes endpoints for triggering DAG runs, and you need to authenticate with this API. The authentication component is quite essential; it's usually handled via service account credentials, which is something I'll highlight in the example code.

Now let's delve into a more concrete explanation and look at how we could implement this pattern in code.

First, the Cloud Function will need a way to make authenticated HTTP requests to the Airflow REST API. You'll need a service account with the appropriate permissions for triggering DAGs, and your Cloud Function’s service account will also need the ability to impersonate this service account. This helps maintain a strong least-privilege security model.

Here's an example using Python (the most common choice for Cloud Functions and Airflow):

```python
import google.auth.transport.requests
import google.oauth2.id_token
import requests
import os
from google.cloud import secretmanager

def get_composer_api_credentials(project_id, secret_name):
    """Fetches Airflow API credentials from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    secret_path = f"projects/{project_id}/secrets/{secret_name}/versions/latest"
    response = client.access_secret_version(request={"name": secret_path})
    payload = response.payload.data.decode("UTF-8")
    username, password = payload.split(":", 1)
    return username, password


def trigger_dag(request):
    """Cloud Function to trigger a DAG via Composer API."""
    project_id = os.environ.get("PROJECT_ID") # Set this in environment variables in your cloud function
    composer_region = os.environ.get("COMPOSER_REGION") # Set this in environment variables in your cloud function
    composer_environment = os.environ.get("COMPOSER_ENVIRONMENT") # Set this in environment variables in your cloud function
    dag_id = os.environ.get("DAG_ID") # Set this in environment variables in your cloud function
    secret_name = os.environ.get("SECRET_NAME") # Set this in environment variables in your cloud function


    username, password = get_composer_api_credentials(project_id, secret_name)
    airflow_url = f"https://{composer_environment}.{composer_region}.composer.cloud.google/api/v1/dags/{dag_id}/dagRuns"
    
    
    try:
      
        id_token = google.oauth2.id_token.fetch_id_token(request=google.auth.transport.requests.Request(),
                                                            target_audience = airflow_url)
        
        headers = {
          "Authorization": f"Bearer {id_token}",
          "Content-Type": "application/json"
        }
        
        
        response = requests.post(airflow_url, headers=headers, json={})
        response.raise_for_status() # Raise error on HTTP issues
        return f"DAG '{dag_id}' triggered successfully. Response: {response.text}", 200

    except Exception as e:
        return f"Error triggering DAG: {e}", 500
```

In this example, we are fetching the Airflow API credentials from Google Secret Manager. This is considered a better security practice than hardcoding credentials in the environment variables. The secret is stored as 'username:password'. We construct the Airflow API url and then we use the Google Oauth2 library to acquire a bearer token that we include in our headers. Finally we use the Python 'requests' library to send an authenticated http post request.

It's crucial to remember that the Cloud Function's service account requires the `roles/iam.serviceAccountTokenCreator` role to generate this token and the service account referenced by the secret in secret manager requires the ability to execute dags in your composer environment.

Now let’s discuss another approach. Sometimes you don't want the complexity of direct API calls, especially if you need more decoupling or need to pass data to your DAG. Here’s where Google Cloud Pub/Sub can be a useful intermediary. Your Cloud Function publishes a message to a Pub/Sub topic, and an Airflow DAG, configured to listen to that topic, is triggered by that message.

Let’s exemplify this with a Cloud Function and a simple DAG snippet in Python:

```python
# Cloud Function (Python) to publish a message to Pub/Sub
import os
from google.cloud import pubsub_v1

def publish_pubsub_message(request):
    """Cloud Function to publish a message to a Pub/Sub topic."""
    project_id = os.environ.get("PROJECT_ID") # Set this in environment variables in your cloud function
    topic_name = os.environ.get("PUBSUB_TOPIC") # Set this in environment variables in your cloud function

    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(project_id, topic_name)

    try:
        # Assuming request contains data to be passed to the DAG
        request_json = request.get_json()
        message_data = str(request_json).encode("utf-8")
        
        publish_future = publisher.publish(topic_path, data=message_data)
        publish_future.result()  # Blocks until message is published
        return f"Message published to {topic_name} successfully.", 200
    except Exception as e:
        return f"Error publishing message: {e}", 500

```
This Cloud Function is simpler; it just publishes a message with data as a JSON string to a specific pub/sub topic.

Now let’s have a look at the relevant portion of the Airflow DAG:
```python
# Airflow DAG (Python) to subscribe to Pub/Sub messages
from airflow import DAG
from airflow.providers.google.cloud.sensors.pubsub import PubSubPullSensor
from airflow.operators.python import PythonOperator
from datetime import datetime
import json

def process_pubsub_message(message):
    """Function to process the data from pub/sub."""
    data = json.loads(message.data.decode('utf-8'))
    print(f"Received message: {data}")
    # Do something with the data here (e.g., trigger downstream tasks).

with DAG(
    dag_id="pubsub_triggered_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
    tags=['pubsub'],
) as dag:
    
    pull_pubsub_sensor = PubSubPullSensor(
        task_id='pull_pubsub_message',
        project_id="your-project-id", # Replace with your project ID
        subscription='your-subscription-id',  # Replace with your subscription ID
        ack_messages=True
    )

    process_message_task = PythonOperator(
        task_id="process_message",
        python_callable=process_pubsub_message,
        provide_context = True
    )
    
    pull_pubsub_sensor >> process_message_task

```
This DAG waits for messages to arrive on a specific subscription. When a message appears it’s passed to the *process_pubsub_message* Python function for further processing. The specific data from the message is now within the scope of your DAG and can be used to parameterize downstream tasks. This is very handy when the content of the message is important.

A final less common approach would involve using a cloud scheduler trigger, where you would have your cloud scheduler trigger a cloud function, which in turn triggers a DAG. This can be useful if you need to execute a DAG at set intervals not supported by Airflow's own scheduler. The Cloud Function is similar to the first code block, and the cloud scheduler will execute it according to its scheduled configuration.

In terms of resources, I'd recommend diving into the official Google Cloud documentation, particularly the sections on Cloud Functions, Cloud Composer, and Pub/Sub. The 'Designing Data-Intensive Applications' by Martin Kleppmann is also an invaluable book for understanding the fundamental concepts behind these kinds of integrations. Furthermore the book 'Orchestrating Data Pipelines with Apache Airflow' by Bas P. Harenslak et al will help solidify your knowledge of Airflow specifics, while the official apache airflow documentation is the source of truth for specific operator and library syntax.

In conclusion, triggering Composer DAGs from Cloud Functions is achievable using several methods, but direct API calls and Pub/Sub event-driven approaches tend to be the most versatile and robust in practice. Choose the best approach based on your specific application requirements, weighing trade-offs such as complexity, decoupling, and latency.
