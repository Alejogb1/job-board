---
title: "How do I create a Google BigQuery connection from the Airflow UI (Dockerized)?"
date: "2024-12-23"
id: "how-do-i-create-a-google-bigquery-connection-from-the-airflow-ui-dockerized"
---

Alright, let's unpack this. I’ve been down this particular rabbit hole before, specifically around five years back when we were migrating a large batch processing system to a cloud-native architecture. Setting up a robust connection between a dockerized Airflow instance and Google BigQuery is a crucial step, and it's more nuanced than it might first appear. It’s not just about the code—it’s also about properly configuring permissions and networking within your containerized environment.

The core issue revolves around ensuring that Airflow, running within its docker container, can authenticate with Google Cloud Platform (GCP) and BigQuery. This authentication typically hinges on service account keys, and the secure and reliable management of those keys is absolutely paramount. Let's walk through the necessary steps and some potential pitfalls based on my experience.

Firstly, we need a service account with the necessary permissions on BigQuery. This account will act as the identity of our Airflow instance when interacting with BigQuery. You’ll need to create a new service account within the GCP console or through the `gcloud` CLI tools, and crucially, give it the appropriate roles: `BigQuery User` and `BigQuery Job User` are generally sufficient for common data loading and query scenarios. Once this account is created, generate a JSON key file for it. Treat this key file like you would any critical credential: keep it secure, don’t commit it to version control, and restrict access to it.

Now, the docker container doesn't inherently know about this key. We'll need to make it accessible to the Airflow application running inside. There are multiple strategies to accomplish this; environment variables or mounting a volume containing the key file are two common methods. For this explanation, I prefer to pass the content of the JSON key file as a base64 encoded string as an environment variable because it is more secure than mounting a file directly. I've found it to be the most manageable approach, particularly in CI/CD pipelines.

Here’s a breakdown of how I'd do it, starting with the environment variable setup:

1. **Base64 Encode Your Key File:** On your development machine or wherever you manage your secrets, you'd run something similar to:

```bash
base64 <service_account_key.json > service_account_key.b64
```
Then, copy the content of `service_account_key.b64` to a safe location where it can be referenced.

2. **Set an Environment Variable in your Docker Setup:** This variable will carry the encoded service account key. When launching the airflow docker container, use a `--env` or a similar mechanism provided by your container orchestration tool:

```docker
docker run -d \
    -e AIRFLOW_GCP_KEY_BASE64="<your_base64_encoded_key_contents>" \
    ...  <your_airflow_image>
```
Replace `<your_base64_encoded_key_contents>` with the actual encoded content of the key you copied in step 1.

3. **Configure the Airflow Connection:** Within the Airflow UI, navigate to `Admin` -> `Connections`. Create a new connection. Set `Connection Type` to `Google Cloud`. There, you will see the option to configure the authentication using a json file. We are not using that, instead, specify `Credentials JSON` as an empty string, and add the project_id under `extra` like this `{"project_id":"your-gcp-project-id"}`.

Now within Airflow we'll write the Python code to leverage that key. The key to utilizing that is `json.loads` and `base64.b64decode`. Here's an example snippet of how to initialize the BigQuery hook within an Airflow DAG:

```python
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
from airflow.models import Variable
import base64
import json
import os

def get_bigquery_hook():
    key_base64 = os.getenv("AIRFLOW_GCP_KEY_BASE64")
    if not key_base64:
        raise ValueError("AIRFLOW_GCP_KEY_BASE64 not found in env variables")
    decoded_key = base64.b64decode(key_base64).decode('utf-8')
    credentials = json.loads(decoded_key)
    return BigQueryHook(gcp_conn_id="google_cloud_default", credentials=credentials)

def execute_bq_query():
    bq_hook = get_bigquery_hook()
    sql = "SELECT 1;"
    results = bq_hook.run_query(sql)
    for row in results:
        print(row)

# Example DAG to use this
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG(
    dag_id='bq_connection_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    bq_task = PythonOperator(
        task_id='execute_bq_query',
        python_callable=execute_bq_query
    )
```

In this example, `get_bigquery_hook` decodes the base64 string, loads the JSON key, and instantiates the `BigQueryHook`. The `execute_bq_query` function demonstrates the basic usage of this hook to run a query. It’s important to note that `gcp_conn_id` has to match with the name of the connection you created in Airflow.

Another alternative is to use the Google application default credentials. This is usually achieved via an instance metadata server but can be implemented via the environment as well.  This is particularly relevant when using managed environments like Google Kubernetes Engine, where you can assign service accounts to your pods. This does require you to configure your container with the necessary service account beforehand, outside the scope of the Airflow UI.

Here’s an example of using Application Default Credentials via `GOOGLE_APPLICATION_CREDENTIALS`:

```python
import os
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook

def get_bigquery_hook_adc():
    if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
        raise ValueError("GOOGLE_APPLICATION_CREDENTIALS not found in environment")
    return BigQueryHook(gcp_conn_id="google_cloud_default")

def execute_bq_query_adc():
    bq_hook = get_bigquery_hook_adc()
    sql = "SELECT 2;"
    results = bq_hook.run_query(sql)
    for row in results:
        print(row)

# Example DAG to use this approach
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG(
    dag_id='bq_connection_example_adc',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    bq_task_adc = PythonOperator(
        task_id='execute_bq_query_adc',
        python_callable=execute_bq_query_adc
    )
```

In this snippet, if the environment variable `GOOGLE_APPLICATION_CREDENTIALS` is set by kubernetes for example, the hook will use those credentials implicitly to establish the connection. This approach simplifies the credential management when using managed services.

Finally, if you choose to go with mounting the service account json as a file to the container, you can configure your hook like this:

```python
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook

def get_bigquery_hook_file():
    return BigQueryHook(gcp_conn_id="google_cloud_default", key_path="/path/to/mounted/key.json")

def execute_bq_query_file():
    bq_hook = get_bigquery_hook_file()
    sql = "SELECT 3;"
    results = bq_hook.run_query(sql)
    for row in results:
        print(row)


# Example DAG to use this approach
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

with DAG(
    dag_id='bq_connection_example_file',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    bq_task_file = PythonOperator(
        task_id='execute_bq_query_file',
        python_callable=execute_bq_query_file
    )
```
Here the `key_path` argument is provided to the `BigQueryHook` that points to where the file is mounted in your container's file system.

For a deep dive on Google Cloud authentication, I highly recommend the official Google Cloud documentation. Specifically, review the “Authenticating as a Service Account” section which can be found in “Cloud Authentication: Overview” on the Google Cloud documentation website. For Airflow specifics, the “Apache Airflow Providers Package Google” documentation, especially around the BigQueryHook, is a great resource.

In closing, connecting Airflow to BigQuery within a Dockerized environment demands careful handling of credentials and understanding of the nuances in authentication mechanisms. Through base64 encoding of the key file, leveraging application default credentials or even mounting the key directly, you can establish the necessary secure connection. Remember to align your method with your infrastructure’s security policies, and always adhere to best practices for credential management.
