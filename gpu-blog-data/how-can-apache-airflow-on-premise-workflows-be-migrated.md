---
title: "How can Apache Airflow on-premise workflows be migrated to Google Cloud Composer?"
date: "2025-01-30"
id: "how-can-apache-airflow-on-premise-workflows-be-migrated"
---
Migrating on-premise Apache Airflow workflows to Google Cloud Composer necessitates a careful, phased approach, primarily due to the managed nature of Composer versus the self-managed environment of a typical on-premise Airflow setup. In my experience, having overseen the migration of several complex Airflow deployments, the key difference lies in transitioning from direct infrastructural control to leveraging GCP’s managed services. The process isn't merely a lift-and-shift; it often involves refactoring DAGs, re-evaluating dependencies, and adapting to a different execution environment.

The core challenges revolve around three main areas: infrastructure differences, dependency management, and resource utilization. On-premise Airflow deployments often benefit from custom configurations, specific Python versions, and directly installed dependencies. Composer, on the other hand, operates within Google Kubernetes Engine (GKE) and abstracts much of this complexity. This requires re-evaluating how dependencies are managed and code is executed. The transition also necessitates a shift from reliance on local resources to using Google Cloud services like Cloud Storage, BigQuery, and Cloud SQL.

Firstly, consider the infrastructural variances. On-premise Airflow typically relies on a specific operating system, often with direct access to network resources, file systems, and other locally hosted services. Composer, being a managed service, abstracts away much of this. Its components like the scheduler, webserver, and workers, run inside GKE, pre-configured with certain default settings. Migrating requires us to account for these limitations. This means, for example, if you're mounting a local directory as a volume in your on-premise deployment, you'll need to transition to using Cloud Storage buckets, which requires rewriting DAGs to interact with that service instead of local paths. Similarly, if you are interacting with an on-premise SQL database, you will need to establish connectivity through a secure method using Cloud SQL Proxy or other authorized networks.

Dependency management forms the second, and often more complex, challenge. Your on-premise setup might be leveraging custom Python packages installed via pip, or perhaps rely on specific versions of system libraries. Composer operates in an isolated Python environment. Therefore, replicating the environment involves specifying exact requirements in a `requirements.txt` file. Furthermore, any custom plugins utilized in the on-premise instance must be re-packaged and uploaded to the Composer environment, which often involves creating a dedicated Cloud Storage bucket for this purpose. This demands a thorough audit of all dependencies to ensure that the Composer environment matches the on-premise one as closely as possible.

Resource utilization is the third important factor. On-premise setups often have access to a fixed pool of computational resources. Composer, by contrast, operates using auto-scaling workers. This means you might not need to tune worker counts as meticulously as you would with an on-premise instance, as Composer manages it for you. However, it also means you need to monitor resource usage to prevent over-consumption and associated costs. DAGs that were tuned for a specific on-premise resource capacity will need to be profiled and potentially adjusted for the autoscaling behavior of Composer’s workers. The transition also presents an opportunity to optimize task resource requirements. For example, you can consider running memory-intensive tasks in parallel using parallelized operators like the `TaskGroup`.

Here are three code examples that illustrate key aspects of the migration:

**Example 1: Transitioning from Local File System to Cloud Storage**

The following example demonstrates how to read a file from a local directory in an on-premise Airflow setup and how to adapt it for use with Google Cloud Storage on Composer.

*On-premise (original code snippet, simplified)*

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def read_local_file():
    with open("/path/to/my_file.txt", "r") as f:
        data = f.read()
        print(data)  # Placeholder: Actual use of data
with DAG(
    dag_id='local_file_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    read_task = PythonOperator(
        task_id='read_local_file',
        python_callable=read_local_file,
    )
```

*Composer (modified code snippet)*

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.gcs import GCSDownloadOperator
from datetime import datetime

def read_gcs_file(bucket_name, file_name):
    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(file_name)
    data = blob.download_as_text()
    print(data)

with DAG(
    dag_id='gcs_file_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    download_file_task = GCSDownloadOperator(
       task_id='download_gcs_file',
       bucket='your-bucket-name',
       object='path/to/my_file.txt',
       save_to='/tmp/my_file.txt' # Note: Save to local filesystem of Airflow worker
   )
    read_task = PythonOperator(
        task_id='read_local_file',
        python_callable=read_gcs_file,
        op_kwargs={'bucket_name': 'your-bucket-name', 'file_name': 'path/to/my_file.txt'},
    )
    download_file_task >> read_task
```

**Commentary:** This example illustrates how to refactor a DAG to use Google Cloud Storage instead of the local file system. On Composer, we cannot directly read files from the local file system of the Airflow worker since this file system is ephemeral, we use the `GCSDownloadOperator` to download a file from Google Cloud Storage to a local temporary folder on the worker, then read it using standard Python. Instead of reading directly using a path, we interact with the `google.cloud.storage` API. This is a typical change necessary when moving from on-premise to cloud.

**Example 2: Handling Custom Dependencies**

This example demonstrates how to handle custom Python dependencies that might be present in your on-premise setup.

*On-premise (example of custom package installation)*

```bash
pip install custom_package==1.2.3
```
*Composer (requirements.txt)*

```text
custom_package==1.2.3
google-cloud-storage
```

**Commentary:** Instead of installing dependencies manually on each worker in the on-premise environment, we list the required packages along with their versions in the `requirements.txt` file, and ensure that any GCP specific package is included as well. Composer uses this file to create an isolated environment for your DAG execution. This ensures consistency and reduces the risk of version conflicts.  The `requirements.txt` file must be uploaded to the Composer environment's Cloud Storage bucket via the web UI or gcloud command-line tool, and will be picked up automatically by the Composer environment.

**Example 3: Transitioning from direct SQL connection to Cloud SQL**

This example showcases connecting to a SQL database, transitioning from a direct local connection to accessing Cloud SQL, and a basic example of using the `MySqlHook` provided by Airflow.

*On-premise (original code snippet, simplified)*

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import mysql.connector

def connect_to_local_sql():
    mydb = mysql.connector.connect(
        host="localhost",
        user="yourusername",
        password="yourpassword",
        database="yourdatabase"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM yourtable")
    results = mycursor.fetchall()
    print(results)

with DAG(
    dag_id='sql_local_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    sql_task = PythonOperator(
        task_id='query_sql_local',
        python_callable=connect_to_local_sql,
    )
```

*Composer (modified code snippet)*

```python
from airflow import DAG
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.operators.python import PythonOperator
from datetime import datetime

def connect_to_cloud_sql(sql_conn_id):
    mysql_hook = MySqlHook(mysql_conn_id=sql_conn_id)
    sql = "SELECT * FROM yourtable"
    results = mysql_hook.get_records(sql)
    print(results)

with DAG(
    dag_id='sql_cloud_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
     sql_task = PythonOperator(
        task_id='query_cloud_sql',
        python_callable=connect_to_cloud_sql,
        op_kwargs={"sql_conn_id": "your_sql_connection_id"}
    )
```

**Commentary:** This illustrates how to transition to a managed SQL service, such as Cloud SQL. The code utilizes the provided `MySqlHook` to manage the connection, as opposed to directly creating the database connection. In Composer you will need to configure a connection entry in the Airflow UI pointing to your cloud SQL instance, using either a static IP address, or the Cloud SQL Proxy. The `mysql_conn_id` needs to correspond to the connection ID defined in the Composer UI. This replaces the direct database connection in the on-premise setup with a managed connection, enhancing security and simplifying connection management.

To supplement this transition, explore Google Cloud documentation on Composer, specifically the guides on "Creating and Managing Environments" and "Managing Dependencies." Additionally, consider reviewing the official Apache Airflow documentation to understand how specific operators and hooks function. This combined knowledge base facilitates a smoother and more successful migration process. Investing time in code review and testing ensures the stability of the transitioned workflows, allowing you to fully leverage the benefits of Google Cloud Composer.
