---
title: "How can data be passed between Airflow DAGs?"
date: "2024-12-23"
id: "how-can-data-be-passed-between-airflow-dags"
---

Okay, let's dive into this. I recall a rather tricky situation back in my days at a fintech startup, where we had a suite of airflow dags that needed to communicate quite intricately. The simple solution, passing values directly between tasks within a single DAG, wasn't sufficient. We had multiple dags, each handling a different aspect of our financial pipeline - think customer onboarding, transaction processing, and risk analysis. The challenge was conveying information between these dags reliably and effectively. So, how do we manage data flow across separate airflow dags? There isn't a single 'perfect' method; instead, we have various techniques, each with its own set of trade-offs in terms of complexity, reliability, and scalability.

First off, let’s get one thing clear: directly sharing variables between dags is a no-go. Airflow is not designed to operate that way. Each dag runs in its own isolated execution context. What we need are mechanisms for storing and retrieving data in ways that are accessible across these contexts.

One straightforward approach, particularly suited for smaller datasets or metadata, is to use **Airflow's XCom (Cross Communication)** system. Think of XCom as a simple key-value store built into Airflow's metadata database. A task within a dag can push a value to XCom using a unique key, and then a task in *another* dag (or even the same dag) can subsequently pull that value, given it knows the correct dag id, task id, and key. The data you can store in XCom is technically limited by the database's string length constraints, so this method isn't suitable for large payloads, but it's convenient for configuration parameters, short summary results, or flags indicating a process's status.

Here’s a simplified python snippet, showcasing how we can send and receive values using XCom:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def push_to_xcom(**kwargs):
    kwargs['ti'].xcom_push(key='my_key', value='Hello from dag1!')

def pull_from_xcom(**kwargs):
    val = kwargs['ti'].xcom_pull(dag_id='dag1', task_ids='push_task', key='my_key')
    print(f"Value received from XCom: {val}")

with DAG(
    dag_id='dag1',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag1:
    push_task = PythonOperator(
        task_id='push_task',
        python_callable=push_to_xcom
    )

with DAG(
    dag_id='dag2',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag2:
    pull_task = PythonOperator(
        task_id='pull_task',
        python_callable=pull_from_xcom
    )
```

In the example, `dag1`'s `push_task` writes the string 'Hello from dag1!' to XCom. `dag2`'s `pull_task` then retrieves it. Notice that we explicitly define the source `dag_id` ('dag1'), `task_ids` ('push_task'), and `key` ('my_key') in `xcom_pull`. This demonstrates that dags are not intrinsically linked; we define these links programmatically.

However, for larger datasets or more persistent storage, XCom just won’t cut it. In these cases, we need to rely on more robust external storage systems. This brings us to our second major method: leveraging **external data stores** such as object storage (like AWS S3, Google Cloud Storage, or Azure Blob Storage), databases (relational or NoSQL), or message queues (like RabbitMQ or Kafka).

With object storage, for instance, a task in one dag can write its output to a specific location (a bucket and a file path), and a task in another dag can later retrieve this output. This approach offers high scalability and durability. Here's a very basic conceptual example illustrating how dags can interact using S3 (using the `boto3` library, which you would need to install):

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import boto3

def push_to_s3(**kwargs):
    s3 = boto3.client('s3')
    s3.put_object(
        Bucket='my-s3-bucket',
        Key='my-data-output.txt',
        Body=b'Data from dag3'
    )

def pull_from_s3(**kwargs):
    s3 = boto3.client('s3')
    response = s3.get_object(
        Bucket='my-s3-bucket',
        Key='my-data-output.txt'
    )
    content = response['Body'].read().decode('utf-8')
    print(f"Data retrieved from S3: {content}")

with DAG(
    dag_id='dag3',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag3:
    push_task = PythonOperator(
        task_id='push_task',
        python_callable=push_to_s3
    )

with DAG(
    dag_id='dag4',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag4:
    pull_task = PythonOperator(
        task_id='pull_task',
        python_callable=pull_from_s3
    )
```

This outlines the idea: the first dag writes to S3, and the second reads from S3. However, real-world applications are considerably more complex, including considerations like data partitioning, versioning, and access control, things that are vital when dealing with sensitive data.

Our third technique involves the use of **Airflow's sensor mechanisms** combined with triggers or external systems that communicate through an API. Instead of transferring data, a dag might generate some signal (for instance, inserting a record into a database table or adding a message to a queue), and then another dag is configured to listen for this signal using a sensor. When the sensor detects the event, the second dag proceeds. This approach decouples dag execution and is particularly useful when working with external event-driven systems. Here’s a concept demonstrating a simplified, database-based approach to this. Let’s assume we are just checking for a record in a table:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.sql import SqlSensor
from datetime import datetime
import sqlite3

def insert_trigger(**kwargs):
    conn = sqlite3.connect('my_database.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS triggers (status TEXT)")
    cursor.execute("INSERT INTO triggers (status) VALUES (?)", ('READY',))
    conn.commit()
    conn.close()


with DAG(
    dag_id='dag5',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag5:

    insert_task = PythonOperator(
        task_id='insert_task',
        python_callable=insert_trigger
    )

    sensor_task = SqlSensor(
        task_id='sensor_task',
        conn_id='sqlite_default',
        sql="SELECT 1 FROM triggers WHERE status = 'READY'",
        poke_interval=10,  # Check every 10 seconds
    )

    insert_task >> sensor_task
```

Here, `dag5` has `insert_task` which adds a status 'READY' to a sqlite database (again, this assumes sqlite is installed and configured correctly), and `sensor_task` continually checks the database for that status, then proceeds once found. This is more of an event-driven architecture which enables less tight coupling between dags.

It’s crucial to evaluate your specific use case, the volume of data, and the complexity of the pipeline. XCom is great for simple signaling. External storage provides scalability for large data volumes and robust persistence. Sensors facilitate loose coupling and event-driven execution. Selecting the right technique, or combination thereof, is key to building resilient and manageable airflow implementations. For deeper understanding I recommend examining the documentation of airflow directly and reading into best practices. “Designing Data-Intensive Applications” by Martin Kleppmann is a good general reference for designing data architectures, and for airflow in depth “Data Pipelines with Apache Airflow” by Bas Harenslak and Julian Rutger, offers in depth discussion on these topics. Understanding the trade-offs between different approaches is critical for a robust system.
