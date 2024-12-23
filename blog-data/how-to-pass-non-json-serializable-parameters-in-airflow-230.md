---
title: "How to pass non-JSON-serializable parameters in Airflow 2.3.0?"
date: "2024-12-23"
id: "how-to-pass-non-json-serializable-parameters-in-airflow-230"
---

Alright, let's talk about non-json serializable parameters in Airflow, specifically in the 2.3.0 version. I've bumped into this particular challenge a few times over the years, especially when dealing with more complex data structures, and it can certainly be a point of friction if not approached correctly.

The core issue, as most probably know, is that Airflow's task parameters, when serialized and sent to workers, primarily rely on JSON serialization. This is a practical choice for its relative simplicity and widespread compatibility. However, this mechanism falls apart when you need to pass parameters that don't naturally translate to JSON – things like datetime objects, custom Python objects, or even sets and tuples.

In my experience, I've encountered this most frequently when interacting with custom libraries or APIs that return or expect specialized data types. Once, I was working on a data ingestion pipeline that processed time series data. The custom library I was using would accept a `pandas.Timestamp` object as an argument for specifying the data range, and predictably, trying to pass it directly through Airflow's task parameters resulted in serialization errors. Similarly, I had a case involving a complex graph library that relied heavily on custom graph node objects, and sending those as task parameters was a non-starter as well.

So, what's the workaround? Well, there isn't a magic bullet, but rather a set of techniques and best practices to ensure your non-json serializable data transits effectively. Let me break down three general approaches I've found most useful.

**1. Manual Serialization and Deserialization:**

The most straightforward method involves manually serializing the problematic object into a JSON-friendly representation, such as a string or a dictionary, before passing it to the Airflow task. On the receiving end, inside your task, you would then deserialize it back into its original form. This approach provides the highest level of control and is often necessary for custom objects.

Here's a code snippet demonstrating this using the `pickle` library to handle arbitrary object serialization.

```python
import pickle
from datetime import datetime
from airflow.decorators import task
from airflow.models import DAG
from airflow.utils.dates import days_ago

with DAG(
    dag_id='manual_serialization_dag',
    schedule_interval=None,
    start_date=days_ago(2),
    tags=['example'],
) as dag:
    @task
    def my_task(serialized_datetime):
        deserialized_datetime = pickle.loads(serialized_datetime.encode('latin1'))
        print(f"Received datetime: {deserialized_datetime}")

    my_datetime = datetime.now()
    serialized_datetime = pickle.dumps(my_datetime).decode('latin1')
    my_task(serialized_datetime=serialized_datetime)
```

In this example, we are pickling the `datetime` object before passing it into the task. We decode it with ‘latin1’ to make it json serializable. Inside the task, we reverse the process by unpickling, successfully retrieving the original object. The critical piece here is using `pickle` which can serialize most Python objects. Ensure to use ‘latin1’ encoding to avoid encoding/decoding issues related to the utf-8 encoding that JSON uses.

**2. Using a Data Store as an Intermediate:**

Another strategy involves using an external data store, like a database, a key-value store, or even cloud storage, as a staging area. Instead of passing the object directly, you pass a reference (like an ID or key) that the task can use to fetch the actual object. This is particularly effective for larger, more complex objects that would become unwieldy to serialize directly. It also helps with data governance, since the intermediate data storage provides a clear audit trail.

Let's look at an example using Redis as a key-value store:

```python
import redis
import json
from datetime import datetime
from airflow.decorators import task
from airflow.models import DAG
from airflow.utils.dates import days_ago

redis_client = redis.Redis(host='localhost', port=6379, db=0)  #configure as per your environment

with DAG(
    dag_id='redis_serialization_dag',
    schedule_interval=None,
    start_date=days_ago(2),
    tags=['example'],
) as dag:
    @task
    def my_task(datetime_key):
        serialized_datetime = redis_client.get(datetime_key)
        if serialized_datetime:
           deserialized_datetime = json.loads(serialized_datetime)
           print(f"Received datetime: {datetime.fromisoformat(deserialized_datetime)}")
        else:
           print("Key not found in redis")

    my_datetime = datetime.now()
    datetime_key = f"my_datetime_{my_datetime.isoformat()}"
    redis_client.set(datetime_key, json.dumps(my_datetime.isoformat()))

    my_task(datetime_key=datetime_key)
```

Here, we store a json serializable string of the datetime object in Redis using a unique key. The task then retrieves the data using the same key and restores the original object. This approach avoids direct serialization and deserialization within Airflow itself, offloading the storage and retrieval of complex data.

**3. Utilize XComs with Serialization:**

Airflow's XComs mechanism offers a third avenue. While XComs are also serialized via JSON by default, the framework provides hooks to customize the serializer. XComs are most efficient for sharing data within the same DAG run. You can push the complex object as an XCom from one task and then pull it in another, using custom serialization and deserialization code. Note that this might introduce complexity if the data type changes between the push and pull operations and should be avoided if possible for this reason.

Here is an example demonstrating the use of custom serialization within XCom.

```python
import json
import pickle
from datetime import datetime
from airflow.decorators import task
from airflow.models import DAG
from airflow.utils.dates import days_ago
from airflow.operators.python import PythonOperator


def custom_xcom_serializer(value):
    return json.dumps({'serialized_value': pickle.dumps(value).decode('latin1')})

def custom_xcom_deserializer(value):
    return pickle.loads(json.loads(value)['serialized_value'].encode('latin1'))

with DAG(
    dag_id='custom_xcom_serialization_dag',
    schedule_interval=None,
    start_date=days_ago(2),
    tags=['example'],
) as dag:
    @task
    def push_datetime_task():
        my_datetime = datetime.now()
        return my_datetime

    def pull_datetime_task(**kwargs):
        ti = kwargs['ti']
        pulled_datetime = ti.xcom_pull(task_ids='push_datetime_task')
        print(f"Received datetime: {pulled_datetime}")


    push_task = push_datetime_task()
    pull_task = PythonOperator(
        task_id='pull_datetime_task',
        python_callable=pull_datetime_task,
        op_kwargs = {'serializer': custom_xcom_serializer, 'deserializer': custom_xcom_deserializer}
    )
    push_task >> pull_task

```

In this example, a datetime object is pushed into XCom, using the custom serializer defined, which uses pickle to do the serialization, and is extracted using the same mechanism. XCom serialization is also not advisable when passing large amounts of data as it stores the values in the metadata store, which can potentially cause performance issues.

**Final Thoughts:**

Choosing the appropriate method for handling non-json serializable parameters depends on your specific needs. For small, simple objects, manual serialization via `pickle` or the `json` module as shown, is often sufficient. When dealing with larger or more complex datasets, employing a data store or XComs with custom serialization techniques would be a better approach. Keep an eye on the amount of data pushed into XCom to avoid performance issues.

For further exploration, consider consulting the following resources:

*   **"Programming in Python 3" by Mark Summerfield:** This book provides a very comprehensive overview of Python's core capabilities, including serialization techniques.
*   **"Fluent Python" by Luciano Ramalho:** This is another excellent resource for understanding python best practices, including pickle limitations and json serialization.
*   **The Apache Airflow documentation:** The official documentation provides thorough details about task parameters, XComs, and various configuration options that can be helpful. Search in particular the section on "xcoms" and serialization for the latest details.

By understanding these methods, I’m confident you can effectively navigate the challenges of working with non-json serializable parameters in Airflow. These solutions have consistently served me well in practice, providing a reliable way to maintain the functionality required, and have also helped avoid major disruptions in my workflows.
