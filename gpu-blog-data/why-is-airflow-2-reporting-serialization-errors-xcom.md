---
title: "Why is Airflow 2 reporting serialization errors (XCom JSON) even if tasks have completed?"
date: "2025-01-30"
id: "why-is-airflow-2-reporting-serialization-errors-xcom"
---
XCom serialization errors in Apache Airflow 2, particularly after task completion, often stem from a mismatch between the data being pushed to XCom and the default JSON serializer's capabilities, coupled with Airflow's deferred cleanup mechanisms. I've encountered this frequently, especially with larger datasets or when using custom Python objects that aren't natively JSON serializable. Let me elaborate on the nuances.

The core issue isn't necessarily that your tasks are *failing* to complete; rather, the problem arises when Airflow attempts to serialize the XCom value after a successful task execution. XCom, or Cross-Communication, is the primary way tasks in Airflow communicate data. It essentially functions as a database table where task outputs are stored as key-value pairs. These values are serialized before being stored, typically using JSON. The problem arises because standard JSON has limitations. It directly supports primitive types (strings, numbers, booleans, and null), lists, and dictionaries containing only these primitives or other lists/dictionaries. When you push anything beyond these structures (e.g., custom Python objects, Pandas DataFrames, complex NumPy arrays) through XCom without explicit handling, the default serializer will raise an exception.

A crucial aspect to consider is that Airflow doesn't serialize and immediately write to XCom upon the completion of a task. Instead, it often performs a "lazy" write, or may defer this action to another process or point in the DAG execution. This deferred writing and serialization can result in post-task completion errors that appear confusing. For instance, a worker process may successfully execute a task, but the serialization of its output fails during a later stage when the scheduler or another process attempts to persist that data. This is what creates the perception of serialization errors occurring despite completed tasks. The situation can be amplified by the volume of data pushed through XCom – larger datasets increase the likelihood of encountering serialization limits, such as JSON's limitations on string length or nesting depth.

Beyond the basic serialization constraints, using different types of Airflow executors also has an impact. For example, a Celery executor often involves different processes that handle task execution and serialization separately. This adds complexity when tracing the source of errors. If, for example, a task running in a Docker executor pushes a complex NumPy array, and the scheduler or a different Celery worker attempts the serialization, the serialization error could occur outside the context of the task execution itself.

Let's look at some code examples demonstrating the various failure points:

**Example 1: Unserializable Custom Object**

```python
from airflow import DAG
from airflow.decorators import task
from datetime import datetime

class MyCustomObject:
    def __init__(self, value):
        self.value = value

with DAG(
    dag_id="xcom_serialization_custom_object",
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    @task
    def create_custom_object():
        return MyCustomObject(123)

    @task
    def use_custom_object(obj):
      print(f"Object Value: {obj.value}")

    custom_object = create_custom_object()
    use_custom_object(custom_object)
```

In this scenario, the `create_custom_object` task completes successfully. However, when Airflow attempts to serialize the `MyCustomObject` instance pushed via XCom, a JSON serialization error is raised. The default JSON serializer does not know how to serialize custom Python classes like `MyCustomObject`. This error will typically surface in the logs, even after `create_custom_object` has marked itself as successful. The traceback will usually involve components responsible for XCom serialization within the Airflow infrastructure and not within the actual task execution log.

**Example 2: Large Dictionary**

```python
from airflow import DAG
from airflow.decorators import task
from datetime import datetime
import json

with DAG(
    dag_id="xcom_serialization_large_dict",
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    @task
    def create_large_dict():
        large_dict = {}
        for i in range(100000): #Creating large dict
            large_dict[str(i)] = "value_" + str(i)
        return large_dict

    @task
    def consume_dict(data):
        print(f"Number of keys: {len(data)}")

    large_data = create_large_dict()
    consume_dict(large_data)
```
Here, the `create_large_dict` task generates a dictionary with a large number of entries. While Python itself can handle this, the default JSON serializer might encounter limits on the string size when converting the dictionary keys and values to JSON format. While this specific example may not consistently raise errors, in practice, larger dictionary/string sizes or more complex nested dictionary can hit the edge cases of the JSON serializer and will result in the XCom serialization error.

**Example 3: Handling Serialization with `json.dumps`**

```python
from airflow import DAG
from airflow.decorators import task
from datetime import datetime
import json
import numpy as np

with DAG(
    dag_id="xcom_serialization_json_dumps",
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:

    @task
    def create_numpy_array():
        return np.array([1, 2, 3, 4, 5])

    @task
    def consume_array(array):
        print(f"Array: {array}")

    @task
    def serialize_numpy(array):
        return json.dumps(array.tolist())

    @task
    def deserialize_numpy(json_str):
        return np.array(json.loads(json_str))

    array = create_numpy_array()
    serialized_array = serialize_numpy(array)
    deserialized_array = deserialize_numpy(serialized_array)
    consume_array(deserialized_array)
```

This example shows a workaround for serializing a NumPy array by converting it into a list and then using `json.dumps` to serialize it as a JSON string before pushing it through XCom. The `deserialize_numpy` task then deserializes the JSON string back to a NumPy array using `json.loads`. By explicitly handling the serialization before pushing to XCom, the issues associated with the default JSON serializer are avoided. While this approach works, it requires you to implement custom serialization for non-standard objects.

To address these serialization issues, consider these strategies:

1.  **Pre-Serialization:** Implement explicit serialization within your tasks, as shown in example 3. This involves converting objects to basic JSON-compatible types before pushing to XCom. For complex objects, consider saving them to a persistent store (e.g., blob storage, database) and passing identifiers through XCom instead. This method circumvents the limitations of the JSON serializer entirely.

2.  **Custom XCom Backend:** Implement a custom XCom backend by extending Airflow's `BaseXCom` class. This involves specifying custom serialization and deserialization logic. For instance, you could use a different serializer, such as pickle or cloudpickle, but this requires careful consideration of security implications.

3.  **Reduce Data Volume:** If possible, reduce the amount of data passed through XCom. Passing large datasets contributes to serialization problems. Try passing only essential data like identifiers and retrieve larger datasets directly from a data store within subsequent tasks when needed.

4.  **Verify Data Types:** Ensure that only basic, JSON-compatible data types are being pushed to XCom. Implement proper error handling in your tasks to detect and log any unexpected object types that may be propagated through the XCom system. Using an Airflow Listener, you can also inspect the data being pushed to XCom and debug based on the reported type of that data.

For further reference, I would recommend reading the official Airflow documentation on XCom, the section discussing custom XCom backends and exploring StackOverflow discussions related to Airflow and serialization for similar use cases. Additionally, reviewing the Python documentation related to the `json`, `pickle`, and `cloudpickle` libraries, will provide better insight into their applicability for serialization. These references will help deepen your understanding and troubleshoot the serialization issues you are facing.
