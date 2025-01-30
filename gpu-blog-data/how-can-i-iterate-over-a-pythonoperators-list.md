---
title: "How can I iterate over a PythonOperator's list output in another Airflow 2 operator?"
date: "2025-01-30"
id: "how-can-i-iterate-over-a-pythonoperators-list"
---
The critical aspect of passing data between Airflow operators, particularly from a `PythonOperator` to a subsequent one, lies in understanding XComs and their limitations when dealing with list-like data structures. Unlike simple strings or integers which are readily serialized, complex objects like lists require careful handling to prevent common pitfalls in Airflow 2. The core challenge presented here involves extracting a list produced by a `PythonOperator` and using it to drive the execution of a downstream operator. Direct access to the `PythonOperator`'s return value outside its execution context is not viable; thus, XComs serve as the mechanism for this data transfer.

Specifically, a `PythonOperator` pushes its return value into an XCom with a default key ('return_value'). This default key can be overridden, which I generally recommend for clarity and maintainability, particularly when complex data like lists are involved. The downstream operator can then pull this data via the `ti.xcom_pull` method within its execution context (e.g., within another `PythonOperator` or a custom operator). However, if a large list is pushed to XComs, performance issues can occur because XCom data is generally serialized to the Airflow metadata database. Therefore, if the task is to iterate over a large list provided by an upstream `PythonOperator`, consider other strategies when the list grows excessively large, such as partitioning, but that exceeds the scope of this particular question.

My approach generally involves three key steps: first, ensure the upstream `PythonOperator` pushes the list to XComs under a clearly named key. Second, a downstream `PythonOperator` pulls that XCom data using the same key. Finally, the downstream task iterates over the retrieved list, performing necessary operations. The crucial aspect here is leveraging XCom functionality correctly and understanding its context within an Airflow DAG run. Let's see some examples.

```python
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime

def generate_list():
    """Generates a sample list of integers."""
    return list(range(5))

def process_list(ti):
    """Retrieves the list from XCom and processes each item."""
    my_list = ti.xcom_pull(task_ids='generate_data', key='my_data_list')
    for item in my_list:
        print(f"Processing item: {item}")

with DAG(
    dag_id='list_iteration_example',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
) as dag:
    generate_data_task = PythonOperator(
        task_id='generate_data',
        python_callable=generate_list,
        do_xcom_push=True, # Automatically pushes return value, but key is 'return_value'
    )

    process_data_task = PythonOperator(
        task_id='process_data',
        python_callable=process_list,
    )

    generate_data_task >> process_data_task
```

This first example demonstrates a rudimentary approach. Notice how in the `generate_data_task`, the default XCom push behavior is utilized (meaning it pushes using the key "return_value"), though I generally advise specifying custom keys for maintainability. The downstream `process_data_task` pulls the list by its default key.  While this example functions, It is not the ideal configuration.  I would usually specify `do_xcom_push=False` and specify a custom key using `ti.xcom_push` for explicit control. The `process_list` function iterates through the retrieved list, printing each item.  This example reveals one common mistake: not explicitly defining the XCom push key.

```python
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime

def generate_list_explicit(ti):
    """Generates a sample list of integers and pushes it to XCom with a custom key."""
    my_list = list(range(5))
    ti.xcom_push(key='my_data_list', value=my_list)


def process_list_explicit(ti):
    """Retrieves the list from XCom using custom key and processes each item."""
    my_list = ti.xcom_pull(task_ids='generate_data_explicit', key='my_data_list')
    for item in my_list:
        print(f"Processing item (explicit): {item}")

with DAG(
    dag_id='list_iteration_example_explicit',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
) as dag:
    generate_data_explicit_task = PythonOperator(
        task_id='generate_data_explicit',
        python_callable=generate_list_explicit,
    )

    process_data_explicit_task = PythonOperator(
        task_id='process_data_explicit',
        python_callable=process_list_explicit,
    )

    generate_data_explicit_task >> process_data_explicit_task
```

The second example introduces explicit XCom pushes and pulls, which, from my experience, leads to cleaner and more maintainable code.  In the `generate_data_explicit_task`, `do_xcom_push` is not explicitly set (defaults to `False` in this scenario). Instead, I use `ti.xcom_push` inside the `generate_list_explicit` function to push the list with the key `'my_data_list'`. Correspondingly, the `process_list_explicit` function in the subsequent `process_data_explicit_task` uses the `xcom_pull` method specifying this key to retrieve the list from XComs. This method is better practice because it is much more clear about what information is being passed between tasks. In practice, this provides more clarity when inspecting DAG runs or debugging issues.

```python
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from datetime import datetime

def generate_list_complex(ti):
    """Generates a sample list of dictionaries and pushes it to XCom with a custom key."""
    my_list = [{"id": i, "value": i * 2} for i in range(5)]
    ti.xcom_push(key='complex_data', value=my_list)


def process_list_complex(ti):
    """Retrieves the list from XCom using custom key and processes each dictionary."""
    my_list = ti.xcom_pull(task_ids='generate_data_complex', key='complex_data')
    for item in my_list:
        print(f"Processing dictionary: {item}, id: {item['id']}, value: {item['value']}")


with DAG(
    dag_id='list_iteration_example_complex',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
) as dag:
    generate_data_complex_task = PythonOperator(
        task_id='generate_data_complex',
        python_callable=generate_list_complex,
    )

    process_data_complex_task = PythonOperator(
        task_id='process_data_complex',
        python_callable=process_list_complex,
    )

    generate_data_complex_task >> process_data_complex_task
```

Finally, the third example builds upon the explicit XCom approach but involves a more complex list structure.  Here, the `generate_list_complex` function generates a list of dictionaries. I find that working with lists of dictionaries is very common when operating on data from various systems. The `process_list_complex` task demonstrates how to access elements within the dictionaries, further emphasizing that the full object is transferred through the XCom mechanism, not just a serialized string representation. This is useful when more complex information needs to be passed between operators.

In summary, when iterating over a list passed from a `PythonOperator` to another, always use XComs as the mechanism for data transfer. Avoid relying on the default XCom key of ‘return_value’ for any task other than perhaps a very simple example. Always explicitly push the list to XCom using `ti.xcom_push` with a custom and descriptive key, and then pull it in the subsequent operator with `ti.xcom_pull` using the same key. Always be mindful of the size of the list because large lists can have a performance impact when stored in the Airflow metadatabase, as XCom data is generally stored there. For extensive datasets, consider alternative approaches like transferring data using shared storage volumes or partitioned processing.

For further study on Airflow, I recommend the official Apache Airflow documentation. It offers the most comprehensive resource and a detailed explanation of its concepts. Also, consider exploring the resources available on Astronomer's website, which are usually geared towards practical applications and best practices. Finally, the book "Data Pipelines with Apache Airflow" can be a valuable resource for understanding advanced concepts. These resources will help provide a deeper understanding of Apache Airflow and its intricacies.
