---
title: "How can I access JSON data before an Airflow DAG runs?"
date: "2024-12-23"
id: "how-can-i-access-json-data-before-an-airflow-dag-runs"
---

Okay, let's tackle this. I’ve definitely been down this road before, a few times actually, and it’s a surprisingly common challenge. The core issue is accessing json data *before* airflow even kicks off a dag, meaning before any tasks are scheduled and executed. This implies we need to think outside the box a bit, moving beyond the usual airflow operators. It's less about dag *execution* and more about dag *definition* and how we provide those definitions with the necessary data.

The fundamental problem is that airflow dags are, at their core, python scripts. They are parsed and translated into task graphs, but this all happens *before* the actual scheduling process. Therefore, if we want to inject json data into our dag before it's even scheduled, we have to intercept that process. Here’s how I've generally approached this, which has worked well in my experience, especially when dealing with dynamic configurations or parameters that frequently change.

The most effective solution typically involves decoupling your dag definition from the source of your json data. We want to avoid hardcoding json directly in the dag script for maintainability reasons. Instead, think about creating a pre-processing step that reads this json data and makes it available to the dag. There are essentially three methods i frequently fall back to, each with their own use cases.

**Method 1: External Configuration Files & Environment Variables**

This is often the simplest to implement, and good for smaller sets of configurations. Let's assume that your json data describes parameters that change infrequently. In that case, it can be perfectly sufficient to read the json from a dedicated file using python's standard library.

Here’s a simplified example:

```python
import json
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Define path relative to dag file. Assumes json lives in same dir
CONFIG_FILE_PATH = os.path.join(os.path.dirname(__file__), 'config.json')

def load_json_config():
    try:
        with open(CONFIG_FILE_PATH, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"Config file not found at: {CONFIG_FILE_PATH}")
        return {}
    except json.JSONDecodeError:
        print(f"Invalid JSON format in: {CONFIG_FILE_PATH}")
        return {}

def my_task(config, **context):
    # Access json parameters here.
    print(f"task executed with config: {config}")
    task_id = context['ti'].task_id
    print(f"executing task {task_id}")
    print(f"parameter_a from json: {config.get('parameter_a', 'default_a')}")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
}


with DAG(
    dag_id='json_config_example_1',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    config_data = load_json_config()

    task_one = PythonOperator(
        task_id='task_one',
        python_callable=my_task,
        op_kwargs={'config': config_data},
    )
```

In this approach, we use the `os` module to build the path to the `config.json` file relative to the dag's file. The `load_json_config` function then reads that data and we pass it directly as an argument to the python operator. This works well for static configurations, but if you are storing the path of the file in environment variables then you can use `os.getenv("YOUR_ENV_VAR", "path/to/config.json")` to make things even more flexible.

**Method 2: Airflow Variables**

For data that can be updated frequently but is not *massive* , Airflow Variables can be very effective. We can use the airflow cli or the airflow web interface to update them. This allows us to programmatically change the behaviour of dags.

This method involves creating a separate process (outside the dag) that loads json data from external sources, and stores it as an airflow variable using the airflow cli or the api. Then, within the dag, you retrieve this variable. The key advantage here is that the json can be modified separately from the dag file, making dag logic independent of changing configurations.

Here is an example:

```python
import json
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime

def get_config_from_airflow_var():
    try:
       config_str = Variable.get("my_json_config_var")
       return json.loads(config_str)
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Error loading airflow variable: {e}")
        return {}

def my_task(config, **context):
    print(f"task executed with config: {config}")
    task_id = context['ti'].task_id
    print(f"executing task {task_id}")
    print(f"parameter_a from json: {config.get('parameter_a', 'default_a')}")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
}

with DAG(
    dag_id='json_config_example_2',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    config_data = get_config_from_airflow_var()

    task_one = PythonOperator(
        task_id='task_one',
        python_callable=my_task,
        op_kwargs={'config': config_data}
    )
```

In the above code, we fetch an airflow variable named `my_json_config_var`. Before running your dag, you’ll need to set this variable using the cli: `airflow variables set my_json_config_var '{"parameter_a": "value_from_var", "parameter_b": 123}'`. Keep in mind there are size limits on airflow variables. They are usually not suitable for very large json payloads.

**Method 3: Database or Message Queue**

For larger datasets, or frequently changing data, consider a more robust approach involving a database or message queue. Your json data can be stored and updated in a database (such as postgresql) or retrieved from a message queue (such as redis). Your dag can query this source as part of its definition process.

This involves creating a connection to your database or a consumer for your message queue *within* your dag's python file, retrieving the data, and subsequently parsing it.

```python
import json
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
from datetime import datetime

def get_config_from_db():
    pg_hook = PostgresHook(postgres_conn_id="my_postgres_conn")
    sql_query = "SELECT config_json FROM config_table WHERE config_id = 'my_config';"
    records = pg_hook.get_records(sql_query)

    if records and records[0] and records[0][0]:
       try:
            return json.loads(records[0][0])
       except json.JSONDecodeError:
            print(f"invalid JSON retrieved from database")
            return {}

    return {}

def my_task(config, **context):
    print(f"task executed with config: {config}")
    task_id = context['ti'].task_id
    print(f"executing task {task_id}")
    print(f"parameter_a from json: {config.get('parameter_a', 'default_a')}")


default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
}

with DAG(
    dag_id='json_config_example_3',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    config_data = get_config_from_db()

    task_one = PythonOperator(
        task_id='task_one',
        python_callable=my_task,
        op_kwargs={'config': config_data}
    )
```

In this example, we are using the `PostgresHook` to retrieve the json configuration from a postgresql database (make sure to create the airflow connection with the correct id) before the dag is even scheduled. Before running your dag, you’d need to populate your database with the configuration. For message queues, you'd follow a similar approach, fetching the json from the queue instead of a database.

**Choosing the Right Approach**

The choice of method depends entirely on your use case. For static configurations, method 1 is often sufficient. For configurations that change programmatically but are relatively small, Airflow Variables are a strong contender. When dealing with large, constantly updating datasets, you need to move towards a database or message queue based approach.

**Further Study**

For deeper insight, consider reading:

*   *Effective Python* by Brett Slatkin: Especially helpful for understanding the nuances of using Python's standard library and designing robust data loading patterns.
*   *Data Pipelines with Apache Airflow* by Bas Harenslak and Julian de Ruiter: For an in-depth exploration of airflow concepts and best practices, beyond the basics.
*   Relevant sections on the apache airflow documentation about variables and hooks. This is the canonical source of truth on how these things work.
*   Relevant sections in the psycopg2 or similar python drivers for interacting with databases, especially if you opt for method 3.

Finally, remember that clean code and thoughtful separation of concerns are key here. Do not hardcode your configurations. Always choose the tool that makes your workflows more maintainable and easier to debug. These three methods should cover most situations and allow for fairly complex configurations when combined and thoughtfully applied.
