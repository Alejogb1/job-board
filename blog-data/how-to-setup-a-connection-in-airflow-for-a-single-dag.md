---
title: "How to setup a connection in Airflow for a single DAG?"
date: "2024-12-23"
id: "how-to-setup-a-connection-in-airflow-for-a-single-dag"
---

Okay, let's dive into this. I've spent quite a bit of time in the trenches with airflow, and handling connection setups—especially on a per-dag basis—can get a little nuanced. It's not always as straightforward as the documentation might suggest at first glance. I've seen teams struggle with this, ending up with tangled connection definitions that become a maintenance headache. Instead of treating airflow connections as a global resource, focusing on per-dag setups can significantly improve your workflow and make things more manageable in the long run.

The challenge, really, isn't just *how* to configure a connection, but *where* to do it so it's both discoverable and scoped correctly to a single dag. You absolutely want to avoid global connection definitions unless they're truly application-wide. Otherwise, it leads to unintended consequences when dags start interacting with services they shouldn't have access to, or if one dag's connection conflicts with another.

So, we're going to look at a few methods to establish these isolated connections. The first, and generally recommended approach, revolves around utilizing airflow's variable system combined with a custom hook. This gives you fine-grained control. I remember back in 2018, working with an organization where we had a dozen different data warehouses, each used by different teams. That taught me the value of isolated connection definitions the hard way.

Here’s how I usually approach it using variables and a custom hook:

**Method 1: Leveraging Airflow Variables and a Custom Hook**

First, you'll define your connection details (host, user, password, etc.) not directly in the airflow ui or a connections json file, but as *airflow variables*. This is critical. Variables allow you to scope those credentials to the environment where your dag is running and, by structuring the variable keys carefully, make it very clear which dag owns which connection.

For instance, if your dag is named `my_etl_pipeline`, and you need a postgres connection for it, you might set up variables like this, preferably using the airflow command line or an external configuration tool:

*   `my_etl_pipeline_db_host`: `your_postgres_host`
*   `my_etl_pipeline_db_user`: `your_postgres_user`
*   `my_etl_pipeline_db_password`: `your_postgres_password`
*   `my_etl_pipeline_db_database`: `your_postgres_db`
*   `my_etl_pipeline_db_port`: `5432`

Next, you’ll build a custom hook that knows how to extract this information and form a connection object using these variables. Here’s an example, typically placed in a `hooks` subdirectory within your airflow dags folder:

```python
from airflow.hooks.base import BaseHook
from airflow.models import Variable
import psycopg2

class CustomPostgresHook(BaseHook):
    """
    A custom hook to connect to Postgres using Airflow variables.
    """

    def __init__(self, dag_id, conn_type="db"):
        super().__init__()
        self.dag_id = dag_id
        self.conn_type = conn_type

    def get_conn(self):
        host = Variable.get(f"{self.dag_id}_{self.conn_type}_host")
        user = Variable.get(f"{self.dag_id}_{self.conn_type}_user")
        password = Variable.get(f"{self.dag_id}_{self.conn_type}_password")
        database = Variable.get(f"{self.dag_id}_{self.conn_type}_database")
        port = Variable.get(f"{self.dag_id}_{self.conn_type}_port", default=5432)

        conn = psycopg2.connect(
            host=host,
            user=user,
            password=password,
            database=database,
            port=port,
        )
        return conn

```

In this snippet, the hook dynamically pulls the connection settings based on the `dag_id` which is passed in. The `conn_type` allows for more connections in one dag. Now, in your dag file, you'd use this custom hook:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from hooks.custom_postgres_hook import CustomPostgresHook

def my_postgres_task(**kwargs):
    hook = CustomPostgresHook(dag_id=kwargs["dag_id"])
    conn = hook.get_conn()
    cursor = conn.cursor()
    cursor.execute("SELECT 1;")
    result = cursor.fetchone()
    conn.close()
    print(f"Connection successful: {result}")

with DAG(
    dag_id="my_etl_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    run_postgres_task = PythonOperator(
        task_id="check_postgres_conn",
        python_callable=my_postgres_task,
    )

```
This approach cleanly separates connection configuration from your dag code and makes reuse much easier.

**Method 2: Parametrized Dag Definition**

Another approach, useful when you need different connection configurations for a dag instance, is to parametrize the dag with environment variables, allowing different connections depending on your deployment configuration. I found this extremely helpful when we transitioned our environment from testing to staging to production and needed different credentials based on the environment.

You might pass in `db_host`, `db_user`, etc. directly into the airflow environment variables and then retrieve them inside the DAG. Here's a small alteration to the above example to illustrate this (assuming you pass in the connection parameters prefixed with `MY_ETL_` as environment variables):

```python
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import psycopg2

def my_parametrized_postgres_task(**kwargs):
    host = os.environ.get('MY_ETL_DB_HOST')
    user = os.environ.get('MY_ETL_DB_USER')
    password = os.environ.get('MY_ETL_DB_PASSWORD')
    database = os.environ.get('MY_ETL_DB_DATABASE')
    port = os.environ.get('MY_ETL_DB_PORT', 5432)
    
    conn = psycopg2.connect(
        host=host,
        user=user,
        password=password,
        database=database,
        port=port
    )
    
    cursor = conn.cursor()
    cursor.execute("SELECT 1;")
    result = cursor.fetchone()
    conn.close()
    print(f"Connection successful: {result}")


with DAG(
    dag_id="my_parametrized_etl_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    run_postgres_task = PythonOperator(
        task_id="check_parametrized_postgres_conn",
        python_callable=my_parametrized_postgres_task,
    )
```

This method gives you flexibility on deployment but might be harder to debug if you have a large number of environment variables.

**Method 3: Dynamic Connection Definitions (less preferred)**

This is generally the least preferred approach but I've used it for quick prototypes where it was not necessary to persist the connection information beyond the dag runtime. You can define connections on-the-fly using the `Connection` object and the airflow meta-store. *However*, this usually involves storing credentials in the DAG's code, which I strongly advise against.

Here’s how that might look, but again, use with *extreme* caution and only for rapid prototyping/experimental dags. I've avoided storing credentials directly in this example, suggesting it to be dynamically defined elsewhere.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Connection
from airflow.utils.db import provide_session
from datetime import datetime
import psycopg2

@provide_session
def my_dynamic_connection_task(session=None, **kwargs):
  
    # Dynamically retrieve these settings from a secure method in a real project
    connection_details = {
        "host":  "your_dynamic_host", 
        "user":  "your_dynamic_user",
        "password": "your_dynamic_password",
        "database": "your_dynamic_db",
        "port": 5432,
    }

    conn = Connection(
        conn_id="dynamic_connection",
        conn_type="postgres",
        host=connection_details["host"],
        login=connection_details["user"],
        password=connection_details["password"],
        schema=connection_details["database"],
        port=connection_details["port"]
    )
    session.add(conn)
    session.commit()


    # Now use the connection
    airflow_conn = Connection.get_connection_from_secrets("dynamic_connection")
    postgres_conn = psycopg2.connect(
        host=airflow_conn.host,
        user=airflow_conn.login,
        password=airflow_conn.password,
        database=airflow_conn.schema,
        port=airflow_conn.port,
    )

    cursor = postgres_conn.cursor()
    cursor.execute("SELECT 1;")
    result = cursor.fetchone()
    postgres_conn.close()
    print(f"Dynamic connection successful: {result}")



with DAG(
    dag_id="my_dynamic_etl_pipeline",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    run_dynamic_connection_task = PythonOperator(
        task_id="check_dynamic_postgres_conn",
        python_callable=my_dynamic_connection_task,
    )
```
Keep in mind that the `Connection` is added to the Airflow metastore, so this is not purely ephemeral. For production environments, methods 1 and 2 are much better suited.

**Resource Recommendations:**

For diving deeper into airflow, I highly recommend the official documentation, specifically focusing on concepts around variables, hooks, and connections. Also, "Data Pipelines with Apache Airflow" by Bas Penders is a fantastic practical guide. Additionally, "Programming Apache Airflow" by Jesse Anderson and Justin Pihony is very useful as it covers the nuances behind best practices for airflow. Look into papers around "Software Architecture: Patterns, Principles, and Practices" as the concepts of modularity and separation of concerns are very relevant when dealing with these setup issues.

In my experience, the first method using custom hooks and airflow variables provides the best balance between security, reusability, and maintainability for most projects and situations. However, if your circumstances demand it, a parameterized dag or dynamic connection creation could become useful. Always remember that separation of concerns is your friend when dealing with connection details in airflow.
