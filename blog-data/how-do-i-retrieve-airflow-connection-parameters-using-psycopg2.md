---
title: "How do I retrieve Airflow connection parameters using psycopg2?"
date: "2024-12-23"
id: "how-do-i-retrieve-airflow-connection-parameters-using-psycopg2"
---

Alright, let's unpack this. It's a question that often comes up when you're trying to bridge the orchestration power of airflow with the database access capabilities of psycopg2. I've definitely been down this path a few times, usually when building data pipelines that require dynamically configured database connections based on airflow variables. It's not always the most straightforward thing, but it's certainly manageable with the right approach.

The core issue is, of course, that airflow stores its connection parameters in its metadata database, separate from where your python code is executing within your dags. You don't want to hardcode passwords or database uris into your dag files; that's a major security faux pas. The good news is that airflow provides the mechanisms to securely retrieve these connections at runtime. Specifically, we use the `get_connection` method from the airflow hooks module.

The key is to understand the two separate parts involved: airflow connection definition and psycopg2 database connection setup. Airflow connections are abstract; they contain information about a particular resource, such as a database. They may hold the hostname, port, username, password, and database name. Psycopg2, on the other hand, is the python library we'll use to connect to a postgres database, requiring specific keyword arguments corresponding to the database parameters.

Let’s break this down practically.

First, inside your DAG, you'll need to retrieve the connection details from airflow based on the connection id configured in the UI. Let’s assume we have an airflow connection configured, maybe named `postgres_conn`. Here’s how we access it within our Python code (likely a PythonOperator, but could be anywhere in your dag context):

```python
from airflow.hooks.base_hook import BaseHook
from psycopg2 import connect
from psycopg2 import Error
from airflow.models import Variable

def retrieve_and_connect():
    try:
        conn_id = Variable.get("my_db_connection") # Example with variable - good practice
        conn = BaseHook.get_connection(conn_id)
        db_params = conn.extra_dejson

        connection_params = {
            'host': conn.host,
            'port': conn.port,
            'user': conn.login,
            'password': conn.password,
            'dbname': conn.schema if conn.schema else db_params.get('database', 'default_database'), # default handling if schema is not available
        }

        with connect(**connection_params) as pg_conn:
            with pg_conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()
                print(f"Connection test result: {result}")

    except Error as e:
        print(f"Database connection error: {e}")
    except Exception as e:
        print(f"Airflow retrieval error: {e}")
```
Let's break down what's happening here:

1.  **`from airflow.hooks.base_hook import BaseHook`**: we import the base hook, required to interact with airflow's connection definitions.
2.  **`from psycopg2 import connect, Error`**: we import psycopg2 connect method and the error class for proper exception handling.
3.  **`from airflow.models import Variable`**: we are getting the connection id from airflow variables. This makes it much easier to manage different connections for different environments.
4.  **`conn_id = Variable.get("my_db_connection")`**: Here we use airflow variables to inject the right connection id from Airflow UI (e.g. `postgres_conn`). This is a robust way to deal with configuration.
5.  **`conn = BaseHook.get_connection(conn_id)`**: This retrieves the airflow connection object based on the connection id.
6.  **`db_params = conn.extra_dejson`**: This is a critical part. Airflow lets you store extra JSON data with your connection definition. We're unpacking it here. Often times additional database specific connection parameters are stored here.
7.  **`connection_params`**: We create a dictionary `connection_params` suitable for psycopg2, mapping the airflow connection attributes to psycopg2's expected keywords. Notice, I am handling schema with a fallback of `default_database`. It is important to have proper defaults when using connections.
8.  **`with connect(**connection_params) as pg_conn:`**: The actual connection to postgres is created using psycopg2 and managed in a context to properly close the connection.
9. **Error Handling:** Notice the nested `try-except` block. This is important for production code. You want to catch both database and airflow retrieval errors.

Now, you might be thinking: 'That's great for a standard postgres connection, but what about if we want to handle different parameter names or add other options?' That's a valid point. Let’s consider a situation where you’ve added an extra field called 'sslmode' or another parameter into the connection's extra settings. We need to unpack this with robust handling:

```python
from airflow.hooks.base_hook import BaseHook
from psycopg2 import connect
from psycopg2 import Error
from airflow.models import Variable


def retrieve_and_connect_advanced():
    try:
        conn_id = Variable.get("my_db_connection")
        conn = BaseHook.get_connection(conn_id)
        db_params = conn.extra_dejson

        connection_params = {
            'host': conn.host,
            'port': conn.port,
            'user': conn.login,
            'password': conn.password,
            'dbname': conn.schema if conn.schema else db_params.get('database', 'default_database'),
        }
        
        # Add extra parameters if present
        ssl_mode = db_params.get('sslmode') # Try to get sslmode
        if ssl_mode:
            connection_params['sslmode'] = ssl_mode
        
        other_param = db_params.get('other_parameter')
        if other_param:
            connection_params['other_parameter'] = other_param


        with connect(**connection_params) as pg_conn:
            with pg_conn.cursor() as cursor:
                cursor.execute("SELECT 1;")
                result = cursor.fetchone()
                print(f"Advanced connection test result: {result}")

    except Error as e:
        print(f"Database connection error: {e}")
    except Exception as e:
        print(f"Airflow retrieval error: {e}")

```

In the above example:

1.  We're still retrieving the connection parameters the same way, but we're extracting potential additional parameters from `db_params`.
2. We selectively add only if the parameter is present. Note this pattern ensures the application does not break for connections that may not have extra parameters configured.
3. This gives you the flexibility to add additional parameters without breaking your code.

Finally, consider the case where your database parameters are not stored in the `extra` field at all and are all directly in the connection parameters. This is also very common, especially when initially setting up connections. Here's how we would approach that situation:

```python
from airflow.hooks.base_hook import BaseHook
from psycopg2 import connect
from psycopg2 import Error
from airflow.models import Variable

def retrieve_and_connect_direct():
    try:
        conn_id = Variable.get("my_db_connection")
        conn = BaseHook.get_connection(conn_id)


        connection_params = {
            'host': conn.host,
            'port': conn.port,
            'user': conn.login,
            'password': conn.password,
            'dbname': conn.schema if conn.schema else conn.database,
        }

        with connect(**connection_params) as pg_conn:
            with pg_conn.cursor() as cursor:
                 cursor.execute("SELECT 1;")
                 result = cursor.fetchone()
                 print(f"Direct connection test result: {result}")

    except Error as e:
        print(f"Database connection error: {e}")
    except Exception as e:
        print(f"Airflow retrieval error: {e}")
```

Here, we pull the schema directly from the `conn` object, or we use the `database` field, depending on your configuration. We are also pulling all the standard parameters directly from the `conn` object. The advantage is that this is very simple and easy to understand. The disadvantage is that you would have to make the connection information directly in airflow.

For further exploration, I recommend reviewing the official airflow documentation, particularly the sections on hooks and connections. Also, for a deeper understanding of database interactions with python, you should check out the psycopg2 documentation. And if you're interested in more advanced patterns related to accessing databases in airflow, "Data Pipelines with Apache Airflow" by Bas Geerdink and "Effective DevOps" by Jennifer Davis and Katherine Daniels could be helpful for patterns and practices related to this.

In my experience, the flexibility of airflow, when combined with a library like psycopg2, allows for robust and adaptable data pipelines. The key is to grasp the differences in where each library manages connection settings and how to bridge the two. Using variable lookup and proper error handling are critical in production.
