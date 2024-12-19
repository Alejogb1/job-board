---
title: "How to Implement 'WITH .. AS' Statements in Airflow Dags?"
date: "2024-12-15"
id: "how-to-implement-with--as-statements-in-airflow-dags"
---

so, you're looking at how to use `with ... as` statements within airflow dags, right? i've been there, believe me. it’s not always as straightforward as you'd think coming from regular python, but it's totally doable and honestly, it makes your dags much cleaner. i remember when i first started using airflow, i was trying to cram everything into single operators, and the dag files looked like spaghetti code. learning to structure things better with contexts was a game changer.

the core issue here is that airflow dags are really just python scripts. but the execution environment is different from a regular script, especially when you start using more advanced operators and wanting to manage resources. the `with ... as` pattern is great for this because it ensures proper setup and teardown even if things go south in between. in my experience, this has saved me countless hours of debugging when jobs were failing mid-execution and leaving resources hanging.

the most common use cases for `with ... as` in airflow dags tend to revolve around resource management – things like connections, temporary file creation, database sessions, even complex operator configurations. the `with` statement, in essence, provides a clean way to manage that lifecycle. let me show you a couple of code snippets that i’ve used in real projects, that will clarify that:

first, let’s start with a simple example that is based on an airflow connection. imagine you have to interact with some external service through a connection object, for example, a database. it should be a custom connection type, and you need to use the connection through a with statement. here’s how that would typically look like:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.common.sql.hooks.sql import DbApiHook
from contextlib import contextmanager

@contextmanager
def get_my_custom_connection(conn_id):
    """
    context manager to handle a custom connection type
    """
    try:
       hook = DbApiHook(postgres_conn_id=conn_id)
       conn = hook.get_conn()
       yield conn
    finally:
        if conn:
            conn.close()

def execute_query_with_connection(conn_id):
     def _inner_execute(**kwargs):
         with get_my_custom_connection(conn_id) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT 1;")
            result = cursor.fetchone()
            print(f"result: {result}")
         return True
     return _inner_execute

with DAG(
    dag_id='with_connection_example',
    start_date=days_ago(1),
    schedule=None,
    catchup=False
) as dag:
    task_execute_query = PythonOperator(
        task_id='execute_query',
        python_callable=execute_query_with_connection(conn_id="my_postgres_connection"),
    )

```

in this example i've created a context manager `get_my_custom_connection` that encapsulates the setup and teardown logic related to a database connection using an airflow hook. the important parts are the `yield conn` that makes it available in the `with` statement context. and the `conn.close()` in the finally block, this ensures that no matter what happens inside the with block, the connection is always closed. now, every time you need a connection, you just use `with get_my_custom_connection('my_postgres_connection') as conn:` and you're guaranteed a clean, managed connection. this approach has saved me from many connection leakages when scripts failed unexpectedly.

another scenario that i have found myself often dealing with, is when needing to create a temporary file with some data to operate with, using the `with ... as` syntax it is great to manage a file life cycle. let’s say you have to produce some data, write it into a file, process it, and then remove it. this is where `with open(...) as f:` comes in super handy:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import tempfile
import os

def process_data_with_temp_file():
    def _inner_process(**kwargs):
        with tempfile.NamedTemporaryFile(mode="w+", delete=False) as temp_file:
            temp_file_path = temp_file.name
            temp_file.write("some data, in the file\n")
            temp_file.write("more data here.")
            print(f"created file at: {temp_file_path}")

        with open(temp_file_path, 'r') as read_file:
            content = read_file.read()
            print(f"content of file:\n {content}")

        os.remove(temp_file_path)
        print(f"file deleted")
        return True
    return _inner_process


with DAG(
    dag_id='temp_file_example',
    start_date=days_ago(1),
    schedule=None,
    catchup=False
) as dag:
    task_process_data = PythonOperator(
        task_id='process_data',
        python_callable=process_data_with_temp_file(),
    )
```

here, we are using python's built-in `tempfile.namedtemporaryfile` with the delete parameter set to false. this lets us get the path to the created file and do operations before it's automatically deleted. after the `with` block where we write the data, we can still open and read the file because it hasn't been deleted. and then finally with the `os.remove(temp_file_path)` statement, i manually remove the file. that guarantees that we are going to have it cleaned after the file has been used. without the `with` statement and manually handle it, it would be harder to correctly control when the file has to be cleaned up. one time i had a process fail and left thousands of temp files and disk space was gone in minutes. it was not pretty. this method avoids that situation.

finally, let’s talk about a more complex use case: setting up some operator parameters dynamically. i had this need when working with dynamically generated partitions. imagine you need to set up a `bigqueryinsertjoboperator` and you have to create partitions dynamically based on the date the dag runs, that’s where dynamic contexts can help a lot:

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.utils.dates import days_ago
from datetime import datetime

def create_partitioned_job(partition_date_str):
    def _inner_create(**kwargs):
            sql = f"""
            CREATE OR REPLACE TABLE my_dataset.my_partitioned_table
            PARTITION BY DATE(partition_col)
            AS
            SELECT
                'some data' as some_col,
                '{partition_date_str}' as partition_col
            """
            
            job_config = {
                "query": {
                     "query": sql,
                     "useLegacySql": False,
                }
            }
            
            with job_config as job_config_context:
                bigquery_job = BigQueryInsertJobOperator(
                    task_id=f'insert_job_{partition_date_str}',
                    configuration=job_config_context
                )
                bigquery_job.execute(context={})
            return True
    return _inner_create


with DAG(
    dag_id='bigquery_with_context_example',
    start_date=days_ago(1),
    schedule=None,
    catchup=False
) as dag:
    today = datetime.today().strftime('%Y-%m-%d')
    task_create_partition = PythonOperator(
        task_id='create_partition',
        python_callable=create_partitioned_job(partition_date_str=today),
    )

```

here we have the `job_config` dictionary, which is not a true context manager. but using the `with` statement allows for a block to do some operations in place before passing it to the operator. in this case, it does not need to use a `yield`. the operator gets the configuration from the `with` block and runs.

one thing to keep in mind when you’re working with contexts, especially in airflow, is how they interact with airflow’s templating engine. sometimes, you’ll need to use jinja templating along with your `with` contexts to generate parameters dynamically. this can get a little complex but it is necessary depending on the use case.

if you are interested in learning more, i recommend looking into the official python documentation on context managers and the `contextlib` module. it really lays out how all this works. also, the book “fluent python” by luciano ramalho has a great chapter on context managers and descriptors, it's a must-read for anyone doing advanced python programming.

and there you have it! that's how i've used the `with ... as` pattern in airflow dags over the years. it’s definitely one of those things that, once you start using it, you can’t imagine going back. it makes code cleaner, less error prone, and easier to debug. plus, it's kind of like adding a little bit of zen to your workflow - the setup is done, the work is done, cleanup is done. no mess, no fuss. just a nice, well-structured dag. it always reminds me of this one time when i was using a legacy system and had to manually clean resources… oh man what a nightmare. i can tell you that story later if you want. it is just like that joke about the programmer who had to clean his room, "it's not a bug, it's a feature of my environment". anyway, i hope this helps you in your airflow journey!
