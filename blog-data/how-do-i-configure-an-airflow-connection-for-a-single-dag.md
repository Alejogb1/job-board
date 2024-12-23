---
title: "How do I configure an Airflow connection for a single DAG?"
date: "2024-12-23"
id: "how-do-i-configure-an-airflow-connection-for-a-single-dag"
---

Let's talk Airflow connections. I've spent a fair bit of time knee-deep in DAGs and airflow infrastructure, and the matter of connections is something that comes up again and again. It’s not just about plugging credentials in; it’s about doing so in a secure, manageable, and scalable way, particularly when you're dealing with multiple environments. Your question focuses on configuring a connection for *a single DAG*, which, while seemingly straightforward, has layers that deserve attention. We shouldn’t just think about what to do, but also *why* and consider potential pitfalls.

First, it's crucial to understand that Airflow connections are not defined within the DAG itself. They're stored separately in the Airflow metadata database. This is by design. Separating data access configurations from workflow logic enhances security and allows for easier management. Imagine having hardcoded database passwords in each DAG – a nightmare for security audits and version control.

My experience includes a rather memorable project where we transitioned from hardcoded credentials in DAG definitions to leveraging the Airflow UI and environment variables, after a significant security breach during a proof-of-concept phase. Let's just say it was a hard lesson learned and a catalyst for adopting better practices. So, what are the options for your single DAG?

The most straightforward approach involves utilizing the Airflow UI or CLI. You define the connection details there, and then your DAG refers to it by its connection id. This method works well for development or small setups. You can navigate to `Admin` > `Connections` in the Airflow UI, select “Create,” specify the connection type (e.g., postgres, http, s3) and fill in the specific connection details. However, this isn’t scalable for production settings where connections need to be programmatically defined or managed via infrastructure-as-code.

Let’s illustrate how the DAG would reference a preconfigured connection, presuming we've set up a connection named `my_postgres_connection` using the UI or via CLI.

```python
from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from datetime import datetime

with DAG(
    dag_id="single_dag_postgres_example",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    run_sql_query = PostgresOperator(
        task_id="execute_sql_query",
        postgres_conn_id="my_postgres_connection",
        sql="SELECT 1;",
    )
```
Notice that the crucial part is `postgres_conn_id="my_postgres_connection"`. The operator uses this identifier to fetch the connection details from the metadata database. The specifics of how to use the PostgresOperator would require that you consult the apache-airflow-providers documentation, but here you can see the pattern.

Now, this direct UI approach isn't ideal for production, especially if you are practicing infrastructure as code. For that, we’d leverage environment variables and/or use a tool like Airflow’s `connections` file alongside a CI/CD workflow. The idea is to avoid manual updates in the UI. A `connections` file can contain a serialized list of dictionaries that represent your connections, which can be pushed to a container or system where Airflow is running. Airflow can read them in upon start-up, or they can be loaded into the metadata database via CLI.

Suppose we want to programmatically define our Postgres connection via the `connections` file, we would write a json config similar to this:
```json
[
  {
    "conn_id": "my_postgres_connection",
    "conn_type": "postgres",
    "host": "my_db_host",
    "schema": "my_db_schema",
    "login": "my_db_user",
    "password": "my_db_password",
    "port": 5432
   }
]
```

You'd then load it using the CLI: `airflow connections import my_connections.json`. This provides a repeatable way to create connections across all environments. Note that you should handle the credentials carefully; they can be sensitive.

However, storing sensitive information in plain text files, even in a managed environment, is not considered a best practice. A much better approach is to use Airflow's built-in support for environment variables or, even better, secrets managers.

Let's assume you are using environment variables. In this case, you'd still define a connection in the UI/CLI, but you could use placeholders within the connection configuration. Airflow, upon accessing the connection, can substitute placeholders with the corresponding environment variable values.

Imagine setting the following environment variables in your Airflow environment:

`AIRFLOW_CONN_MY_POSTGRES_CONNECTION_HOST=my_db_host`
`AIRFLOW_CONN_MY_POSTGRES_CONNECTION_LOGIN=my_db_user`
`AIRFLOW_CONN_MY_POSTGRES_CONNECTION_PASSWORD=my_db_password`
`AIRFLOW_CONN_MY_POSTGRES_CONNECTION_SCHEMA=my_db_schema`

In this situation, the connection configuration defined in the UI, CLI or `connections` file might look like this:
```json
[
  {
    "conn_id": "my_postgres_connection",
    "conn_type": "postgres",
    "host": "{{ var.value.AIRFLOW_CONN_MY_POSTGRES_CONNECTION_HOST }}",
    "schema": "{{ var.value.AIRFLOW_CONN_MY_POSTGRES_CONNECTION_SCHEMA }}",
    "login": "{{ var.value.AIRFLOW_CONN_MY_POSTGRES_CONNECTION_LOGIN }}",
    "password": "{{ var.value.AIRFLOW_CONN_MY_POSTGRES_CONNECTION_PASSWORD }}",
    "port": 5432
   }
]
```
Now, instead of hardcoding, your credentials are securely retrieved from environment variables. This is a major improvement, but this process can also become a bit challenging to manage as the number of secrets grow or if you are managing multiple environments. For many people, the "Variable" feature is sufficient but you should be aware of all options.

To extend this, a third code snippet demonstrates an approach using placeholders and a variable lookup in the DAG itself (but it's not recommended):

```python
from airflow import DAG
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.models import Variable
from datetime import datetime

with DAG(
    dag_id="single_dag_postgres_example_env_var",
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    run_sql_query_using_env_vars = PostgresOperator(
        task_id="execute_sql_query_env_vars",
        postgres_conn_id="my_postgres_connection",
        sql="SELECT 1;",
        hook_params={
           "host": Variable.get("AIRFLOW_CONN_MY_POSTGRES_CONNECTION_HOST"),
           "login": Variable.get("AIRFLOW_CONN_MY_POSTGRES_CONNECTION_LOGIN"),
           "password": Variable.get("AIRFLOW_CONN_MY_POSTGRES_CONNECTION_PASSWORD"),
           "schema": Variable.get("AIRFLOW_CONN_MY_POSTGRES_CONNECTION_SCHEMA")
       }
    )
```
Here, instead of relying entirely on placeholders within the connection config, we are fetching the parameters via airflow variable functions. This is not a common approach but it illustrates how dynamic connections could also be utilized. You'd still define a connection ID ( `"my_postgres_connection"` ) in Airflow, but we are overriding key parameters at runtime. As mentioned, this is not a best-practice as the parameters are not encrypted, nor centralized. Use this only if you are experimenting.
While this is feasible, it’s generally less preferable than relying on properly set environment variables or secret managers, since, once again, we are starting to pull values from the dag itself.

For a deep understanding of these topics, I would recommend reading "Designing Data-Intensive Applications" by Martin Kleppmann for broader architectural patterns and then exploring the official Apache Airflow documentation, particularly the sections on connections, security, and the cli tools. The Apache Airflow provider documentation should also be consulted whenever a specific data store operator is being used (like the Postgres example I have given).

In summary, managing connections for an Airflow DAG involves several options. Using the UI is a quick way to get started, but for production workflows, programmatically managing them via environment variables, files, or secrets managers is essential. Always be careful about how you handle sensitive connection information. It's not about *making it work*, it's about making it secure, repeatable, and manageable in the long run. This is an area where having a well-planned approach is well worth the initial effort.
