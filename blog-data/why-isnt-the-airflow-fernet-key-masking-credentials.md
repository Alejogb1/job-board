---
title: "Why isn't the Airflow Fernet key masking credentials?"
date: "2024-12-23"
id: "why-isnt-the-airflow-fernet-key-masking-credentials"
---

Alright,  I've seen this problem pop up more times than I care to count, and it's usually a head-scratcher for those diving into airflow’s security model. The issue isn’t that airflow *can't* mask credentials using Fernet; it’s more a nuanced situation where the implementation can be misunderstood or incorrectly configured, leading to the impression that it’s simply not working. I recall one particularly gnarly case back at *TechSolutions Inc.* where we spent a whole afternoon tracing connection string leakage – a real lesson in the importance of config management.

At its core, the Fernet key in airflow is designed to encrypt sensitive information, and this definitely includes credentials. However, the masking you see in the airflow UI, specifically in places like task logs or the webserver's variable view, is a *separate* process from the actual encryption happening behind the scenes. The Fernet key is responsible for securing your credentials at rest, making them indecipherable to anyone without the key, typically in the metadata database. The masking in the UI, on the other hand, is a UI-level concern to ensure secrets aren't inadvertently displayed to users. These two processes are often confused.

The disconnect frequently occurs due to several factors. First, *not all credential storage methods automatically leverage Fernet encryption*. For instance, if you’re manually populating variables directly through the UI or via the command line without explicitly triggering encryption, the value may be stored in the database as plaintext. The UI then masks the view, but the data itself isn't encrypted. Additionally, if you're using a custom secrets backend (beyond the database), that backend’s integration with Fernet needs to be carefully implemented to achieve proper encryption of secrets at rest. Another aspect is the timing. If your airflow deployment wasn’t initially set up with a Fernet key (or if you started before it was mandated in airflow 2.0), any secrets loaded prior to the enforcement may exist unencrypted even if you add the key later.

Let's break this down with a few code examples demonstrating common scenarios and how to handle them.

**Example 1: Setting an unencrypted variable directly**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def print_variable(**kwargs):
    var_value = kwargs['ti'].xcom_pull(task_ids='set_variable', key='my_secret')
    print(f"Variable value: {var_value}")


with DAG(
    dag_id="unencrypted_var_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    set_variable_task = PythonOperator(
        task_id="set_variable",
        python_callable=lambda **kwargs: kwargs['ti'].xcom_push(key='my_secret', value='my_plain_secret_value')
    )

    print_variable_task = PythonOperator(
        task_id="print_variable",
        python_callable=print_variable
    )

    set_variable_task >> print_variable_task
```

In this first example, the PythonOperator `set_variable` pushes a plain string to xcom, which isn't encrypted. Even though the airflow ui will mask this when you inspect the xcom value through task details, the data isn't encrypted in the metadata database, and is vulnerable if the db is exposed. This illustrates why merely setting a variable without leveraging airflow’s encryption features leaves a security gap.

**Example 2: Using `os.environ` for credentials and why it *can* be problematic**

```python
import os
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def print_env_var():
    secret = os.environ.get('MY_SECRET')
    print(f"Secret from env: {secret}")


with DAG(
    dag_id="env_var_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    print_env_task = PythonOperator(
        task_id="print_env_variable",
        python_callable=print_env_var
    )
```

While this example doesn't directly interact with airflow variables, many engineers initially reach for environment variables to manage secrets. If those aren't encrypted at the OS level and simply passed through, airflow isn't going to encrypt them, either. The output (and indeed, the data in the OS) isn't protected using Fernet encryption. It's essential to understand the distinction: airflow's Fernet key protects data *within airflow’s ecosystem*, not necessarily credentials that bypass its direct encryption features. Furthermore, environment variables can leak in logs if not properly handled in your python code and often do.

**Example 3: Using an encrypted connection (example configuration)**

Let’s say we're setting up a database connection in airflow. We'd configure this in the airflow UI under 'Admin' -> 'Connections' with the connection type, host, username, and password. Importantly, when we save a connection via the UI or via the cli, the password value *is encrypted by airflow* using the fernet key before it's stored in the database. So, if we now try to get it in our dags:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime

def print_connection_password():
    conn = BaseHook.get_connection("my_db_connection")
    print(f"Connection password : {conn.password}")

with DAG(
    dag_id="connection_password_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    print_conn_task = PythonOperator(
        task_id="print_connection_password",
        python_callable=print_connection_password
    )
```

In this scenario, airflow utilizes the fernet key to secure the password. If you were to inspect the metadata table directly (e.g., via `select * from connection`), the `password` field will be encrypted. However, note that the output of the python script will output the unmasked password value (assuming you have appropriate permissions). So it is essential to log to production safely and not simply print passwords directly to the console. Note however that in the airflow UI, you would still see the password masked during configuration and when viewing connections. The *storage* is protected, and the *display* is masked. The data in memory is *not* and this is an important distinction to understand.

The primary take-away here is that merely having a Fernet key configured doesn’t automatically cloak all your credentials in impenetrable security. *You have to actively use airflow’s mechanisms designed for credential encryption*. This often involves using connections, variables configured through appropriate airflow methods that trigger encryption (such as the cli commands or using the providers secret backends), and ensuring any custom credential stores respect the Fernet key.

For those looking to deepen their understanding, I highly recommend reviewing the official Apache Airflow documentation on Security and Variable Management (specifically those sections on Encryption), and also consult "Programming Apache Airflow" by Bas Pijls and Andreas Sjödin for a thorough exploration of these concepts. Another useful resource is the apache airflow enhancement proposal on Fernet encryption itself, which is often the most up-to-date information on the implementation nuances. Additionally the documentation for any specific secrets backends you are using in airflow should be carefully reviewed to understand how they interact with fernet.

Finally, always remember that security is a layered approach. Fernet key encryption is a vital component of airflow's overall security posture, but it's not a silver bullet. Regularly review your security policies and infrastructure, ensuring best practices are consistently applied to protect those credentials, both in storage and in use.
