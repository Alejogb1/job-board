---
title: "How to mask or hide config parameters in Airflow?"
date: "2024-12-16"
id: "how-to-mask-or-hide-config-parameters-in-airflow"
---

Alright, let's talk about securing configuration parameters in Apache Airflow. I've seen this trip up many a team, and it’s certainly something I’ve grappled with extensively over the years. It’s crucial to keep sensitive data like database credentials, api keys, or other secrets out of plain sight in your dags and airflow configurations. Letting these sit around unencrypted is practically an invitation for trouble.

One of the most fundamental steps, and perhaps where many initially stumble, is realizing that merely avoiding hardcoding values in the DAG file itself is only part of the battle. Simply loading from a config file or environment variable isn't sufficient protection. Those values still exist in an accessible format on the filesystem or within your environment, which is not ideal, especially if you're dealing with a multi-user setup.

My first encounter with this challenge was when deploying a data pipeline for a healthcare company. We initially used environment variables to manage database passwords, and while this was better than hardcoding, it wasn't nearly secure enough, especially given the stringent regulations concerning personal health information. It quickly became apparent that we needed a robust solution to avoid exposing credentials to anyone who had access to the environment or the machine where airflow was running.

So, what are some viable techniques? Let's dive into some detailed strategies, and I'll illustrate each with some python examples.

**1. Using Airflow Connections:**

This is generally considered the best practice within the Airflow ecosystem. Airflow connections are not inherently a masking technique, but rather a way to securely store sensitive information *external* to your DAGs. You can configure connections via the airflow UI, command-line, or the API. Importantly, airflow stores these connections encrypted in its metadata database.

The key here is using the airflow connection id in your dags instead of directly using credentials. Consider a database connection:

```python
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.models import Connection

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
}

with DAG(
    dag_id='postgres_example',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    def fetch_data():
        # Assuming you've configured a connection named 'my_postgres_conn'
        hook = PostgresHook(postgres_conn_id='my_postgres_conn')
        conn = hook.get_conn()
        cur = conn.cursor()
        cur.execute("SELECT * FROM my_table;")
        results = cur.fetchall()
        cur.close()
        conn.close()
        print(f"Data fetched: {results}")


    task_fetch_data = PythonOperator(
        task_id='fetch_data_from_postgres',
        python_callable=fetch_data,
    )
```

Here, `'my_postgres_conn'` is the connection id you've configured outside of the dag file, ideally through a secure mechanism. You are storing database information like hostname, database name, username and the sensitive password within the airflow metadata database, not in any plaintext configuration files.

**2. Backend Secrets Management with a Secrets Backend:**

Airflow natively supports integration with various secret backends, like HashiCorp Vault or AWS Secrets Manager. Instead of storing secrets directly in the Airflow metadata database, which itself could be a security concern if compromised, these backends provide a more secure, centralized secrets management platform. I’ve extensively used HashiCorp vault in past projects and can attest to the added security and flexibility it brings.

The way this generally works is you configure airflow to talk to the backend by defining the secret backend class, along with its specific configuration in your `airflow.cfg` file. Then, within your dag, you can access the secrets by their given key paths, without exposing the values in your code.

Here's a snippet, assuming you are using HashiCorp Vault:

```python
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.secrets.vault import VaultBackend
import os

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
}


with DAG(
    dag_id='vault_secrets_example',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    def fetch_vault_secret():
        # Note: Vault configuration is done in airflow.cfg
        backend = VaultBackend(
            url = os.environ.get("VAULT_ADDR"),
            token = os.environ.get("VAULT_TOKEN"),
            kv_path='secret/data'
        )
        secret = backend.get_secret("my_secret/my_key")
        print(f"Retrieved secret value: {secret}")
    task_get_vault_secret = PythonOperator(
        task_id='get_secret_vault',
        python_callable=fetch_vault_secret,
    )
```
In this example, `my_secret/my_key` represents the path in your Vault where you have stored your specific secret. The backend's `get_secret` method will securely retrieve it at runtime. Notice how we are *not* exposing the sensitive value in this DAG file. Important to note that `os.environ.get("VAULT_ADDR")` and `os.environ.get("VAULT_TOKEN")` should be present only in the execution environment. Ensure your vault access is done over a secure TLS connection.

**3. Utilizing Environment Variables Carefully:**

While relying solely on environment variables isn't sufficient, they can be used strategically *in conjunction with other methods*. For instance, instead of storing the actual credentials in environment variables, you can use them to store *references* to secrets, or parameters that are *less sensitive* but could benefit from configuration outside of code files. A typical use case is storing backend addresses.
Here’s how environment variables would help with the previous example

```python
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.secrets.vault import VaultBackend
import os

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
}


with DAG(
    dag_id='vault_env_secrets_example',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
) as dag:

    def fetch_vault_secret():
        # Vault configuration is done in airflow.cfg
        backend = VaultBackend(
            url = os.environ.get("VAULT_ADDR"), # getting backend address from environment
            token = os.environ.get("VAULT_TOKEN"), #getting token from env
            kv_path='secret/data'
        )
        secret = backend.get_secret("my_secret/my_key")
        print(f"Retrieved secret value: {secret}")
    task_get_vault_secret = PythonOperator(
        task_id='get_secret_vault',
        python_callable=fetch_vault_secret,
    )
```
Notice, `os.environ.get("VAULT_ADDR")` and `os.environ.get("VAULT_TOKEN")` are *not* the actual secrets, they are the vault address and token which are still considered sensitive, but may need to be changed between various environments. This technique allows for greater environmental flexibility, while still keeping core secrets within a vault.

**Important Considerations:**

*   **Least Privilege:** Always adhere to the principle of least privilege. Grant only the necessary permissions to your services and users. Do not give read access to Vault path or Airflow connections to every user.
*   **Auditing:** Make sure to have thorough audit logs and monitoring setup for your secrets management platform. Be able to trace access and modification of sensitive data.
*   **Rotation:** Regularly rotate your secrets. This prevents a compromised secret from staying valid indefinitely.
*   **Secure Storage:** Even your secrets backend should be stored securely. Follow best practices for infrastructure security.
*   **Avoid direct access:** Try to avoid using python `os` module in your dags as much as possible to fetch environment variables. Always use connections or secrets backends to fetch sensitive information.
*   **Code Review:** Code reviews are an invaluable tool to catch insecure practices. Enforce reviews before deploying code with sensitive parameters.

**Recommendations:**

To further deepen your understanding, I recommend reading the following:

*   **"Secrets Management" chapter from the "Kubernetes in Action, 2nd Edition" book by Marko Lukša:** Provides a thorough understanding of secure secrets handling in Kubernetes and other cloud-native environments, which translates well to managing secrets in any distributed application, including Airflow.
*   **HashiCorp Vault Documentation:** Go through the official documentation of HashiCorp Vault for in depth practical knowledge, as this is a very popular option for Secrets management.
*   **Airflow Documentation:** Review the official Airflow documentation for up to date information about supported backend integrations and features.

Securing secrets in Airflow is a multi-faceted problem, requiring a combination of the above strategies. Remember, security is not a one-time task but a continuous process. By implementing these approaches carefully, you can significantly reduce your risk and enhance the overall robustness of your data pipeline.
