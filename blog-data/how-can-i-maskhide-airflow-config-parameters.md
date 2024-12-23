---
title: "How can I Mask/Hide Airflow Config parameters?"
date: "2024-12-23"
id: "how-can-i-maskhide-airflow-config-parameters"
---

,  I've definitely been down this road before, particularly when scaling airflow deployments across various teams with varying degrees of access. Exposing configuration parameters directly, especially things like database connection strings or api keys, is a definite non-starter for any production environment. It's a security headache waiting to happen. Let's break down some reliable strategies, focusing on both the 'how' and the 'why'.

First, it's crucial to understand that airflow's configuration, by default, isn't inherently secure. It relies on a configuration file, typically `airflow.cfg`, and environment variables, all of which can be vulnerable if not handled carefully. The goal, therefore, is to decouple the actual values from these accessible locations and introduce an abstraction layer.

My own experience with this issue arose during a massive data ingestion project where multiple teams were building their DAGs. Initially, we naively used environment variables, which quickly turned into a nightmare of access control and potential leaks. The moment one team needed a specific database connection, we had to expose it to the entire infrastructure, which was far from ideal. We needed a granular, secure, and auditable method.

The first and arguably simplest method involves leveraging airflow's built-in variables system, accessible via the web interface and `airflow.models.Variable` objects. You can think of these as a key-value store specific to airflow. This is a step up from environment variables directly, as you’re abstracting the sensitive details and managing them via airflow’s authentication layer. However, that too has caveats. These variables are still stored in the metadata database, which needs its own access control. Nevertheless, it’s often good enough for simple cases, and you can integrate secrets management using Airflow’s built-in secret backends. Here’s how that would look in a DAG:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime


def print_database_credentials(**kwargs):
    db_user = Variable.get("database_user")
    db_password = Variable.get("database_password", default='default_password_if_not_set') #default value to prevent fail if not found
    print(f"Database user: {db_user}")
    print(f"Database password: {db_password}")

with DAG(
    dag_id="variable_example",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    print_vars = PythonOperator(
        task_id="print_variables",
        python_callable=print_database_credentials,
    )
```

In this example, instead of hardcoding the `database_user` and `database_password`, the DAG fetches them from airflow’s variable store. You would populate these using either the web interface or the airflow CLI (e.g., `airflow variables set database_user your_username`). Notice the use of `default='default_password_if_not_set'`; a crucial practice, as it allows your tasks to continue if a variable hasn't yet been defined, preventing unpredicted errors at runtime.

Now, while airflow variables improve security over just using env vars, this can still be less than ideal, and for truly sensitive information, I generally recommend utilizing an external secrets manager. This is where things get significantly more robust. Many options exist, including Hashicorp Vault, AWS Secrets Manager, GCP Secret Manager, Azure Key Vault, and even basic password management tools. The principle remains the same: store the secrets outside of airflow and retrieve them at runtime.

To illustrate, let's consider using an AWS secrets manager, as it's fairly common. We can use a custom Airflow hook for seamless interaction with AWS. Below is a simplified example:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.hooks.base import BaseHook
from datetime import datetime
import boto3
import json
from airflow.exceptions import AirflowException

class AwsSecretsManagerHook(BaseHook):
    def __init__(self, region_name="us-east-1", aws_conn_id="aws_default", *args, **kwargs):
        self.region_name = region_name
        self.aws_conn_id = aws_conn_id
        super().__init__(*args, **kwargs)

    def get_conn(self):
        conn = self.get_connection(self.aws_conn_id)
        config = {}
        if conn.extra_dejson.get("region_name"):
            config['region_name'] = conn.extra_dejson.get("region_name")
        
        if conn.extra_dejson.get("aws_access_key_id"):
            config["aws_access_key_id"] = conn.extra_dejson.get("aws_access_key_id")
            config["aws_secret_access_key"] = conn.extra_dejson.get("aws_secret_access_key")

        session = boto3.Session(**config)
        client = session.client("secretsmanager")

        return client

    def get_secret(self, secret_name):
        client = self.get_conn()
        try:
           response = client.get_secret_value(SecretId=secret_name)
        except Exception as e:
            raise AirflowException(f"Error fetching secret {secret_name} from AWS Secrets Manager: {e}")

        if 'SecretString' in response:
          return json.loads(response['SecretString'])
        else:
            return response['SecretBinary']

def print_database_credentials_from_secrets_manager(**kwargs):
    hook = AwsSecretsManagerHook()
    secret = hook.get_secret("my-database-credentials-secret")
    print(f"Database user: {secret['username']}")
    print(f"Database password: {secret['password']}")


with DAG(
    dag_id="aws_secrets_manager_example",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    print_secrets = PythonOperator(
        task_id="print_secrets_from_manager",
        python_callable=print_database_credentials_from_secrets_manager,
    )
```

Here, the `AwsSecretsManagerHook` abstracts the interaction with AWS Secrets Manager. You'd store a json object as secret such as `{"username": "your_username", "password": "your_password"}` under the secret name `my-database-credentials-secret` in aws. The DAG then retrieves it by name and extracts the necessary values, leaving no trace of the actual credentials within the airflow environment itself. Crucially, the hook will use the provided AWS connection within airflow which should include your aws credentials. *You will also need to ensure that the service account used by airflow has permissions to access the specified secrets in AWS secrets manager*.

Finally, I'd like to touch on the use of encrypted connections and environment variables as well, since they form part of a good 'defense in depth' strategy. Even when secrets are retrieved from a secrets manager, ensure that all connections, especially database ones, are encrypted using TLS/SSL where possible. This protects data in transit. In regards to environment variables, utilize them solely to configure secrets manager access - *not to store actual secrets*. For example, you might store an AWS region or secret arn in env variables - but never the sensitive values.

For further reading, I would recommend researching the following: Hashicorp's documentation on Vault and its integration with Airflow; AWS documentation for Secrets Manager and the python boto3 library; and, more generally, the OWASP recommendations on secrets management. Also, the Apache Airflow documentation itself is invaluable for understanding its built-in features, as mentioned above.

In my experience, a combination of airflow variables for less sensitive info, combined with a strong secrets manager and a defense in depth approach is essential for any production workflow. This will not only keep your data secure but will also lead to a much more maintainable setup in the long run. I hope this helps you on your journey.
