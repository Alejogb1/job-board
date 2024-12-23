---
title: "How can I access dag_run.conf values and pass to SSHOperator?"
date: "2024-12-16"
id: "how-can-i-access-dagrunconf-values-and-pass-to-sshoperator"
---

,  I’ve definitely been down this road before, more than once, and it’s a common hurdle when you start building more dynamic airflow workflows. Getting those `dag_run.conf` values into an `SSHOperator`—or really, any operator that needs dynamic configuration—requires a bit of thought about templating and Airflow’s execution context.

The challenge, at its core, is that `dag_run.conf` isn't automatically available as simple variables within your tasks. It’s a dictionary, and you need to specifically extract and then pass its contents. The key thing here is utilizing Jinja templating, which Airflow natively supports. The `context` object, automatically available in Airflow templates, holds all sorts of useful information, including the `dag_run`. Let me explain with some specifics and some code examples based on patterns I've seen and used successfully.

The fundamental approach revolves around accessing the configuration dictionary stored within the `dag_run` object of the context passed to the template and extracting the desired key-value pair. We can then pass this extracted value using Jinja within the parameters of the SSH operator.

**Example 1: Basic Access and Templating**

Let’s say you have a `dag_run.conf` that looks something like this when you trigger the dag: `{"target_server": "my-server-01", "script_path": "/opt/scripts/my_script.sh"}`. Here's how you'd utilize those values in your `SSHOperator`:

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='ssh_dag_with_conf',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:

    run_remote_script = SSHOperator(
        task_id='run_script_on_remote',
        ssh_conn_id='my_ssh_connection', # your connection name, defined in airflow
        command="""
            sshpass -p '{{ dag_run.conf.password }}' ssh -o StrictHostKeyChecking=no user@{{ dag_run.conf.target_server }} 'bash {{ dag_run.conf.script_path }}'
        """,
        dag=dag
    )
```
In this example, the crucial part is `{{ dag_run.conf.target_server }}` and `{{ dag_run.conf.script_path }}` in the command.  Airflow’s templating engine will process this before the operator executes, substituting the actual values from your `dag_run.conf`.  Note the direct access of `dag_run.conf` and nested access to get the respective dictionary values. For security considerations, it is not advised to pass secrets such as passwords through the command string itself. In a real world scenario, it would be preferable to use an ssh key or an Airflow variable for that. I have intentionally included this to showcase how a string can be interpolated.

**Example 2: Handling Missing Keys (With a Default)**

Sometimes your `dag_run.conf` might not always have all the values you expect. In such scenarios you will want to avoid the task failing due to a missing key in the config dictionary. Here's how you could provide defaults or handle missing keys using Jinja filters:

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='ssh_dag_with_conf_defaults',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:

    run_remote_script = SSHOperator(
        task_id='run_script_on_remote',
        ssh_conn_id='my_ssh_connection',
        command="""
            bash {{ dag_run.conf.get("script_path", "/default/path/script.sh") }} 
            && echo "Host is : {{ dag_run.conf.get("target_server", "default_server") }}"
        """,
        dag=dag
    )
```

Here, we use the Jinja `get()` filter on the `dag_run.conf` dictionary. The `get()` method is called using standard python access notation `.` for dictionary keys. If a key (`script_path` or `target_server`) isn't present in `dag_run.conf`, then `get()` will return the default value we've provided as the second argument – `/default/path/script.sh` or `default_server`.

**Example 3: More Complex Configuration with Nested Values**

Let's imagine a more complex `dag_run.conf` like this:

```json
{
  "deployment": {
    "target_env": "prod",
    "server_details": {
        "host": "my-prod-server",
        "path": "/opt/prod_deploy"
      }
   },
   "app_name": "my_app_v2"
}
```

Here's how we could access the nested dictionary with `target_env`, `host` and `path` values:

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='ssh_dag_nested_conf',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:

    run_remote_script = SSHOperator(
        task_id='run_script_on_remote',
        ssh_conn_id='my_ssh_connection',
       command="""
        echo "Deploying to {{ dag_run.conf.deployment.target_env }} environment" && 
        ssh -o StrictHostKeyChecking=no user@{{ dag_run.conf.deployment.server_details.host }} 'bash {{ dag_run.conf.deployment.server_details.path }}/deploy_script.sh --app-name {{ dag_run.conf.app_name }}'
        """,
        dag=dag
    )
```
As we can see, it is quite straightforward to access these nested config values using the dot operator and jinja templates.  We’re just drilling down through the nested structure, `dag_run.conf.deployment.server_details.host` for instance, to get the actual value. This demonstrates the flexibility of using `dag_run.conf` with more complex configurations.

**Things to Keep in Mind:**

1. **Data Types**: `dag_run.conf` values are treated as strings by Jinja templates. If you need to handle numbers or boolean values, you might need to use Jinja filters to convert them. Usually, this is handled by the scripting language you are passing into the command e.g., if the value is expected to be an integer it should be parsed as one using the scripting language being executed remotely.

2. **Security**: Never include sensitive information like passwords or API keys directly in your `dag_run.conf` values. It is better to use Airflow connections or variables for those. In the example I've provided with the password, I did so for illustration only. Ideally, you should be leveraging ssh keys and storing secrets appropriately.

3. **Error Handling:** It is usually best practice to check if the configuration variable exist or define a default before using. This will avoid your tasks from failing because of a missing key.

4. **JSON Serialization:** The `dag_run.conf` is typically passed via the UI, CLI, or API in JSON format, which means the values are already serialized.

**Recommended Reading:**

For a deeper understanding, I'd suggest exploring these resources:

*   **"Programming Apache Airflow" by Jarek Potiuk and Marcin Ziemniak:** This book offers an in-depth look at Airflow’s core concepts, including templating and the context object. Pay special attention to the chapters on task execution and Jinja templating for a clear understanding of the concepts we have discussed.
*   **Apache Airflow Documentation:** The official documentation is an invaluable resource. Specifically, focus on the sections about Jinja templating and the context variables available within task execution. The documentation will provide the authoritative source for all features that have been highlighted here.
*   **Jinja Documentation:** While Airflow provides the context, understanding the Jinja template engine itself can help create more dynamic templates. I find diving deeper into the Jinja documentation very valuable in these types of cases.
*   **"Effective Python" by Brett Slatkin:** While not strictly about Airflow, this book offers great insights into using Python in an effective and idiomatic way which can greatly help when you're writing your dag.

In my experience, mastering this interaction between `dag_run.conf` and templating is essential for building truly dynamic and reusable Airflow workflows. It enables you to handle a vast array of configuration scenarios and keep your pipelines flexible and robust. The most important thing is to always think about how Airflow’s templating engine processes your code and how context can be utilized in a secure and robust way.
