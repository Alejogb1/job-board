---
title: "How do I access dag_run.conf values and pass them to an SSHOperator command?"
date: "2024-12-23"
id: "how-do-i-access-dagrunconf-values-and-pass-them-to-an-sshoperator-command"
---

Alright, let's tackle this. I've bumped into this scenario more times than I care to count over the years, primarily when orchestrating complex infrastructure deployments via Airflow. The need to dynamically feed configuration parameters into SSH commands, drawn directly from a `dag_run.conf`, is a common requirement. The key is understanding how Airflow structures its contexts and templating engine, and then applying that knowledge strategically.

At its core, the `dag_run.conf` allows you to supply custom configuration parameters when manually triggering a DAG or through the API. These parameters become accessible to your tasks as part of the Jinja templating context. Now, when using `SSHOperator`, you need to leverage this templating capability to inject those values into the command you intend to execute.

Essentially, the challenge comes down to making the `dag_run.conf` values available within the `ssh_command` argument of the `SSHOperator`. You're not directly handing parameters to the operator function, but rather specifying a command string that Airflow will then interpret using the Jinja context. The relevant context for us includes the `dag_run` object itself, from which we can then access its associated `conf` dictionary.

Let me lay out a few practical examples, drawing from my past projects to highlight different approaches and the trade-offs.

**Example 1: Direct Access to Conf Values in Command**

In this scenario, we directly embed the `dag_run.conf` values into the command string. This is the most straightforward approach, but it requires caution because, if you try to access a non-existent key, it'll cause an error during templating.

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='ssh_conf_direct',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    ssh_task = SSHOperator(
        task_id='execute_command_with_conf',
        ssh_conn_id='my_ssh_connection',
        command="""
            echo "Environment: {{ dag_run.conf['environment'] }}"
            echo "Application Name: {{ dag_run.conf['app_name'] }}"
            echo "Version: {{ dag_run.conf['version'] }}"
            # Actual command you want to run, using conf values
            /path/to/your/script --environment {{ dag_run.conf['environment'] }} --app_name {{ dag_run.conf['app_name'] }} --version {{ dag_run.conf['version'] }}
        """,
        do_xcom_push=False
    )
```

In this snippet, the command string leverages Jinja templating syntax `{{ }}` to embed values from the `dag_run.conf` dictionary. If, during dag execution, you pass a `conf` like `{"environment": "production", "app_name": "MyWebApp", "version": "1.2.3"}` then the generated command that executes on the remote host would be:

```bash
echo "Environment: production"
echo "Application Name: MyWebApp"
echo "Version: 1.2.3"
/path/to/your/script --environment production --app_name MyWebApp --version 1.2.3
```

**Example 2: Using a Conditional Default Value**

What if a certain key within the `dag_run.conf` might be optional? Directly accessing a missing key results in an error. To avoid this, we can use Jinja's `get` method to specify a default value when a key is missing.

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='ssh_conf_optional',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    ssh_task = SSHOperator(
        task_id='execute_command_with_conf',
        ssh_conn_id='my_ssh_connection',
        command="""
            echo "Environment: {{ dag_run.conf.get('environment', 'development') }}"
            echo "Application Name: {{ dag_run.conf.get('app_name', 'default_app') }}"
            echo "Version: {{ dag_run.conf.get('version', 'latest') }}"
             # Actual command you want to run, using conf values, handling missing keys.
             /path/to/your/script --environment {{ dag_run.conf.get('environment', 'development') }} --app_name {{ dag_run.conf.get('app_name', 'default_app') }} --version {{ dag_run.conf.get('version', 'latest') }}
        """,
        do_xcom_push=False
    )
```

Here, if the `environment` key is not present in `dag_run.conf`, it will use ‘development’ instead. This `get` method adds resilience to your workflow when you're dealing with optional parameters, preventing the DAG from breaking simply because a specific configuration was omitted. If the `conf` now is only `{"app_name": "WebAppA"}`, the resulting command on the remote server would be:

```bash
echo "Environment: development"
echo "Application Name: WebAppA"
echo "Version: latest"
/path/to/your/script --environment development --app_name WebAppA --version latest
```

**Example 3: Complex Configuration Handling with a Python Function**

For more intricate setups, direct templating can become unwieldy. In such cases, I often find it more maintainable to pre-process the configuration inside a Python function and pass the result to the operator. This also enables you to perform validation, complex transformations, or generate the command dynamically based on the parameters.

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from airflow.operators.python import PythonOperator
from datetime import datetime

def build_ssh_command(dag_run_conf, **kwargs):
    environment = dag_run_conf.get('environment', 'development')
    app_name = dag_run_conf.get('app_name', 'default_app')
    version = dag_run_conf.get('version', 'latest')

    # You could also perform other operations based on parameters

    command = f"""
        echo "Environment: {environment}"
        echo "Application Name: {app_name}"
        echo "Version: {version}"
        /path/to/your/script --environment {environment} --app_name {app_name} --version {version}
    """
    return command

with DAG(
    dag_id='ssh_conf_function',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    prepare_command = PythonOperator(
        task_id='prepare_ssh_command',
        python_callable=build_ssh_command,
        provide_context=True
    )

    ssh_task = SSHOperator(
        task_id='execute_command_with_conf',
        ssh_conn_id='my_ssh_connection',
        command="{{ ti.xcom_pull(task_ids='prepare_ssh_command') }}",
        do_xcom_push=False
    )

    prepare_command >> ssh_task
```

This approach utilizes `PythonOperator` to build the command. The `build_ssh_command` function takes the `dag_run.conf` dictionary, extracts its required values (with default values if required), constructs the final command, and then the result of this is pushed to XCom (Airflow's mechanism for inter-task communication). We then use `ti.xcom_pull` to retrieve it and use it as the `command` value for `SSHOperator`. The advantages are clearer separation of logic, better readability, and greater flexibility to implement validation and transformations.

**Important Considerations**

1. **Security:** Be extremely careful about how you construct your command strings, especially if the `dag_run.conf` comes from untrusted sources. Avoid directly interpolating strings that have not been explicitly validated.
2. **Error Handling:** The Jinja templating engine can throw exceptions if it can’t evaluate a template expression, like in example 1 if a key is missing. Use the `get` method, as shown in example 2, or preprocess your configuration in Python, as demonstrated in example 3, to manage these scenarios effectively.
3. **Readability and Maintainability:** When things start to get complex, it's very beneficial to leverage Python code to handle configurations as in Example 3. It helps in keeping your DAGs organized and much easier to maintain.

For in-depth understanding of Airflow's concepts, especially templating, I recommend referring to the official Apache Airflow documentation, especially sections on templating with Jinja and Context variables. I'd also recommend the book "Data Pipelines with Apache Airflow" by Bas P. Harenslak et al., which dives deeply into all aspects of Airflow, which includes context handling and templating, and is extremely helpful for mastering these nuances. Furthermore, looking at the source code of `SSHOperator` itself is informative (you can usually find it by looking for the class definition in the Airflow provider packages). Lastly, testing your templates independently can save you a lot of time. I often use a small Python script for template evaluation to avoid iterative runs on the actual DAG.

In short, accessing and passing `dag_run.conf` values to an `SSHOperator` boils down to understanding Jinja templating and how Airflow’s context objects (like `dag_run`) integrate within it, and then choosing the appropriate method that best fits the complexity and maintainability requirements of your specific use case.
