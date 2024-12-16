---
title: "How to pass values from Airflow's dag_run.conf to SSHOperator?"
date: "2024-12-16"
id: "how-to-pass-values-from-airflows-dagrunconf-to-sshoperator"
---

Okay, let’s talk about passing values from Airflow’s `dag_run.conf` to the `SSHOperator`. It's a situation I’ve encountered countless times, particularly when dealing with dynamically generated configurations or orchestrating tasks that require parameters at runtime. It’s not as straightforward as simply passing a variable, but the mechanics are fairly logical once you understand how Airflow context works. This is a critical ability for building robust, data-driven workflows.

The heart of the matter lies in understanding that Airflow templates its arguments. These templates use Jinja2 templating engine and can access the execution context of the task. The `dag_run.conf` dictionary, which is what you get when you trigger a DAG with a configuration, is part of this context. So, instead of thinking about passing a variable *directly*, we actually construct strings that, when evaluated, contain the values we need.

From past experience, I recall working on a project involving nightly data processing on a remote cluster. We used a single DAG to trigger various processing pipelines based on the configuration passed at dag trigger time. For instance, we'd use the `dag_run.conf` to specify the location of input data, processing parameters, and output paths. Without the ability to dynamically use these configurations, our workflow would have been far less flexible and maintainable.

Now, let's break it down into a practical approach. The `SSHOperator` itself doesn't directly accept a dictionary or an object as an argument for the command; it expects a string that contains the command to execute, and we construct that string by injecting values from `dag_run.conf` within the Jinja2 template.

Here’s the basic technique: When defining the `SSHOperator`, use the Jinja2 template syntax `{{ dag_run.conf['your_key'] }}` inside the `command` argument. Airflow's templating engine will substitute the actual value present in the `dag_run.conf` at runtime. If the key `your_key` doesn’t exist or `dag_run.conf` is empty, it will result in a template error; to manage that, we’ll look at providing default values in code snippets below.

Let me illustrate with a few code examples:

**Example 1: Basic Usage - Passing a single value**

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='ssh_conf_example1',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:
    execute_command = SSHOperator(
        task_id='execute_remote_command',
        ssh_conn_id='my_ssh_connection',  # Defined in Airflow UI
        command="""
            echo "Processing data for path: {{ dag_run.conf['data_path'] }}"
            # Your actual command here...
        """,
    )
```

In this example, assuming you trigger the DAG with a `dag_run.conf` like `{"data_path": "/data/input/20240520"}`, the actual command executed on the remote server would be something akin to: `echo "Processing data for path: /data/input/20240520"`. Notice that `my_ssh_connection` would have been pre-configured in Airflow UI.

**Example 2: Default Values - Handling missing keys**

To make the DAG more resilient and avoid runtime errors from missing values, use Jinja2’s `get` method combined with a default argument. This is crucial for real-world scenarios where not every dag run might have a comprehensive configuration. Here is the updated code snippet:

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='ssh_conf_example2',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:
    execute_command = SSHOperator(
        task_id='execute_remote_command',
        ssh_conn_id='my_ssh_connection',  # Defined in Airflow UI
        command="""
            echo "Processing data for path: {{ dag_run.conf.get('data_path', '/default/path') }}"
            echo "Processing date: {{ dag_run.conf.get('processing_date', 'today') }}"
            # Your actual command here...
        """,
    )
```
Here, if ‘data_path’ is not specified in `dag_run.conf`, `/default/path` will be used, similarly ‘today’ would be used if ‘processing_date’ is absent. This adds an important element of stability.

**Example 3: Passing Multiple Parameters & Formatted Commands**

In many practical situations you may need to pass multiple parameters. You can format these into a single command string for convenience and clarity. Here is an example showcasing it:

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='ssh_conf_example3',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:
    execute_command = SSHOperator(
        task_id='execute_remote_command',
        ssh_conn_id='my_ssh_connection',  # Defined in Airflow UI
        command="""
            script_path={{ dag_run.conf.get('script_path', '/opt/scripts/default.sh') }};
            input_dir={{ dag_run.conf.get('input_dir', '/data/input') }};
            output_dir={{ dag_run.conf.get('output_dir', '/data/output') }};
            echo "Running $script_path with input from $input_dir and output to $output_dir"
            $script_path --input "$input_dir" --output "$output_dir"
            # Your actual command here...
        """,
    )
```

This demonstrates the passing of multiple parameters that are used in an actual command, complete with default values.

**Important Considerations:**

*   **Security:** Be cautious about directly passing sensitive information through `dag_run.conf`. Consider using Airflow variables for sensitive information, and retrieve those variables within your task using Jinja templates, or use secure secrets management systems.
*   **Complex Data Structures:** If your `dag_run.conf` contains complex data structures like nested dictionaries or lists, you might want to serialise them (e.g., using `json.dumps`) when setting the `dag_run.conf` before using `json.loads` within your Jinja template. This is more suited to situations where complex configurations need to be processed within the remote task, rather than simply passed as arguments. You’d want to avoid that when simpler solutions will do.
*   **Command Structure:** Ensure the command you build through Jinja templating is properly escaped for the shell environment. Misplaced quotes or other special characters can break your commands.
*   **Error Handling:** If your command depends on the existence of specific parameters, utilize the defaults or use other mechanism to prevent the `SSHOperator` failing with errors. I've found it useful to add validation within the remote script itself, not just the Airflow DAG. This allows you to handle any issues directly on the remote server and provides more robust logging.
*   **Testing**: Thoroughly test with different `dag_run.conf` values to ensure your templates behave as intended and handle unexpected input.

**Further Resources:**

For anyone looking to deepen their understanding, I highly recommend the official Airflow documentation, specifically on Jinja templating and the various available context variables. For understanding the finer points of Jinja, “Jinja Documentation”, you can find it on the official website. For a broader perspective, the book "Data Pipelines with Apache Airflow" by Bas P. Harenslak is an excellent resource. Also, I would suggest reviewing "Programming Apache Airflow" by Jesse Anderson and Greg Wilson, for a more conceptual overview. Also, checking specific provider's documentation for SSH Operator is always advisable.

In essence, the combination of `dag_run.conf` with Jinja templating in the `SSHOperator` provides the capability to create highly configurable and adaptable workflows. By using these techniques, you can handle complex data processing scenarios efficiently, with proper consideration to security and error handling. Remember that while it is powerful, always maintain vigilance and adopt best practices for building production ready pipelines.
