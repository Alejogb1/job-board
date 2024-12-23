---
title: "How can I access dag_run.conf values in Airflow and pass to SSHOperator?"
date: "2024-12-23"
id: "how-can-i-access-dagrunconf-values-in-airflow-and-pass-to-sshoperator"
---

Alright,  Been there, done that, quite a few times actually. Passing `dag_run.conf` values to an `SSHOperator` in Airflow is a common requirement, especially when you need to dynamically configure your remote execution based on user input or external triggers. It's not complicated, but you need to understand a few core concepts to get it right and avoid some potential pitfalls. Let's break down the approach, and I'll give you some practical code examples based on some projects where I’ve had to handle this sort of thing.

First, understand that `dag_run.conf` is a dictionary available when a dag run is triggered, either manually or via the scheduler. This dictionary can hold arbitrary key-value pairs, which makes it extremely versatile for parameterizing your dags. Now, the key to getting those values into your `SSHOperator` is understanding Jinja templating, which Airflow uses extensively.

The `SSHOperator` accepts `command` argument, and this field is templatable using Jinja. This allows you to access `dag_run.conf` values within the command string. Airflow makes several objects available to your templated fields, including `dag_run`, which has the `conf` attribute.

Let's dive into our first code example, which is a straightforward implementation using a simple dictionary:

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='ssh_with_conf_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    ssh_task = SSHOperator(
        task_id='execute_remote_command',
        ssh_conn_id='my_ssh_connection',
        command="""
            echo "Received config value: {{ dag_run.conf.get('my_key', 'default_value') }}"
            echo "Another value from config: {{ dag_run.conf.get('another_key', 'fallback') }}"
            """,
    )
```

In this example, the `command` argument contains Jinja templated expressions using `{{ dag_run.conf.get('my_key', 'default_value') }}` and `{{ dag_run.conf.get('another_key', 'fallback') }}`. `dag_run.conf` accesses the configuration dictionary, and the `get()` method is used to safely retrieve values, providing a default value if the key doesn't exist. In this specific case, if the `dag_run` does not contain 'my_key', 'default_value' will be used.

Now, imagine a slightly more complex situation. I worked on a system that required passing configuration for a specific application running on a remote server based on user inputs. The config was an arbitrarily nested dictionary, and we needed to serialize it to JSON before passing it to the remote script. That's where we moved beyond a simple echo command. Here's an example of what that looked like:

```python
import json
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='ssh_with_complex_conf',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    ssh_task = SSHOperator(
        task_id='execute_remote_script',
        ssh_conn_id='my_ssh_connection',
        command="""
            CONFIG_JSON='{{ dag_run.conf | tojson | string }}'
            python3 /path/to/my/remote_script.py --config "$CONFIG_JSON"
        """,
        dag=dag,
    )
```

Here, `dag_run.conf` is passed through the `tojson` filter, ensuring it is properly formatted as a JSON string. Furthermore, the `string` filter guarantees it will be passed as a proper string, not an object. The generated JSON string is then assigned to the shell variable `CONFIG_JSON` and passed as an argument to `python3 /path/to/my/remote_script.py`. This approach is crucial when dealing with complex configurations as it avoids potential issues with quoting and escaping. You have to ensure you handle this conversion properly in your remote script (`my_remote_script.py` in this case) by deserializing it with `json.loads()`. This is where you can encounter issues; if your data doesn't handle the serialisation properly, you’ll end up with a confusing error.

Now, let's look at a scenario I had to deal with that involved files. Sometimes, you don’t want to pass an entire configuration, but rather refer to specific files based on an entry in the `dag_run.conf`. Imagine a situation where we had to process user-uploaded files, and the location of these files was specified in the `dag_run.conf`.

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='ssh_with_file_location',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    ssh_task = SSHOperator(
        task_id='execute_file_processing',
        ssh_conn_id='my_ssh_connection',
        command="""
            FILE_PATH='{{ dag_run.conf.get('file_location', '/default/path/file.txt') }}'
            python3 /path/to/my/remote_file_processor.py --file "$FILE_PATH"
            """,
        dag=dag,
    )
```

In this case, the `command` argument contains an expression retrieving 'file_location' from the `dag_run.conf` with a default path if the key is absent. The value is stored in the `FILE_PATH` shell variable and passed to the python script using the argument `--file`. This demonstrates that values can also represent file locations, which can be useful in different scenarios.

It’s important to ensure the value that you’re passing in your `dag_run.conf` matches the expected data type. For example, if your remote script is expecting a boolean and it receives the string 'True' instead, you'll encounter errors. Careful type checking and validation should be performed in the remote script. Furthermore, error handling is critical. If a key doesn't exist or a value isn't correct, your remote script should handle the situation appropriately.

For further reading and a deeper dive into these concepts, I’d suggest consulting the official Airflow documentation, especially the sections on Jinja templating and `SSHOperator`. Also, "Fluent Python" by Luciano Ramalho can prove exceptionally helpful to ensure you have a proper grip on Python syntax and semantics, especially when handling more complex data structures before processing them in your remote scripts. Another resource worth exploring is the book "Designing Data-Intensive Applications" by Martin Kleppmann. It provides a solid understanding of data processing in distributed environments, which often relates to the type of tasks you might run on remote servers via SSH.

In conclusion, accessing `dag_run.conf` values within `SSHOperator` is achieved using Jinja templating, providing flexibility for dynamic configuration. You can retrieve these values using `dag_run.conf.get()`, serialize complex objects with `tojson`, and pass it as part of a command string. Just be sure to handle your data types correctly, and remember to always provide appropriate defaults and error handling. With a bit of practice, you’ll be able to make use of these concepts to streamline your Airflow workflows.
