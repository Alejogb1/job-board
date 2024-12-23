---
title: "How do I access dag_run.conf values and pass them to an SSHOperator command in Airflow?"
date: "2024-12-23"
id: "how-do-i-access-dagrunconf-values-and-pass-them-to-an-sshoperator-command-in-airflow"
---

, let's dive into this, because I've certainly been down that particular rabbit hole a few times. The core challenge here, as you've pointed out, is accessing those `dag_run.conf` values within your Airflow tasks, specifically when you need to dynamically pass parameters to an `SSHOperator` command. It's not inherently intuitive, but with a bit of understanding of Jinja templating and Airflow's context variables, it becomes quite manageable.

My first experience with this wasn’t on a greenfield project. It was a legacy system where scheduled jobs depended on configuration data passed into the workflow, which then needed to be used in shell scripts executed on remote machines. We had a lot of hard-coded values scattered across various dag files, making them difficult to maintain and scale. It became evident that we needed a more dynamic way to manage configuration, and that's how I really got to grips with using `dag_run.conf`.

The essential idea is that the `dag_run.conf` allows you to provide configuration values when triggering a dag. These values are then exposed within the Airflow context as part of a templating environment. Airflow, under the hood, utilizes Jinja2 for its templating engine, which lets you embed Python-like expressions inside strings that get evaluated at runtime. When using an `SSHOperator`, the command string can leverage these templated values.

The first key aspect is ensuring that you're calling the `SSHOperator` correctly to allow template rendering. The `command` argument is where the Jinja magic happens. The `dag_run.conf` values are accessible through the `dag_run` dictionary within the templating context. So, a simplified example might be something like this:

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime

with DAG(
    dag_id='ssh_conf_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    run_ssh_command = SSHOperator(
        task_id='run_ssh_command',
        ssh_conn_id='your_ssh_connection',  # replace with your connection id
        command="""
            echo "The environment is: {{ dag_run.conf.environment }}"
            echo "The input file is: {{ dag_run.conf.input_file }}"
            /path/to/your/script.sh {{ dag_run.conf.input_file }}
        """,
    )
```

In this scenario, if you were to trigger this dag and include a `conf` dictionary like `{"environment": "dev", "input_file": "/data/input.txt"}`, those values would replace the corresponding Jinja expressions within the `command`. Notice the triple quotes used for the command string, which allow multi-line statements and avoid having to escape quotes within the command. This is a crucial detail to avoid parsing issues.

Now, let's consider a more practical scenario where you might want to include defaults. It's common that some parameters aren't always provided when triggering the dag. In this case, it’s useful to use Jinja’s `get` function with a default value. Take this example:

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime
from airflow.utils.dates import days_ago

with DAG(
    dag_id='ssh_conf_example_defaults',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False
) as dag:

    run_ssh_command_defaults = SSHOperator(
        task_id='run_ssh_command_defaults',
        ssh_conn_id='your_ssh_connection', # replace with your connection id
        command="""
            echo "Processing file: {{ dag_run.conf.get('file_path', '/default/input.csv') }}"
            echo "Using batch size: {{ dag_run.conf.get('batch_size', 1000) }}"
            /path/to/your/processor.py --input_file "{{ dag_run.conf.get('file_path', '/default/input.csv') }}" --batch_size {{ dag_run.conf.get('batch_size', 1000) }}
        """,
    )
```

Here, the Jinja expression `dag_run.conf.get('file_path', '/default/input.csv')` will first attempt to retrieve the value of `file_path` from `dag_run.conf`. If `file_path` is not provided, it defaults to `/default/input.csv`. The same logic applies for `batch_size`, using `1000` as a default. This provides a level of resilience and allows you to trigger your dag without supplying all arguments every time.

Another important point to note is how you handle more complex data structures within your `dag_run.conf`. If you're passing nested dictionaries or lists, you need to access them with the proper dot notation or array indexing within your Jinja expressions. For instance, if you send in `{"database": {"host": "db.example.com", "port": 5432}}`, you would access the `host` as `{{ dag_run.conf.database.host }}`. This method can become less readable for complicated structures.

Finally, let's consider a more dynamic use case where you might want to construct the command string based on a dictionary passed via `dag_run.conf`. Consider the following example:

```python
from airflow import DAG
from airflow.providers.ssh.operators.ssh import SSHOperator
from datetime import datetime
from airflow.utils.dates import days_ago
import json

with DAG(
    dag_id='ssh_conf_dynamic',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False
) as dag:
    
    run_ssh_dynamic = SSHOperator(
        task_id='run_ssh_dynamic',
        ssh_conn_id='your_ssh_connection', # replace with your connection id
        command="""
        {% set command_args = dag_run.conf.get('command_args', {}) %}
        {% set arg_string = [] %}
        {% for key, value in command_args.items() %}
            {% if value %}
                {% set _ = arg_string.append('--' + key + ' ' + value | string) %}
            {% endif %}
        {% endfor %}
        /path/to/your/cli.py {{ arg_string | join(' ') }}
        """,
    )
```

In this more complex example, we're using Jinja's loop constructs to iterate over a dictionary of command-line arguments. The assumption is that your `dag_run.conf` might contain something like `{"command_args": {"verbose": "true", "output_dir": "/data/output/", "limit": null}}`. Notice the null value, this showcases how the script will skip parameters with a falsey value. The result is that we generate a command that will include parameters only when provided and they are truthy.

To further expand your knowledge on these concepts, I'd highly recommend diving into the official Jinja2 documentation. It’s incredibly comprehensive and explains the intricacies of the templating engine in great detail. Also, thoroughly explore the Airflow documentation regarding task context variables; it contains a wealth of information on what's accessible during task execution. In particular, pay attention to the section on Jinja templating. These are your fundamental resources. Consider also reading "Data Pipelines with Apache Airflow" by Bas Pijls, which will provide additional practical guidance.

The key takeaway here is that `dag_run.conf` is a flexible and powerful mechanism for injecting dynamic configuration data into your Airflow workflows. By mastering Jinja templating and understanding how to access the dag context, you can build much more robust and maintainable pipelines. Don't underestimate the power of defaults when handling optional configuration values, and remember to structure your command strings to be both readable and maintainable as your configurations get more complex.
