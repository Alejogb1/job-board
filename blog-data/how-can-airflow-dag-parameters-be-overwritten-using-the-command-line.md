---
title: "How can Airflow DAG parameters be overwritten using the command line?"
date: "2024-12-23"
id: "how-can-airflow-dag-parameters-be-overwritten-using-the-command-line"
---

, let's talk about overriding airflow dag parameters from the command line. It’s a capability I’ve leaned on heavily, particularly during those messy migration projects where constants seem to shift daily. I’ve seen many folks initially stumble with this, so let's break it down systematically. The core idea, when you get past the documentation jargon, is straightforward: you're essentially leveraging Jinja templating in your DAG definition combined with command-line arguments that airflow then makes available to the rendering context.

Think of airflow dag parameters, or configurations, as variables that can influence the behavior of your tasks. These parameters can be default values specified within your dag definition itself, but to inject flexibility, we want the ability to adjust these values externally, specifically from the command line when triggering the dag. This is where airflow's templating and command-line execution intertwine.

Let me illustrate this with a common scenario I encountered a few years ago. We had a complex data processing pipeline, and the input file path was consistently changing depending on the environment we were targeting – development, staging, or production. Rather than hard-code these paths within our dag, we chose to use a configurable parameter, allowing us to switch input locations with ease.

Here's how we structured it in the dag. We define the default parameter within the `default_args` section of the dag instantiation.

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'input_path': '/default/input/path.csv' # Default parameter
}

dag = DAG(
    'parameterized_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
)

process_data = BashOperator(
    task_id='process_data',
    bash_command='echo "Processing data from {{ dag_run.conf[\'input_path\'] }}"',
    dag=dag
)
```

In this example, the `input_path` is initially set to `/default/input/path.csv`. The magic happens within the `BashOperator` where `{{ dag_run.conf['input_path'] }}` is used. This is Jinja syntax that allows airflow to evaluate this value at runtime. The `dag_run.conf` dictionary contains any parameters passed in from the command line, and if there isn’t a command-line override, airflow uses the default value declared in the `default_args`.

Now, let's see how to overwrite it from the command line. Airflow uses the `airflow dags trigger` command followed by the name of the dag and then parameters passed using the `-c` (or `--conf`) flag, and it takes in json. Here is how we would run this dag, overriding the `input_path`.

```bash
airflow dags trigger parameterized_dag -c '{"input_path": "/override/path/input.csv"}'
```

This command launches the `parameterized_dag` dag, replacing the default `input_path` with `/override/path/input.csv`. We can confirm that the task now echoes out the overriden path. This is how one could switch from one environment to another when triggering the DAG.

A slightly more complex, albeit highly practical use case is when we want to pass more than one configuration to the dag. Let's say in addition to the `input_path`, we also want to pass a configuration flag that determines how the data should be processed. Here's how our DAG definition could look:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'input_path': '/default/input/path.csv',
    'process_mode': 'standard'  # Additional parameter
}

dag = DAG(
    'multi_param_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
)

process_data = BashOperator(
    task_id='process_data',
    bash_command='echo "Processing data from {{ dag_run.conf[\'input_path\'] }} in {{ dag_run.conf[\'process_mode\'] }} mode."',
    dag=dag
)

```

Here, in addition to `input_path`, we now have the parameter `process_mode`. The default mode is `standard`. We use the same templating technique in the `bash_command`, extracting both parameters from `dag_run.conf`.

Now, to override both, from the command line:

```bash
airflow dags trigger multi_param_dag -c '{"input_path": "/alternate/input/file.csv", "process_mode": "optimized"}'
```

This command triggers the `multi_param_dag`, providing new values for both the `input_path` and `process_mode`. We can now see the task output showing data being processed from `/alternate/input/file.csv` in `optimized` mode. This is where the flexibility shines.

One more scenario to further highlight the utility, is when needing to pass parameters that are not strings, or more complex nested parameters.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'config': {
        'retries': 3,
        'timeout': 60
    }
}

dag = DAG(
    'complex_param_dag',
    default_args=default_args,
    schedule_interval=None,
    catchup=False
)

def print_config(config):
    print(f"Retries set to: {config['retries']}")
    print(f"Timeout set to: {config['timeout']}")

process_data = PythonOperator(
    task_id='process_config',
    python_callable=print_config,
    op_kwargs={'config': "{{ dag_run.conf.get('config', dag.default_args.config) }}"},
    dag=dag
)

```

Here, we have the `config` parameter that holds a dictionary as it's value. If the `config` dictionary exists in `dag_run.conf`, the PythonOperator will use that, otherwise, it will fall back to the dag's default args.

To override this from command line:
```bash
airflow dags trigger complex_param_dag -c '{"config": {"retries": 5, "timeout": 120}}'
```

The python task will print that retries are set to 5 and that the timeout is 120 seconds. It is not necessary to send the complete structure in the command, you can send any level of nesting from the command line. If you specify a key that doesn't exist on the default dictionary, it will just add a new key-value pair to the `dag_run.conf`, and can be accessed in the tasks through jinja.

The important thing to always consider is: what is the data type of what you send through the command line? It will be a string when passed through the command line, even if that string represents a dictionary, a list, or a number. Airflow will interpret this string as a JSON, and will treat the structure as native python objects in `dag_run.conf` dictionary.

For further depth, I highly recommend looking at the following material: *“Jinja2 Documentation”*, available online, which details the templating engine used by airflow; the official airflow documentation specifically the *‘dag_run’* and *’Command line interface’* sections; and for a broader understanding of system design patterns, *“Patterns of Enterprise Application Architecture”* by Martin Fowler can provide a useful background. This, combined with practical application, is how I’ve come to manage increasingly complex workflows. Being able to override these parameters has proven invaluable in maintaining agility in our deployment process.
