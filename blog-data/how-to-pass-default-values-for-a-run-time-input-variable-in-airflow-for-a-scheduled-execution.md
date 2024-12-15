---
title: "How to pass default values for a run time input variable in airflow for a scheduled execution?"
date: "2024-12-15"
id: "how-to-pass-default-values-for-a-run-time-input-variable-in-airflow-for-a-scheduled-execution"
---

hey there,

i've seen this particular question pop up a bunch, and i get it. it's one of those "seems simple, but then you're knee-deep in dag files" kinda things. dealing with default values for runtime variables in airflow, specifically for scheduled runs, has been a thing for me, and i think i can break it down in a way that's useful.

let's first talk about why this even matters. when you're scheduling dags, especially those that take input parameters, you're often dealing with cases where certain parameters should have defaults. if, for example, the dag is processing some data that usually comes from a specific place or has a defined batch size, you dont want to re-enter this *every single time*, particularly for scheduled runs which should just run reliably. you want airflow to fallback on some reasonable value and only have to change it when needed.

i've personally been burned before because of this, back when i was less careful with parameter handling. i had a dag that was processing daily sensor data. it had a ‘data_path’ variable. in manual runs i’d always pass it the correct value, but when i deployed it for scheduled runs, i failed to provide a default and everything broke because the dag was not receiving that variable at all. it took me a good few hours of debugging to figure that out. i even thought the code itself had a bug. i was so proud when i finally figured out it was the lack of default values, hahaha, i felt like i had cracked a cypher. but yeah, a lesson learned.

anyway, airflow gives you several ways to handle default values for runtime variables, and the best way to approach it depends on exactly what you're trying to accomplish. let me give you a couple of options i have used myself.

**option 1: leveraging the `default_var` parameter in trigger_dag run**

one option is to define defaults directly in your dag file by defining a variable using the function `Variable.get`, and providing the `default_var` parameter. i personally don't use it often because it requires that the variable exist on the airflow ui, which i think it's a bit redundant, but sometimes it's useful if you want to provide a variable in the user interface or if you think other users could try to change this parameter. let's see a simple example:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable
from datetime import datetime


def print_runtime_parameters(**kwargs):
    data_path = Variable.get("data_path", default_var="/default/data/path")
    batch_size = int(Variable.get("batch_size", default_var=1000))
    print(f"Data path: {data_path}")
    print(f"Batch size: {batch_size}")


with DAG(
    dag_id="default_values_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    print_params_task = PythonOperator(
        task_id="print_parameters",
        python_callable=print_runtime_parameters,
    )
```

in this code block, if the variables `"data_path"` and `"batch_size"` are not present in airflow variables, then the python code will fallback to `"default/data/path"` and `1000` respectively. the default value will be used if the variable is not found, which is great when no parameter has been set via the airflow ui.

**option 2: using jinja templates with a dictionary**

another very cool approach, and my favorite if i'm honest, is to provide a dictionary of defaults and use the power of jinja templating to resolve the parameters from this dictionary. this method is very powerful because you can centralize your defaults in a single location, making it easy to manage and change. you can also introduce logic to the resolution process. here’s an example.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime


default_runtime_parameters = {
    "data_path": "/default/data/path_2",
    "batch_size": 2000,
    "processing_date": "{{ dag_run.logical_date.strftime('%Y-%m-%d') }}",
}


def print_runtime_parameters(**kwargs):
    data_path = kwargs["dag_run"].conf.get("data_path", default_runtime_parameters["data_path"])
    batch_size = int(
        kwargs["dag_run"].conf.get("batch_size", default_runtime_parameters["batch_size"])
    )
    processing_date = kwargs["dag_run"].conf.get(
        "processing_date", default_runtime_parameters["processing_date"]
    )
    print(f"Data path: {data_path}")
    print(f"Batch size: {batch_size}")
    print(f"Processing date: {processing_date}")


with DAG(
    dag_id="default_jinja_values_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    print_params_task = PythonOperator(
        task_id="print_parameters",
        python_callable=print_runtime_parameters,
    )
```

in this case, i've established a dictionary `default_runtime_parameters`, which holds all the default values i want to use. the `print_runtime_parameters` function then uses the `get` method of the `dag_run.conf` dictionary to retrieve the values; using this pattern, if it exists the given parameter is extracted from the configuration passed when the dag was triggered, otherwise the value of the `default_runtime_parameters` dictionary is taken.

the benefit here is the flexibility: you can get the default parameter from a dictionary, environment variables or any external system as long as the code has access to it.

**option 3: using templated fields in operators**

another option is to use airflow's templating engine to directly manage the parameters within the operators. if you are using operators like the `bashoperator` or the `pythonoperator`, you can define templated fields where you can assign your variables. this will make it simpler, especially if you want to pass the parameters to a bash script, for instance. this is how i do it if i need to run a bash script:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime


default_runtime_parameters = {
    "data_path": "/default/data/path_3",
    "batch_size": 3000,
    "processing_date": "{{ dag_run.logical_date.strftime('%Y-%m-%d') }}",
}


with DAG(
    dag_id="templated_field_dag",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    print_params_task = BashOperator(
        task_id="print_parameters",
        bash_command="""
            echo "data path: {{ dag_run.conf.get('data_path', params.data_path) }}"
            echo "batch size: {{ dag_run.conf.get('batch_size', params.batch_size) }}"
            echo "processing date: {{ dag_run.conf.get('processing_date', params.processing_date) }}"
        """,
        params=default_runtime_parameters,
    )
```

here, i'm making use of the `params` attribute of the `bashoperator`. airflow will make the content of `params` available to the `bash_command` via the jinja templating engine. just like before, if `dag_run.conf` has the variable i'm looking for it will be used, otherwise airflow will fallback to use the content of `params`. i use this pattern a lot when working with bash scripts. it is really helpful.

**some final thoughts**

when it comes to scheduled dags, think carefully about how your parameters are sourced. these defaults should be a safe fallback, not the primary method for setting variables in your dag. for scheduled runs, you might want to consider storing default parameters in airflow variables or a configuration file, allowing them to be changed separately from your dag code. it will be less error-prone in the long run.

if you're looking to delve deeper, i’d recommend looking into airflow's documentation on templating, particularly the jinja section. the official documentation is always a good starting point. also, the “data pipeline with airflow” book by bas van sumeren offers a more in-depth discussion about parameter management strategies.

let me know if you have any more specific questions or you need to cover other topics. i've spent a considerable amount of time working with airflow and i'm always happy to share my experience.
