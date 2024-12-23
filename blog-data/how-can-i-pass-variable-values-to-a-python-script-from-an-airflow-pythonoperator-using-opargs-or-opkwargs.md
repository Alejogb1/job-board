---
title: "How can I pass variable values to a Python script from an Airflow PythonOperator using `op_args` or `op_kwargs`?"
date: "2024-12-23"
id: "how-can-i-pass-variable-values-to-a-python-script-from-an-airflow-pythonoperator-using-opargs-or-opkwargs"
---

Alright, let’s tackle this. Passing values into your Python scripts from an Airflow `PythonOperator` is a core task, and while it might seem straightforward, subtle nuances can trip you up. I’ve certainly had my share of debugging sessions where I’ve overlooked something simple, so let me share some insights drawn from those experiences.

The `PythonOperator` in Airflow gives you two primary mechanisms to feed external data into the python callable it executes: `op_args` and `op_kwargs`. The choice between them often comes down to how you prefer to structure your function’s input: positional arguments (`op_args`) or keyword arguments (`op_kwargs`). Let’s break down how they function and when to favor one over the other.

First, consider `op_args`. This parameter expects a list of values. When the operator executes, these values are unpacked and passed to your Python function as positional arguments. Think of it as passing parameters to a function in the order they are expected. This works well when you have a small, well-defined set of parameters. Let me give you a simple example:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def my_function(param1, param2, param3):
    print(f"Received: param1 = {param1}, param2 = {param2}, param3 = {param3}")

with DAG(
    dag_id='op_args_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    task = PythonOperator(
        task_id='my_task',
        python_callable=my_function,
        op_args=['value1', 123, True]
    )
```

In this case, `my_function` is called with ‘value1’ assigned to `param1`, `123` to `param2`, and `True` to `param3`. As a word of caution, remember that the order and count of elements in `op_args` must precisely match the positional parameters of your target function. Errors due to mismatches are not uncommon, and troubleshooting them means paying close attention to your function signature and the order in `op_args`.

Now, let's turn to `op_kwargs`. This is where we deal with keyword arguments. This parameter accepts a dictionary. The keys in the dictionary are mapped to the keyword arguments of your callable. This approach is especially useful when dealing with a larger set of parameters, when default values are used, or when the order of the parameters is less relevant. Suppose we modify our previous example to utilize `op_kwargs`.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def my_function_kwargs(param1, param2=None, param3=False):
    print(f"Received: param1 = {param1}, param2 = {param2}, param3 = {param3}")

with DAG(
    dag_id='op_kwargs_example',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    task = PythonOperator(
        task_id='my_task_kwargs',
        python_callable=my_function_kwargs,
        op_kwargs={'param1': 'keyword_value', 'param2': 456}
    )
```

Here, the function `my_function_kwargs` will be invoked with `param1` set to ‘keyword_value’ and `param2` set to `456`. The `param3` keyword argument uses the default, `False`, because it was not provided in the `op_kwargs` dictionary. This is where `op_kwargs` demonstrates flexibility—the order of key-value pairs in the dictionary doesn't matter, and parameters with default values can easily be omitted.

In my experience managing complex workflows, I found that using `op_kwargs` tends to result in more readable and maintainable code, especially when a function has many parameters, or when the parameter set tends to change more frequently. Positional arguments are convenient for simple functions, but for more complex scripts, keyword arguments via `op_kwargs` are just more robust.

Now, let’s tackle a more advanced scenario incorporating template values, which you will likely encounter in any practical usage of Airflow. Airflow uses Jinja templating for many of its values, and these templates can be powerful. Let's see how this works when interacting with `op_kwargs`.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.utils.dates import days_ago

def template_function(my_date, logical_date):
    print(f"My Date: {my_date}, Logical Date: {logical_date}")


with DAG(
    dag_id='template_kwargs_example',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
) as dag:
    task = PythonOperator(
        task_id='template_task',
        python_callable=template_function,
        op_kwargs={
            'my_date': '{{ ds }}',
            'logical_date': '{{ dag_run.logical_date }}',
            },
    )

```

In this case, we’re not passing static string values but rather Jinja templates. Airflow processes these before passing them to the python callable. `{{ ds }}` resolves to the execution date as a string (`YYYY-MM-DD`), while `{{ dag_run.logical_date }}` resolves to a datetime object, also representing the execution date. Using Jinja templating allows your data pipelines to dynamically adjust to various contexts and provides the ability to work with run-specific details. This is crucial for pipelines that operate on date-partitioned data, as an example.

It is critical to be aware that when using Jinja templates within `op_kwargs`, these templates must be strings. Attempting to pass a non-string representation of a Jinja template can cause unexpected issues during task execution. Airflow evaluates the strings using Jinja, replacing placeholder strings with dynamic values from the execution context.

A few things to consider beyond basic examples: when dealing with more complex parameter structures, think about whether a dictionary passed via `op_kwargs` should be serializable using JSON or similar methods. It’s common to encounter situations where you need to handle lists, nested dictionaries, or custom objects. Serializing them to JSON will simplify passing them along to your python scripts via `op_kwargs`.

For those diving deeper into Airflow, I highly recommend “Programming Apache Airflow,” by Bas P. Harenslak and Julian J. Gonzalez. It’s an excellent resource that covers these nuances in detail. Additionally, the official Apache Airflow documentation is invaluable for understanding the specific parameters and configurations available. Another beneficial resource is the book "Data Pipelines with Apache Airflow", which also presents practical guidance on constructing reliable data processing pipelines. I also suggest consulting the Jinja templating engine documentation; this will clarify usage rules that apply when working with Jinja templates within Airflow.

In summary, both `op_args` and `op_kwargs` are instrumental tools for passing parameters to python functions in Airflow. While `op_args` handles positional arguments directly, `op_kwargs` adds the flexibility of keyword arguments. For any kind of real-world complexity, `op_kwargs` is generally a better option, and the combination of `op_kwargs` and jinja templating will give you considerable control over how your python callables get their data. Just pay close attention to the types, order, and whether templates are strings or not, and you should be able to manage this with ease.
