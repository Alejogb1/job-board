---
title: "How can I log 'logical_date' or 'ds' in an Airflow task?"
date: "2024-12-23"
id: "how-can-i-log-logicaldate-or-ds-in-an-airflow-task"
---

Okay, let's talk about logging the 'logical_date' or 'ds' within an Airflow task. It's a frequent hurdle, and I've definitely been down that rabbit hole more times than I care to count. When you're debugging or trying to understand the context of a task's execution, having that specific execution date readily available in your logs is crucial.

Here's the thing: 'logical_date' (or its string representation, 'ds') isn't inherently a globally available variable within the context of your task's execution. Airflow uses the concept of templating, and you need to leverage that to get the information you're after. Instead of imagining it's magically floating around, you have to actively request it.

I’ll recall a project from some years ago. We were processing daily financial reports, and having the execution date explicitly logged inside each task became vital when we needed to backfill several months of data following a system migration. Without clear execution dates in the log output, deciphering which task instance produced a particular log line became a nightmare, requiring tedious correlation with Airflow's web interface. We needed a consistent and robust solution that could easily integrate into various tasks.

The key to accessing 'logical_date' is using Airflow's Jinja templating capabilities. Jinja is a powerful templating engine, which Airflow uses for parameterizing many things, including variables passed to your tasks, and task commands. Think of it as a mechanism to inject dynamic information into your task definitions.

Here's how you approach it. The standard method is to use the `{{ ds }}` or `{{ dag_run.logical_date }}` template variables, where 'ds' is a string representation of the date in 'YYYY-MM-DD' format, and `dag_run.logical_date` is a datetime object. You can insert these directly into your Python code, inside the context of an Airflow operator. It's critical to understand that these aren't just regular python variables; they’re templates that Airflow evaluates *before* the task executes.

Let's move on to some concrete examples to illustrate this.

**Example 1: Using `{{ ds }}` directly in a BashOperator**

If you're executing a bash script, perhaps you want to add the logical date to a log message inside that shell context. You can pass the variable as an environment variable, or directly use it inside your script using template expansion through the `bash_command` variable in the `BashOperator`.

```python
from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

with DAG(
    dag_id="bash_logging_example",
    start_date=datetime(2023, 10, 26),
    schedule_interval=None,
    catchup=False
) as dag:

    log_task = BashOperator(
        task_id='log_date_bash',
        bash_command='echo "Task running for date: {{ ds }}" > /tmp/log_output.txt && cat /tmp/log_output.txt'
    )
```

In this example, the bash command itself will be executed on an executor. Before the command is executed, Airflow will evaluate the template `{{ ds }}` and replace it with the actual logical date string associated with the current dag run. If you examine the log, you'll find a line similar to: `Task running for date: 2023-10-26`. This is where Airflow’s templating power really shines. It provides context to that task from a date perspective.

**Example 2: Accessing `{{ dag_run.logical_date }}` in a PythonOperator**

When you're dealing with a `PythonOperator` and your actual python code, you need to pass the template variable as a parameter. It's still not available as a raw Python variable, but you can use it when defined as a keyword argument within your python_callable function parameters. You must define this as a parameter in your python function so Airflow can pass the value as parameter.

```python
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

def log_date(**kwargs):
    logical_date = kwargs['logical_date']
    print(f"Task running for logical date: {logical_date}")


with DAG(
    dag_id="python_logging_example",
    start_date=datetime(2023, 10, 26),
    schedule_interval=None,
    catchup=False
) as dag:

    log_task = PythonOperator(
        task_id='log_date_python',
        python_callable=log_date,
        op_kwargs={'logical_date': '{{ dag_run.logical_date }}'}
    )
```

Here, we're passing the `{{ dag_run.logical_date }}` template as a string value to an op_kwargs called `logical_date`. Airflow will evaluate the template and pass it as a value in your defined callable. Inside the `log_date` function, you then receive this as part of the `**kwargs`, allowing you to use it within your Python logic. The output in the log will be something like: `Task running for logical date: 2023-10-26 00:00:00+00:00`, a python datetime object (with UTC timezone, in this example).

**Example 3: Using a custom log handler**

In more sophisticated scenarios, you might need to add the logical date to every log message. Instead of having to use op_kwargs, which is not ideal, you can configure a custom log handler that includes the templated variable as context in your logs. This is often more efficient and less error prone than manually handling each log line.

```python
import logging
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime
from airflow.utils.log.logging_mixin import LoggingMixin

class CustomLogging(LoggingMixin):
    def __init__(self, dag_run):
        self.log = self.log
        self.dag_run = dag_run

    def info(self, message):
        self.log.info(f"LOG DATE: {self.dag_run.logical_date} | {message}")

def log_date_custom(dag_run):
    custom_log = CustomLogging(dag_run)
    custom_log.info("this is my log message")

with DAG(
    dag_id="custom_logging_example",
    start_date=datetime(2023, 10, 26),
    schedule_interval=None,
    catchup=False
) as dag:
   log_task = PythonOperator(
      task_id="log_with_custom_handler",
      python_callable=log_date_custom,
      op_kwargs={"dag_run":"{{ dag_run }}"}
   )
```

Here, we created our own log wrapper that receives the `dag_run` object as a parameter. The `dag_run` object has several useful attributes available at run-time, including `logical_date`. Within our custom logger we can then include this in each log message that uses the `info` method. This will keep your log format consistent, avoid having to remember to add the variable at each logging instance, and provide a consistent way to log any dag context variable. In practice, you may also want to consider an Airflow plugin if you plan to extend the Airflow logging functionality.

Regarding more information on templating and dag objects: I would suggest diving into the Airflow documentation; it's pretty comprehensive and provides all the specifics on Jinja templating with Airflow. For a deeper understanding of Jinja itself, refer to the official Jinja documentation – it's an incredibly useful resource. Additionally, "Programming Airflow" by J.B. Rubinovitz is a great book that helps with all the concepts around Airflow, and its best practices.

In conclusion, the key to logging 'logical_date' or 'ds' in Airflow is understanding and using the templating engine. Once you grasp how to access the template variables and how to pass them through your tasks, you'll find logging relevant task metadata a straightforward process. Always leverage templates, and if you need more consistent logging, a customized logger is the best solution. It's all about extracting information that is already present within Airflow's execution context, rather than relying on a global variable that doesn't exist.
