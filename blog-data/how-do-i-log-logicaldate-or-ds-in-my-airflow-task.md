---
title: "How do I log 'logical_date' or 'ds' in my Airflow task?"
date: "2024-12-23"
id: "how-do-i-log-logicaldate-or-ds-in-my-airflow-task"
---

Alright, let's tackle logging the ‘logical_date’ or ‘ds’ within an Airflow task. This is a common need, and I've found myself addressing this numerous times in various pipeline setups, from batch processing of financial data to real-time sensor ingestion. The challenge isn't so much *how* it's done, but *how to do it correctly* in a way that’s both reliable and informative. We aren't just printing strings; we’re trying to create meaningful audit trails and debug information.

First off, let's clear up some terminology. 'logical_date' and 'ds' (which stands for *date string*) refer to the execution date of your airflow dag run. Crucially, this isn't always the real-time when the task executes. It's the date the dag *is scheduled to run for*, based on your schedule intervals. This concept is central to Airflow’s backfilling capabilities and it’s something that can easily trip up newcomers if not understood clearly. Think of 'logical_date' as the timestamp for the *data period* your task is processing, not necessarily the time when it's processing it.

Now, on to how you access this information. Airflow provides this through the execution context, which is a dictionary passed to your tasks. The context keys we're interested in are `dag_run` and the `execution_date`. Within `dag_run`, you'll find various metadata about that specific DAG run, including `logical_date`. The ‘ds’ is a formatted string version of the `execution_date` object, providing you with a convenient string formatted as `YYYY-MM-DD`.

Here's the initial hurdle many face. We don't just blindly access this context, or we run the risk of encountering issues if the context isn’t fully available (especially when testing or executing outside of a dag run context). It's best to use the parameters passed by Airflow to tasks through the context variable.

Let's look at some practical examples. I'll showcase python-based tasks, as that's where the majority of airflow workflows tend to be.

**Example 1: Basic Logging Within A Python Operator**

```python
from airflow.decorators import task
from airflow.utils.log.logging_mixin import LoggingMixin
import logging

@task
def log_date_basic(ds=None, execution_date=None, dag_run=None):
  log = LoggingMixin().log
  log.info(f"Task started processing for logical_date: {execution_date.isoformat()} using 'execution_date' object")
  log.info(f"Task started processing for logical_date: {ds} using 'ds' variable")
  log.info(f"Task started processing for logical_date: {dag_run.logical_date.isoformat()} via dag_run")
  return 'logged'
```

This example uses the `@task` decorator (introduced in recent Airflow versions), and it directly receives `ds`, `execution_date`, and `dag_run` as parameters, injected by Airflow. I'm pulling the logger from the `LoggingMixin` class here to keep things consistent with Airflow standards and the output format you would expect in the UI. Using `isoformat()` allows a structured approach to the datetime object, as opposed to just a `str()` conversion, making it easier for automated parsing further down your processing chains. This helps when using something like the Elastic stack to process the logs.

**Example 2: Accessing from within a Bash Operator (using macros)**

Sometimes you’re dealing with bash scripts or external commands. Here's how to pass the logical_date via environment variables:

```python
from airflow.operators.bash import BashOperator

bash_log_date = BashOperator(
    task_id="bash_log_date",
    bash_command="echo 'Executing for logical_date: $AIRFLOW_CTX_DAG_RUN_LOGICAL_DATE'",
    env={
          "AIRFLOW_CTX_DAG_RUN_LOGICAL_DATE":"{{ dag_run.logical_date }}"
        }
)
```

Notice how we're accessing the `dag_run.logical_date` via jinja templating. Airflow evaluates this within its engine, making it readily available within the Bash context. I tend to define these constants in my `airflow.cfg` file as jinja functions that are available across the configuration of an airflow instance to ensure there's consistency on how these fields are being handled in different parts of our codebase. This can prevent unexpected results due to inconsistent usage across teams.

**Example 3: Passing the Logical Date to an External System (e.g., API Call)**

Let’s say you need to pass the logical date to a downstream service as a query parameter. I've had to do this countless times when pushing data to RESTful APIs that require the processing date.

```python
from airflow.decorators import task
import requests

@task
def api_log_date(ds=None, execution_date=None, dag_run=None):
  url = "https://api.example.com/ingest" #replace with your actual URL
  formatted_date = dag_run.logical_date.isoformat()
  params = {"processing_date": formatted_date}
  response = requests.get(url, params=params)

  if response.status_code != 200:
    logging.error(f"Error calling API: {response.status_code}, {response.text}")
    raise Exception(f"Failed API call")

  logging.info(f"API call successful with logical_date: {formatted_date}")
  return 'API Call complete'
```

In this example, the  `dag_run.logical_date` is passed to a fictitious API endpoint in a parameter called `processing_date`. Again, using `.isoformat()` ensures a standardized date format. Handling of API errors with a more detailed logging and exception is crucial in robust production scenarios.

Now, some practical insights I've gathered over the years.

*   **Consistency is key**: Always use the `.isoformat()` or `.strftime()` method for consistent date formatting. It avoids ambiguity and simplifies debugging later on.
*   **Avoid direct string manipulation**: Prefer the use of methods provided with the python datetime object for date formatting; it's less error-prone.
*   **Test locally**: Use airflow's `cli` to `test` your tasks to make sure your code isn't just compiling, but that the logic works as expected before going to a production system.
*   **Log level**: Be mindful of your log levels (`debug`, `info`, `warning`, `error`, `critical`). Use `info` for general execution information, `error` for failures, and potentially `debug` for detailed intermediate states, which will need to be configured separately.
*   **Don’t overdo it**: Only log the information that’s genuinely useful for debugging or auditing. Too much noise makes it harder to identify critical issues. This is especially important for more complex DAGs where logs can rapidly become very large.

For further reading, consider delving into the core Airflow documentation concerning execution context and the built-in Jinja templating mechanism. The book “Data Pipelines with Apache Airflow” by Bas Pijnenburg, is a great practical reference. Also, the “Effective Python” book by Brett Slatkin has many examples of best practices in formatting datetime objects and when to use specific string manipulation methods.

In summary, accessing the logical_date in Airflow is straightforward, but it requires careful handling of the execution context and a focus on standardized logging and formatting. By using the examples above and the recommended resources, you can ensure you're doing it in a way that is reliable and insightful for your data pipelines. Remember, well structured logs are a critical foundation for robust pipelines that are easy to debug and maintain.
