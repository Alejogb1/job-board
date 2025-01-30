---
title: "How do I view Airflow output?"
date: "2025-01-30"
id: "how-do-i-view-airflow-output"
---
Airflow’s output, specifically task logs, constitutes a crucial component for debugging and monitoring workflows; direct access and effective interpretation are essential for maintaining operational health. Having spent considerable time troubleshooting complex pipelines involving data transformations and external API integrations, I’ve developed a practical understanding of how to navigate Airflow’s logging system and extract actionable insights. This response will outline the various methods for viewing task output, offer concrete examples, and highlight pertinent resources for further exploration.

Fundamentally, Airflow stores task logs in the directory defined by the `logging_base_dir` configuration setting, typically located within the Airflow installation path (e.g., `/opt/airflow/logs`). These logs are structured hierarchically, reflecting the DAG's organization. Each DAG run creates a subdirectory, containing further subdirectories for individual tasks, then attempting to execute the command using the specific task's date. Files within these task directories are named according to the attempt number, meaning if a task fails and retries, you can observe logs from previous runs. This structure makes identifying specific issues easy.

The most common access point is the Airflow web UI, which offers several mechanisms for examining logs. On the "Graph View" or "Tree View," after a DAG run has started, you’ll notice task instances displaying different colors; these colors reflect the state of the task. If a task fails, its color will change accordingly, and you can click on the task instance to access the task details. These details include the “Log” tab, which renders the log output directly in the browser. Here, you can access previous attempts of that task, if retries were configured, and you can typically see the logs from any tasks upstream if they used the traditional logging framework. The web UI also allows you to download the log files as text, which can be useful for offline analysis or for sending log output to a support team.

Another approach to viewing logs is via the command-line interface (CLI). The `airflow logs` command is particularly useful, allowing you to specify a DAG id, task id, and execution date to retrieve logs directly to your terminal. The CLI will return both the standard output and standard error of the task's execution, often providing very detailed error messages. This is particularly beneficial when performing initial debugging or when needing to quickly sift through a large amount of log data without the overhead of the web UI. This access method often feels faster, and it is particularly useful for scripted use cases where it is better to avoid the UI.

While the UI and CLI are the primary methods for viewing output, Airflow also provides flexibility in how logs are handled. Log handlers are configured in `airflow.cfg`, allowing you to modify log output destination. While standard output is typically sent to the local file system, custom log handlers can be configured. This enables logs to be directed towards external logging systems, like Elasticsearch or Splunk, for centralized aggregation and analysis. These platforms provide advanced search and filtering capabilities, facilitating more comprehensive monitoring of Airflow’s operational health. I’ve found these integrations invaluable when debugging complex production setups, as they allow for correlation between different parts of the system.

Furthermore, the output of tasks can be accessed programmatically via Airflow’s xcom mechanism. If tasks explicitly push data to xcom (using `task_instance.xcom_push`), this data can be retrieved by downstream tasks. This mechanism isn't about logs themselves, but is invaluable when needing the output of a task to be used by other tasks. It's critical to understand that xcom has size limitations and is not intended for transferring large data payloads.

Below are three code examples demonstrating logging and accessing it within Airflow tasks:

**Example 1: Basic Task Logging**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def log_message():
    print("This is a message from the task.")

with DAG(
    dag_id="basic_logging_example",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    log_task = PythonOperator(
        task_id="log_task",
        python_callable=log_message
    )
```

In this example, the `log_message` function utilizes a simple `print` statement. This output will be captured by Airflow’s logging system and be available through the web UI or command-line interface. When examining the logs for `log_task`, the line "This is a message from the task." will be present. This represents the simplest form of log generation in an Airflow context, demonstrating that standard output is captured.

**Example 2: Logging With the Logger Object**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.log.logging_mixin import LoggingMixin
from datetime import datetime

class LogTest(LoggingMixin):
   def __init__(self):
      super(LogTest, self).__init__()

   def log_some_stuff(self):
      self.log.info("This is an info message.")
      self.log.warning("This is a warning message.")
      self.log.error("This is an error message.")

def log_with_logger():
    log_test = LogTest()
    log_test.log_some_stuff()

with DAG(
    dag_id="logger_logging_example",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    log_task = PythonOperator(
        task_id="log_task",
        python_callable=log_with_logger
    )
```

This example demonstrates a more sophisticated approach to logging. Using the `LoggingMixin` class, a task has access to `self.log`, which is a standard logging object. This facilitates different log levels (info, warning, error) and structures messages more effectively. Each logged statement will appear in the task log with a prepended timestamp and log level. This level of granularity is beneficial for advanced debugging, particularly when using filtering capabilities in external logging platforms. The message text is also much clearer to the reader and provides more information than print statements.

**Example 3: XCom and Task Output**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def push_value_to_xcom(ti):
    ti.xcom_push(key="my_value", value="Hello from first task!")

def pull_value_from_xcom(ti):
    retrieved_value = ti.xcom_pull(key="my_value", task_ids="push_value_task")
    print(f"Retrieved from XCom: {retrieved_value}")

with DAG(
    dag_id="xcom_example",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    push_value_task = PythonOperator(
        task_id="push_value_task",
        python_callable=push_value_to_xcom
    )
    pull_value_task = PythonOperator(
       task_id="pull_value_task",
       python_callable=pull_value_from_xcom
    )
    push_value_task >> pull_value_task

```
Here, the first task pushes a string value to xcom using the `xcom_push` function. The second task retrieves the value using `xcom_pull`, specifying the key and task id of the pushing task.  The printed output of `pull_value_task` contains the data pushed using xcom in a different task. While not logs in the traditional sense, `xcom` effectively transmits task outputs between Airflow tasks.

To solidify your understanding, several resources are available. Airflow’s official documentation is the primary source for comprehensive information on logging and task output.  The official documentation will also detail how to configure advanced log handlers for specific use cases. Tutorials and blog posts by companies that use Airflow extensively can also offer practical examples and highlight common issues. Additionally, consulting the source code of the `apache-airflow` project can provide a deep technical understanding of internal mechanisms. Finally, the Airflow community maintains an active discussion forum, which is excellent for getting help when facing unexpected problems. I've found a variety of solutions to complex problems by simply searching through these forums.

By exploring these options and practicing regularly, any user can gain proficiency in viewing and understanding task output within Airflow, thus improving their workflow management capabilities.
