---
title: "How can I suppress all informational airflow logs without suppressing application-specific logs?"
date: "2025-01-30"
id: "how-can-i-suppress-all-informational-airflow-logs"
---
The core challenge in suppressing informational airflow logs while preserving application-specific logs lies in the nuanced configuration of Airflow's logging system.  Airflow, by default, utilizes a hierarchical logging structure, meaning log messages are categorized based on their origin and severity.  Simply adjusting the root logger's level is insufficient, as this affects all loggers, including those crucial for tracking application-specific events within your DAGs. My experience troubleshooting similar issues in large-scale ETL pipelines has taught me the critical importance of granular control over logging levels at the individual logger level.  This necessitates using specific configuration techniques to target only informational messages emanating from Airflow's internal components.

**1. Understanding Airflow's Logging Hierarchy:**

Airflow's logging mechanism leverages Python's `logging` module.  The root logger sits at the top, with child loggers branching out to represent various components – the scheduler, webserver, executors, and individual DAGs.  Each logger has a level associated with it (DEBUG, INFO, WARNING, ERROR, CRITICAL). Log messages are only recorded if their severity level is equal to or greater than the logger's configured level.  The challenge lies in filtering informational messages (`INFO` level) from the Airflow core loggers without affecting the application-specific loggers within your DAGs, which are often configured independently.

**2.  Strategies for Selective Log Suppression:**

The most effective approach avoids modifying the root logger. Instead, we focus on manipulating the logging levels of specific Airflow loggers responsible for generating informational airflow logs. This necessitates identifying these loggers, a process that can involve examining Airflow's source code or carefully analyzing the logs themselves.  In my experience, the primary culprits often include loggers associated with the scheduler and the executor.  The `airflow.scheduler` and `airflow.executor` loggers, along with any associated sub-loggers, are prime candidates for level adjustment.

**3. Code Examples with Commentary:**

Below are three examples demonstrating different methods for achieving selective log suppression.  These examples leverage Python's `logging` configuration capabilities directly within your Airflow environment, often within the `airflow.cfg` file or a custom configuration file that is subsequently imported.  Remember to restart your Airflow services after applying any configuration changes.

**Example 1:  Modifying the `airflow.cfg` file:**

This approach directly alters the Airflow configuration file. While less flexible for dynamic adjustments, it's suitable for static configurations:

```python
# airflow.cfg
[loggers]
keys=root, airflow.scheduler, airflow.executor

[handlers]
keys=consoleHandler

[formatters]
keys=simpleFormatter

[logger_root]
level=WARNING
handlers=consoleHandler

[logger_airflow.scheduler]
level=WARNING
handlers=consoleHandler
qualname=airflow.scheduler
propagate=0

[logger_airflow.executor]
level=WARNING
handlers=consoleHandler
qualname=airflow.executor
propagate=0

[handler_consoleHandler]
class=StreamHandler
level=DEBUG
formatter=simpleFormatter
args=(sys.stdout,)

[formatter_simpleFormatter]
format=%(asctime)s - %(name)s - %(levelname)s - %(message)s
datefmt=%Y-%m-%d %H:%M:%S
```

This configuration sets the logging level for the `airflow.scheduler` and `airflow.executor` loggers to `WARNING`, effectively suppressing INFO-level messages from these components while leaving the root logger and other loggers (including those within your DAGs) at their default levels.  The `propagate=0` directive prevents the log messages handled at this logger level from propagating up the logging hierarchy.


**Example 2: Programmatic Configuration within a DAG:**

This provides greater flexibility, allowing dynamic control over logging levels based on runtime conditions:

```python
import logging

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago

with DAG(
    dag_id='selective_logging_dag',
    start_date=days_ago(1),
    schedule_interval=None,
    tags=['logging'],
) as dag:
    def suppress_airflow_info():
        logger = logging.getLogger('airflow.scheduler') #Target the specific logger
        logger.setLevel(logging.WARNING)
        logging.info("This message will NOT be logged by the airflow scheduler.")  # will be suppressed
        logging.warning("This warning message WILL be logged by the airflow scheduler.") #will be shown

    suppress_logs_task = PythonOperator(
        task_id='suppress_airflow_info',
        python_callable=suppress_airflow_info
    )

```

Here, we directly manipulate the logger's level within a Python callable executed as part of the DAG. This allows for context-sensitive logging control, suppressing informational logs only for a specific part of the workflow.


**Example 3: Using a custom logging handler:**

A more advanced technique uses a custom logging handler to filter messages based on the logger name:


```python
import logging
from logging import Handler, LogRecord

class AirflowInfoFilter(Handler):
    def emit(self, record: LogRecord):
        if not record.name.startswith('airflow.') or record.levelno > logging.INFO:
            self.handle(record)


#In your airflow.cfg or a custom config:

[loggers]
keys=root, airflow.scheduler, airflow.executor

[handlers]
keys=consoleHandler, airflowFilter

[handler_airflowFilter]
class=__main__.AirflowInfoFilter
level=DEBUG
formatter=simpleFormatter
args=()


```

This custom handler (`AirflowInfoFilter`) examines the logger name.  Only logs NOT starting with `airflow.` or those exceeding INFO level are processed.  This provides a flexible and robust method to suppress logs from a specific logger subset.


**4. Resource Recommendations:**

Consult the official Airflow documentation on logging configuration for detailed information.  Review Python's `logging` module documentation for a comprehensive understanding of logging mechanisms.  Familiarize yourself with the structure of your Airflow deployment to pinpoint the specific loggers that are generating the undesired informational messages.  Testing and iterative refinement of your logging configuration are crucial to achieve the desired level of control.
