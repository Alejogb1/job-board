---
title: "Why aren't Airflow DAG end dates and durations consistently reported in Slack on successful execution?"
date: "2025-01-30"
id: "why-arent-airflow-dag-end-dates-and-durations"
---
In my experience managing large-scale Airflow deployments, the inconsistent reporting of DAG end dates and durations in Slack notifications, particularly upon successful execution, often stems from a confluence of factors related to Airflow's internal scheduling mechanisms and the configuration of the notification system itself. This is not typically a bug in Airflow, but rather a consequence of how certain variables are resolved and how notifications are triggered. The root issue frequently lies in a misunderstanding of when templated values are rendered and the precise lifecycle stage of a DAG run where a notification is initiated.

The core problem is that the values we expect to represent the "end time" and "duration" are usually derived from the `execution_date` variable and the timestamps of task states within the DAG run. Airflow uses a combination of these internal variables and the task metadata stored in the metastore to track the progress of a DAG. The `execution_date`, in particular, is often confused with the actual start time of a DAG run. This distinction is critical. The `execution_date` refers to the logical date for which the DAG is running, not necessarily the exact moment it started. For instance, if a DAG is scheduled to run every day at midnight, the `execution_date` for a run that begins at 2 AM will still be the preceding midnight.

Furthermore, the 'end time' and 'duration' information is generally not part of the immediate context available when the DAG completes. Instead, this data needs to be retrieved after the DAG has finished, often by querying the metastore. The notification mechanism, therefore, has to be programmed to properly extract this information. The way Airflow's callbacks and listeners operate also plays a crucial role. When a DAG is marked as successful, the success callback is triggered and it's *here* that we often attempt to extract the execution information for Slack. However, depending on the exact hook or method used, the data might not yet be fully written to the metastore, or the appropriate context may not be readily available.

Let's consider a few concrete examples:

**Example 1: Incorrect Usage of Template Variables**

A common mistake is relying solely on template variables within the Slack notification message, without explicitly retrieving the execution information from the DAG run.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from datetime import datetime

def my_task():
    print("Task is running")

with DAG(
    dag_id="incorrect_slack_notification",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=["example"],
) as dag:
    task1 = PythonOperator(task_id="task1", python_callable=my_task)

    send_slack_notification = SlackWebhookOperator(
        task_id="slack_notification",
        slack_webhook_conn_id="my_slack_connection",
        message="DAG {{ dag.dag_id }} completed at {{ ds }}",  #  using ds, not runtime information
        trigger_rule="all_success"
    )

    task1 >> send_slack_notification
```
In this instance, the Slack message uses `{{ ds }}`, which is a convenient alias for the `execution_date`. While this *appears* to be the completion time, it is not. The `ds` variable reflects the logical date of the run, not the actual time it finished. Similarly, without specifically accessing the duration, that information will be missing. Consequently, the notification may report the run as finishing at a prior date/time, leading to the illusion of inconsistencies.

**Example 2: Using the `on_success_callback` without proper context**

Another approach involves the `on_success_callback` within a DAG's definition. If not implemented correctly, it can lead to the same inaccuracies.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.models import DagRun
from datetime import datetime

def slack_success_callback(context):
    dag_run = context.get('dag_run')
    if dag_run:
      start_time = dag_run.start_date.strftime('%Y-%m-%d %H:%M:%S')
      end_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S') # Incorrect: Should fetch real end time
      duration = (datetime.now() - dag_run.start_date).total_seconds()  # Incorrect calculation
      message = f"DAG {dag_run.dag_id} completed. Start time: {start_time}, End time: {end_time}, Duration: {duration} seconds"

    else:
      message = "DAG success callback triggered, but unable to get dag_run context"
    
    slack_notif = SlackWebhookOperator(
        task_id="slack_notification",
        slack_webhook_conn_id="my_slack_connection",
        message=message,
    )
    return slack_notif.execute(context=context)
    

def my_task():
    print("Task is running")

with DAG(
    dag_id="incorrect_success_callback",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    on_success_callback=slack_success_callback,
    tags=["example"],
) as dag:
    task1 = PythonOperator(task_id="task1", python_callable=my_task)
```

Here, while we retrieve the `start_date` from the `dag_run`, we are incorrectly assigning the current time (`datetime.now()`) as the end time. Furthermore, the duration calculation is therefore also incorrect since we are subtracting the current time from the start time. This approach does not query the metastore for the true completion time of the DAG and duration and introduces inconsistencies since it relies on the time the callback executes as opposed to when the DAG *actually* finished.

**Example 3: Correct Retrieval of End Time and Duration**

A more accurate approach involves accessing the Airflow metastore to retrieve the actual end time of the DAG.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.models import DagRun
from airflow.utils.timezone import utc
from datetime import datetime
from airflow.settings import Session
from sqlalchemy import func

def slack_success_callback(context):
    session = Session()
    dag_run = context.get('dag_run')

    if dag_run:
        try:
          query = session.query(
              DagRun.start_date,
              DagRun.end_date,
              (func.extract('epoch', DagRun.end_date) - func.extract('epoch', DagRun.start_date)).label('duration')
            ).filter(DagRun.dag_id == dag_run.dag_id, DagRun.run_id == dag_run.run_id)
          result = query.first()
          if result and result[0] and result[1]:
            start_time = result[0].strftime('%Y-%m-%d %H:%M:%S UTC')
            end_time = result[1].strftime('%Y-%m-%d %H:%M:%S UTC')
            duration =  result[2]
          else:
            start_time = "Unknown"
            end_time = "Unknown"
            duration = "Unknown"
          message = f"DAG {dag_run.dag_id} completed. Start time: {start_time}, End time: {end_time}, Duration: {duration} seconds"
        except Exception as e:
          message = f"Error getting end time and duration: {e}"
    else:
      message = "DAG success callback triggered, but unable to get dag_run context"


    slack_notif = SlackWebhookOperator(
        task_id="slack_notification",
        slack_webhook_conn_id="my_slack_connection",
        message=message,
    )
    session.close()
    return slack_notif.execute(context=context)


def my_task():
    print("Task is running")

with DAG(
    dag_id="correct_slack_notification",
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    on_success_callback=slack_success_callback,
    tags=["example"],
) as dag:
    task1 = PythonOperator(task_id="task1", python_callable=my_task)
```
This refined example directly queries the `dag_run` table in the Airflow metastore using an SQLAlchemy query to retrieve the correct start time, end time and the difference between the two to correctly determine the DAG run duration. This eliminates the discrepancies caused by using template variables that don't reflect the actual runtime.

To further improve consistency, I would recommend thoroughly examining the Airflow logs, specifically those related to the DAG run execution and the notification tasks, to understand the timing and context in which variables are resolved. Resources like the official Airflow documentation (especially the sections on templating, callbacks, and metastore access), along with books focusing on the intricacies of workflow management platforms like Airflow, can provide a much deeper understanding of these complexities. Additionally, the source code of Airflow, available on Github, can often reveal the details of how values are computed and made available during a run. Carefully inspecting how the `DagRun` model is populated within the metastore is very beneficial for correctly retrieving accurate timing information. Understanding these underlying details is essential for implementing reliable and consistent Slack notifications in Airflow environments.
