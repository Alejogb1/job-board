---
title: "How can airflow exceptions be wrapped to trigger on-failure notifications?"
date: "2025-01-30"
id: "how-can-airflow-exceptions-be-wrapped-to-trigger"
---
Airflow's exception handling, while robust, often requires customization to achieve granular control over notification triggers.  My experience debugging complex DAGs across various production environments has highlighted the critical need for context-aware error handling, moving beyond simple email alerts to targeted notifications based on the specific exception type and context.  This is not simply about catching errors; it's about generating actionable insights from failures.

The core challenge lies in decoupling the exception handling logic from the task itself.  Directly embedding notification logic within each task leads to duplicated code and maintenance headaches.  A more elegant and scalable solution involves a custom operator or a wrapper function that intercepts exceptions, extracts relevant information, and triggers notifications based on predefined criteria.

**1. Clear Explanation:**

The proposed approach leverages Airflow's `on_failure_callback` parameter available to most operators.  Instead of directly setting a notification function here, we construct a wrapper function that receives the context of the failed task. This function then analyzes the exception, filters based on exception types or custom logic, and sends appropriate notifications.  This allows for differentiating between expected exceptions (e.g., a retryable transient network error) and critical failures requiring immediate intervention. The critical element is to pass relevant context — the task instance, the exception type, and potentially custom log data — to the notification function. This enriched context ensures the notification is highly informative.


**2. Code Examples with Commentary:**

**Example 1: Basic Exception Wrapping with Email Notification**

```python
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
from airflow.decorators import task
import smtplib
from email.mime.text import MIMEText

def send_email_notification(context):
    """Sends email notification on task failure."""
    task_instance = context['ti']
    try:
        exception = context['exception']
        msg = MIMEText(f"Task {task_instance.task_id} failed: {exception}")
        msg['Subject'] = f"Airflow Task Failure: {task_instance.task_id}"
        msg['From'] = 'your_email@example.com'
        msg['To'] = 'recipient_email@example.com'

        with smtplib.SMTP('your_smtp_server', 587) as server:
            server.starttls()
            server.login('your_email@example.com', 'your_password')
            server.send_message(msg)
    except Exception as e:
        print(f"Error sending email notification: {e}")

@task(on_failure_callback=send_email_notification)
def my_task():
    # Your task logic here
    raise AirflowException("Task failed intentionally.")

# DAG definition using the decorated task
# ...
```

This example demonstrates a straightforward implementation.  The `send_email_notification` function extracts the exception details and task ID from the context dictionary provided by Airflow.  It then constructs and sends a simple email.  Error handling is incorporated within the notification function itself for robustness.


**Example 2:  Conditional Notification Based on Exception Type**

```python
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException, AirflowSkipException
from airflow.decorators import task

def conditional_notification(context):
    """Sends notification only for specific exception types."""
    task_instance = context['ti']
    exception = context['exception']
    if isinstance(exception, AirflowException):
        # Log critical failure.  Consider more sophisticated notification mechanisms here
        print(f"Critical failure in task {task_instance.task_id}: {exception}")
    elif isinstance(exception, AirflowSkipException):
        print(f"Task {task_instance.task_id} skipped as expected.")
    else:
        print(f"Unexpected exception in task {task_instance.task_id}: {exception}")

@task(on_failure_callback=conditional_notification)
def my_conditional_task():
    #Task logic here, potentially raising different exceptions
    raise AirflowSkipException("Task skipped intentionally.")


# DAG definition using the decorated task
# ...
```

This example introduces conditional logic. Notifications are tailored based on the type of exception.  Critical failures receive detailed logging (which could easily be extended to more robust alert systems like Slack or PagerDuty), while expected skips produce a less urgent message.


**Example 3:  Custom Exception Class and Advanced Notification**

```python
from airflow.operators.python import PythonOperator
from airflow.exceptions import AirflowException
from airflow.decorators import task
import logging

class CustomAirflowException(AirflowException):
    """Custom exception for specific scenarios."""
    def __init__(self, message, extra_context=None):
        super().__init__(message)
        self.extra_context = extra_context


def advanced_notification(context):
    """Handles custom exceptions and provides rich context."""
    task_instance = context['ti']
    exception = context['exception']
    log = logging.getLogger(__name__)

    if isinstance(exception, CustomAirflowException):
        log.error(f"Custom exception in task {task_instance.task_id}: {exception.message} - Extra context: {exception.extra_context}")
        # Send a more detailed notification (e.g., Slack message with context)
    else:
        log.error(f"Generic failure in task {task_instance.task_id}: {exception}")


@task(on_failure_callback=advanced_notification)
def my_custom_task():
    #Task logic here
    raise CustomAirflowException("Database connection failed", extra_context={"database":"mydb"})

# DAG definition using the decorated task
# ...

```

This example showcases creating a custom exception type to encapsulate specific error situations within a given task, enabling the advanced_notification function to handle these scenarios with richer context information.  The extra_context is especially useful for conveying detailed diagnostic data not readily available from the base exception.  Integration with more sophisticated alerting systems would involve adding another layer to this function.

**3. Resource Recommendations:**

* **Airflow documentation:**  Thoroughly review the official Airflow documentation on operators, exception handling, and the `on_failure_callback` mechanism.
* **Python logging library:**  Master the Python `logging` module to implement comprehensive and structured logging for better debugging and monitoring.  Consider different log handlers for various needs.
* **Alerting systems integration:**  Familiarize yourself with the APIs and SDKs of popular alerting systems like PagerDuty, Slack, or Opsgenie to seamlessly integrate with Airflow for advanced notification capabilities.  Consider the advantages of each in terms of features and cost.


By applying these techniques and incorporating robust logging practices, you can dramatically improve the effectiveness of your Airflow exception handling, transforming error monitoring from a reactive process to a proactive, insight-driven approach.  Remember that the choice of notification mechanism should align with your specific operational needs and the criticality of the tasks involved.
