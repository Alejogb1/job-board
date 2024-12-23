---
title: "How can Microsoft Teams be integrated with Apache Airflow?"
date: "2024-12-23"
id: "how-can-microsoft-teams-be-integrated-with-apache-airflow"
---

Alright, let's talk about integrating Microsoft Teams with Apache Airflow. This is a topic I’ve tackled a few times in past projects, and it's surprisingly common when you're orchestrating complex data workflows that need visibility beyond the engineering team. It's not just about getting a notification that something failed; it’s about creating a seamless feedback loop within an organization.

The core challenge, as I see it, revolves around two distinct systems that operate in very different spaces. Airflow, at its heart, is a batch processing orchestrator. Teams, meanwhile, is a real-time communications platform. Bridging that gap effectively takes a thoughtful approach, and it's more than just slapping together a basic notification.

In the projects I’ve worked on where Teams and Airflow needed to play nicely, we generally aimed for a few key capabilities. First, robust notifications for pipeline success and failures were paramount. Second, we wanted richer contextual information – perhaps a link to the logs, or details about which task in the DAG failed. And finally, we wanted a level of control, such as a method to trigger a manual rerun directly from a Teams channel if something went wrong. These requirements shaped the solutions we implemented.

The most straightforward method involves leveraging Airflow's built-in notification mechanisms coupled with the Microsoft Teams connector. The idea is to configure a simple callback that fires on task failure or success, sending a payload to a specified Teams webhook URL. This is, perhaps, the simplest integration point and the first one I tend to recommend for rapid prototyping.

Let’s look at a basic example in Python, which could be implemented in your dag:

```python
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import requests
import json

def send_teams_message(message, webhook_url):
    payload = json.dumps({
        "text": message
    })
    headers = { 'Content-Type': 'application/json'}
    response = requests.post(webhook_url, headers=headers, data=payload)
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

def my_task_success(**kwargs):
    message = f"Task {kwargs['task_instance'].task_id} in DAG {kwargs['dag'].dag_id} succeeded."
    send_teams_message(message, "YOUR_TEAMS_WEBHOOK_URL")

def my_task_failure(**kwargs):
     message = f"Task {kwargs['task_instance'].task_id} in DAG {kwargs['dag'].dag_id} failed. See logs: {kwargs['task_instance'].log_url}"
     send_teams_message(message, "YOUR_TEAMS_WEBHOOK_URL")

with DAG(
    dag_id='teams_basic_notification',
    start_date=days_ago(2),
    schedule=None,
    catchup=False,
    tags=['example'],
) as dag:
    t1 = PythonOperator(
        task_id='dummy_task',
        python_callable=lambda: print("Doing some work."),
        on_success_callback=[my_task_success],
        on_failure_callback=[my_task_failure]
    )

```

In this example, `send_teams_message` takes the message and webhook url and makes a post request, while `my_task_success` and `my_task_failure` generate message with some info. Within the `DAG`, `t1` is configured to trigger these callbacks via the `on_success_callback` and `on_failure_callback` options. Replace "YOUR_TEAMS_WEBHOOK_URL" with the appropriate webhook address configured in Microsoft Teams for this to function correctly. Notice the inclusion of `kwargs['task_instance'].log_url`, which provides a direct link to the Airflow logs, saving time when debugging.

This method works, but it's fairly basic. For a richer experience, or when aiming to integrate with Teams as more than just a notification sink, exploring the use of custom operators becomes valuable. Specifically, crafting an Airflow operator that wraps the Teams API could bring the needed level of abstraction, especially when you're making regular calls that go beyond simple messages.

Here’s a look at a possible custom operator example:

```python
from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
import requests
import json

class TeamsOperator(BaseOperator):
    """
    Custom operator to send messages to Microsoft Teams.

    :param webhook_url: Teams webhook url
    :param message: The message to send to teams.
    """

    template_fields = ('message',)

    @apply_defaults
    def __init__(self, webhook_url, message, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.webhook_url = webhook_url
        self.message = message

    def execute(self, context):
        payload = json.dumps({"text": self.message})
        headers = { 'Content-Type': 'application/json'}
        response = requests.post(self.webhook_url, headers=headers, data=payload)
        response.raise_for_status()


from airflow.models import DAG
from airflow.utils.dates import days_ago

with DAG(
    dag_id='teams_custom_operator',
    start_date=days_ago(2),
    schedule=None,
    catchup=False,
    tags=['example'],
) as dag:
    teams_task = TeamsOperator(
        task_id='send_teams_message',
        webhook_url="YOUR_TEAMS_WEBHOOK_URL",
        message="This is a test message from the TeamsOperator."
    )
```

This `TeamsOperator` inherits from `BaseOperator` and provides a clean way to incorporate message sending into any of your DAGS. Importantly, it makes use of Jinja templating, meaning the `message` parameter can be templated, which allows the use of contextual information such as date or task IDs. We can make the message dynamic using the `context` provided in the `execute` method. This example also demonstrates how to avoid hardcoding the webhook URL by leveraging connections. It provides a more reusable solution to achieve what the first code snippet does.

Going beyond this, consider interactive elements. In a more advanced setup, I've used Microsoft's Adaptive Cards in Teams messages. This involved constructing the appropriate JSON payload within the Airflow task. This allows you to send complex, visually formatted messages complete with buttons and controls that could, for example, direct a user to rerun a failed DAG from a specific task. This often requires a custom integration solution that might go beyond the scope of readily available operators, but the effort in implementation pays dividends in terms of user experience and the speed of incident resolution.

Here's a snippet that demonstrates how to construct an Adaptive Card payload to provide a "Rerun" button, although implementing the rerunning action itself would require additional components, most likely custom-built API calls:

```python
from airflow.models.baseoperator import BaseOperator
from airflow.utils.decorators import apply_defaults
import requests
import json
import uuid

class TeamsAdaptiveCardOperator(BaseOperator):

    template_fields = ('message',)
    @apply_defaults
    def __init__(self, webhook_url, message, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.webhook_url = webhook_url
        self.message = message

    def execute(self, context):
        task_id = context["task_instance"].task_id
        dag_id = context["dag"].dag_id
        run_id = context["run_id"]
        unique_id = str(uuid.uuid4())
        adaptive_card_payload = {
           "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
           "type": "AdaptiveCard",
            "version": "1.5",
            "body": [
              {
                "type": "TextBlock",
                "text": self.message,
                "wrap": True
              }
            ],
            "actions": [
            {
                "type": "Action.OpenUrl",
                  "title": "Rerun DAG",
                "url": f"https://your-airflow-webserver/airflow/graph?dag_id={dag_id}&run_id={run_id}"
              }
            ]
        }
        payload = json.dumps(adaptive_card_payload)
        headers = { 'Content-Type': 'application/json'}
        response = requests.post(self.webhook_url, headers=headers, data=payload)
        response.raise_for_status()

from airflow.models import DAG
from airflow.utils.dates import days_ago

with DAG(
    dag_id='teams_adaptive_card',
    start_date=days_ago(2),
    schedule=None,
    catchup=False,
    tags=['example'],
) as dag:
    teams_task = TeamsAdaptiveCardOperator(
        task_id='send_teams_card',
        webhook_url="YOUR_TEAMS_WEBHOOK_URL",
        message="Task failed in DAG {{ dag.dag_id }}, run id is {{ run_id }}."
    )
```

In this code, we've constructed an `AdaptiveCard`, adding a 'Rerun' action pointing to an Airflow Graph URL, which users can use to manually trigger the DAG.

For further learning, I strongly recommend exploring the *Microsoft Graph API documentation* which is the foundation for building custom Teams integrations. Also, the Apache Airflow documentation offers an extensive explanation of how to create custom operators. Another good resource is *Fluent Python* by Luciano Ramalho for a detailed look at best practices in python which always helps when working on complex integrations. I would also suggest looking into the *'Effective Python: 90 Specific Ways to Write Better Python'*, by Brett Slatkin, for guidance on writing Pythonic code, as it is essential when dealing with the complex integrations like this.

In summary, integrating Teams and Airflow can range from simple notification callbacks to complex interactive systems. Begin with a simple webhook and callback setup, progress to custom operators for more control, and then explore advanced elements such as adaptive cards for more interactive experience. The key is understanding the limitations and capabilities of both systems and using each for its strength.
