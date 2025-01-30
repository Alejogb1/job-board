---
title: "How do I connect an Airflow task to an AWS EMR Notebook?"
date: "2025-01-30"
id: "how-do-i-connect-an-airflow-task-to"
---
The crux of connecting an Airflow task to an AWS EMR Notebook lies in leveraging the EMR Serverless Application API and its inherent orchestration capabilities within the Airflow DAG.  My experience building large-scale data pipelines using this approach highlighted the importance of precise control over the notebook execution lifecycle to ensure reliable and repeatable data processing.  Directly executing notebook code from Airflow is generally avoided due to security and manageability concerns; instead, the preferred method is to initiate a notebook execution through the EMR Serverless API and then monitor its progress.

**1. Clear Explanation:**

Connecting an Airflow task to an EMR Serverless Notebook involves several key steps.  First, you need proper authentication to access your AWS resources.  This usually involves configuring your Airflow environment with AWS credentials, often through environment variables or an IAM role assumed by the Airflow worker.  Next, the Airflow task utilizes the `boto3` library to interact with the EMR Serverless Application API.  The task constructs a request to start a new application execution, specifying the notebook path and any necessary parameters, such as the application configuration (which might include environment variables, input data locations, and output locations). After submission, the task enters a polling loop, using the `boto3` client to retrieve the application execution status.  This loop continues until the application completes successfully, fails, or reaches a timeout. The Airflow task’s status is then updated based on the EMR Serverless application execution status.  Successful completion may trigger subsequent Airflow tasks, while failure may trigger alerts or rollback mechanisms.  Error handling is critical at each stage, encompassing network errors, authentication failures, and application execution errors.  Proper logging throughout the process is essential for debugging and monitoring.


**2. Code Examples with Commentary:**

**Example 1: Basic Notebook Execution**

This example demonstrates a basic Airflow task that starts an EMR Serverless notebook application and polls for its completion.

```python
from airflow import DAG
from airflow.providers.amazon.aws.operators.emr import EmrServerlessStartJobOperator
from airflow.operators.python import PythonOperator
from airflow.decorators import task
from datetime import datetime
import boto3

with DAG(
    dag_id='emr_serverless_notebook_example',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    start_notebook = EmrServerlessStartJobOperator(
        task_id='start_notebook',
        application_id='your_emr_serverless_application_id', # Replace with your application ID
        execution_properties={'notebookParams': {'path': 's3://your-bucket/notebooks/my_notebook.ipynb'}} #Replace with your notebook path
    )

    # Example of checking the status using boto3 (alternative to EmrServerlessStartJobOperator's built-in polling, offering more fine-grained control):

    @task
    def check_notebook_status(application_id):
        client = boto3.client('emr-serverless')
        response = client.get_application(applicationId=application_id)
        return response['application']['status']

    notebook_status = check_notebook_status(start_notebook.output['applicationId'])

    #Further Airflow tasks based on notebook_status
```

**Commentary:** This example leverages the `EmrServerlessStartJobOperator` for simplicity. The `application_id` and notebook path must be replaced with your actual values. The optional `check_notebook_status` task demonstrates using `boto3` directly for more control,  allowing for custom logic based on the returned status.


**Example 2: Handling Execution Failures**

This example expands upon the previous one by including error handling and logging.

```python
import logging
from airflow.exceptions import AirflowException

# ... (previous imports and DAG definition) ...

with dag:
    start_notebook = EmrServerlessStartJobOperator(
        task_id='start_notebook',
        application_id='your_emr_serverless_application_id',
        execution_properties={'notebookParams': {'path': 's3://your-bucket/notebooks/my_notebook.ipynb'}},
        retries=3,
        retry_delay=timedelta(minutes=5)
    )

    @task
    def handle_notebook_status(application_id):
        client = boto3.client('emr-serverless')
        response = client.get_application(applicationId=application_id)
        status = response['application']['status']
        if status != 'COMPLETED':
            logging.error(f"Notebook execution failed with status: {status}")
            raise AirflowException(f"EMR Serverless Notebook execution failed: {status}")

    handle_notebook_status(start_notebook.output['applicationId']) >> further_tasks
```

**Commentary:** This version adds retries and uses `AirflowException` to signal failure. The `handle_notebook_status` task checks for failures and raises an exception, halting the workflow. Logging provides crucial information for debugging.


**Example 3: Parameterized Notebook Execution**

This example demonstrates passing parameters to the EMR Serverless notebook.

```python
from airflow.models.param import Param

with DAG(
    dag_id='emr_serverless_notebook_parameterized',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    start_notebook = EmrServerlessStartJobOperator(
        task_id='start_notebook_param',
        application_id='your_emr_serverless_application_id',
        execution_properties={
            'notebookParams': {
                'path': 's3://your-bucket/notebooks/my_notebook.ipynb',
                'args': ['param1', '{{ dag_run.conf["param2"] }}']
            }
        },
        params={'param2': Param(type='string', description='Second parameter')}
    )
```

**Commentary:**  This example uses `dag_run.conf` to pass parameters from the Airflow UI.  The notebook must be written to accept these parameters as command-line arguments or environment variables.  The use of `Param` in the `params` argument allows for type checking and parameter description within the Airflow UI.



**3. Resource Recommendations:**

*   The official AWS documentation for EMR Serverless.
*   The `boto3` library documentation.
*   Airflow documentation, focusing on the `EmrServerlessStartJobOperator` and custom operator development.
*   A comprehensive guide on AWS IAM roles and permissions.
*   Best practices for logging and error handling within Airflow DAGs.


This approach, leveraging the EMR Serverless Application API, provides a robust and scalable solution for integrating EMR Serverless Notebooks into your Airflow workflows.  Remember to always adhere to security best practices when handling AWS credentials and managing access to your resources.  Proper error handling and logging are paramount for maintaining a stable and monitorable data pipeline.  Through careful planning and execution, you can effectively manage the complete lifecycle of your EMR Serverless notebook executions within your Airflow environment.
