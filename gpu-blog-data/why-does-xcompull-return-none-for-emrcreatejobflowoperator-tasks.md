---
title: "Why does xcom_pull return None for EmrCreateJobFlowOperator tasks in Airflow?"
date: "2025-01-30"
id: "why-does-xcompull-return-none-for-emrcreatejobflowoperator-tasks"
---
In my experience debugging complex Airflow DAGs involving EMR, a frequent source of frustration is the seemingly unpredictable behavior of `xcom_pull` when attempting to retrieve values from `EmrCreateJobFlowOperator` tasks. The root cause typically lies in the specific structure of data returned by this operator and how XCom (cross-communication) handles it. The operator, upon successful execution, does not directly return the job flow ID as a readily available string that can be pulled. Instead, it returns a dictionary, where the JobFlowId is nested within a response object. Failing to account for this nested structure results in `xcom_pull` returning `None`.

Let's delve into why this happens and how to resolve it. `xcom_pull` works by accessing values stored in the XCom database associated with a specific task instance, identified by its task ID and DAG run ID. When an operator completes successfully, any returned value is serialized and stored as an XCom message. The issue arises because the `EmrCreateJobFlowOperator` doesn’t return the raw JobFlowId string directly but rather a detailed API response containing numerous fields. This response is serialized and stored in XCom, and unless the pulling task specifically accesses the nested job flow id value, it will receive a `None` value because it doesn’t receive the expected data structure.

Understanding this nested structure is crucial. When an `EmrCreateJobFlowOperator` executes successfully, the response is structured like this (truncated for clarity):

```json
{
  "ResponseMetadata": {
    "RequestId": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
    "HTTPStatusCode": 200,
    "HTTPHeaders": {
      "x-amzn-requestid": "xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx",
      "content-type": "application/x-amz-json-1.1",
      "content-length": "xxx",
      "date": "Wed, 02 Aug 2024 12:00:00 GMT"
    },
    "RetryAttempts": 0
  },
  "JobFlowId": "j-XXXXXXXXXXXXXXXXX"
}
```

Notice that the actual `JobFlowId` resides at the top level of this response object, rather than the response object being nested within another. We want to extract this `JobFlowId` and pass it to other tasks within the DAG. If we blindly try to `xcom_pull` from the creating task, we will not receive the JobFlowId value, as the key doesn't correspond to the entire response object.

To illustrate this, consider three different code examples. The first example demonstrates the incorrect approach which often results in `None`, the second shows a correct extraction method within a PythonOperator, and the third provides an alternative way to achieve the same result using a custom operator.

**Example 1: Incorrect `xcom_pull` Usage (Resulting in None)**

This example shows how one might intuitively attempt to retrieve the JobFlowId, which leads to a `None` value when printing it in a downstream task.

```python
from airflow import DAG
from airflow.providers.amazon.aws.operators.emr_create_job_flow import EmrCreateJobFlowOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.utils.dates import days_ago


JOB_FLOW_OVERRIDES = {
    "Name": "my-emr-jobflow",
    "ReleaseLabel": "emr-6.15.0",
    "Applications": [{"Name": "Hadoop"}, {"Name": "Spark"}],
    "Instances": {
        "InstanceGroups": [
            {
                "Name": "Master nodes",
                "Market": "ON_DEMAND",
                "InstanceRole": "MASTER",
                "InstanceType": "m5.xlarge",
                "InstanceCount": 1,
            },
             {
                "Name": "Core nodes",
                "Market": "ON_DEMAND",
                "InstanceRole": "CORE",
                "InstanceType": "m5.xlarge",
                "InstanceCount": 1,
            }
        ],
        "KeepJobFlowAliveWhenNoSteps": True,
        "TerminationProtected": False
    },
    "VisibleToAllUsers": True
}


def print_job_id_incorrect(**kwargs):
    job_flow_id = kwargs['ti'].xcom_pull(task_ids='create_emr_cluster')
    print(f"Job Flow ID (Incorrect): {job_flow_id}") # Will likely print None or the whole response metadata

with DAG(
    dag_id='emr_incorrect_pull',
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['emr'],
) as dag:

    create_emr_cluster = EmrCreateJobFlowOperator(
        task_id='create_emr_cluster',
        job_flow_overrides=JOB_FLOW_OVERRIDES
    )

    print_job_id_task = PythonOperator(
        task_id='print_job_id',
        python_callable=print_job_id_incorrect,
        provide_context=True
    )
    
    create_emr_cluster >> print_job_id_task
```

Here, the `print_job_id_incorrect` function directly tries to pull the output of `create_emr_cluster`. As discussed before, this will return the entire response JSON rather than just the job ID. This demonstrates how simply attempting to retrieve the value without accounting for structure will result in `None`.

**Example 2: Correct Extraction Using a Python Operator**

This example uses a `PythonOperator` to extract the `JobFlowId` from the nested dictionary returned by `EmrCreateJobFlowOperator`.

```python
from airflow import DAG
from airflow.providers.amazon.aws.operators.emr_create_job_flow import EmrCreateJobFlowOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.utils.dates import days_ago


JOB_FLOW_OVERRIDES = {
    "Name": "my-emr-jobflow",
    "ReleaseLabel": "emr-6.15.0",
    "Applications": [{"Name": "Hadoop"}, {"Name": "Spark"}],
    "Instances": {
        "InstanceGroups": [
            {
                "Name": "Master nodes",
                "Market": "ON_DEMAND",
                "InstanceRole": "MASTER",
                "InstanceType": "m5.xlarge",
                "InstanceCount": 1,
            },
             {
                "Name": "Core nodes",
                "Market": "ON_DEMAND",
                "InstanceRole": "CORE",
                "InstanceType": "m5.xlarge",
                "InstanceCount": 1,
            }
        ],
        "KeepJobFlowAliveWhenNoSteps": True,
        "TerminationProtected": False
    },
    "VisibleToAllUsers": True
}


def print_job_id_correct(**kwargs):
    response = kwargs['ti'].xcom_pull(task_ids='create_emr_cluster')
    job_flow_id = response.get("JobFlowId")
    print(f"Job Flow ID (Correct): {job_flow_id}")

with DAG(
    dag_id='emr_correct_pull_python',
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['emr'],
) as dag:

    create_emr_cluster = EmrCreateJobFlowOperator(
        task_id='create_emr_cluster',
        job_flow_overrides=JOB_FLOW_OVERRIDES
    )

    print_job_id_task = PythonOperator(
        task_id='print_job_id',
        python_callable=print_job_id_correct,
        provide_context=True
    )

    create_emr_cluster >> print_job_id_task
```

In this corrected example, the `print_job_id_correct` function correctly pulls the entire response, extracts the `JobFlowId` using a dictionary lookup (`response.get("JobFlowId")`), and then prints the intended value. This demonstrates the required approach to access nested values returned by the operator.

**Example 3: Custom Operator for Direct Job Flow ID Extraction**

Here, we create a custom operator that inherits from `EmrCreateJobFlowOperator`, overrides the `execute` method, and specifically returns the `JobFlowId`.

```python
from airflow import DAG
from airflow.providers.amazon.aws.operators.emr_create_job_flow import EmrCreateJobFlowOperator
from airflow.operators.python import PythonOperator
from datetime import datetime
from airflow.utils.dates import days_ago


JOB_FLOW_OVERRIDES = {
    "Name": "my-emr-jobflow",
    "ReleaseLabel": "emr-6.15.0",
    "Applications": [{"Name": "Hadoop"}, {"Name": "Spark"}],
    "Instances": {
        "InstanceGroups": [
            {
                "Name": "Master nodes",
                "Market": "ON_DEMAND",
                "InstanceRole": "MASTER",
                "InstanceType": "m5.xlarge",
                "InstanceCount": 1,
            },
             {
                "Name": "Core nodes",
                "Market": "ON_DEMAND",
                "InstanceRole": "CORE",
                "InstanceType": "m5.xlarge",
                "InstanceCount": 1,
            }
        ],
        "KeepJobFlowAliveWhenNoSteps": True,
        "TerminationProtected": False
    },
    "VisibleToAllUsers": True
}


class EmrCreateJobFlowOperatorWithId(EmrCreateJobFlowOperator):
    def execute(self, context):
        response = super().execute(context)
        return response["JobFlowId"]


def print_job_id_custom(**kwargs):
    job_flow_id = kwargs['ti'].xcom_pull(task_ids='create_emr_cluster_custom')
    print(f"Job Flow ID (Custom): {job_flow_id}")

with DAG(
    dag_id='emr_custom_operator',
    schedule=None,
    start_date=days_ago(1),
    catchup=False,
    tags=['emr'],
) as dag:

    create_emr_cluster_custom = EmrCreateJobFlowOperatorWithId(
        task_id='create_emr_cluster_custom',
        job_flow_overrides=JOB_FLOW_OVERRIDES
    )

    print_job_id_task = PythonOperator(
        task_id='print_job_id',
        python_callable=print_job_id_custom,
        provide_context=True
    )

    create_emr_cluster_custom >> print_job_id_task
```

This approach, using the custom operator, avoids having to extract the value from the response object by modifying the `execute` method to return the `JobFlowId` directly. The downstream task can now directly `xcom_pull` and receive the desired job ID without further processing. While this is more code, it makes downstream tasks cleaner by having the ID readily available.

In summary, `xcom_pull` returning `None` with `EmrCreateJobFlowOperator` stems from the operator returning a nested dictionary rather than the job flow ID directly. The first example highlights the issue, while the second demonstrates how a `PythonOperator` can correctly retrieve the value and the third illustrates how to customize the operator's return value to suit the need.

For further understanding of Airflow's XCom mechanism and Amazon EMR operator intricacies, consult the official Airflow documentation, specifically the section on XComs, and the detailed documentation of the Amazon EMR provider. Additionally, examining the source code of the `EmrCreateJobFlowOperator` class can also provide valuable insight into its internal workings and return data structure. Consulting books dedicated to data engineering on AWS may also be useful.
