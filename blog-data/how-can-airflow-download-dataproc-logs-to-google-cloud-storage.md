---
title: "How can Airflow download Dataproc logs to Google Cloud Storage?"
date: "2024-12-23"
id: "how-can-airflow-download-dataproc-logs-to-google-cloud-storage"
---

,  I've certainly seen my share of headaches around log management in distributed environments, and Dataproc is definitely one area where it's crucial to get this process nailed down correctly. In my past experience, especially with larger-scale pipelines, the ability to efficiently retrieve and store Dataproc logs for auditing and debugging has been absolutely vital. If the logging pipeline isn't working smoothly, troubleshooting becomes a nightmare. So, let's break down how to pull those logs from Dataproc and archive them into Google Cloud Storage (GCS) using Airflow.

The core idea centers around using the Airflow's operators designed to interact with Google Cloud Platform, in particular the `dataproc_operator.DataprocWorkflowTemplateInstantiateOperator` or the `dataproc_operator.DataprocSubmitJobOperator` and subsequently employing Google Cloud Storage operators to fetch the logs. Specifically, it's about programmatically accessing the Cloud Logging API, identifying the logs associated with your specific Dataproc clusters and jobs, then downloading and staging them into GCS. This isn’t a direct, single-step operation, it’s more of a process which can be wrapped into an airflow pipeline.

First, let's consider the scenario where you're using `DataprocWorkflowTemplateInstantiateOperator` to launch a Dataproc workflow. When a workflow template runs, logs are automatically written to Cloud Logging. What we need is a mechanism to reliably copy those logs to GCS after the workflow completes. This involves multiple steps within your Airflow DAG:

1.  **Workflow instantiation:** You'll start with the `DataprocWorkflowTemplateInstantiateOperator` to trigger the workflow. This operator returns the cluster ID and operation ID when complete.
2.  **Log identification:** Next, you'll need to identify the specific log entries associated with your cluster and its jobs. This can be done via a custom PythonOperator or use the `GoogleCloudLoggingHook` within a custom Operator class. The logs are associated with your Dataproc cluster id.
3.  **GCS staging:** Finally, we need to use the `GoogleCloudStorageHook` within a custom operator to pull the log data using the Cloud Logging API and write it to a GCS bucket.

Here's a basic code snippet illustrating this process, focusing on log identification and GCS staging (assuming you have the workflow instantiation setup already). Please note this example code provides an overview and assumes some familiarity with Airflow concepts, custom operators and hooks.:

```python
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.google.cloud.hooks.logging import GoogleCloudLoggingHook
from airflow.providers.google.cloud.hooks.gcs import GoogleCloudStorageHook
from datetime import datetime
import json

# Custom Airflow Operator for Log Extraction and Upload
class DataprocLogUploaderOperator(PythonOperator):
    def __init__(self, cluster_id, gcs_bucket, gcs_log_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_id = cluster_id
        self.gcs_bucket = gcs_bucket
        self.gcs_log_path = gcs_log_path

    def execute(self, context):
        log_hook = GoogleCloudLoggingHook()
        gcs_hook = GoogleCloudStorageHook()
        gcs_path = f"{self.gcs_log_path}/cluster_{self.cluster_id}.json"

        query = f'resource.type="cloud_dataproc_cluster" resource.labels.cluster_name="{self.cluster_id}"'

        log_entries = log_hook.list_entries(
             filter_=query,
            order_by="timestamp desc"
        )

        # Serializing to JSON for easy storage
        log_json = json.dumps([entry.to_api_repr() for entry in log_entries])

        gcs_hook.upload_string(
            bucket_name=self.gcs_bucket,
            filename=gcs_path,
            string_data=log_json,
            mime_type="application/json"
         )
        self.log.info(f"Logs for cluster {self.cluster_id} uploaded to {gcs_path}")


with DAG(
    dag_id='dataproc_log_extraction',
    schedule_interval=None,
    start_date=days_ago(1),
    tags=['dataproc', 'logging'],
    catchup=False,
) as dag:

    # Assume you have a Dataproc workflow initiation task called 'start_dataproc'
    # and it sets a variable 'cluster_id'
    # cluster_id = context['ti'].xcom_pull(task_ids='start_dataproc', key='cluster_id')

    log_upload_task = DataprocLogUploaderOperator(
        task_id="upload_logs_to_gcs",
        cluster_id="{{ ti.xcom_pull(task_ids='start_dataproc', key='cluster_id') }}",
        gcs_bucket="your-gcs-bucket-name", # Replace with your bucket name
        gcs_log_path="dataproc_logs", # Replace with your desired gcs path
        dag=dag,
    )

    # Add dependency
    # start_dataproc >> log_upload_task
```
In this Python Operator example, I've used `GoogleCloudLoggingHook` to retrieve the logs associated with the Dataproc cluster id, and `GoogleCloudStorageHook` to upload the serialized JSON representation of the logs to GCS. The `filter_` parameter of `list_entries()` is crucial because it allows you to pull logs just for your specific cluster. This helps keep our log retrieval targeted and reduces unnecessary fetching of logs. The uploaded log is formatted as a JSON string to ease readability and further analysis.

Now, let's consider a slightly different approach where you're using `DataprocSubmitJobOperator`. This operator executes jobs within a running Dataproc cluster. Log retrieval here is similar, but you'd use the job ID instead of the cluster ID to identify specific log entries.

```python
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.google.cloud.hooks.logging import GoogleCloudLoggingHook
from airflow.providers.google.cloud.hooks.gcs import GoogleCloudStorageHook
from datetime import datetime
import json

# Custom Airflow Operator for Job Log Extraction and Upload
class DataprocJobLogUploaderOperator(PythonOperator):
    def __init__(self, job_id, gcs_bucket, gcs_log_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.job_id = job_id
        self.gcs_bucket = gcs_bucket
        self.gcs_log_path = gcs_log_path

    def execute(self, context):
        log_hook = GoogleCloudLoggingHook()
        gcs_hook = GoogleCloudStorageHook()
        gcs_path = f"{self.gcs_log_path}/job_{self.job_id}.json"

        query = f'resource.type="cloud_dataproc_job" resource.labels.job_id="{self.job_id}"'

        log_entries = log_hook.list_entries(
             filter_=query,
             order_by="timestamp desc"
        )

        # Serializing to JSON for easy storage
        log_json = json.dumps([entry.to_api_repr() for entry in log_entries])

        gcs_hook.upload_string(
            bucket_name=self.gcs_bucket,
            filename=gcs_path,
            string_data=log_json,
            mime_type="application/json"
         )
        self.log.info(f"Logs for job {self.job_id} uploaded to {gcs_path}")



with DAG(
    dag_id='dataproc_job_log_extraction',
    schedule_interval=None,
    start_date=days_ago(1),
    tags=['dataproc', 'logging'],
    catchup=False,
) as dag:

    # Assume you have a Dataproc job submission task called 'submit_dataproc_job'
    # and it sets a variable 'job_id'
    # job_id = context['ti'].xcom_pull(task_ids='submit_dataproc_job', key='job_id')

    log_upload_task = DataprocJobLogUploaderOperator(
        task_id="upload_job_logs_to_gcs",
        job_id="{{ ti.xcom_pull(task_ids='submit_dataproc_job', key='job_id') }}",
        gcs_bucket="your-gcs-bucket-name", # Replace with your bucket name
        gcs_log_path="dataproc_job_logs", # Replace with your desired gcs path
        dag=dag,
    )
    # Add dependency
    # submit_dataproc_job >> log_upload_task

```
Here, the key difference is the log query, targeting `cloud_dataproc_job` and the specific `job_id`. The rest of the logic, using the same hooks for log retrieval and GCS upload, remains largely the same. This highlights the versatility of this pattern.

For a more complex use-case with multiple jobs submitted to multiple clusters, you might consider a more generalized custom operator that utilizes `XCom` to manage identifiers, but the core logic for log retrieval and GCS upload would remain the same.  Remember, all this uses the Google Cloud client libraries behind the scenes, so proper authentication is critical – your Airflow environment should have the necessary permissions to interact with Dataproc and GCS.

Lastly, let's tackle log aggregation in a more realistic scenario, where multiple worker nodes generate log files on Dataproc clusters.  In this case, log aggregation on each node will send logs to Cloud Logging automatically and you don't need to move the log files out of the node file system.
However, this scenario also assumes your cluster is set up to forward its logs to Cloud Logging.

```python
from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
from airflow.providers.google.cloud.hooks.logging import GoogleCloudLoggingHook
from airflow.providers.google.cloud.hooks.gcs import GoogleCloudStorageHook
from datetime import datetime
import json

# Custom Airflow Operator for Log Extraction and Upload
class DataprocClusterLogUploaderOperator(PythonOperator):
    def __init__(self, cluster_name, gcs_bucket, gcs_log_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cluster_name = cluster_name
        self.gcs_bucket = gcs_bucket
        self.gcs_log_path = gcs_log_path

    def execute(self, context):
        log_hook = GoogleCloudLoggingHook()
        gcs_hook = GoogleCloudStorageHook()
        gcs_path = f"{self.gcs_log_path}/cluster_logs_{self.cluster_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"

        query = f'resource.type="cloud_dataproc_cluster" resource.labels.cluster_name="{self.cluster_name}"'
        log_entries = log_hook.list_entries(
             filter_=query,
            order_by="timestamp desc"
        )

        log_json = json.dumps([entry.to_api_repr() for entry in log_entries])

        gcs_hook.upload_string(
            bucket_name=self.gcs_bucket,
            filename=gcs_path,
            string_data=log_json,
            mime_type="application/json"
        )
        self.log.info(f"Aggregated logs for cluster {self.cluster_name} uploaded to {gcs_path}")


with DAG(
    dag_id='dataproc_cluster_log_aggregation',
    schedule_interval=None,
    start_date=days_ago(1),
    tags=['dataproc', 'logging'],
    catchup=False,
) as dag:

    # Assume you have a Dataproc cluster creation task called 'create_dataproc_cluster'
    # and it sets a variable 'cluster_name'
    # cluster_name = context['ti'].xcom_pull(task_ids='create_dataproc_cluster', key='cluster_name')

    log_aggregator_task = DataprocClusterLogUploaderOperator(
        task_id="aggregate_cluster_logs_to_gcs",
        cluster_name="{{ ti.xcom_pull(task_ids='create_dataproc_cluster', key='cluster_name') }}",
        gcs_bucket="your-gcs-bucket-name",  # Replace with your bucket name
        gcs_log_path="dataproc_cluster_logs",  # Replace with your desired gcs path
        dag=dag,
    )

    # Add dependency
    # create_dataproc_cluster >> log_aggregator_task
```

In this example, I'm aggregating logs from a full Dataproc cluster using the `cluster_name` to query the Cloud Logging api and saving all the logs for this cluster.  The timestamped file name ensures you have a history of log aggregations. The choice of JSON as the storage format ensures compatibility with other tools for analysis.

For further study, I recommend focusing on the official Google Cloud documentation for both Cloud Logging and Dataproc, along with Airflow’s documentation specifically on the Google provider.  Additionally, "Designing Data-Intensive Applications" by Martin Kleppmann provides crucial background on managing distributed systems and data pipelines, and is a must-read for anyone working in this space. "Cloud Native Patterns" by Cornelia Davis is also very valuable to understanding and crafting repeatable solutions for cloud environments.

In essence, the mechanism for downloading Dataproc logs involves programmatic interaction with Cloud Logging via Airflow. This allows you to automate the crucial task of log collection, ensuring that you have the audit trails you need without the manual headache. The key lies in understanding how to use the Airflow hooks and operators to make the connection and automate the entire process.
