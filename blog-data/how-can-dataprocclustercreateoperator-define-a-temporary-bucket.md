---
title: "How can DataprocClusterCreateOperator define a temporary bucket?"
date: "2024-12-23"
id: "how-can-dataprocclustercreateoperator-define-a-temporary-bucket"
---

Alright, let's talk about temporary buckets when you're spinning up Dataproc clusters with the `DataprocClusterCreateOperator`. It’s a common enough requirement, especially in ephemeral data processing workflows, and I’ve definitely bumped into it a fair few times over the years. The challenge, generally, isn't that it’s impossible; it’s about ensuring that the temporary bucket is both automatically created *and* cleaned up properly, without leaving orphaned resources hanging around to rack up costs. The `DataprocClusterCreateOperator` itself doesn’t have a dedicated parameter called “temporary_bucket.” Instead, you need to manage this via configuration specifics within the cluster definition it accepts.

First, understand that Dataproc clusters typically use a staging bucket. This is where things like job dependencies, scripts, and configuration files are stored during cluster operation. It's effectively the workspace the cluster uses. While the operator doesn’t explicitly manage a "temporary bucket," the standard staging bucket *can* function as one if handled appropriately. Let’s get into how we approach making that work effectively.

My typical approach involves using the cluster configuration to set up the staging bucket's lifecycle. This way, when the cluster is no longer needed, the bucket (and everything in it) gets tidied away. This revolves around setting a deletion policy or adding specific lifecycle rules to the staging bucket. The key here is to avoid using a pre-existing, essential bucket as the staging location. You want a dedicated, temporary space.

Here's how I’d do it, typically starting with the python operator definition:

```python
from airflow.providers.google.cloud.operators.dataproc import DataprocCreateClusterOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow import DAG
from airflow.models import TaskInstance


PROJECT_ID = "your-gcp-project-id"
REGION = "us-central1"
CLUSTER_NAME = "my-temp-cluster"
BUCKET_NAME = f"temp-staging-bucket-{CLUSTER_NAME.lower()}"
GCS_BUCKET = f"gs://{BUCKET_NAME}"

CLUSTER_CONFIG = {
    "project_id": PROJECT_ID,
    "cluster_name": CLUSTER_NAME,
    "config": {
        "gce_cluster_config": {
            "zone_uri": f"https://www.googleapis.com/compute/v1/projects/{PROJECT_ID}/zones/{REGION}-a",
        },
        "master_config": {
            "num_instances": 1,
            "machine_type_uri": "n1-standard-2",
        },
        "worker_config": {
            "num_instances": 2,
            "machine_type_uri": "n1-standard-2",
        },
        "software_config": {
            "image_version": "1.5-debian10",
        },
        "staging_bucket": BUCKET_NAME,
    }
}


with DAG(
    dag_id="dataproc_temp_bucket",
    start_date=days_ago(1),
    schedule_interval=None,
    tags=["dataproc", "temporary", "bucket"],
    catchup=False,
    default_args={"project_id": PROJECT_ID},
) as dag:
    create_cluster = DataprocCreateClusterOperator(
        task_id="create_dataproc_cluster",
        cluster_name=CLUSTER_NAME,
        cluster_config=CLUSTER_CONFIG,
        region=REGION,
    )

    delete_cluster = DataprocDeleteClusterOperator(
         task_id="delete_dataproc_cluster",
        cluster_name=CLUSTER_NAME,
        region=REGION,
    )
    create_cluster >> delete_cluster
```

This code illustrates a basic `DataprocClusterCreateOperator` usage. The critical part for our discussion is within the `CLUSTER_CONFIG` dictionary, particularly this line: `"staging_bucket": BUCKET_NAME`. Here, we're explicitly defining where Dataproc will stage its files. We're constructing the bucket name in a way that should make it unique and obviously associated with this cluster. However, this alone won’t trigger an automatic cleanup; it's just setting the location of the bucket.

To get that cleanup behaviour, we need to add some form of lifecycle management to this staging bucket *outside* of the cluster creation definition. Here’s the usual method I follow:

First, I use the `google-cloud-storage` library to programmatically set lifecycle rules on the created bucket. I’d often do this as part of a prior task in the same airflow dag, creating the bucket with the desired name if it doesn't exist. I'll also add an explicit deletion task later. I generally wrap this logic in a custom operator, but the code can be adapted as needed.

```python
from google.cloud import storage
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowException

class CreateStagingBucketWithLifecycleOperator(BaseOperator):

    @apply_defaults
    def __init__(self, bucket_name, project_id, days_to_delete=7, **kwargs):
      super().__init__(**kwargs)
      self.bucket_name = bucket_name
      self.project_id = project_id
      self.days_to_delete = days_to_delete
      self.storage_client = storage.Client(project=self.project_id)


    def execute(self, context):
      bucket = self.storage_client.bucket(self.bucket_name)
      if not bucket.exists():
        bucket = self.storage_client.create_bucket(self.bucket_name, location="US")
        self.log.info(f"Bucket {self.bucket_name} created")
      else:
        self.log.info(f"Bucket {self.bucket_name} already exists.")


      lifecycle_rule = {
          "action": {"type": "Delete"},
          "condition": {"age": self.days_to_delete},
      }
      bucket.patch({"lifecycle": {"rule": [lifecycle_rule]}})
      self.log.info(f"Lifecycle policy set to delete bucket {self.bucket_name} after {self.days_to_delete} days.")


class DeleteStagingBucketOperator(BaseOperator):

    @apply_defaults
    def __init__(self, bucket_name, project_id, **kwargs):
      super().__init__(**kwargs)
      self.bucket_name = bucket_name
      self.project_id = project_id
      self.storage_client = storage.Client(project=self.project_id)


    def execute(self, context):
      bucket = self.storage_client.bucket(self.bucket_name)
      if bucket.exists():
          bucket.delete(force=True)
          self.log.info(f"Bucket {self.bucket_name} deleted.")
      else:
        self.log.info(f"Bucket {self.bucket_name} did not exist")

```

In this snippet, the `CreateStagingBucketWithLifecycleOperator` checks for the bucket’s existence and creates it if needed. Critically, it adds a lifecycle rule so that any objects within the bucket, and subsequently, the bucket itself, will be deleted after a specified number of days. Using a deletion task instead or lifecycle rules helps more immediately. The `DeleteStagingBucketOperator` will remove the bucket.

So, to integrate this all into an Airflow DAG, the workflow would look like this:

```python
from airflow.providers.google.cloud.operators.dataproc import DataprocCreateClusterOperator, DataprocDeleteClusterOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
from airflow import DAG
from airflow.models import TaskInstance
from airflow.operators.python import PythonOperator
from google.cloud import storage
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from airflow.exceptions import AirflowException


PROJECT_ID = "your-gcp-project-id"
REGION = "us-central1"
CLUSTER_NAME = "my-temp-cluster"
BUCKET_NAME = f"temp-staging-bucket-{CLUSTER_NAME.lower()}"
GCS_BUCKET = f"gs://{BUCKET_NAME}"

CLUSTER_CONFIG = {
    "project_id": PROJECT_ID,
    "cluster_name": CLUSTER_NAME,
    "config": {
        "gce_cluster_config": {
            "zone_uri": f"https://www.googleapis.com/compute/v1/projects/{PROJECT_ID}/zones/{REGION}-a",
        },
        "master_config": {
            "num_instances": 1,
            "machine_type_uri": "n1-standard-2",
        },
        "worker_config": {
            "num_instances": 2,
            "machine_type_uri": "n1-standard-2",
        },
        "software_config": {
            "image_version": "1.5-debian10",
        },
        "staging_bucket": BUCKET_NAME,
    }
}


class CreateStagingBucketWithLifecycleOperator(BaseOperator):

    @apply_defaults
    def __init__(self, bucket_name, project_id, days_to_delete=7, **kwargs):
      super().__init__(**kwargs)
      self.bucket_name = bucket_name
      self.project_id = project_id
      self.days_to_delete = days_to_delete
      self.storage_client = storage.Client(project=self.project_id)


    def execute(self, context):
      bucket = self.storage_client.bucket(self.bucket_name)
      if not bucket.exists():
        bucket = self.storage_client.create_bucket(self.bucket_name, location="US")
        self.log.info(f"Bucket {self.bucket_name} created")
      else:
        self.log.info(f"Bucket {self.bucket_name} already exists.")


      lifecycle_rule = {
          "action": {"type": "Delete"},
          "condition": {"age": self.days_to_delete},
      }
      bucket.patch({"lifecycle": {"rule": [lifecycle_rule]}})
      self.log.info(f"Lifecycle policy set to delete bucket {self.bucket_name} after {self.days_to_delete} days.")


class DeleteStagingBucketOperator(BaseOperator):

    @apply_defaults
    def __init__(self, bucket_name, project_id, **kwargs):
      super().__init__(**kwargs)
      self.bucket_name = bucket_name
      self.project_id = project_id
      self.storage_client = storage.Client(project=self.project_id)


    def execute(self, context):
      bucket = self.storage_client.bucket(self.bucket_name)
      if bucket.exists():
          bucket.delete(force=True)
          self.log.info(f"Bucket {self.bucket_name} deleted.")
      else:
        self.log.info(f"Bucket {self.bucket_name} did not exist")


with DAG(
    dag_id="dataproc_temp_bucket_lifecycle",
    start_date=days_ago(1),
    schedule_interval=None,
    tags=["dataproc", "temporary", "bucket", "lifecycle"],
    catchup=False,
    default_args={"project_id": PROJECT_ID},
) as dag:
    create_bucket = CreateStagingBucketWithLifecycleOperator(
        task_id="create_staging_bucket",
        bucket_name=BUCKET_NAME,
        project_id=PROJECT_ID,
        days_to_delete=1
    )

    create_cluster = DataprocCreateClusterOperator(
        task_id="create_dataproc_cluster",
        cluster_name=CLUSTER_NAME,
        cluster_config=CLUSTER_CONFIG,
        region=REGION,
    )

    delete_cluster = DataprocDeleteClusterOperator(
        task_id="delete_dataproc_cluster",
        cluster_name=CLUSTER_NAME,
        region=REGION,
    )

    delete_bucket = DeleteStagingBucketOperator(
      task_id="delete_staging_bucket",
      bucket_name=BUCKET_NAME,
      project_id=PROJECT_ID
    )

    create_bucket >> create_cluster >> delete_cluster >> delete_bucket

```

This workflow is set up to:

1.  **Create the staging bucket:** This is where we create the staging bucket (if needed), and also set the lifecycle policy to delete the bucket and objects after one day or other desired timeframe.
2.  **Create the Dataproc cluster:** This operator is configured to use the dynamically created staging bucket.
3.  **Delete the Dataproc cluster:** We remove the Dataproc cluster.
4.  **Delete the staging bucket:** Finally, we explicitly remove the bucket, which also removes all objects in it, if it exists.

To reinforce this, let me suggest some further reading: For the specifics of the google-cloud-storage library, the official Google Cloud documentation is always your best bet. Similarly, the Apache Airflow documentation on the `DataprocCreateClusterOperator` is invaluable. For more complex lifecycle policies or management, I would suggest exploring *Google Cloud Platform Cookbook* by Rui Costa and *Programming Google Cloud Platform* by Drew Hodun, and reviewing the official GCP documentation related to object lifecycle management.

In summary, while the `DataprocClusterCreateOperator` doesn't have a direct 'temporary bucket' parameter, you can achieve this functionality quite effectively by using the standard staging bucket and attaching lifecycle rules or using a deletion task following cluster deletion. It's a pattern I've used many times and found it to be a very robust solution, avoiding unnecessary costs while keeping our environment clean.
