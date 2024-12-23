---
title: "How can I move a GCS to GCS operator between Airflow/Composer projects?"
date: "2024-12-23"
id: "how-can-i-move-a-gcs-to-gcs-operator-between-airflowcomposer-projects"
---

Alright, let's talk about shuffling data between Google Cloud Storage (GCS) locations across different Airflow/Composer projects – a scenario I've definitely encountered a few times, and it usually requires a bit more thought than a simple copy within the same project. It's not inherently difficult, but understanding the underlying permissions and configuration is crucial to avoid headaches down the line. In my experience, this often arises when you're dealing with segmented pipelines where data is ingested in one project, processed in another, and perhaps archived in a third.

The key to a successful GCS-to-GCS operation between projects lies in correctly setting up service account permissions and appropriately configuring the `GCSToGCSOperator`. Forget the idea of directly accessing buckets across projects with a single, global service account – that's a recipe for security issues and unexpected failures. We need to think granularly.

The general process involves these core steps:

1.  **Service Account Setup**: First, identify the service account that your Airflow/Composer environment uses in the *source* project (where your original data resides). We'll call this `source_service_account@source-project.iam.gserviceaccount.com`. You will need to grant this service account the *storage.objects.get* and *storage.objects.list* permissions on the *source* GCS bucket.

2. **Service Account Impersonation:** Next, we need to set up the *target* project where we want our data to end up. In that project, create a service account, such as `target_service_account@target-project.iam.gserviceaccount.com`. This service account needs *storage.objects.create* permission on the *target* GCS bucket. Then, crucially, grant the `source_service_account` the `roles/iam.serviceAccountTokenCreator` permission on the `target_service_account`. This grants `source_service_account` the ability to "impersonate" the `target_service_account`.

3. **Airflow Configuration:** Finally, we'll configure the `GCSToGCSOperator` within your Airflow DAG in the *source* project. This is where we tell it to use the impersonated service account to perform the copy operation into the target project.

Let's look at some code examples using the Python Airflow SDK. These are intended for the source Airflow project DAG, and we will assume you have the necessary GCP provider setup in Airflow and all needed libraries are installed.

**Example 1: Basic cross-project copy with impersonation**

```python
from airflow import DAG
from airflow.providers.google.cloud.transfers.gcs_to_gcs import GCSToGCSOperator
from datetime import datetime

with DAG(
    dag_id='gcs_cross_project_copy',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    copy_data = GCSToGCSOperator(
        task_id='copy_gcs_data',
        source_bucket='source-bucket',
        source_object='source-folder/*',
        destination_bucket='target-bucket',
        destination_object='target-folder/',
        impersonation_chain=['target_service_account@target-project.iam.gserviceaccount.com'], # List of service accounts to impersonate, can be multiple
        gcp_conn_id='google_cloud_default', # Replace with connection if needed
    )
```

In this first example, we specify the `impersonation_chain` parameter. This tells the `GCSToGCSOperator` to obtain temporary credentials for the `target_service_account` using the `source_service_account` credentials, assuming permissions for the impersonation are already set, and subsequently use these temporary credentials when copying to the `target-bucket`.  It's important to use `source-folder/*` to copy everything inside `source-folder` rather than `source-folder` itself which would copy the folder as an object into the target bucket. The wildcard is needed for matching all files in a specific folder.

**Example 2: Copying single object with specific destination name**

```python
from airflow import DAG
from airflow.providers.google.cloud.transfers.gcs_to_gcs import GCSToGCSOperator
from datetime import datetime

with DAG(
    dag_id='gcs_single_object_copy',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    copy_single_file = GCSToGCSOperator(
        task_id='copy_single_gcs_file',
        source_bucket='source-bucket',
        source_object='my_file.txt',
        destination_bucket='target-bucket',
        destination_object='new_file_name.txt',
        impersonation_chain=['target_service_account@target-project.iam.gserviceaccount.com'],
        gcp_conn_id='google_cloud_default',
    )

```

Here, we illustrate the process of copying a specific file (`my_file.txt`) and renaming it during the copy operation within the target bucket. This is very useful when you need to process data and write it with a different convention. Again, impersonation is configured to operate in a cross project scenario. It is worth pointing out that this operation does not change the original file, just creates a new copy on the target bucket.

**Example 3: Advanced copy with metadata and overwrite handling**

```python
from airflow import DAG
from airflow.providers.google.cloud.transfers.gcs_to_gcs import GCSToGCSOperator
from datetime import datetime

with DAG(
    dag_id='gcs_advanced_copy',
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False
) as dag:
    copy_with_metadata = GCSToGCSOperator(
        task_id='copy_gcs_with_metadata',
        source_bucket='source-bucket',
        source_object='data_folder/*',
        destination_bucket='target-bucket',
        destination_object='archive_folder/',
        impersonation_chain=['target_service_account@target-project.iam.gserviceaccount.com'],
        gcp_conn_id='google_cloud_default',
        replace=True,
        gzip=False,
        move_object=False, # Keep the file in the source bucket
        metadata = {'processed_by_airflow': 'true', 'timestamp': '{{ ts }}'},
    )
```

This example demonstrates a few more parameters that `GCSToGCSOperator` offers. We use `replace=True` to handle the scenario where a file with the same name already exists in the destination and thus replace it. We are not using compression `gzip=False`. We also make sure we don't move the files using `move_object=False` and instead we just copy.  Finally, we add custom metadata to the target file, which is very useful for downstream processes or auditing.  Here we can see some templating with airflow `{{ ts }}` variable, but you can define more complex dictionary objects.

A few important notes based on past experiences:

*   **Network Considerations:** Ensure that there's proper network connectivity between your source and target Google Cloud projects. If they're in separate VPCs, you might need to set up VPC peering or use shared VPC for them to properly communicate.
*   **Performance:** For large datasets, consider using the Google Storage Transfer Service. This service is optimized for large data transfers and offers features like parallel transfers. You can trigger it using an Airflow operator like `GoogleCloudStorageTransferServiceStartJobOperator`.
*  **Connection Management:** Depending on your Airflow setup, you may need to create a specific GCP connection rather than using the default, the `gcp_conn_id` param can be used to specify the connection in the operator configuration.
*   **Monitoring:** Always enable logging and monitoring for your Airflow tasks. This will help you quickly diagnose any errors during the data transfer. Also, monitor the GCS bucket logs in both projects.
*   **Error Handling**: Implement proper error handling within your DAG. Use retry policies and alerting to handle transient network issues.
* **Permissions:** Ensure the service accounts have only the permissions they need. Avoid using overly broad permissions for security.

For further understanding, I would highly recommend delving into the official Google Cloud documentation on IAM, particularly on service accounts and impersonation. "The Cloud Operations Handbook" by Google, while focused on broader ops principles, is a good reference for best practices in managing GCP. Additionally, the "Designing Data-Intensive Applications" by Martin Kleppmann provides fundamental insights into data transfer and consistency, which are beneficial to understand for building reliable data pipelines.

Moving data across projects might seem complex initially, but with careful planning and attention to detail – especially around permissions – you should be able to handle these scenarios without too much of a hassle. Remember to test thoroughly in a non-production environment before deploying to production!
