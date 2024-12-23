---
title: "Why am I getting `GCP| Composer Dataproc submit job| Auth credential not found`?"
date: "2024-12-23"
id: "why-am-i-getting-gcp-composer-dataproc-submit-job-auth-credential-not-found"
---

Let's address this "GCP| Composer Dataproc submit job| Auth credential not found" issue, shall we? I’ve seen this particular error pop up more times than I care to remember, and it usually boils down to a few common culprits, which aren't always immediately apparent. It isn’t usually a direct coding mistake but rather a misconfiguration of authentication between your composer environment and the dataproc cluster. My experience, particularly around late 2019, when we were heavily scaling our data pipelines, involved constant tweaking of these settings. So, let me walk you through the common causes and their solutions, based on what I’ve encountered, making sure to include practical code examples to bring the abstract into something concrete.

The core issue here is that when your Composer DAG attempts to submit a job to Dataproc, the underlying execution doesn't have the necessary credentials to authenticate with the Dataproc api. The "auth credential not found" error essentially means the service account your composer environment or airflow worker is using doesn't have the required permissions (IAM bindings) or the correct service account key files for google application default credentials aren’t available to it.

The first, and perhaps most frequent, cause stems from how the underlying worker nodes within your composer environment authenticate. Composer relies on Google Cloud Platform’s (GCP) service accounts. In short, these are special types of accounts used by applications and services, instead of individual users, to make API calls.

If, during the creation of your composer environment, you didn't explicitly provide a service account with the appropriate Dataproc permissions (specifically the "Dataproc Worker" role, at a minimum) the underlying google compute engine instances running your airflow workers won't be able to talk to the Dataproc api. They default to the compute engine default service account, which often lacks the necessary permissions for Dataproc interaction.

**Here’s how to check if your composer environment has the appropriate permissions:**

1.  Navigate to the composer environment page in your google cloud console.
2.  Look for the “environments” panel.
3. Click on your environment to go to the details page.
4. In the configuration pane, you will see which service account is associated with your composer environment. It's critical to make sure this service account has the Dataproc Worker (or equivalent custom role) role. If it's the default compute engine service account (ending with “@developer.gserviceaccount.com”), it is very likely the cause.
5. Navigate to the "IAM" section of the project.
6. Look for the service account under "principals" and ensure it has the necessary "Dataproc Worker" permissions bound.

If the service account doesn't have these roles, you’ll need to update its IAM bindings. You can do this through the console or via the gcloud cli. That’s the most common cause when interacting with dataproc directly from airflow, especially if you have a "minimal permissions" setup.

Now, let's move to the second area where I have often found misconfigurations—the service account context when creating a Dataproc job via the `dataproc_submit_job_operator`. A frequent mistake is implicitly relying on the default application credentials, which might not always behave as expected inside a DAG’s context, especially when the workflow runs in a less traditional way like an airflow worker within Kubernetes.

**Code Example 1: Basic Dataproc Submit Job Operator – potential issue.**

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.dataproc import DataprocSubmitJobOperator
from datetime import datetime

with DAG(
    dag_id="basic_dataproc_job",
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    submit_pyspark_job = DataprocSubmitJobOperator(
        task_id="submit_pyspark_job",
        job={
            "reference": {"job_id": "test_pyspark_job"},
            "placement": {"cluster_name": "your-dataproc-cluster-name"},
            "pyspark_job": {
                "main_python_file_uri": "gs://your-bucket/pyspark_script.py"
            },
        },
        region="your-region",
    )
```

In the above example, no service account is explicitly defined in the `DataprocSubmitJobOperator`. This implies the operator will try to use default application credentials, which might be the default google compute service account, and as explained above, that might not have enough permission to interact with the dataproc api. The fix to avoid this ambiguity, would be to explicitly declare the service account you want to use with the `gcp_conn_id` parameter in the `DataprocSubmitJobOperator` and define it in the Airflow UI.

This leads me to the third crucial issue, related to using service account impersonation or custom credentials explicitly.

**Code Example 2: Dataproc Submit Job Operator with explicit service account**

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.dataproc import DataprocSubmitJobOperator
from datetime import datetime

with DAG(
    dag_id="explicit_service_dataproc_job",
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
    submit_pyspark_job_with_sa = DataprocSubmitJobOperator(
        task_id="submit_pyspark_job_with_sa",
        job={
            "reference": {"job_id": "test_pyspark_job"},
            "placement": {"cluster_name": "your-dataproc-cluster-name"},
            "pyspark_job": {
                "main_python_file_uri": "gs://your-bucket/pyspark_script.py"
            },
        },
        region="your-region",
        gcp_conn_id="my_gcp_connection" # This tells airflow to use the defined connection with credentials
    )

```

Here, `gcp_conn_id` points to an Airflow connection setup to use explicit key file authentication. This means that you have defined this connection to use a service account JSON key, as opposed to the default application credentials and hence it will use that for the API calls. Make sure that the service account associated with this credential has all the required permission, and that the key file is properly configured under this connection in the Airflow UI.
This approach is very useful if you need more granular control over the credentials used or to implement service account impersonation. In this specific case, make sure your service account defined under this connection has the needed permissions.

A fourth, less common but important issue involves network configurations. Sometimes, if you have private clusters or custom networking setups, the worker nodes of your composer environment might not be able to reach the dataproc api endpoint due to firewall rules or network segmentation. It’s critical to ensure the composer environment and the dataproc cluster are in networks that can communicate. Typically this is solved with the same vpc and proper egress and ingress rules.

**Code Example 3: Dataproc Submit Job Operator with Service Account Impersonation**

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.dataproc import DataprocSubmitJobOperator
from airflow.providers.google.common.hooks.base_google import GoogleBaseHook
from datetime import datetime
from google.oauth2 import service_account


def get_impersonated_credentials():
    # Loads default airflow connection credentials for impersonation.
    base_hook = GoogleBaseHook(gcp_conn_id='my_gcp_connection')
    credentials = base_hook.get_credentials()
    impersonated_sa = 'your-impersonated-service-account@your-project.iam.gserviceaccount.com' # Service account you wish to impersonate
    impersonated_credentials = credentials.with_subject(impersonated_sa)
    return impersonated_credentials

with DAG(
    dag_id="impersonated_service_dataproc_job",
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
) as dag:
   submit_pyspark_job_with_impersonation = DataprocSubmitJobOperator(
        task_id="submit_pyspark_job_with_impersonation",
        job={
            "reference": {"job_id": "test_pyspark_job"},
            "placement": {"cluster_name": "your-dataproc-cluster-name"},
            "pyspark_job": {
                "main_python_file_uri": "gs://your-bucket/pyspark_script.py"
            },
        },
        region="your-region",
        gcp_conn_id="my_gcp_connection",
        impersonation_chain=get_impersonated_credentials() # Use impersonation credentials
    )
```

This example illustrates service account impersonation, a more advanced topic, but one often needed in large projects. Here the base service account defined in the gcp connection has the `roles/iam.serviceAccountTokenCreator` role to be able to impersonate another one.

To dive deeper, I recommend referring to the official Google Cloud documentation for IAM (Identity and Access Management) and specifically the documentation on service accounts and roles. The book "Cloud Native Patterns" by Cornelia Davis, specifically the sections on identity and authorization, could also provide more context. For more practical implementation details, the official Apache Airflow documentation for the Google provider is crucial, especially for specifics on the `DataprocSubmitJobOperator` and how to set up connections and service accounts.

Finally, debugging such issues requires meticulous logging review, both from Airflow logs and Google Cloud Operation Logs. You will find more detailed information there about which specific action is being rejected due to authorization failure.

To summarize, the error "GCP| Composer Dataproc submit job| Auth credential not found" stems from missing or misconfigured credentials between your composer environment and the dataproc api, be it from inadequate permissions granted, default credential assumptions, lack of explicit connection configuration or even network isolation. Following these steps and code examples should put you on the path to resolution. Remember, always explicitly configure service accounts, check for the correct IAM bindings, and use logs to help guide your troubleshooting process.
