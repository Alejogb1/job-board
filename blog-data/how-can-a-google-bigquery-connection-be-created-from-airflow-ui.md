---
title: "How can a Google BigQuery connection be created from Airflow UI?"
date: "2024-12-16"
id: "how-can-a-google-bigquery-connection-be-created-from-airflow-ui"
---

Alright, let’s unpack this one. I’ve been down this road a few times, especially when we were transitioning from a legacy ETL system to a more modern data orchestration setup back at Chronos Analytics. Connecting BigQuery from Airflow’s UI isn’t inherently complex, but getting it *robust* and *maintainable* requires a bit of care, particularly around authentication and configuration management. Let's break down the process and discuss the nuances.

The central concept here revolves around Airflow’s connection mechanisms. Essentially, Airflow needs credentials and connection details to interact with BigQuery. This isn’t simply about plugging in a username and password. We leverage Airflow's connection management system, often configured via environment variables or directly through the UI itself. I generally favor the former for production environments due to its benefits in security and version control, but let’s explore the UI path, as that's the focus here.

The primary resource that makes this connection possible is the `google.cloud.bigquery` library, which you can use in Airflow's tasks (using either the `BigQueryHook` or more recently, the `BigQueryTransferOperator`). But before that, we need to configure the connection itself within the Airflow UI.

To achieve this, you will first need to log into the Airflow UI. On the main menu, usually accessible by hovering over a 'gear' icon, or similar, you will find the 'Admin' menu where you'll see the 'Connections' entry. This is where we define the specifics for our BigQuery interaction. Clicking 'Create' will prompt a form where you need to fill in several critical details.

For the 'Conn Id', something descriptive like `bigquery_default` or something tailored to your project (e.g., `bq_my_analytics_project`). This 'Conn Id' is what you’ll refer to in your DAGs and task definitions, so make it meaningful. Under ‘Conn Type’, you will obviously select ‘Google BigQuery’. The 'Project Id' field is where you will input the name of your google cloud project id that houses your bigquery datasets.

Crucially, you’ll need to provide authentication credentials, and the preferred approach here is using a service account key file. I’d absolutely recommend a service account with narrowly scoped permissions for least privilege security. This service account JSON key isn’t directly entered into a field. Instead, it needs to be placed inside Airflow’s environment variables or mounted as a secret and referred by the UI. For the UI path, you'd fill the ‘Keyfile Path’ with the exact path in the Airflow worker node where your JSON key resides. Alternatively, If you are deploying Airflow with Kubernetes, storing your secrets in Kubernetes secrets would be a great option instead of mounting files. Note that the path should be absolute.

For this example, let’s assume the file is located at `/opt/airflow/secrets/my-bq-service-account.json`. This is the path we input into the 'Keyfile Path' field. The rest of the fields are less often changed in a basic setup, like ‘Location’ and ‘Credentials Path’, and you can generally leave the defaults.

Once this connection is set up, Airflow can interact with BigQuery. Let’s take a look at some code snippets to demonstrate how this plays out when you are creating DAGs.

**Snippet 1: Using `BigQueryHook` to Query Data**

Here's an example of how to use the `BigQueryHook` to execute a simple SQL query within a PythonOperator.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
from datetime import datetime

def query_bigquery(**kwargs):
    bq_hook = BigQueryHook(
        gcp_conn_id='bigquery_default'  # Referencing our defined connection
    )
    sql = "SELECT * FROM `your-project-id.your_dataset.your_table` LIMIT 10;"
    results = bq_hook.run_query(sql, use_legacy_sql=False)
    print(results) # output results into Airflow's logs
    return results

with DAG(
    dag_id="bigquery_query_example",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    query_task = PythonOperator(
        task_id="run_query",
        python_callable=query_bigquery,
    )
```

In this snippet, the `BigQueryHook` is initialized with our `conn_id` and executes the query. Note how `gcp_conn_id` references the name we assigned to the connection in the Airflow UI. You should always prefer `use_legacy_sql=False` to execute standard SQL rather than legacy SQL, which is now mostly deprecated. The results of your query will then be available in the Airflow logs for that task.

**Snippet 2: Transferring Data using `BigQueryTransferOperator`**

The following is a code example showing how to use the `BigQueryTransferOperator` to load data from a source table into another destination table.

```python
from airflow import DAG
from airflow.providers.google.cloud.transfers.bigquery_to_bigquery import BigQueryToBigQueryOperator
from datetime import datetime

with DAG(
    dag_id="bigquery_transfer_example",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    transfer_task = BigQueryToBigQueryOperator(
        task_id='transfer_bq_table',
        source_project_dataset_tables='your-project-id.source_dataset.source_table',
        destination_project_dataset_table='your-project-id.destination_dataset.destination_table',
        gcp_conn_id='bigquery_default',
        write_disposition='WRITE_TRUNCATE' # other options would be: WRITE_APPEND, WRITE_EMPTY
        )
```

This snippet illustrates a more direct form of interaction with BigQuery. It copies data directly from one table to another, utilizing all configurations defined in the connection we created in the Airflow UI. `WRITE_TRUNCATE` will overwrite the table if exists. Other options for the write_disposition include `WRITE_APPEND` and `WRITE_EMPTY`.

**Snippet 3: Using the `BigQueryInsertJobOperator` to perform data load from Google Cloud Storage**

Here's an example of how you might use `BigQueryInsertJobOperator` to load data from Google Cloud Storage into Bigquery.

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from datetime import datetime

with DAG(
    dag_id="bigquery_load_from_gcs_example",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    load_from_gcs = BigQueryInsertJobOperator(
        task_id="load_gcs_to_bq",
        configuration={
            "load": {
                "sourceUris": ["gs://your-bucket-name/your-file.csv"],
                "schema": {
                    "fields": [
                        {"name": "field1", "type": "STRING"},
                        {"name": "field2", "type": "INTEGER"},
                        # define the schema for your columns
                    ]
                },
                "destinationTable": {
                    "projectId": "your-project-id",
                    "datasetId": "your_dataset",
                    "tableId": "your_table",
                },
               "skipLeadingRows": 1,
               "sourceFormat": "CSV",
            }
        },
        gcp_conn_id='bigquery_default',
    )
```

This final example displays a common real world scenario - loading data from GCS into Bigquery. As with the earlier examples, it is crucial to set the `gcp_conn_id` to the connection you configured in the Airflow UI. Ensure that the service account you are using in your configuration has read access to the GCS bucket and write access to your destination Bigquery table.

For a more detailed understanding of the inner workings of `BigQueryHook`, I would highly recommend reviewing the official Apache Airflow documentation for Google Cloud Providers and specifically, the BigQuery module documentation. Another great resource is the "Google Cloud Platform Cookbook" by Rui Costa and Jose Luis Martinez, it offers practical examples and deeper dive into such integrations. Finally, the "Designing Data-Intensive Applications" by Martin Kleppmann will help you understand design considerations for large data workflows that you are more likely to find with Bigquery data. These documents and books will provide you with a more thorough grounding in both theory and practice.

In short, connecting BigQuery from the Airflow UI is straightforward. But remember, it's less about just getting it to work, and more about achieving it in a secure, maintainable, and professional way. Focus on proper service account management, least privilege access, and environment configuration. This careful planning will significantly reduce future headaches and promote a more robust data pipeline.
