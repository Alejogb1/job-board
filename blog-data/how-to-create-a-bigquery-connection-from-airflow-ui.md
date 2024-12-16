---
title: "How to create a BigQuery connection from Airflow UI?"
date: "2024-12-16"
id: "how-to-create-a-bigquery-connection-from-airflow-ui"
---

Okay, let’s tackle connecting to BigQuery from the Airflow ui. I’ve wrangled this particular beast a few times, and while the process is conceptually straightforward, a few practical considerations tend to crop up, so let's get into the details.

It's not merely about clicking a button; it's about establishing a robust, secure, and manageable connection that Airflow can reliably leverage for data orchestration. Typically, when I approach this, I'm mindful of a few key aspects: authentication, connection configuration, and best practices for security. Let’s break down how I’ve handled this in past projects, specifically from the perspective of initiating everything from the Airflow ui itself.

First, authentication. In my experience, the most prevalent method, and usually the safest, involves using a service account key. Avoid embedding credentials directly in connection parameters—it's a security no-no. Instead, we leverage the credentials file's path. Think of it as a secure key to your BigQuery kingdom. Airflow will use these credentials to authorize its interactions. This method is superior to manually inputting credentials, offering both better security and manageability.

To configure this in Airflow’s UI, navigate to the "Admin" section and select "Connections." Then, hit the “Create” button. From the connection type dropdown, select “Google Cloud Platform.” You’ll now face several fields requiring your attention. Here's a breakdown of the important ones:

* **Conn Id:** This is the logical name by which your connection will be referenced in your DAG definitions. I'd typically stick with something descriptive, like `bigquery_default`, or if connecting to a specific project, `bigquery_project_x`, for better organization.
* **Conn Type:** As I mentioned, select "Google Cloud Platform." This selection will present a set of specific fields relevant to Google’s infrastructure.
* **Project Id:** Enter the google cloud project id associated with the BigQuery resources. This is the key that identifies the project within the gcp ecosystem.
* **Keyfile path:** This is the path within the Airflow environment where the json service account key file is stored. It is *critical* that this file be accessible to the airflow worker processes. This path can be relative to the Airflow DAGs folder, or an absolute path accessible within the container.
* **Keyfile JSON:** *Do not* paste the json content directly here. Use the file path above instead. This ensures better security by avoiding storing sensitive credentials directly within the database records.
* **Scopes:** This is where you specify the permissions Airflow requires. Usually, you'll need `https://www.googleapis.com/auth/bigquery` for typical BigQuery operations, as well as `https://www.googleapis.com/auth/cloud-platform` if you plan on doing other gcp related operations within your DAG.

Once you have all those parameters set, clicking “Test” is highly recommended. This will trigger a validation process ensuring Airflow can establish a connection using the supplied credentials. It’s a good sanity check to confirm everything is configured properly before you move forward. If you're dealing with network restrictions, or are behind some kind of proxy, you'll need to add additional settings within airflow.cfg. For example, a `proxy_host` setting.

Okay, let’s dive into some practical examples.

**Example 1: Simple Query Execution**

Here's how I might structure a basic DAG that queries BigQuery:

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryExecuteQueryOperator
from datetime import datetime

with DAG(
    dag_id='bigquery_simple_query',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    query_task = BigQueryExecuteQueryOperator(
        task_id='execute_bigquery_query',
        sql="SELECT COUNT(*) FROM `your-project.your_dataset.your_table`;",
        use_legacy_sql=False,
        gcp_conn_id='bigquery_default'
    )
```

In this scenario, the `BigQueryExecuteQueryOperator` is used. Notice how the `gcp_conn_id` parameter refers back to the connection we defined in the Airflow ui – `bigquery_default`. The sql parameter holds your query. *Ensure that `your-project`, `your_dataset`, and `your_table` are updated to match your actual project details*. This is about as straightforward as it gets for a simple query.

**Example 2: Loading Data From GCS into BigQuery**

Here’s an example of loading data from google cloud storage into a table:

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryCreateEmptyTableOperator
from airflow.providers.google.cloud.transfers.gcs_to_bigquery import GCSToBigQueryOperator
from datetime import datetime

with DAG(
    dag_id='bigquery_gcs_load',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    create_table = BigQueryCreateEmptyTableOperator(
        task_id='create_table',
        project_id="your-project",
        dataset_id="your_dataset",
        table_id="your_new_table",
        schema_fields=[
              {'name': 'column_1', 'type': 'STRING', 'mode': 'NULLABLE'},
              {'name': 'column_2', 'type': 'INTEGER', 'mode': 'NULLABLE'}
        ],
        gcp_conn_id='bigquery_default'
    )
    load_gcs_to_bq = GCSToBigQueryOperator(
        task_id='load_data',
        bucket='your-gcs-bucket',
        source_objects=['path/to/your/file.csv'],
        destination_project_dataset_table='your-project.your_dataset.your_new_table',
        write_disposition='WRITE_TRUNCATE',
        source_format='CSV',
        skip_leading_rows=1, #adjust if needed
        gcp_conn_id='bigquery_default'
    )
    create_table >> load_gcs_to_bq
```

Here, we're not only using the `bigquery_default` connection but also orchestrating a two-step process, leveraging two distinct operators. The first one, `BigQueryCreateEmptyTableOperator` is used to create the empty table, whilst the second one `GCSToBigQueryOperator` loads the data from a google cloud storage file. Again, ensure all placeholders, including the path to your gcs source file, are replaced with the appropriate names. `write_disposition` dictates how writing is handled if the table already exists. `WRITE_TRUNCATE` deletes the existing table and writes the new data. Other options like `WRITE_APPEND` exist too.

**Example 3: Using Templated Queries**

Finally, let’s consider a more advanced scenario – templated queries. This is particularly powerful when you need to parameterize your queries based on the execution context:

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryExecuteQueryOperator
from datetime import datetime

with DAG(
    dag_id='bigquery_templated_query',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:
    templated_query = """
    SELECT *
    FROM `your-project.your_dataset.your_table`
    WHERE date_column = '{{ ds }}'
    """

    query_with_template = BigQueryExecuteQueryOperator(
        task_id='execute_templated_query',
        sql=templated_query,
        use_legacy_sql=False,
        gcp_conn_id='bigquery_default'
    )
```
Here, the query leverages Jinja templating. `{{ ds }}` becomes the execution date, which provides a dynamic approach to query data within specific ranges. This technique is incredibly useful when processing data incrementally. This templating functionality is an important feature of airflow, and not just limited to Bigquery.

For further in-depth exploration, I'd recommend reviewing the official Apache Airflow documentation. Specifically, look into the provider documentation for google, and particularly the `airflow.providers.google.cloud.operators.bigquery` and `airflow.providers.google.cloud.transfers.gcs_to_bigquery` documentation. The google cloud documentation on bigquery best practices will also offer useful insights on how to effectively manage your data in the service. While these resources don't cover all the edge cases you might encounter, understanding them will provide a strong foundation for any data orchestration task between Airflow and BigQuery. Also consider examining the source code of these operators on GitHub for even greater granularity if needed.

Establishing this BigQuery connection through the Airflow ui involves a combination of careful configuration of credentials and a deep understanding of your service account's permissions. Doing this methodically ensures that your data pipelines operate securely and reliably.
