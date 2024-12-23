---
title: "How can I use XCOM pushed values as parameters in BigQueryToGCSOperator?"
date: "2024-12-23"
id: "how-can-i-use-xcom-pushed-values-as-parameters-in-bigquerytogcsoperator"
---

Alright, let’s tackle this. The dance of passing values from XCOM to operators, especially to something like `BigQueryToGCSOperator`, can sometimes feel a bit like choreography, but with clear steps, it's quite manageable. I've definitely seen my share of these scenarios in data pipelines. In fact, just last year, I was working on a project where we were generating daily BigQuery tables based on dynamically calculated dates, and we needed to export these tables to GCS. We heavily relied on this pattern of pulling values from XCOM to feed parameters to downstream operators. It’s a common scenario, and there are several straightforward ways to handle it.

The challenge essentially boils down to correctly structuring your Airflow DAG so that the appropriate values are pushed to XCOM, and then subsequently retrieved and used as arguments for the `BigQueryToGCSOperator`. The key is to understand that XCOM is essentially a key-value store, and the push and pull operations need to be aligned between tasks. The core approach involves pushing the dynamic values from a task using the `xcom_push` method or task return values in a task function, followed by referencing these XCOM values in the downstream operator through Jinja templating.

Here’s how I generally approach this, with a few concrete examples to illustrate different methods:

**Method 1: Using `task_instance.xcom_push` and Jinja Templating**

This method is good for when you need very explicit control over what and when you push to XCOM. The steps are:

1.  A task pushes a value to XCOM using `task_instance.xcom_push` with a key-value pair within an Airflow PythonOperator.
2.  A subsequent task references this XCOM value as a Jinja template within its parameters, like in `BigQueryToGCSOperator`.

Here’s the code example:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.transfers.bigquery_to_gcs import BigQueryToGCSOperator
from airflow.utils.dates import days_ago
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
}


def push_xcom_value(**kwargs):
    ti = kwargs['ti']
    date_str = datetime.now().strftime('%Y%m%d')
    ti.xcom_push(key='dynamic_date', value=date_str)

with DAG(
    'xcom_example_1',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:
    push_date_task = PythonOperator(
        task_id='push_dynamic_date',
        python_callable=push_xcom_value,
    )

    export_to_gcs = BigQueryToGCSOperator(
        task_id='export_to_gcs',
        source_project_dataset_table="{{ var.value.gcp_project }}.mydataset.mytable_{{ ti.xcom_pull(task_ids='push_dynamic_date', key='dynamic_date') }}",
        destination_cloud_storage_uris=[
            f"gs://my-bucket/my_output_{{{{ ti.xcom_pull(task_ids='push_dynamic_date', key='dynamic_date') }}}}.json",
        ],
        export_format='JSON',
        gcp_conn_id='gcp_default',
    )

    push_date_task >> export_to_gcs

```

In this example, the `push_dynamic_date` task calculates a current date and pushes it to XCOM with the key `dynamic_date`. The `export_to_gcs` task then accesses this pushed value using Jinja template syntax `{{ ti.xcom_pull(task_ids='push_dynamic_date', key='dynamic_date') }}` within both `source_project_dataset_table` and `destination_cloud_storage_uris` parameters, creating dynamic filepaths and tables based on this date. I recommend looking at the Airflow documentation specifically around templating, as mastering this is key for dynamic DAG creation.

**Method 2: Using Task Return Values and Jinja Templating**

This approach simplifies XCOM interactions, where a PythonOperator's returned value is automatically pushed to XCOM (keyed by the task_id).

1.  A PythonOperator returns the desired value.
2.  A downstream task retrieves this value using its task id in Jinja template.

Here's the code:

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.transfers.bigquery_to_gcs import BigQueryToGCSOperator
from airflow.utils.dates import days_ago
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
}

def generate_xcom_value():
    date_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"my_table_{date_str}"


with DAG(
    'xcom_example_2',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:
    generate_table_name_task = PythonOperator(
        task_id='generate_table_name',
        python_callable=generate_xcom_value,
    )

    export_to_gcs = BigQueryToGCSOperator(
        task_id='export_to_gcs',
         source_project_dataset_table="{{ var.value.gcp_project }}.mydataset.{{ ti.xcom_pull(task_ids='generate_table_name') }}",
        destination_cloud_storage_uris=[
           f"gs://my-bucket/my_output_{{{{ ti.xcom_pull(task_ids='generate_table_name') }}}}.csv",
        ],
        export_format='CSV',
        gcp_conn_id='gcp_default',
    )

    generate_table_name_task >> export_to_gcs
```

Here, the `generate_table_name` task returns a dynamically created table name string. Airflow implicitly pushes this string to XCOM with the task id as the key. The `BigQueryToGCSOperator` then pulls the value using  `{{ ti.xcom_pull(task_ids='generate_table_name') }}` in the table name and GCS output filename. This is arguably a cleaner approach, especially if you are only dealing with single values.

**Method 3: Using a Dictionary to structure XCOM values and Jinja Templating**

Sometimes, you need to pass multiple values, this method is suitable. It is an extension of Method 2.
1. PythonOperator returns a dictionary.
2. Downstream tasks use dictionary keys within Jinja templates.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.transfers.bigquery_to_gcs import BigQueryToGCSOperator
from airflow.utils.dates import days_ago
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
}

def generate_xcom_dict():
    date_str = datetime.now().strftime('%Y%m%d')
    table_name = f"my_table_{date_str}"
    output_filename = f"my_output_{date_str}.parquet"
    return {"table_name": table_name, "filename": output_filename}

with DAG(
    'xcom_example_3',
    default_args=default_args,
    schedule_interval=None,
    catchup=False,
) as dag:
    generate_data_dict_task = PythonOperator(
        task_id='generate_data_dict',
        python_callable=generate_xcom_dict,
    )

    export_to_gcs = BigQueryToGCSOperator(
        task_id='export_to_gcs',
        source_project_dataset_table="{{ var.value.gcp_project }}.mydataset.{{ ti.xcom_pull(task_ids='generate_data_dict')['table_name'] }}",
        destination_cloud_storage_uris=[
            f"gs://my-bucket/{{{{ ti.xcom_pull(task_ids='generate_data_dict')['filename'] }}}}",
        ],
        export_format='PARQUET',
        gcp_conn_id='gcp_default',
    )

    generate_data_dict_task >> export_to_gcs
```

Here, the `generate_data_dict` returns a dictionary, including a table name and filename. The `BigQueryToGCSOperator` references each field in the dictionary by its key.

**Important Considerations and Recommendations**

*   **Error Handling:** Always ensure proper error handling and default values in your Jinja templates to prevent DAG failures if XCOM values are missing. Consider adding checks like `{{ ti.xcom_pull(task_ids='my_task', key='my_key', default='fallback_value') }}`.
*   **Type Safety:** Be mindful of data types passed through XCOM, especially with JSON or other complex structures. Ensure data is serializable. If dealing with particularly complex values, you can explore custom XCOM backends.
*   **Documentation:** Read and digest the official Airflow documentation thoroughly, especially regarding XCOM and Jinja templating. The official documentation is your friend here. There are also good resources on the topic available on the "Astronomer" website.
*  **Code Clarity:** Prioritize code readability by using well-defined task names and keys for your XCOM values. This helps maintain a clear understanding of your DAG logic. I've had to debug some really convoluted pipelines in my time, and clear naming makes a world of difference.

To further expand your knowledge I recommend focusing on Airflow's documentation on templating and XCOM. It is important to get a solid understanding of these underlying concepts. I recommend reading up on *Effective Python* by Brett Slatkin for best practices on structuring your python code which is used heavily within the python operator.

In summary, using XCOM pushed values as parameters in `BigQueryToGCSOperator` is all about understanding how to push and pull values using the correct keys and references within Airflow DAGs. Mastering Jinja templating in Airflow allows for flexible and dynamic DAGs. By utilizing one of the methods discussed and ensuring proper error handling, you can build robust and reliable data pipelines.
