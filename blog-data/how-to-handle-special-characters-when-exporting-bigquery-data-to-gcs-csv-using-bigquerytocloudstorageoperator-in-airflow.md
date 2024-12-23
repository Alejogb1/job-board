---
title: "How to handle special characters when exporting BigQuery data to GCS CSV using BigQueryToCloudStorageOperator in Airflow?"
date: "2024-12-23"
id: "how-to-handle-special-characters-when-exporting-bigquery-data-to-gcs-csv-using-bigquerytocloudstorageoperator-in-airflow"
---

Alright,  I remember back in '18, we ran into a similar issue while migrating a large e-commerce dataset. Special characters, specifically those beyond the standard ASCII range, would wreak havoc on our data exports from BigQuery to GCS via Airflow. It manifested as mangled text, encoding errors, and, in some cases, complete data loss when we’d try to process the resulting csv files downstream. Dealing with this properly, it turns out, requires a firm grasp on encoding nuances and how BigQuery and GCS interact. The `BigQueryToCloudStorageOperator` itself, while powerful, isn't magical; it needs explicit guidance on how to handle these characters.

The core of the problem lies in character encodings. By default, many systems, including legacy data stores or older file formats, default to single-byte encodings like ASCII or ISO-8859-1. These encodings simply lack the capacity to represent the wide range of characters used in various languages (think accents, emojis, or non-Latin scripts), leading to the aforementioned issues when encountering them. BigQuery, thankfully, stores data internally using UTF-8, which is designed to handle all these characters. But this doesn't automatically propagate to the export process. If we don't specify an appropriate encoding for the export, BigQuery can default to something less comprehensive, and the resulting CSV file ends up a mess.

To address this, we must explicitly define the encoding during the export operation. We need to tell the `BigQueryToCloudStorageOperator` to write the CSV file using UTF-8, ensuring that all characters, including special ones, are correctly represented. The operator has an `export_format_options` parameter that allows us to configure this. Specifically, we need to set the `encoding` key within this parameter.

Here’s the first snippet, demonstrating a basic, though insufficient, approach:

```python
from airflow.providers.google.cloud.operators.bigquery import BigQueryToCloudStorageOperator
from airflow import DAG
from datetime import datetime

with DAG(
    dag_id='basic_export',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    export_task = BigQueryToCloudStorageOperator(
        task_id='export_bq_to_gcs',
        source_project_dataset_table='your-project.your_dataset.your_table',
        destination_cloud_storage_uris=['gs://your-bucket/output/data.csv'],
        export_format='csv'
    )
```
This snippet, while functional for ASCII characters, is going to cause the issue we're talking about when special chars are included. Notice there’s no encoding specified here. It defaults.

Now, let’s move on to the corrected approach. This is how you *should* be doing it:

```python
from airflow.providers.google.cloud.operators.bigquery import BigQueryToCloudStorageOperator
from airflow import DAG
from datetime import datetime

with DAG(
    dag_id='utf8_encoded_export',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    export_task = BigQueryToCloudStorageOperator(
        task_id='export_bq_to_gcs_utf8',
        source_project_dataset_table='your-project.your_dataset.your_table',
        destination_cloud_storage_uris=['gs://your-bucket/output/data.csv'],
        export_format='csv',
        export_format_options={'encoding': 'UTF-8'}
    )
```

In the second snippet, we've added the `export_format_options` parameter and explicitly set the `encoding` to 'UTF-8'. This small change is the key to correctly exporting special characters. This ensures that BigQuery encodes the output CSV in a way that can represent all your data accurately.

But, I’ve seen cases where merely setting the encoding isn’t enough if the data itself contains inconsistent characters or even if the source had some encoding issues pre-BigQuery. So, a more robust approach might also involve specifying a 'quote_character', which I’ll show in the third example. Sometimes, fields may include delimiters that confuse the CSV parser. Enclosing fields with a quote character prevents this. It will also help when you use other systems for consuming data. A comma in a text field might cause problems if you don't have these in place.

```python
from airflow.providers.google.cloud.operators.bigquery import BigQueryToCloudStorageOperator
from airflow import DAG
from datetime import datetime

with DAG(
    dag_id='robust_export',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    export_task = BigQueryToCloudStorageOperator(
        task_id='export_bq_to_gcs_robust',
        source_project_dataset_table='your-project.your_dataset.your_table',
        destination_cloud_storage_uris=['gs://your-bucket/output/data.csv'],
        export_format='csv',
         export_format_options={
            'encoding': 'UTF-8',
             'quote_character': '"'
        }

    )

```

This third example not only sets UTF-8 encoding but also defines double quotes as the quote character. This gives you a solid foundation for handling complex text data within CSV exports. It's a defensive approach that saves headaches down the line.

Beyond code, there’s some reading that can enhance your understanding. For a deep dive into character encoding, I highly recommend "Unicode Explained" by Jukka Korpela. It’s a very comprehensive guide to the topic. If you're interested in the specifics of CSV file formats, check out the RFC 4180 which is the official specification, and not quite exciting but useful nonetheless. Regarding best practices with BigQuery, the official Google Cloud documentation is always a valuable resource, especially the sections dealing with data export and encoding.

In summary, dealing with special characters during BigQuery exports to GCS via Airflow isn’t something complex, but it requires awareness of character encoding. By explicitly setting the `encoding` to 'UTF-8' and even specifying a `quote_character` in the `export_format_options` of the `BigQueryToCloudStorageOperator`, you’ll sidestep most of the common pitfalls. The first code snippet fails to handle these specifics while the other two do. Remember to be proactive and anticipate potential issues. It can save you a lot of time and frustration down the line. These small configuration adjustments can mean the difference between a clean, usable dataset, and a frustrating, error-prone one.
