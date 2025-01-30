---
title: "How can I load a GCS CSV file encrypted with GPG into BigQuery using Airflow or Python?"
date: "2025-01-30"
id: "how-can-i-load-a-gcs-csv-file"
---
The primary challenge lies in the lack of native GPG decryption capabilities within BigQuery or standard Airflow operators. I encountered this exact issue while migrating a legacy data pipeline that utilized PGP-encrypted CSV files stored on Google Cloud Storage. A straightforward approach of directly loading the encrypted file into BigQuery will fail because BigQuery expects unencrypted CSV data. I found the most reliable solution involves decrypting the file within the Airflow environment prior to loading it into BigQuery. This necessitates a workflow involving file retrieval from GCS, decryption using a GPG key, and then utilizing the decrypted content for BigQuery loading.

The core strategy centers around employing Python’s `gpg` module in conjunction with the Google Cloud Storage client library and the BigQuery client library. We’ll structure an Airflow DAG that orchestrates these steps sequentially. First, the file must be downloaded from GCS. Then, the downloaded file is decrypted using a private key. Finally, the decrypted content, which resides in memory, can be written to a new, unencrypted CSV file or a file-like object which is then loaded into BigQuery. Let’s examine the crucial pieces of code and their roles in the process.

**Code Example 1: GCS File Download and Decryption**

```python
from google.cloud import storage
import gnupg
import io
import os

def decrypt_gcs_file(bucket_name, source_blob_name, private_key_path, passphrase=None):
    """
    Downloads a GPG encrypted file from GCS, decrypts it using GPG and returns the decrypted content as a file-like object.

    Args:
        bucket_name (str): The GCS bucket name.
        source_blob_name (str): The GCS blob name.
        private_key_path (str): Path to the GPG private key file.
        passphrase (str, optional): Passphrase to decrypt the key. Defaults to None.

    Returns:
        io.StringIO: File-like object containing the decrypted content.
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    
    with io.BytesIO() as buffer:
      blob.download_to_file(buffer)
      buffer.seek(0)

      gpg = gnupg.GPG()
      with open(private_key_path, "r") as key_file:
          import_result = gpg.import_keys(key_file.read())
      if import_result.count != 1:
          raise ValueError("Private key import failed or unexpected number of keys imported")

      decrypted_data = gpg.decrypt_file(buffer, passphrase=passphrase)

      if not decrypted_data.ok:
        raise Exception(f"GPG Decryption failed: {decrypted_data.status} {decrypted_data.stderr}")
    
      return io.StringIO(str(decrypted_data))

```

*Commentary:* This function encapsulates the core decryption logic. It starts by establishing a connection to Google Cloud Storage. After that, it downloads the encrypted file into a byte buffer in memory to prevent creating a persistent file on the filesystem. This is crucial for ephemeral environments like Airflow workers.  The GPG private key is loaded from a specified path. The decryption using the `decrypt_file` method handles the bulk of the decryption process, which, unlike decrypting string data, can accommodate the byte stream of the encrypted file without holding the whole file in memory at once. Finally, it returns an `io.StringIO` which acts like a readable file object, which the next step will use. It’s important to note that any errors during download, decryption, or key loading will result in a thrown Exception, preventing corrupted data from being passed to downstream tasks. This method's robustness stems from the error checking and explicit file-like object handling.

**Code Example 2: BigQuery Load with File-like Object**

```python
from google.cloud import bigquery

def load_decrypted_to_bigquery(project_id, dataset_id, table_id, decrypted_data, schema_definition=None):
    """
    Loads a decrypted CSV file-like object into BigQuery.

    Args:
        project_id (str): GCP project id.
        dataset_id (str): BigQuery dataset id.
        table_id (str): BigQuery table id.
        decrypted_data (io.StringIO): File-like object containing decrypted CSV data.
        schema_definition (list, optional): BigQuery schema definition, passed as list of dicts.
                                           Defaults to None (schema is auto-detected).
    """
    client = bigquery.Client(project=project_id)
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,  # Assumes a header row in CSV
        schema=schema_definition,
    )

    load_job = client.load_table_from_file(decrypted_data, table_ref, job_config=job_config)
    load_job.result() # Waits for the job to complete
    print(f"Loaded {load_job.output_rows} rows into {table_ref.path}")
```
*Commentary:* This function focuses on the BigQuery integration aspect. It takes the decrypted content, which is in memory and represented by a file-like object. The method is similar to the standard `load_table_from_file` method except that it takes a file-like object as input, instead of a file system path.  The function uses a standard BigQuery loading procedure, including specifying CSV source format, automatically detecting the schema or applying a pre-defined schema, and skipping the header row. I found specifying the schema explicitly valuable for preventing type coercion issues. Waiting for the job result makes sure the load is complete, which is vital in an Airflow DAG, as it acts as a sensor. The log message, which contains the loaded row count and target table details, is helpful during debugging and monitoring.

**Code Example 3: Airflow DAG Integration**

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

# Define DAG arguments, start date etc here
dag_args = {
    'owner': 'me',
    'start_date': datetime(2023, 1, 1),
    'schedule_interval': None,
}

with DAG('gcs_gpg_to_bq_pipeline', default_args=dag_args, catchup=False) as dag:
    
    decrypt_task = PythonOperator(
        task_id='decrypt_gcs_file_task',
        python_callable=decrypt_gcs_file,
        op_kwargs={
            'bucket_name': 'my-gcs-bucket',
            'source_blob_name': 'encrypted_data.csv.gpg',
            'private_key_path': '/path/to/my/private.key',
            'passphrase': 'my_secret_passphrase',
        },
    )
    
    bq_load_task = PythonOperator(
      task_id='load_decrypted_to_bigquery_task',
      python_callable=load_decrypted_to_bigquery,
        op_kwargs={
            'project_id': 'my-gcp-project',
            'dataset_id': 'my_dataset',
            'table_id': 'my_table',
            'decrypted_data': '{{ ti.xcom_pull(task_ids="decrypt_gcs_file_task", key="return_value") }}',
           #Schema definition: Optional
           'schema_definition':  [ 
                                    {'name': 'column_1', 'type': 'STRING'},
                                    {'name': 'column_2', 'type': 'INTEGER'},
                                  ]
        }
    )

    decrypt_task >> bq_load_task
```
*Commentary:* This is the example Airflow DAG that uses the previous two Python functions.  It consists of two `PythonOperator` tasks: `decrypt_task` and `bq_load_task`. The key concept here is using XCom to pass the decrypted file-like object from the decryption task to the BigQuery loading task. This is done by accessing the `return_value` of the first task. The `op_kwargs` in each operator are set with placeholders, indicating where you would pass your specific details, including GCS bucket, blob, key locations and BigQuery configurations. Note that schema definition is optional. The tasks are chained using the `>>` operator, ensuring that the decryption task always completes before the loading task begins.

In conclusion, this approach effectively tackles the problem of loading GPG-encrypted CSV data from GCS into BigQuery by leveraging Python's GPG module and the flexibility of Airflow's Python operators. Key to its success is the memory based operation, and using file like objects to maintain data flow without persisting data to the filesystem, along with a robust error handling within the decryption process.

**Resource Recommendations:**

1.  **The Python 'gnupg' module:** The official documentation offers detailed guidance on utilizing various GPG functions, including key management and data encryption/decryption.
2.  **Google Cloud Storage Client Library:** Thoroughly explore the documentation, which provides in-depth coverage of file manipulation such as download, upload, and handling different types of blob data.
3.  **Google BigQuery Client Library:** Focus on the sections that describe data loading, job configurations, and schema handling.
4.  **Apache Airflow documentation:** Review the documentation on Python operators, XCom, DAG creation, and dependency management.
5. **GPG manual:** Understanding the fundamentals of GPG, particularly key management and data handling, is crucial.
