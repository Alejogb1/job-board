---
title: "Can the SimpleHttpOperator in Airflow handle binary file uploads?"
date: "2025-01-30"
id: "can-the-simplehttpoperator-in-airflow-handle-binary-file"
---
The fundamental nature of the SimpleHttpOperator in Apache Airflow, relying on the `requests` library, necessitates careful handling of binary data. While seemingly straightforward, directly passing binary file content as a request payload without proper encoding often leads to failures or corrupt transfers.

I’ve encountered this specific issue several times while automating data pipelines that involved extracting information from web APIs. My team, tasked with daily data synchronization between a proprietary cloud platform and our data lake, initially attempted to use the SimpleHttpOperator for uploading image files and compressed datasets. This led to a pattern of unpredictable HTTP 400 errors and, even worse, corrupted files being stored in our target location. Through a process of debugging and testing, we determined the root cause centered on how the operator, and specifically the underlying `requests` library, processes binary data.

The SimpleHttpOperator, at its core, constructs an HTTP request using the `requests` library. When a request payload is provided via the `data` argument, `requests` attempts to serialize this data based on its detected content type. For strings or dictionaries, it usually works flawlessly, often automatically encoding them as JSON or using URL-encoding. However, when encountering raw byte streams representing binary files, `requests` frequently attempts string encoding which leads to data corruption when interpreted at the receiving end. This corruption often manifests as changes in file checksum or difficulty in file parsing. It is essential to instruct `requests` not to perform any such encoding for the binary data. This is achieved through explicitly setting the content type and using the `files` parameter.

A key method for handling binary uploads is leveraging the `files` parameter of the `requests` library within the `SimpleHttpOperator`. Instead of placing the raw binary data directly into the `data` argument, we provide it as a file-like object within the `files` parameter. The library then appropriately packages this data for transmission. When dealing with a binary file, the request's content type should reflect the file's type (e.g., `image/png`, `application/zip`). If the server expects a specific form-data upload, setting the `Content-Type` to multipart/form-data is essential.

Here is the first example, demonstrating the *incorrect* approach that causes problems:

```python
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow import DAG
from datetime import datetime
import os

with DAG(
    dag_id='binary_upload_incorrect',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    with open('/path/to/my_image.png', 'rb') as f:
        image_data = f.read()

    upload_task = SimpleHttpOperator(
        task_id='upload_image_incorrect',
        http_conn_id='my_http_connection',
        endpoint='/upload',
        method='POST',
        data=image_data,
        headers={"Content-Type": "image/png"}
    )
```
This approach treats the binary data `image_data` as a simple data blob. The `requests` library might then attempt to encode this as a string, leading to transmission issues. Even with the specified `Content-Type`, the underlying processing is not optimized for raw binary data.  The server receives corrupted data.

Now, the correct method utilizing the `files` parameter of `requests` is shown below:

```python
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow import DAG
from datetime import datetime
import os

with DAG(
    dag_id='binary_upload_correct',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    with open('/path/to/my_image.png', 'rb') as f:
        upload_task = SimpleHttpOperator(
            task_id='upload_image_correct',
            http_conn_id='my_http_connection',
            endpoint='/upload',
            method='POST',
            files={'file': ('my_image.png', f, 'image/png')}
        )
```
In this version, `files` is a dictionary. The key is the parameter name as expected by the server ("file").  The value is a tuple containing the file's name, the file-like object, and the content type. The `requests` library handles this tuple correctly, packaging the data within a multipart/form-data request, preserving the binary data integrity. The result is a correctly uploaded file. The server receives a valid file as a multipart message.

If the receiving end does *not* expect form data and only needs the raw bytes, we can set a custom `data` argument and explicitly avoid encoding:

```python
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow import DAG
from datetime import datetime
import os

with DAG(
    dag_id='binary_upload_raw',
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    with open('/path/to/my_compressed_data.zip', 'rb') as f:
        zip_data = f.read()

    upload_task = SimpleHttpOperator(
        task_id='upload_zip_raw',
        http_conn_id='my_http_connection',
        endpoint='/raw_upload',
        method='PUT',
        data=zip_data,
        headers={'Content-Type': 'application/zip'},
        extra_options={'data': zip_data, "allow_redirects": True, 'decode_content':False}
    )
```
Here, while using the `data` parameter, we also specify the raw data and the correct `Content-Type`. Crucially, `extra_options` are utilized to avoid the standard `requests` behavior for processing the payload. The `decode_content=False` parameter is important, preventing automatic decoding of the response by `requests`. This approach was necessary for my team when uploading large compressed datasets as a raw stream.

When implementing solutions, the server-side API’s specifications dictate the precise approach. If the server expects multipart/form-data, the `files` parameter should be used. If the server receives raw data and the client is allowed to specify the `Content-Type`, passing the raw bytes as `data` with the `extra_options` is the preferred alternative.

For resources, I recommend researching the official `requests` library documentation for its handling of files and data payloads. Additionally, understanding the specifics of multipart form data is beneficial for web API development. There are many books and academic articles detailing these methods that provide useful background. Finally, a deep dive into the Apache Airflow documentation, specifically the SimpleHttpOperator and related providers, illuminates the available options and limitations.
