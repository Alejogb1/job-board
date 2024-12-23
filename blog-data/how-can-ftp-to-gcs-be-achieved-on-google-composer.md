---
title: "How can FTP to GCS be achieved on Google Composer?"
date: "2024-12-16"
id: "how-can-ftp-to-gcs-be-achieved-on-google-composer"
---

,  Funny enough, I remember a project a few years back where we had a legacy system still churning out data via FTP, and the mandate was to get everything into Google Cloud Storage (GCS) as quickly and reliably as possible. We ended up using Google Composer, and it wasn’t completely straightforward, but we got there. Here’s the breakdown of how you can achieve FTP to GCS transfers on Composer, drawing on some of the practical lessons I picked up along the way.

The core challenge isn’t just moving files; it’s about orchestrating that movement securely and reliably within a scalable environment. Composer, being a managed Apache Airflow service, is perfectly suited for this kind of task. Airflow’s DAG (Directed Acyclic Graph) structure allows us to define the workflow as a sequence of operations, each one represented by an operator. We’ll use this to orchestrate the FTP download and GCS upload.

At the most fundamental level, we’ll need operators that can handle FTP connections and interact with GCS. Airflow doesn't have a built-in 'FTP to GCS' operator; we'll need to build a DAG using individual operators and potentially some custom code. It's important to remember that Composer environments are relatively containerized. So, you need to make sure that necessary libraries are available in your environment.

Here's how we can structure a typical pipeline:

1.  **FTP Download:** Use the `FTPSensor` to wait for a file to exist on an FTP server. Then, use the `FTPHook` to download it to the local worker's filesystem within the composer environment.
2.  **GCS Upload:** Following the successful FTP download, use the `GCSToGCSOperator` or `GCSHook` to upload the file to a specific GCS bucket.
3.  **Cleanup (Optional):** Delete the file from the local filesystem if you don’t need it around, using the `BashOperator` running an `rm` command.

Here's where we get into the code. I'm using Python with Airflow's library structure here, and this isn’t necessarily a copy/paste ready solution due to differing authentication details, but it will illustrate the main points, using `datetime` for scheduling:

**Code Snippet 1: Basic FTP Download and GCS Upload**

```python
from airflow import DAG
from airflow.providers.ftp.sensors.ftp import FTPSensor
from airflow.providers.ftp.hooks.ftp import FTPHook
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id="ftp_to_gcs_basic",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    ftp_sensor = FTPSensor(
        task_id="ftp_file_sensor",
        path="/path/to/your/file.txt", # Adjust this
        ftp_conn_id="your_ftp_connection", # Ensure the connection is properly configured in Airflow UI
    )

    ftp_download = FTPHook(
        task_id="ftp_download_file",
        ftp_conn_id="your_ftp_connection"
        ).retrieve_file(
            remote_full_path="/path/to/your/file.txt",
            local_full_path="/tmp/file.txt" # Local path to download on worker
        )

    gcs_upload = LocalFilesystemToGCSOperator(
        task_id="gcs_upload_file",
        src="/tmp/file.txt",
        dst="gs://your-gcs-bucket/file.txt", # Adjust this to GCS target bucket and file
        bucket="your-gcs-bucket"
    )

    cleanup = BashOperator(
        task_id="cleanup_local_file",
        bash_command="rm /tmp/file.txt"
    )

    ftp_sensor >> ftp_download >> gcs_upload >> cleanup
```

This first snippet showcases a fundamental approach: Use `FTPSensor` to wait for the file, then download via the hook and upload it to GCS, finally cleanup the local copy of the file. Note that you'll need to have an 'ftp_connection' created on your Airflow connection settings which specifies all of the necessary authentication information.

Now, let’s talk about some of the real-world hiccups we encountered. Often, the remote FTP server might not just drop a file and leave it; there’s frequently a processing delay or a chance the file is still being written to.

**Code Snippet 2: Handling FTP File Incomplete Status**

```python
from airflow import DAG
from airflow.providers.ftp.sensors.ftp import FTPSensor
from airflow.providers.ftp.hooks.ftp import FTPHook
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import time
from airflow.exceptions import AirflowException

def check_file_size(ftp_hook, file_path):
    """Checks if an FTP file size is stable."""
    initial_size = ftp_hook.get_size(file_path)
    time.sleep(10)
    final_size = ftp_hook.get_size(file_path)

    if initial_size != final_size:
       raise AirflowException(f"File size changed during check, not ready. Initial:{initial_size}, Final:{final_size}")
    return True

with DAG(
    dag_id="ftp_to_gcs_stable_size",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    ftp_hook = FTPHook(
        task_id="ftp_hook",
        ftp_conn_id="your_ftp_connection"
    )

    ftp_sensor = FTPSensor(
        task_id="ftp_file_sensor",
        path="/path/to/your/file.txt", # Adjust this
        ftp_conn_id="your_ftp_connection",
        poke_interval=60, # Increase the poke interval to reduce load
        timeout=3600 # Adjust the timeout based on the expected wait time for files
    )

    wait_for_stable_file = python_callable = check_file_size(ftp_hook, "/path/to/your/file.txt")

    ftp_download = FTPHook(
        task_id="ftp_download_file",
        ftp_conn_id="your_ftp_connection"
        ).retrieve_file(
            remote_full_path="/path/to/your/file.txt",
            local_full_path="/tmp/file.txt"
        )

    gcs_upload = LocalFilesystemToGCSOperator(
        task_id="gcs_upload_file",
        src="/tmp/file.txt",
        dst="gs://your-gcs-bucket/file.txt",
        bucket="your-gcs-bucket"
    )

    cleanup = BashOperator(
        task_id="cleanup_local_file",
        bash_command="rm /tmp/file.txt"
    )

    ftp_sensor >> wait_for_stable_file >> ftp_download >> gcs_upload >> cleanup
```

In this second example, we introduce a file size check to confirm that the file isn't still being written. I've included a Python callable task, `wait_for_stable_file` that checks file size. You'll need to install the `apache-airflow-providers-ftp` package in your environment, by the way. This helps to ensure a complete transfer. We introduced a short delay via `time.sleep(10)`. This is a simple approach; more complex scenarios might involve a more sophisticated algorithm or checksum checks, depending on the nature of the data.

Finally, sometimes you might need to perform transformations during the transfer. Let's say, for example, that you are dealing with compressed files from the FTP server.

**Code Snippet 3: Uncompressing FTP Files before GCS Upload**

```python
from airflow import DAG
from airflow.providers.ftp.sensors.ftp import FTPSensor
from airflow.providers.ftp.hooks.ftp import FTPHook
from airflow.providers.google.cloud.transfers.local_to_gcs import LocalFilesystemToGCSOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import subprocess

def uncompress_file(local_path, output_path):
    """Uncompresses the local file"""
    try:
        subprocess.run(["gzip", "-d", local_path, "-c", output_path], check=True)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Failed to uncompress file: {e}")

with DAG(
    dag_id="ftp_to_gcs_uncompress",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    ftp_sensor = FTPSensor(
        task_id="ftp_file_sensor",
        path="/path/to/your/file.gz",  # Adjust this to expect the compressed file
        ftp_conn_id="your_ftp_connection"
    )

    ftp_download = FTPHook(
        task_id="ftp_download_file",
        ftp_conn_id="your_ftp_connection"
        ).retrieve_file(
            remote_full_path="/path/to/your/file.gz",
            local_full_path="/tmp/file.gz"
        )

    uncompress_task = BashOperator(
        task_id="uncompress_file",
        bash_command=f"gzip -d /tmp/file.gz -c > /tmp/file.txt"
    )
    gcs_upload = LocalFilesystemToGCSOperator(
        task_id="gcs_upload_file",
        src="/tmp/file.txt",
        dst="gs://your-gcs-bucket/file.txt",
        bucket="your-gcs-bucket"
    )

    cleanup = BashOperator(
        task_id="cleanup_local_files",
        bash_command="rm /tmp/file.gz && rm /tmp/file.txt"
    )

    ftp_sensor >> ftp_download >> uncompress_task >> gcs_upload >> cleanup
```

This final snippet introduces a BashOperator using `gzip` to uncompress a file before uploading it to GCS. The same principle can be applied to other transformations using bash commands or custom python code via `PythonOperator`.

In terms of resource material, I'd suggest looking at "Data Pipelines with Apache Airflow" by Bas Penders for a comprehensive guide on building data pipelines. Also, the official Airflow documentation (specifically on hooks and operators) is crucial. I've also found "Google Cloud Platform Cookbook" by Rui Costa, and Drew Hodun to be helpful for understanding the GCS component.

These examples should provide a solid foundation. The key takeaways are to handle errors gracefully, monitor your DAG runs within Composer carefully, and adapt your code to the specific nuances of your FTP server. Remember, robust data pipelines require a blend of technical expertise and real-world experience, and building these kinds of integration workflows successfully often involves a bit of trial and error.
