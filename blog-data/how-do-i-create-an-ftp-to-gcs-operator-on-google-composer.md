---
title: "How do I create an FTP to GCS Operator on Google Composer?"
date: "2024-12-16"
id: "how-do-i-create-an-ftp-to-gcs-operator-on-google-composer"
---

Let's tackle this. Creating an FTP to Google Cloud Storage (GCS) operator in Google Composer isn’t exactly a plug-and-play affair, but it's a common enough requirement that I’ve spent a fair bit of time refining the process. I've had to build this several times, often starting from scratch when dealing with legacy systems or clients that aren't quite up-to-date. It's not terribly complicated, but some nuances demand attention. I remember wrestling with an authentication issue on one project that took a good few hours of debugging until I realised the server was using an older, non-standard ftp configuration. The key is to break down the process and choose the appropriate approach based on your needs.

Fundamentally, we’re dealing with the need to pull files from an FTP server and place them into GCS. Airflow (the underlying orchestration engine used by Google Composer) doesn't have a dedicated FTP to GCS operator out of the box. We'll need to orchestrate the following general steps: First, establish an FTP connection; next, fetch the desired files; and finally, upload those files to GCS. We can approach this either by utilizing existing operators alongside bash scripts or by creating a custom operator. I will provide code snippets for each approach.

The most straightforward, although potentially the least elegant, way is to leverage a combination of the `BashOperator` and readily available command-line tools like `wget` or `curl` for the ftp operations and the `gsutil` command for GCS interaction. I’ve used this method quite frequently when dealing with quick turnaround projects or environments where custom Python packages are hard to manage. It keeps the dependency management simple.

Here's an example of that using `wget` and `gsutil`:

```python
from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    dag_id='ftp_to_gcs_bash',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    download_ftp_task = BashOperator(
        task_id='download_from_ftp',
        bash_command="""
            wget -m ftp://<ftp_username>:<ftp_password>@<ftp_server_address>/<remote_path_to_files>/ -P /tmp
            ls /tmp
        """
    )

    upload_gcs_task = BashOperator(
        task_id='upload_to_gcs',
        bash_command="""
            gsutil -m cp /tmp/<files_to_upload> gs://<gcs_bucket>/<gcs_destination_path>/
        """
    )

    cleanup_task = BashOperator(
        task_id='cleanup_local',
        bash_command="""
            rm /tmp/<files_to_upload>
        """
    )
    download_ftp_task >> upload_gcs_task >> cleanup_task

```

In this snippet, `<ftp_username>`, `<ftp_password>`, `<ftp_server_address>`, `<remote_path_to_files>`, `<files_to_upload>`, `<gcs_bucket>`, and `<gcs_destination_path>` placeholders are, of course, parameters you would need to customize based on your specific context. This approach is quick to set up but can become cumbersome if you need complex file handling, dynamic file path generation, or proper logging. The `wget -m` downloads the entire folder, which might not be what you want if you have a large folder, and the `ls /tmp` step is mostly for debugging purposes.

A better, more scalable method involves using python and the built-in python operators of airflow. Here's an example that utilizes the `ftplib` and `google.cloud.storage` libraries, along with the `PythonOperator`.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from ftplib import FTP
from google.cloud import storage
import os

def ftp_to_gcs(**kwargs):
    ftp_server = "<ftp_server_address>"
    ftp_user = "<ftp_username>"
    ftp_password = "<ftp_password>"
    ftp_remote_path = "<remote_path_to_files>"
    gcs_bucket_name = "<gcs_bucket>"
    gcs_destination_path = "<gcs_destination_path>"
    
    try:
      ftp = FTP(ftp_server)
      ftp.login(user=ftp_user, passwd=ftp_password)
      ftp.cwd(ftp_remote_path)
      
      file_list = []
      ftp.retrlines('NLST', file_list.append)
      
      for file in file_list:
        if file.endswith('.txt'):  #example filter, you can change as needed
            local_file_path = f"/tmp/{file}"
            with open(local_file_path, 'wb') as local_file:
              ftp.retrbinary(f'RETR {file}', local_file.write)
            
            gcs_client = storage.Client()
            bucket = gcs_client.bucket(gcs_bucket_name)
            blob = bucket.blob(os.path.join(gcs_destination_path, file))
            blob.upload_from_filename(local_file_path)
            os.remove(local_file_path)
      
      ftp.quit()
    except Exception as e:
      print(f"An Error Occured: {e}")


with DAG(
    dag_id='ftp_to_gcs_python',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:

    ftp_to_gcs_task = PythonOperator(
        task_id='ftp_to_gcs_function',
        python_callable=ftp_to_gcs
    )

```

This example is a more robust way to handle transfers.  Again, you'll need to fill in the placeholders. Crucially, this snippet demonstrates a crucial approach: using the `ftplib` library for FTP connection and download and the `google.cloud.storage` for GCS uploading. This Python approach is significantly more flexible and allows for better error handling, log management, and file filtering compared to the `BashOperator` method. It’s also much more maintainable in the long run because you’re not relying on calling external command-line utilities.

Lastly, for complex use-cases, you might consider developing a custom operator. This approach requires a more involved setup but gives maximum control. It's what I gravitate towards for more involved data engineering projects where repeated actions with specific features are required.

```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from ftplib import FTP
from google.cloud import storage
import os

class FtpToGcsOperator(BaseOperator):
    @apply_defaults
    def __init__(self, ftp_server, ftp_username, ftp_password, ftp_remote_path, gcs_bucket, gcs_destination_path, **kwargs):
        super().__init__(**kwargs)
        self.ftp_server = ftp_server
        self.ftp_username = ftp_username
        self.ftp_password = ftp_password
        self.ftp_remote_path = ftp_remote_path
        self.gcs_bucket = gcs_bucket
        self.gcs_destination_path = gcs_destination_path
    
    def execute(self, context):
        try:
            ftp = FTP(self.ftp_server)
            ftp.login(user=self.ftp_username, passwd=self.ftp_password)
            ftp.cwd(self.ftp_remote_path)
            
            file_list = []
            ftp.retrlines('NLST', file_list.append)
            
            for file in file_list:
              if file.endswith('.txt'):  #again, filter as needed.
                  local_file_path = f"/tmp/{file}"
                  with open(local_file_path, 'wb') as local_file:
                    ftp.retrbinary(f'RETR {file}', local_file.write)
                
                  gcs_client = storage.Client()
                  bucket = gcs_client.bucket(self.gcs_bucket)
                  blob = bucket.blob(os.path.join(self.gcs_destination_path, file))
                  blob.upload_from_filename(local_file_path)
                  os.remove(local_file_path)
            ftp.quit()
        except Exception as e:
            print(f"An Error Occured: {e}")
            raise
          
from airflow import DAG
from datetime import datetime
from airflow.operators.python import PythonOperator

with DAG(
    dag_id='ftp_to_gcs_custom_operator',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False
) as dag:
    
    ftp_to_gcs_task = FtpToGcsOperator(
      task_id='ftp_to_gcs_custom',
        ftp_server="<ftp_server_address>",
        ftp_username="<ftp_username>",
        ftp_password="<ftp_password>",
        ftp_remote_path="<remote_path_to_files>",
        gcs_bucket="<gcs_bucket>",
        gcs_destination_path="<gcs_destination_path>"
    )

```

Here, we have defined a `FtpToGcsOperator` which encapsulates the logic we wrote before in the `PythonOperator` example, making it reusable across different DAGs. This method makes it easy to maintain, test, and extend your data pipeline. The initialization variables are defined in such a way that any new instance will be passed to the parent class, enabling inheritance features.

For deeper understanding, consider reading "Fluent Python" by Luciano Ramalho for advanced Python concepts and "Cloud Native Patterns" by Cornelia Davis for best practices in cloud architecture. Additionally, exploring Google Cloud's official documentation for their storage and Composer products is invaluable.

In my experience, starting with a Python-based operator and using `ftplib` and the GCS SDK libraries provides the most versatile and maintainable solutions for FTP-to-GCS transfers, especially as the complexity of the project grows. The custom operator approach is suitable for specific projects where repeated functionality needs to be packaged as a component for re-use, however, you should assess if the extra complexity makes sense for your particular use case. Whichever approach you choose, careful error handling and logging are critical, and remember to always secure your FTP credentials, preferably utilizing environment variables managed by Airflow’s secrets backend rather than embedding them directly in the code.
