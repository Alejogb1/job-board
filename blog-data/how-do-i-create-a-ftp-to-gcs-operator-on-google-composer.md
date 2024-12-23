---
title: "How do I create a FTP to GCS operator on Google Composer?"
date: "2024-12-23"
id: "how-do-i-create-a-ftp-to-gcs-operator-on-google-composer"
---

 I’ve certainly spent my fair share of time moving data around, and transitioning from FTP to Google Cloud Storage (GCS) via Composer is a common enough need. It's not always straightforward, but understanding the underlying pieces will make your life much easier. Essentially, you're looking to build an Apache Airflow operator that automates the process of transferring files from an FTP server to a GCS bucket within your Google Composer environment. Forget manual uploads; that’s what we’re moving away from.

When I first encountered this a few years back, we had a legacy system pushing out nightly data dumps via FTP. Integrating that with our cloud-based analytics pipelines was crucial, and that involved precisely what you're asking. We ended up crafting a custom operator because the existing ones didn't quite fit our specific requirements. It required a decent amount of thought and quite a bit of iterative testing, so trust me, I get the problem you're facing.

The core challenge revolves around orchestrating two distinct interactions: one with an FTP server, the other with GCS. Airflow, as you know, excels at handling these kinds of complex, multi-stage operations. To construct an appropriate operator, you’ll primarily need the `ftplib` library for FTP interaction and the `google-cloud-storage` library for GCS access. These are usually readily available in a standard Composer environment.

Let’s break this down into the key steps of building that operator, alongside concrete code examples:

**1. Custom Operator Foundation:**

First, you’ll need to define a new class that inherits from Airflow’s `BaseOperator`. This will encapsulate the functionality of your FTP-to-GCS transfer. We'll also want to parametrize this for flexibility.

```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from ftplib import FTP
from google.cloud import storage
import os

class FTPtoGCSOperator(BaseOperator):

    @apply_defaults
    def __init__(self,
                 ftp_host,
                 ftp_user,
                 ftp_password,
                 ftp_remote_path,
                 gcs_bucket,
                 gcs_blob_prefix,
                 *args, **kwargs):

        super(FTPtoGCSOperator, self).__init__(*args, **kwargs)
        self.ftp_host = ftp_host
        self.ftp_user = ftp_user
        self.ftp_password = ftp_password
        self.ftp_remote_path = ftp_remote_path
        self.gcs_bucket = gcs_bucket
        self.gcs_blob_prefix = gcs_blob_prefix

    def execute(self, context):
        ftp = FTP(self.ftp_host)
        ftp.login(user=self.ftp_user, passwd=self.ftp_password)
        ftp.cwd(self.ftp_remote_path)

        files = ftp.nlst()

        storage_client = storage.Client()
        bucket = storage_client.bucket(self.gcs_bucket)

        for file_name in files:
            if not file_name.startswith('.'): #Avoid hidden files
               self._upload_file(ftp, file_name, bucket)

        ftp.quit()


    def _upload_file(self, ftp, file_name, bucket):
        blob_name = f"{self.gcs_blob_prefix}/{file_name}"
        blob = bucket.blob(blob_name)

        local_file_path = f"/tmp/{file_name}" #Temporary local file
        with open(local_file_path, 'wb') as local_file:
            ftp.retrbinary(f'RETR {file_name}', local_file.write)

        blob.upload_from_filename(local_file_path)
        os.remove(local_file_path) #Clean up the temp file
```

This basic structure handles authentication, listing files, and a barebones file upload. We’ll refine it further below, but this sets the stage. Notice that the operator takes the essential credentials and paths as parameters during initialization. The `execute` method is where the core logic lives, and it's broken down into a separate `_upload_file` method.

**2. Adding Error Handling & Logging:**

Proper error handling is paramount, especially when dealing with file transfers across different systems. Let’s add that into the mix along with detailed logging so you can track what is happening.

```python
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
from ftplib import FTP, all_errors as ftperrors
from google.cloud import storage
import os
import logging

class FTPtoGCSOperator(BaseOperator):

    @apply_defaults
    def __init__(self,
                 ftp_host,
                 ftp_user,
                 ftp_password,
                 ftp_remote_path,
                 gcs_bucket,
                 gcs_blob_prefix,
                 *args, **kwargs):

        super(FTPtoGCSOperator, self).__init__(*args, **kwargs)
        self.ftp_host = ftp_host
        self.ftp_user = ftp_user
        self.ftp_password = ftp_password
        self.ftp_remote_path = ftp_remote_path
        self.gcs_bucket = gcs_bucket
        self.gcs_blob_prefix = gcs_blob_prefix

    def execute(self, context):
        logging.info(f"Starting transfer from FTP: {self.ftp_host}:{self.ftp_remote_path} to GCS: {self.gcs_bucket}/{self.gcs_blob_prefix}")
        ftp = FTP() # create instance here.
        try:
            ftp.connect(self.ftp_host)
            ftp.login(user=self.ftp_user, passwd=self.ftp_password)
            ftp.cwd(self.ftp_remote_path)

            files = ftp.nlst()
            logging.info(f"Found {len(files)} file(s) on FTP server.")

            storage_client = storage.Client()
            bucket = storage_client.bucket(self.gcs_bucket)

            for file_name in files:
                if not file_name.startswith('.'): #Avoid hidden files
                   self._upload_file(ftp, file_name, bucket)

            logging.info(f"Transfer complete from FTP: {self.ftp_host}:{self.ftp_remote_path} to GCS: {self.gcs_bucket}/{self.gcs_blob_prefix}")

        except ftperrors as e:
            logging.error(f"FTP Error: {e}")
            raise
        except Exception as e:
            logging.error(f"General Error: {e}")
            raise
        finally:
            if ftp.sock: # check socket existence first
                ftp.quit()

    def _upload_file(self, ftp, file_name, bucket):
        logging.info(f"Transferring file: {file_name}")
        blob_name = f"{self.gcs_blob_prefix}/{file_name}"
        blob = bucket.blob(blob_name)
        local_file_path = f"/tmp/{file_name}"

        try:
            with open(local_file_path, 'wb') as local_file:
                 ftp.retrbinary(f'RETR {file_name}', local_file.write)

            blob.upload_from_filename(local_file_path)
            logging.info(f"Successfully uploaded {file_name} to {blob_name}")

        except ftperrors as e:
              logging.error(f"FTP Error transferring {file_name}: {e}")
              raise

        except Exception as e:
            logging.error(f"Error uploading {file_name}: {e}")
            raise
        finally:
             os.remove(local_file_path)
```

With the additions, we wrapped the FTP interactions in a `try...except` block, catching `ftplib` specific exceptions. We've also added logging statements to track file progress and flag potential errors, both at the task and individual file level. This will aid significantly in debugging and monitoring. Note that I've also added a `finally` clause in the `execute` function to ensure ftp closes even in error conditions and added `ftp.sock` check to ensure ftp instance is open. The `_upload_file` function now has a similar try...except block.

**3. Utilizing in a DAG:**

Finally, let's see how you’d use this operator in an Airflow DAG:

```python
from airflow import DAG
from airflow.utils.dates import days_ago
from your_custom_operators import FTPtoGCSOperator #Ensure this import matches your module path

default_args = {
    'owner': 'me',
    'start_date': days_ago(1),
}

with DAG('ftp_to_gcs_dag',
         default_args=default_args,
         schedule_interval=None,
         catchup=False) as dag:


    transfer_files = FTPtoGCSOperator(
        task_id='transfer_ftp_files',
        ftp_host='your_ftp_host',
        ftp_user='your_ftp_user',
        ftp_password='your_ftp_password',
        ftp_remote_path='/remote/ftp/path',
        gcs_bucket='your-gcs-bucket',
        gcs_blob_prefix='ftp_data',
    )
```

You’ll need to replace the placeholder strings with your specific FTP and GCS details. This DAG demonstrates a basic setup: A single task using our custom operator. The important bit here is how you are passing your parameters to the operator constructor, making this operator re-usable across many similar data transfer tasks.

To enhance your understanding beyond these snippets, I strongly suggest consulting "Python Cookbook" by David Beazley and Brian K. Jones, especially for deeper dives into networking and file handling techniques in Python, and "Cloud Native Data Pipelines" by Matt Fuller and Mark Grover for a broader understanding of data pipeline architectures, including considerations for error handling and performance optimization. Furthermore, the official Apache Airflow documentation is invaluable for more details on creating custom operators. You could also find useful examples on Github where many users might share similar use cases.

Creating an operator like this takes a little time upfront, but you'll find it incredibly beneficial for robust, automated data pipelines. It brings a crucial layer of control and observability to your data transfers, making life much easier in the long run. Let me know if anything is unclear, or if there's any specific variation you would like to explore.
