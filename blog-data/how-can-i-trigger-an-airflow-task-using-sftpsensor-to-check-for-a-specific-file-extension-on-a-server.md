---
title: "How can I trigger an Airflow task using SFTPSensor to check for a specific file extension on a server?"
date: "2024-12-23"
id: "how-can-i-trigger-an-airflow-task-using-sftpsensor-to-check-for-a-specific-file-extension-on-a-server"
---

Alright, let's talk about triggering Airflow tasks based on file extensions using an SFTPSensor. I’ve encountered this particular challenge more than a few times over the years, and it's usually a common pain point in data pipelines dealing with external systems. It's not just about ‘seeing’ a file; it's about waiting for the *correct* file before proceeding.

The core problem is that SFTPSensor only checks for the *presence* of a file or directory, not its content, name, or specific extension. So, you can’t directly configure it to trigger only on, say, `.csv` files. That said, there’s an elegant workaround using a combination of the SFTPSensor and some custom logic, and I'm going to walk you through that. Essentially, we’ll leverage the sensor’s core functionality to detect a file change and then augment that with our criteria check.

Let’s break down how this unfolds into a structured approach.

First, we'll use the `SFTPSensor` to monitor a directory, just as you normally would. We’ll be less concerned about the exact file initially, but rather any file change. The key is to understand that the sensor polls the directory according to its polling interval, and, if a file is detected or has been modified (depending on configuration), it will proceed past the sensor task in your DAG. We don't yet care about the extension here - it is simply an initial “something has changed” step.

Next, *after* the `SFTPSensor` has completed, we'll utilize a PythonOperator to perform more detailed file inspection, specifically targeting the file extension. This operator will retrieve a list of file(s) from the remote directory, filter for our desired file type and then, importantly, set an XCOM variable which can be consumed by subsequent tasks to process the file(s) or, if the filter fails, cause the dag to stop or fail gracefully.

Here's a breakdown with three Python code examples:

**Example 1: Basic SFTPSensor Setup**

This is how we might configure the sftpsensor in our dag:

```python
from airflow.providers.ssh.sensors.sftp import SFTPSensor
from airflow.models import DAG
from datetime import datetime

with DAG(
    dag_id='sftp_file_extension_trigger',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    sftp_sensor_task = SFTPSensor(
        task_id='sftp_file_check',
        path='/remote/directory/',
        sftp_conn_id='my_sftp_connection', # replace with your sftp connection
        file_pattern=".*", # this is a basic pattern, but you could use something more specific if you have other restrictions
        poke_interval=60,
        timeout=3600,
    )
```

In this first step, you can see we are not at all concerned about a particular file type, rather, we are just looking for a file appearing or being updated. We set `file_pattern` to `".*"` to pick up any file.

**Example 2: PythonOperator with Extension Check**

Now comes the crucial bit. Below, we'll craft a PythonOperator, using the `paramiko` library to list and filter the files:

```python
from airflow.operators.python import PythonOperator
from airflow.models import Variable
import paramiko
import logging
from os import path

def check_sftp_file_extension(sftp_conn_id, remote_dir, file_extension, **kwargs):
    """
    Checks for files with a specific extension in an sftp directory, setting an xcom if found
    """
    sftp_conn = Variable.get(sftp_conn_id, deserialize_json=True)
    host = sftp_conn['host']
    port = sftp_conn.get('port', 22)
    username = sftp_conn['login']
    password = sftp_conn.get('password')
    key_file = sftp_conn.get('key_file')
    key_file_password = sftp_conn.get('key_file_password')

    transport = paramiko.Transport((host, port))
    try:
        if key_file:
            transport.connect(username=username, password=password, pkey=paramiko.RSAKey.from_private_key_file(key_file, password=key_file_password))
        else:
            transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)
        files = sftp.listdir(remote_dir)
        qualified_files = []
        for file in files:
            if path.splitext(file)[1].lower() == file_extension.lower():
               qualified_files.append(path.join(remote_dir,file))
        if qualified_files:
            kwargs['ti'].xcom_push(key='qualified_files', value=qualified_files)
            logging.info(f"Found files with extension {file_extension}: {qualified_files}")
        else:
            logging.warning(f"No files found with extension {file_extension}")
    except Exception as e:
        logging.error(f"Error during file check: {e}")
        raise
    finally:
        if transport.is_active():
            transport.close()
        if sftp:
          sftp.close()

    return None


with DAG(
    dag_id='sftp_file_extension_trigger',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    sftp_sensor_task = SFTPSensor(
        task_id='sftp_file_check',
        path='/remote/directory/',
        sftp_conn_id='my_sftp_connection',
        file_pattern=".*",
        poke_interval=60,
        timeout=3600,
    )

    check_extension_task = PythonOperator(
        task_id='check_file_extension',
        python_callable=check_sftp_file_extension,
        op_kwargs={
            'sftp_conn_id':'my_sftp_connection',
            'remote_dir': '/remote/directory/',
            'file_extension': '.csv'
        }
    )

    sftp_sensor_task >> check_extension_task
```

Here, the `check_sftp_file_extension` function connects to the SFTP server, lists the contents of the directory specified, and filters the results using `path.splitext()`. It then pushes the list of matching files (or an empty list if none are found) to an XCOM variable called `qualified_files`. This is crucial because now, downstream tasks can access that variable and act upon it.

**Example 3: Downstream Task Consumption (optional)**

Here's a simple example of how a downstream task might consume the output from the `check_extension_task`:

```python
from airflow.operators.python import PythonOperator
from airflow.models import DAG
from datetime import datetime

def process_qualified_files(**kwargs):
    """
    Processes files fetched from previous task
    """
    ti = kwargs['ti']
    qualified_files = ti.xcom_pull(key='qualified_files', task_ids='check_file_extension')
    if qualified_files:
        for file in qualified_files:
            print(f"Processing file: {file}")
            #add your specific processing logic here
    else:
        print("No files found")


with DAG(
    dag_id='sftp_file_extension_trigger',
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
) as dag:

    sftp_sensor_task = SFTPSensor(
        task_id='sftp_file_check',
        path='/remote/directory/',
        sftp_conn_id='my_sftp_connection',
        file_pattern=".*",
        poke_interval=60,
        timeout=3600,
    )

    check_extension_task = PythonOperator(
        task_id='check_file_extension',
        python_callable=check_sftp_file_extension,
        op_kwargs={
            'sftp_conn_id':'my_sftp_connection',
            'remote_dir': '/remote/directory/',
            'file_extension': '.csv'
        }
    )

    process_files_task = PythonOperator(
        task_id='process_files',
        python_callable=process_qualified_files,
    )

    sftp_sensor_task >> check_extension_task >> process_files_task
```

In this final snippet, we have added a new task, `process_files_task` which pulls the XCOM variable `qualified_files` from the previous task, and then iterates through that list of files performing specific processing logic on each file.

Now, regarding further reading, you'll want to dive deeper into Paramiko's documentation. Paramiko handles SSH and SFTP connections securely, and that will give you a really solid foundation. Look at sections pertaining to `SFTPClient`. Secondly, I strongly suggest looking into the Airflow documentation on providers, specifically how to configure and manage sftp connections as well as how the SFTPSensor works. Finally, a good resource, although not specific to Airflow, would be the *Python Cookbook* by David Beazley and Brian K. Jones. It is an invaluable companion for any Python development tasks like file operations and system interactions we’ve touched on here.
I hope this breakdown is both thorough and helpful. Let me know if you have any more questions – always happy to share what I've learned.
