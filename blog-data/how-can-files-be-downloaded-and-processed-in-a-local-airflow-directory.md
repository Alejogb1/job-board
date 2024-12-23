---
title: "How can files be downloaded and processed in a local Airflow directory?"
date: "2024-12-23"
id: "how-can-files-be-downloaded-and-processed-in-a-local-airflow-directory"
---

Alright, let's talk file downloads and processing within an airflow environment. It's a situation I’ve encountered more times than I'd care to count, and there are nuances that aren't always obvious at first glance. The core idea is pretty straightforward: get the file from an external location, stash it somewhere airflow can access locally, and then operate on it. But like most things in distributed systems, the devil is in the details. Over the years, i've seen a lot of variations, from simple single-file pulls to complex multi-source ingestions. So let's break this down methodically, starting with the basic workflow, and moving towards more intricate patterns.

At the heart of this process, you'll find the `airflow.providers.http.operators.http.HttpOperator` (or its equivalents for other protocols like sftp, or even cloud storage). Essentially, you're using these operators to transfer files into the airflow environment. The key thing to remember is that airflow tasks run in worker environments, so it’s essential to ensure that the downloaded files are within the accessible filesystem of the worker executing your task. The most common strategy involves downloading to the local filesystem of a worker, often within a directory dedicated to the execution of the dag, which can be found under the `airflow.conf` configuration parameters `base_log_folder` and `dags_folder`.

The first, and probably most common issue, is managing cleanup. Airflow's default behavior doesn't automatically remove these downloaded files once a task completes. This can quickly lead to a disk space issue, particularly with larger files or multiple daily runs. The most basic approach is to add a cleanup step to your dag after the processing completes. This cleanup involves a `BashOperator` executing a simple `rm` command. Here's an example of how that could look, including a small processing task to simulate real work:

```python
from airflow import DAG
from airflow.providers.http.operators.http import HttpOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

with DAG(
    dag_id='http_download_process_cleanup',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=['example'],
) as dag:
    download_file = HttpOperator(
        task_id='download_file',
        http_conn_id='my_http_conn',
        endpoint='/myfile.csv',
        method='GET',
        log_response=True,
        response_check=lambda response: response.status_code == 200,
        out_filename='/tmp/downloaded_file.csv'
    )

    process_file = BashOperator(
        task_id='process_file',
        bash_command='''
            sleep 5 # Simulate some processing
            echo "File processed." >> /tmp/processed.log
            cat /tmp/downloaded_file.csv  > /tmp/processed_copy.csv
        '''
    )


    cleanup = BashOperator(
        task_id='cleanup_files',
        bash_command='rm /tmp/downloaded_file.csv /tmp/processed_copy.csv'
    )

    download_file >> process_file >> cleanup
```

In this snippet, we're downloading a file `/myfile.csv` from a pre-configured http connection defined in airflow’s connections called 'my_http_conn' (which has to be set up in the airflow UI or through an env variable). The downloaded file is saved as `/tmp/downloaded_file.csv`. The `process_file` task then simulates doing some work with this file, and lastly, the `cleanup` task removes the temporary files created. This setup is a decent starting point but it doesn’t handle situations when you want to handle files with unique names.

For scenarios where the filename changes dynamically, such as including a date or timestamp in the download URL, you need a more sophisticated approach. Here, you can employ Jinja templating within your `HttpOperator`. This allows you to dynamically generate the URL, and also ensures the output filename is unique. Let’s say, for example, you want to download daily files named as `data_YYYYMMDD.csv`. You could change your code like so:

```python
from airflow import DAG
from airflow.providers.http.operators.http import HttpOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta

with DAG(
    dag_id='http_download_dynamic_filename',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=['example'],
) as dag:
    download_file = HttpOperator(
        task_id='download_file',
        http_conn_id='my_http_conn',
        endpoint='/data_{{ dag_run.logical_date.strftime("%Y%m%d") }}.csv',
        method='GET',
        log_response=True,
        response_check=lambda response: response.status_code == 200,
        out_filename='/tmp/data_{{ dag_run.logical_date.strftime("%Y%m%d") }}.csv'
    )


    process_file = BashOperator(
        task_id='process_file',
        bash_command='''
            sleep 5
            echo "File processed." >> /tmp/processed.log
            cat /tmp/data_{{ dag_run.logical_date.strftime("%Y%m%d") }}.csv > /tmp/processed_copy_{{ dag_run.logical_date.strftime("%Y%m%d") }}.csv
        '''
    )


    cleanup = BashOperator(
        task_id='cleanup_files',
        bash_command='''
            rm /tmp/data_{{ dag_run.logical_date.strftime("%Y%m%d") }}.csv \
            /tmp/processed_copy_{{ dag_run.logical_date.strftime("%Y%m%d") }}.csv
        '''
    )

    download_file >> process_file >> cleanup

```

Notice the usage of `{{ dag_run.logical_date.strftime("%Y%m%d") }}`. This is a Jinja template that provides the logical date of the current dag run formatted as "YYYYMMDD". It's crucial to remember that `logical_date` reflects when the dag run *should have* run, not necessarily when it did, this is a fundamental distinction within airflow scheduling. This setup ensures that each daily run fetches the appropriate file and stores it locally with a unique filename. In a real production setting, the `/tmp` directory may not be ideal since it can be cleaned unexpectedly by the host OS. It’s better to use a directory managed by airflow using the `dags_folder` or a custom designated location.

Now, consider more elaborate scenarios where you might have a sequence of files to download, or perhaps need to perform additional checks after a download. In these cases, it's often useful to use PythonOperators. These operators give you the full power of Python to control the workflow. A python operator can be used to download the file using the `requests` library, which offers more flexibility. This would enable you to handle complex headers, status codes, and even implement retry logic if downloads fail, while also allowing for dynamic naming conventions.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
from datetime import timedelta
import requests
import os

def download_file_python(**context):
    target_date = context['dag_run'].logical_date.strftime("%Y%m%d")
    url = f"http://my-server.com/data_{target_date}.csv"
    local_file = f"/tmp/data_{target_date}.csv"

    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
    with open(local_file, 'wb') as fd:
        for chunk in response.iter_content(chunk_size=8192):
            fd.write(chunk)
    context['ti'].xcom_push(key='local_file', value=local_file)


def process_file_python(**context):
    local_file = context['ti'].xcom_pull(task_ids='download_file', key='local_file')
    print(f"Processing file: {local_file}")
    with open(local_file, 'r') as infile, open(local_file.replace(".csv",".processed.csv"), 'w') as outfile:
        for line in infile:
            outfile.write(line.upper())
    context['ti'].xcom_push(key='processed_file', value=local_file.replace(".csv",".processed.csv"))



def cleanup_files_python(**context):
    downloaded_file = context['ti'].xcom_pull(task_ids='download_file', key='local_file')
    processed_file = context['ti'].xcom_pull(task_ids='process_file', key='processed_file')
    os.remove(downloaded_file)
    os.remove(processed_file)


with DAG(
    dag_id='http_download_python_operator',
    start_date=days_ago(1),
    schedule_interval=None,
    catchup=False,
    tags=['example'],
) as dag:
    download_file = PythonOperator(
        task_id='download_file',
        python_callable=download_file_python
    )


    process_file = PythonOperator(
        task_id='process_file',
        python_callable=process_file_python
    )


    cleanup = PythonOperator(
        task_id='cleanup_files',
        python_callable=cleanup_files_python
    )

    download_file >> process_file >> cleanup
```
This example utilizes `xcom` to pass the file path between the tasks. `xcom` stands for "cross communication," it is a way for tasks to communicate with each other. The python operator gives you much finer control but does come with the responsibility to manage errors and file cleanup explicitly.

To deepen your understanding, I'd strongly suggest exploring "Designing Data-Intensive Applications" by Martin Kleppmann for broader concepts on handling large datasets. Also, "Data Pipelines Pocket Reference" by James Densmore provides practical insights into building and maintaining data workflows. If you’re dealing with web data, it would be useful to familiarize yourself with the `requests` library’s documentation thoroughly.

In summary, you've got several options to download and process files within an airflow setup. Start with simpler operators if the use case allows, but don't hesitate to move to PythonOperators when the complexity demands it. Always bear in mind the need for proper cleanup and a robust error-handling strategy, which are crucial for reliable and sustainable airflow deployments.
