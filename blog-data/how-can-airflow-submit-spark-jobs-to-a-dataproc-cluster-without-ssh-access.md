---
title: "How can Airflow submit Spark jobs to a Dataproc cluster without SSH access?"
date: "2024-12-23"
id: "how-can-airflow-submit-spark-jobs-to-a-dataproc-cluster-without-ssh-access"
---

,  I've been down this path a few times, so I can offer some practical insights. The challenge of submitting Spark jobs to a Dataproc cluster from Airflow, particularly without resorting to the usual SSH gymnastics, is a common one and requires a good understanding of Google Cloud Platform's (GCP) ecosystem. It's not just about getting the job to run; it's also about doing it securely and in a maintainable manner.

The fundamental issue is that Airflow, typically running outside the Dataproc cluster's network, needs a secure and authorized way to interact with the cluster’s resources. Relying on SSH for this in production environments introduces a myriad of security and management complexities that are best avoided. Luckily, GCP provides alternative pathways that circumvent the need for direct SSH access. Specifically, we'll be focusing on utilizing the Dataproc API, which is the recommended approach.

The core strategy revolves around leveraging a combination of service accounts and the Dataproc API client. The service account that Airflow uses to interact with GCP needs the necessary permissions to create and manage jobs on the Dataproc cluster. Let's assume we already have an Airflow environment set up with a dedicated service account that possesses, at minimum, `roles/dataproc.worker` (for submitting jobs) and ideally `roles/storage.objectAdmin` (if your spark jobs read from or write to cloud storage). If not, this setup will be the first thing you'll need to tackle. Once permissions are in place, we can use the Dataproc API client library to craft and submit our Spark jobs.

I've personally implemented this approach using the official Python client library and a custom Airflow operator. Here's an illustrative breakdown, using python code snippets:

**Snippet 1: Defining a Dataproc Job Configuration**

First, we need to define the Spark job configuration. Instead of shelling out to `spark-submit`, we build a python dictionary that represents the configuration we want for the spark job that we're about to submit. This configuration specifies the main application jar file, any dependent jars, the main class, and other relevant arguments.

```python
def create_spark_job_config(jar_file, main_class, arguments, dependencies=None):
    job_config = {
        'placement': {
            'cluster_name': 'your-dataproc-cluster-name'  # Replace with your cluster name
        },
        'spark_job': {
            'jar_file_uris': [jar_file],
            'main_class': main_class,
            'args': arguments,
        }
    }
    if dependencies:
        job_config['spark_job']['jar_file_uris'].extend(dependencies)
    return job_config
```

This function constructs the job configuration, which is then passed on to the client. Note how we're specifying the 'cluster_name'. We avoid explicitly providing connection details, relying on the service account’s permissions to determine the scope of our operation.

**Snippet 2: Submitting the Job via Dataproc API**

Next, we'll use the Dataproc API client to submit the job using the configuration we created above. The core of this functionality can be embedded in a custom airflow operator.

```python
from google.cloud import dataproc_v1 as dataproc
import time

def submit_dataproc_job(job_config, project_id='your-gcp-project-id', region='your-region'): # Replace with your values
    client = dataproc.JobControllerClient()
    job = client.submit_job(
        request={
            "project_id": project_id,
            "region": region,
            "job": job_config
        }
    )
    print(f"Submitted job with id: {job.reference.job_id}")

    return job.reference.job_id

def poll_job_status(job_id, project_id='your-gcp-project-id', region='your-region'):
    client = dataproc.JobControllerClient()
    while True:
      job = client.get_job(
            request={"project_id": project_id, "region": region, "job_id": job_id}
        )
      if job.status.state in [dataproc.JobStatus.State.DONE, dataproc.JobStatus.State.ERROR, dataproc.JobStatus.State.CANCELLED]:
          return job.status.state
      time.sleep(30)

```

This code uses the `dataproc.JobControllerClient` to submit the job. It then polls the job status every 30 seconds until it's either completed successfully, errored or cancelled. You might also include timeout features and more robust error handling into this. Crucially, it does not require any SSH connections. The client authenticates based on the service account set in the environment running the code (which, in this scenario, would be the Airflow worker).

**Snippet 3: Integrating within an Airflow DAG**

Now let's integrate this into an Airflow DAG. The `PythonOperator` from airflow is used to encapsulate the submit_dataproc_job and poll_job_status logic within a custom function.

```python
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def dataproc_task(**kwargs):
  job_config = create_spark_job_config(
        'gs://your-bucket/path/to/your.jar', # Replace with your path to the jar file
        'com.example.YourMainClass', # Replace with your main class
        ['--input', 'gs://your-bucket/input.txt', '--output', 'gs://your-bucket/output'] # Replace with your arguments
      )
  job_id = submit_dataproc_job(job_config)
  final_state = poll_job_status(job_id)
  if final_state != dataproc.JobStatus.State.DONE:
      raise Exception(f"Job finished with state: {final_state}")

with DAG(
    dag_id="dataproc_no_ssh",
    start_date=datetime(2023, 1, 1),
    schedule=None,
    catchup=False,
) as dag:
    run_dataproc_job = PythonOperator(
      task_id='submit_dataproc_job',
      python_callable=dataproc_task,
  )

```

This DAG defines a single task that, when triggered, constructs a job configuration, submits the job to Dataproc, polls for its completion, and if the job doesn't succeed, raises an exception. This is a simple example, but it illustrates the full chain of operations.

In my experience, this method has proven reliable and scalable, particularly for production deployments. It aligns well with Google Cloud’s best practices and eliminates many of the operational overheads associated with SSH-based solutions. It also makes it easier to implement proper logging and monitoring, as we are interacting via a managed API. The key is to ensure proper service account management and keep the underlying job configurations portable and reusable.

For further exploration, I'd recommend consulting the official Google Cloud Dataproc documentation, paying special attention to the section on the Dataproc API. The book, “Google Cloud Platform in Action” by JJ Geewax also provides a good overview on the various services that make this setup possible. A deep dive into the `google-cloud-dataproc` Python client library’s reference documentation will be essential. Another useful resource would be the *Designing Data Intensive Applications* book by Martin Kleppmann, which can help you understand the concepts of distributed computing and how systems like dataproc work under the hood.

Remember to always prioritize security by adhering to the principle of least privilege, giving only the necessary permissions to the service account. I hope that, having walked through this setup, you'll find it a solid approach for submitting your Dataproc jobs without ssh.
