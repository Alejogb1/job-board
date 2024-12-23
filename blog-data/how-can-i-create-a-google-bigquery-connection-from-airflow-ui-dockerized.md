---
title: "How can I Create a google bigquery connection from Airflow UI (Dockerized)?"
date: "2024-12-23"
id: "how-can-i-create-a-google-bigquery-connection-from-airflow-ui-dockerized"
---

Okay, let’s talk about connecting to Google BigQuery from within a Dockerized Airflow environment, specifically through the Airflow UI. This is something I’ve tackled a fair few times, and getting it set up reliably involves a few crucial steps beyond the initial 'plug-and-play' expectation. It’s not rocket science, but you have to nail the details.

First and foremost, understand that Airflow, when dockerized, is essentially a separate system from your development machine or local network, and it needs its own credentials to interact with Google Cloud Platform (GCP). The core challenge isn’t so much about 'connecting', but about *authenticating* correctly. Think of it as making sure Airflow has the keys to the kingdom, and those keys have to be passed through securely and correctly configured in the docker context.

When I first started, I tripped over the simple, yet crucial, point that using your personal GCP credentials directly within a docker container is not just insecure, it's generally not the right way to operate in a production setting or even a robust development environment. Instead, you need to use service accounts. A service account is a specific Google account that belongs to your application (in this case, Airflow) and not to any individual. It’s much safer and more manageable.

The first step, then, is to create a service account in GCP with the appropriate permissions. At minimum, for BigQuery, this will include roles like `bigquery.dataViewer`, `bigquery.jobUser`, and `bigquery.user` depending on what actions your Airflow workflows will perform – reading, querying, or writing data. Once you’ve created the service account, download its JSON key file. This file contains the private key needed for authentication. It's absolutely crucial to keep this file safe; never commit it to a repository, for example. Treat it like a password.

Next, you'll need to get this key to your Airflow docker container. I typically recommend creating a custom Dockerfile that copies the key file into the container during the build process and sets it as an environment variable that can be used by the Airflow google provider. There is another, possibly even better, method to mount the key as a volume but I find this to be slightly less ideal in terms of initial simplicity. The following snippets show both ways.

Let's break down the custom dockerfile first:

```dockerfile
# Assuming you are building from a standard airflow image
FROM apache/airflow:2.8.1

# Install any additional providers you might need
RUN pip install apache-airflow-providers-google

# Copy the service account key
COPY service_account.json /opt/airflow/
# Set an env variable to make the key accessible to the provider
ENV GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/service_account.json

# Rest of your Dockerfile setup might follow here.
```

In this snippet, `service_account.json` is the filename of the key you downloaded. This makes the key directly available to your provider. Building the docker image with this Dockerfile, you would place your service_account.json file in the same directory.

Now, for the second, volume mount method of providing the key, which might be more flexible for updates, you don’t need changes in the dockerfile, but rather rely on docker commands:
```bash
docker run -d \
    -p 8080:8080 \
    --name airflow \
    -e "AIRFLOW__CORE__EXECUTOR=LocalExecutor" \
    -v $PWD/service_account.json:/opt/airflow/service_account.json \
    -e "GOOGLE_APPLICATION_CREDENTIALS=/opt/airflow/service_account.json" \
    apache/airflow:2.8.1
```

This command, executed at your project root where `service_account.json` is located, mounts the file as a volume inside the container making it available at `/opt/airflow/service_account.json` and then sets the `GOOGLE_APPLICATION_CREDENTIALS` environment variable. I typically use the docker compose for this to manage all the services.

Now for the Airflow configuration from within the UI. You do not do this through the UI for the provider configurations, instead you configure via environment variables, as demonstrated in the docker commands. To verify, you can create an Airflow connection for BigQuery.

Assuming you have an appropriate airflow DAG file created, let’s do a basic example of a BigqueryOperator which uses this authenticated connection. Here’s the python code:

```python
from airflow import DAG
from airflow.providers.google.cloud.operators.bigquery import BigQueryExecuteQueryOperator
from datetime import datetime

with DAG(
    dag_id="bigquery_example",
    schedule=None,
    start_date=datetime(2023, 10, 26),
    catchup=False,
    tags=["example"],
) as dag:
    query_job = BigQueryExecuteQueryOperator(
        task_id="run_query",
        sql="SELECT 1;",  # This is the query to run. Replace with your own SQL.
        use_legacy_sql=False, # Standard SQL
        location = "US", # Specify your Bigquery location.
        gcp_conn_id="google_cloud_default"  # Connection id that is created by airflow
    )
```

The `gcp_conn_id` in the above code refers to `google_cloud_default` which comes pre-configured with airflow to use the environment variable `GOOGLE_APPLICATION_CREDENTIALS`.

It's important to remember that the service account has to have sufficient permissions for the operations you’re doing. If your queries involve creating or updating data, you might need to add other roles like `bigquery.dataEditor`. Check the BigQuery documentation for the exact required permissions. Also, if dealing with large datasets ensure that your environment has the necessary memory and network bandwidth for the operation. It's also essential to monitor logs for any authentication issues, as those will often point to key issues or permission problems.

For further reading, I highly recommend delving into Google’s official documentation on using service accounts with Cloud APIs. Also, explore the official Airflow documentation for the Google provider, which often has the most up-to-date configuration details. Specifically, look into the `apache-airflow-providers-google` documentation. Additionally, the book "Google Cloud Platform for Data Engineers" by Michael C. Wiles provides a very comprehensive overview that could be useful. Finally, you might also want to look at the "Docker Deep Dive" by Nigel Poulton for in depth docker insights. These resources helped me tremendously when I was initially setting up such integrations.

In summary, connecting to BigQuery from a Dockerized Airflow environment through the UI requires a secure authentication method using service account keys, correct environment variable settings, and appropriate Airflow configuration, combined with the right permissions. It’s not a trivial process, but with attention to detail, it becomes straightforward and manageable. The key is to remember that Airflow needs its own identity and that needs to be managed and controlled correctly, not hardcoded.
