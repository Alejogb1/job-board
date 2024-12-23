---
title: "How does S3KeySensor function in Airflow 2?"
date: "2024-12-23"
id: "how-does-s3keysensor-function-in-airflow-2"
---

, let’s unpack the functionality of the `S3KeySensor` in Apache Airflow 2. I've encountered this particular sensor quite a few times in my projects, and it has proven to be an invaluable tool for triggering downstream tasks based on the presence (or absence) of files in Amazon S3. It's more nuanced than a simple file existence check, and understanding its intricacies can save you a lot of debugging time.

First off, let’s establish what a sensor generally does in Airflow. It's a task that repeatedly checks a condition. Instead of completing a single unit of work, sensors monitor external systems and only complete when that condition is met. Think of it as an active listener rather than a direct processor. The `S3KeySensor` specifically listens for keys (objects) within an S3 bucket.

Now, moving onto the `S3KeySensor` itself. It's designed to be a polling sensor, which means it's not event-driven in the traditional sense. It periodically queries S3 to determine if the specified key exists. This is crucial because S3 itself doesn't send out notifications that a file has appeared (though you can configure it to do so, that's a separate discussion). The sensor achieves this using the boto3 library under the hood, which provides an interface to interact with AWS services.

One key parameter is `bucket_key`. This is a string or a callable (usually a Jinja template), specifying the key within the S3 bucket to check. This allows for dynamic key generation based on task execution context, which makes this sensor remarkably flexible. You might be looking for a file with a timestamped naming convention, or a file that results from a previous task. Another crucial setting is `wildcard_match`. If this is set to `True`, the `bucket_key` is treated as a prefix, and the sensor will succeed if any key in the bucket starts with that prefix. This is useful when waiting for a set of files with similar prefixes.

Another useful parameter is `check_mode`, which defaults to "exists". You can also set it to "non_exist", making the sensor wait for the absence of a file. This is less common, but can be important in scenarios like waiting for a cleanup process before starting the next steps. The `timeout` parameter dictates how long the sensor will run before giving up and marking the task as failed. The polling interval is governed by `poke_interval`, which defines the frequency of checks to S3, specified in seconds.

Let's go through some examples. Imagine a scenario where a file with a timestamped name needs to be present in S3 before a data processing job begins:

```python
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow import DAG
from datetime import datetime, timedelta

with DAG(
    dag_id='s3_key_sensor_example_1',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:

    sensor_task = S3KeySensor(
        task_id='wait_for_file',
        bucket_key="data/my_data_{{ ds }}.csv",
        bucket_name='my-s3-bucket',
        poke_interval=60, # Poll S3 every 60 seconds
        timeout=3600,    # Timeout after 1 hour
    )
```

In this example, the `bucket_key` is dynamically generated using Jinja templating, incorporating the execution date (represented by `ds`). The sensor task will repeatedly check for a file named `data/my_data_YYYY-MM-DD.csv` in the `my-s3-bucket`.

Now, let's look at a scenario involving wildcard matching. Suppose you have a series of log files that are uploaded to S3 with the common prefix, `logs/web_server_`. Let’s wait for at least one to be present:

```python
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow import DAG
from datetime import datetime, timedelta

with DAG(
    dag_id='s3_key_sensor_example_2',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:

    sensor_task = S3KeySensor(
        task_id='wait_for_logs',
        bucket_key="logs/web_server_",
        bucket_name='my-s3-bucket',
        wildcard_match=True,
        poke_interval=30,
        timeout=3600
    )
```

Here, the `wildcard_match=True` parameter instructs the sensor to check for any object that has the prefix `logs/web_server_`. It doesn’t care what follows this, as long as *something* matches that prefix.

Finally, let’s consider a more unusual scenario where we're waiting for a file to be *removed* from the bucket (again, not super common but occasionally necessary).

```python
from airflow.providers.amazon.aws.sensors.s3 import S3KeySensor
from airflow import DAG
from datetime import datetime, timedelta

with DAG(
    dag_id='s3_key_sensor_example_3',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['example'],
) as dag:

    sensor_task = S3KeySensor(
        task_id='wait_for_file_removal',
        bucket_key="temp_files/old_data.csv",
        bucket_name='my-s3-bucket',
        check_mode='non_exist',
        poke_interval=30,
        timeout=1800
    )
```

In this example, the sensor will continue to poll S3 until the key `temp_files/old_data.csv` is no longer present in the `my-s3-bucket`.

Now, a few important points I have picked up from practice. Firstly, be mindful of the polling interval. Too frequent and you might generate unnecessary API calls to AWS, potentially increasing costs. Too infrequent, and your workflow might suffer from unnecessary delays. Secondly, thoroughly test your Jinja templating if you're using dynamic key generation. Small syntax errors in the templates can be easily missed and can lead to sensor tasks perpetually waiting. Lastly, be aware of how AWS IAM permissions are set up, because the Airflow worker process running the sensor needs permission to access your S3 bucket.

For those who are keen to dive deeper into sensors and how they operate in Airflow, I would highly recommend spending time with the official Airflow documentation, especially the section on sensors. Furthermore, for a solid understanding of the interaction with S3, a thorough look through boto3's documentation for the S3 service is advisable. I also found "Designing Data-Intensive Applications" by Martin Kleppmann particularly useful for understanding the underlying distributed system considerations for this kind of operation, even though it's not exclusively focused on Airflow.
The goal is not just to get the sensor working, but to understand *why* it works and how it interacts with the underlying systems, which is often what separates proficiency from merely making something function.
