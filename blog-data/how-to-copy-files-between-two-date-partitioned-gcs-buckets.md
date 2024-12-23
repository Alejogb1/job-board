---
title: "How to copy files between two date-partitioned GCS buckets?"
date: "2024-12-23"
id: "how-to-copy-files-between-two-date-partitioned-gcs-buckets"
---

Okay, let’s tackle this. I’ve certainly seen my share of data wrangling scenarios involving google cloud storage (gcs), and copying between date-partitioned buckets is a common one, often with its own set of quirks. The key isn't just blindly copying, but doing so efficiently and reliably, especially when dealing with large datasets. I’ll break down my approach, including some code examples and things I’ve learned the hard way.

Firstly, understanding the nature of date partitioning in gcs is paramount. Typically, you'll see directory structures like `gs://my-bucket/year=2023/month=10/day=26/`. The advantage is that it makes querying and processing data much faster, but it does present a unique challenge for copying. Simply using wildcards can sometimes be inefficient or lead to unintended consequences if the partitions are highly imbalanced, or if you have metadata mixed in at different levels.

The naive approach, just using gsutil `cp -r`, while seemingly simple, often falls short when dealing with massive datasets or complex partition structures. It’s important to be specific about what you're copying, especially if the target bucket has a different partitioning scheme or you need to transform the data in transit. I learned this acutely during a project migrating a multi-petabyte dataset. We started with a simple `gsutil cp -r`, and it was a catastrophic failure of both time and resource usage – a very costly mistake.

For a reliable process, I prefer scripting the copy process, incorporating mechanisms for validation and retries, and leaning heavily on the google cloud sdk. Here's a python snippet that illustrates this:

```python
from google.cloud import storage
from google.api_core import exceptions
import datetime
import os

def copy_partition(source_bucket_name, dest_bucket_name, date_obj):
    """Copies a single date partition from one GCS bucket to another.

    Args:
        source_bucket_name: Name of the source bucket.
        dest_bucket_name: Name of the destination bucket.
        date_obj: A datetime.date object representing the date partition.
    """
    storage_client = storage.Client()
    source_bucket = storage_client.bucket(source_bucket_name)
    dest_bucket = storage_client.bucket(dest_bucket_name)

    source_prefix = f"year={date_obj.year}/month={date_obj.month:02}/day={date_obj.day:02}/"
    dest_prefix = source_prefix  # Destination can be different if needed.

    blobs = source_bucket.list_blobs(prefix=source_prefix)

    for blob in blobs:
        try:
            if not blob.name.endswith("/"): # Prevent copying the directory object itself.
                source_blob = source_bucket.blob(blob.name)
                destination_blob = dest_bucket.blob(os.path.join(dest_prefix, os.path.basename(blob.name)))
                destination_blob.rewrite(source_blob)
                print(f"Copied {blob.name} to {destination_blob.name}")
        except exceptions.NotFound as e:
            print(f"Blob not found: {blob.name} - {e}")
        except Exception as e:
            print(f"Error copying {blob.name}: {e}")

if __name__ == '__main__':
    source_bucket = "source-bucket-name" # replace with your actual source bucket name.
    destination_bucket = "destination-bucket-name"  # replace with your actual destination bucket name.
    start_date = datetime.date(2023, 10, 25)
    end_date = datetime.date(2023, 10, 28)

    current_date = start_date
    while current_date <= end_date:
         copy_partition(source_bucket, destination_bucket, current_date)
         current_date += datetime.timedelta(days=1)

```

This script takes a date range and copies each partition individually. It’s more robust because it handles file-by-file copy using `rewrite`, which is generally more efficient than downloading and re-uploading, particularly with large files. The `exceptions.NotFound` helps handle cases where files might be absent in some partitions, which can occur in real-world datasets.

However, sometimes we also want to perform data filtering while copying. For example, let's say we only want to copy files with a specific extension. Here is a modified snippet:

```python
from google.cloud import storage
from google.api_core import exceptions
import datetime
import os

def copy_partition_with_filter(source_bucket_name, dest_bucket_name, date_obj, file_extension):
    """Copies a single date partition with a file filter from one GCS bucket to another.

    Args:
        source_bucket_name: Name of the source bucket.
        dest_bucket_name: Name of the destination bucket.
        date_obj: A datetime.date object representing the date partition.
        file_extension: The file extension to filter, e.g., ".parquet".
    """
    storage_client = storage.Client()
    source_bucket = storage_client.bucket(source_bucket_name)
    dest_bucket = storage_client.bucket(dest_bucket_name)

    source_prefix = f"year={date_obj.year}/month={date_obj.month:02}/day={date_obj.day:02}/"
    dest_prefix = source_prefix

    blobs = source_bucket.list_blobs(prefix=source_prefix)

    for blob in blobs:
        try:
            if not blob.name.endswith("/") and blob.name.endswith(file_extension):
                source_blob = source_bucket.blob(blob.name)
                destination_blob = dest_bucket.blob(os.path.join(dest_prefix, os.path.basename(blob.name)))
                destination_blob.rewrite(source_blob)
                print(f"Copied {blob.name} to {destination_blob.name}")
        except exceptions.NotFound as e:
            print(f"Blob not found: {blob.name} - {e}")
        except Exception as e:
            print(f"Error copying {blob.name}: {e}")

if __name__ == '__main__':
    source_bucket = "source-bucket-name" # replace with your actual source bucket name.
    destination_bucket = "destination-bucket-name" # replace with your actual destination bucket name.
    start_date = datetime.date(2023, 10, 25)
    end_date = datetime.date(2023, 10, 28)
    target_extension = ".parquet"

    current_date = start_date
    while current_date <= end_date:
        copy_partition_with_filter(source_bucket, destination_bucket, current_date, target_extension)
        current_date += datetime.timedelta(days=1)
```

This extended version adds a `file_extension` filter in the `copy_partition_with_filter` function. You can now target specific file types, making your copy operation more specific and tailored to the target use case.

Finally, in environments where massive data migration is necessary, it's wise to think about parallelism. We can leverage google cloud functions to scale this process out. A cloud function can handle one or more date partitions, and be triggered by a pub/sub message. Let's demonstrate with a pseudo code.

```python
# Python Code for a google cloud function
from google.cloud import storage
from google.api_core import exceptions
import os
import json

def gcf_copy_partition(event, context):
    """ Google Cloud Function to copy a date partition from one bucket to another.

    Args:
        event: Event payload containing source and destination bucket information
            and date parameters in json string.
        context: Cloud Function context.
    """

    try:
        message_data = json.loads(event["data"])

        source_bucket_name = message_data["source_bucket_name"]
        dest_bucket_name = message_data["dest_bucket_name"]
        date_string = message_data["date"]

        date_obj = datetime.datetime.strptime(date_string, "%Y-%m-%d").date()

        storage_client = storage.Client()
        source_bucket = storage_client.bucket(source_bucket_name)
        dest_bucket = storage_client.bucket(dest_bucket_name)

        source_prefix = f"year={date_obj.year}/month={date_obj.month:02}/day={date_obj.day:02}/"
        dest_prefix = source_prefix

        blobs = source_bucket.list_blobs(prefix=source_prefix)

        for blob in blobs:
            try:
                 if not blob.name.endswith("/"):
                     source_blob = source_bucket.blob(blob.name)
                     destination_blob = dest_bucket.blob(os.path.join(dest_prefix, os.path.basename(blob.name)))
                     destination_blob.rewrite(source_blob)
                     print(f"Copied {blob.name} to {destination_blob.name}")
            except exceptions.NotFound as e:
                print(f"Blob not found: {blob.name} - {e}")
            except Exception as e:
                print(f"Error copying {blob.name}: {e}")

        print(f"Partition for date: {date_obj} completed.")

    except Exception as e:
        print(f"Function failed: {e}")

# And how to publish to Pub/Sub
import datetime
from google.cloud import pubsub_v1
import json

if __name__ == '__main__':
        project_id = "your-gcp-project" # replace with your actual GCP project id.
        topic_name = "your-pubsub-topic" # replace with your pubsub topic name
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(project_id, topic_name)

        source_bucket = "source-bucket-name" # replace with your actual source bucket name.
        destination_bucket = "destination-bucket-name"  # replace with your actual destination bucket name.
        start_date = datetime.date(2023, 10, 25)
        end_date = datetime.date(2023, 10, 28)
        current_date = start_date

        while current_date <= end_date:
                message = {
                     "source_bucket_name": source_bucket,
                     "dest_bucket_name": destination_bucket,
                     "date": str(current_date)
                }
                message_json = json.dumps(message).encode("utf-8")
                future = publisher.publish(topic_path, message_json)
                print(f"Published message for date {current_date}, id: {future.result()}")
                current_date += datetime.timedelta(days=1)

```
This snippet illustrates a function that can be triggered by a pub/sub message. Each message contains the source and destination bucket names and a date object. The publisher code shows how to generate these messages and send them to the pub/sub topic, which in turn triggers the cloud functions to copy the data partitions in parallel. This setup allows a considerable performance improvement.

For further exploration on gcs and its best practices, I would recommend reviewing google cloud's official documentation, particularly their guides on data transfer options. Additionally, “Designing Data-Intensive Applications” by Martin Kleppmann provides a good understanding of general data management principles. For specific google cloud sdk tips and tricks, the documentation for the `google-cloud-storage` library on pypi is indispensable. These resources have formed the base of my own experience and will provide a comprehensive view of the technology.
