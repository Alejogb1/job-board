---
title: "How can I track new records in an Azure blob CSV file?"
date: "2024-12-23"
id: "how-can-i-track-new-records-in-an-azure-blob-csv-file"
---

Let's dive straight into it, shall we? Tracking changes in an Azure blob CSV file might seem simple on the surface, but it quickly becomes nuanced when you consider scalability and efficiency. Over the years, I've encountered this particular problem more times than I’d like to remember, often when dealing with data ingestion pipelines for large-scale analytics. One instance involved a complex telemetry system where new sensor readings were appended to a CSV file hourly. We needed to process *only* the new data, not the entire file every time. Just grabbing the whole thing for each processing run is, of course, highly inefficient and leads to unnecessary resource consumption.

The fundamental issue here is that CSV files, by their very nature, aren’t built for change tracking. They're simple text files, and Azure blob storage treats them as such. There’s no inherent mechanism that says “these are the new lines since last time.” So, we need to introduce some sort of external tracking mechanism. The methods that follow, I've found, work well in various situations with different tradeoffs in terms of cost and complexity.

Firstly, the simplest approach, and perhaps the most common starting point, involves **tracking the file size or last modified timestamp**. In many scenarios, particularly with append-only workflows, these two are often correlated with new data. Here’s how that could work conceptually, and in a straightforward Python snippet:

```python
import os
from azure.storage.blob import BlobServiceClient

def process_new_data_file_size(connection_string, container_name, blob_name, last_processed_size):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    blob_properties = blob_client.get_blob_properties()
    current_size = blob_properties.size

    if current_size > last_processed_size:
        print(f"File size has increased. New data available. Previous size: {last_processed_size}, Current size: {current_size}")
        downloaded_blob = blob_client.download_blob()
        blob_content = downloaded_blob.content_as_text()
        lines = blob_content.splitlines()
        new_lines = lines[int(last_processed_size/(len(lines[0]) if lines else 1)):] #approx. line split based on file size to get new lines
        for line in new_lines:
           process_line(line) #replace with your actual processing
        return current_size
    else:
        print("No new data available.")
        return last_processed_size

def process_line(line):
  print(f"Processing: {line}") #replace with your actual processing logic

#example
connection_str = "YOUR_CONNECTION_STRING"
container_str = "YOUR_CONTAINER_NAME"
blob_str = "YOUR_BLOB_NAME"
last_size = 0 # Initialize to 0 for the first run
last_size = process_new_data_file_size(connection_str, container_str, blob_str, last_size)
#on subsequent calls to this, it will use the updated value of last_size to retrieve only new content
```

In this code, we maintain `last_processed_size` outside the function call. Each time the function is invoked, we get the current blob size. If the current size is greater than the last processed size, it's inferred that the file has been updated. We use the previous size to approximate which lines to download. Note, this approach of using file size to estimate the change will fail if lines can be removed and does not guarantee to provide the correct changes; however it is lightweight and useful in many append only cases. It’s simple, it's quick, and it avoids reading the entire file when possible.

However, this method has its limitations. If new data is appended to the file without increasing the overall file size – through overwriting or truncation, for instance, this method will completely fail. We might also run into problems with very small increments to file size or slow network conditions, making the size comparison insufficient. This can also prove error-prone if your csv file contains different length rows. So, while it’s an easy starting point, you may need something more robust for production environments. This method is appropriate for use cases where files are created and appended to in a relatively straightforward manner, where new data is largely append only and not a modification of existing data.

The second approach I’ve found highly useful, and perhaps the most reliable way for production environments, is to **maintain a separate tracking file**. This file can hold information about the last processed line number or a checksum of the last processed data. While slightly more complex, this gives us much more granular control and resilience. This tracking file could be an Azure blob (often a small JSON or TXT file), or better yet, a table in an Azure database service. Here's a Python example using an Azure blob as the tracking file:

```python
import json
from azure.storage.blob import BlobServiceClient, BlobClient
import hashlib


def process_new_data_tracking_file(connection_string, container_name, blob_name, tracking_blob_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    tracking_client = blob_service_client.get_blob_client(container=container_name, blob=tracking_blob_name)

    try:
      tracking_blob = tracking_client.download_blob()
      tracking_data = json.loads(tracking_blob.content_as_text())
      last_processed_line = tracking_data.get('last_processed_line', 0)
      last_processed_hash = tracking_data.get('last_processed_hash','')
    except:
        last_processed_line = 0
        last_processed_hash = '' # default

    downloaded_blob = blob_client.download_blob()
    blob_content = downloaded_blob.content_as_text()
    lines = blob_content.splitlines()

    new_lines = lines[last_processed_line:]
    for i, line in enumerate(new_lines):
      current_hash = hashlib.sha256(line.encode()).hexdigest()
      if current_hash != last_processed_hash:
        process_line(line) # Replace with your processing logic
        last_processed_hash = current_hash
        last_processed_line +=1

    #update tracking file
    tracking_data_update = {
        'last_processed_line': last_processed_line,
        'last_processed_hash' : last_processed_hash
    }
    tracking_client.upload_blob(json.dumps(tracking_data_update), overwrite = True)


def process_line(line):
  print(f"Processing: {line}") #replace with your actual processing logic


#example usage
connection_str = "YOUR_CONNECTION_STRING"
container_str = "YOUR_CONTAINER_NAME"
blob_str = "YOUR_BLOB_NAME"
tracking_blob_str = "YOUR_TRACKING_BLOB_NAME.json"
process_new_data_tracking_file(connection_str, container_str, blob_str, tracking_blob_str)

```

Here, we first try to load the tracking file, which is a json file, and we get the `last_processed_line` and `last_processed_hash`. We then read the blob data and iterate through the lines we’ve not processed. We use a sha256 hash to determine if we have processed it. If not, we process the new line. Finally we update the tracking file with our current state. This provides much more accurate tracking even if the file size doesn’t change and ensures each new line is processed precisely once.

The tracking method provides a much more robust approach, however, it has the extra overhead of maintaining the tracking file and potentially database interactions if one chooses to store the tracking in such a location. It adds a bit more complexity, but in systems where data integrity is crucial, it pays off.

The third approach, which is more appropriate for very high-volume scenarios or when near real-time data processing is required, would be to utilize **event-driven architecture using Azure Event Grid**. When the blob changes, Event Grid triggers an event, and that event can then trigger an Azure function to process the changes. Using event grid allows for a more reactive approach. This avoids the need to poll the blob storage on a schedule.

```python
import logging
import json
import os
from azure.storage.blob import BlobServiceClient


def main(event):
    logging.info(f"event: {event}")
    connection_string = os.environ["AzureWebJobsStorage"]
    container_name = event.get('data', {}).get('containerName')
    blob_name = event.get('data', {}).get('blobName')

    if not container_name or not blob_name:
       logging.error("Invalid Event Data")
       return

    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    downloaded_blob = blob_client.download_blob()
    blob_content = downloaded_blob.content_as_text()
    lines = blob_content.splitlines()
    for line in lines:
      process_line(line)

def process_line(line):
    print(f"Processing line via Event Grid: {line}")#replace with your actual processing logic
```

Here, I show a basic Azure function, triggered by an Event Grid event. The function retrieves the blob information from the event metadata and processes the lines of the modified blob. While it's not *strictly* "tracking" in the same way as our previous examples, the event-driven nature means your function will only run when there’s something new to handle. It doesn't handle the concept of processing only new data, and it processes all data each event; however, this is more suited to realtime and high frequency processing. You could, for example, couple this approach with the tracking file mechanism described earlier.

When choosing which approach to take, think about how often new data arrives, how critical it is to process *only* the new data, what level of fault tolerance you require and what the performance implications will be. For deeper understanding of these techniques, consider exploring resources such as *Designing Data-Intensive Applications* by Martin Kleppmann for broader architectural patterns related to data processing. Also the *Microsoft Azure documentation* regarding Azure Blob Storage and related services is invaluable.

In summary, tracking new records in an Azure blob CSV isn't directly supported by the storage service itself. It requires you to implement some form of change tracking. The best solution depends heavily on your requirements and constraints. Whether using simple size comparisons, a dedicated tracking file, or an event-driven approach via Azure Event Grid, I hope you now have a solid foundation upon which to move forward with your particular scenario.
