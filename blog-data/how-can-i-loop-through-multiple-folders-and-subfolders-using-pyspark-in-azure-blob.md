---
title: "How can I loop through multiple folders and subfolders using PySpark in Azure Blob?"
date: "2024-12-23"
id: "how-can-i-loop-through-multiple-folders-and-subfolders-using-pyspark-in-azure-blob"
---

Okay, let's tackle this. I've spent quite a bit of time dealing with complex data ingestion pipelines, especially those involving Azure Blob storage and PySpark. Accessing files nested within multiple directories is a common scenario, and it can become quite tricky if not handled efficiently. Let me walk you through my approach, which has evolved over numerous projects, along with some crucial considerations.

The core challenge, as I see it, isn't just about listing files; it's about doing it scalably within the PySpark framework, making the most of its distributed processing capabilities. Azure Blob storage uses a flat namespace, but we can conceptually create a folder structure using prefixes within the blob names. This is important to understand as we plan our data access strategy. I've often seen naive approaches that treat blob storage like a hierarchical filesystem, leading to performance bottlenecks. We need to think differently here.

Instead of recursively navigating folders—a technique that would be very inefficient in a distributed environment—PySpark excels at working with patterns. We leverage wildcard characters to specify the blobs we need to access.

Here's the general principle: I typically start by constructing a base path, which usually points to the root container and a parent folder, if applicable. Then, I incorporate a wildcard to catch files at different folder depths. Spark is very adept at parallelizing this pattern-based file discovery process.

Let's illustrate with some concrete code. Assume, for this example, that your Azure Blob storage is structured something like this:

`container/
  year=2021/
    month=01/
      day=01/
        data_1.csv
        data_2.csv
      day=02/
        data_3.csv
        data_4.csv
    month=02/
      day=03/
        data_5.csv
  year=2022/
    month=03/
      day=04/
         data_6.csv
      day=05/
        data_7.csv`

**Snippet 1: Basic Wildcard Access**

```python
from pyspark.sql import SparkSession

def access_blob_data_wildcard(container_name, root_path, file_type):
  """
  Accesses blob data from multiple subfolders using wildcard characters.
  """

  spark = SparkSession.builder.appName("BlobAccess").getOrCreate()

  # Construct the pattern to access all files within our base path.
  file_pattern = f"wasbs://{container_name}@{storage_account}.blob.core.windows.net/{root_path}/*/*/*/*.{file_type}"
    # Replace <storage_account> with the appropriate name

  try:
      # Read the files from the defined pattern
    df = spark.read.format(file_type).load(file_pattern)
    return df
  except Exception as e:
      print(f"Error accessing blob data: {e}")
      return None
  finally:
      spark.stop()


if __name__ == '__main__':
    container = "your-container"
    base_folder = "year=*/month=*/day="
    file_type = "csv" #or "parquet" or "json" or others
    data = access_blob_data_wildcard(container, base_folder, file_type)
    if data:
        data.show(5)
```

In this code, I'm using `*` as a wildcard, which expands to any subdirectory or filename. This is a very efficient way to get a large amount of data. Importantly, spark will scan the blob path at the location specified in the path, and collect the list of files for processing. This process is very efficient as spark is able to distribute and parallelize the work. The specific structure of the path to search will depend on your organizational logic for your blob storage. For example, using the "year=2021/month=01" syntax in our path will allow us to filter data by year and month. Note, the `wasbs` protocol is required when working with Azure Blob Storage in Spark.

**Snippet 2: Filtering with a List of Subfolders**

Sometimes, you might need to selectively ingest data from specific subfolders. You can achieve that without resorting to recursive operations by specifying the complete path, or path pattern using a loop or list comprehension.

```python
from pyspark.sql import SparkSession
import re

def access_blob_data_filtered(container_name, subfolders, file_type, storage_account):
    """
    Accesses blob data from multiple subfolders.
    """
    spark = SparkSession.builder.appName("BlobAccessFiltered").getOrCreate()
    all_dataframes = []

    for subfolder in subfolders:

       file_pattern = f"wasbs://{container_name}@{storage_account}.blob.core.windows.net/{subfolder}/*.{file_type}"
        try:
            df = spark.read.format(file_type).load(file_pattern)
            all_dataframes.append(df)
        except Exception as e:
            print(f"Error accessing path {subfolder}, error {e}")

    try:
        if len(all_dataframes) > 0:
          merged_df = all_dataframes[0]
          for i in range(1, len(all_dataframes)):
             merged_df = merged_df.union(all_dataframes[i])
          return merged_df
        else:
          print("No dataframes found")
          return None
    except Exception as e:
        print(f"Error merging dataframes, error: {e}")
        return None
    finally:
        spark.stop()

if __name__ == '__main__':
    container = "your-container"
    storage_account = "your-storage-account" #Replace
    subfolders = ["year=2021/month=01/day=01", "year=2021/month=02/day=03","year=2022/month=03/day=04" ]
    file_type = "csv"
    filtered_data = access_blob_data_filtered(container,subfolders,file_type, storage_account)
    if filtered_data:
        filtered_data.show(5)
```

Here, I provide the explicit subfolders I need as a list. The code iterates over the list, constructing the full path to the specific data files, and reads the files into separate dataframes which are then unioned together into a single dataframe. This approach is useful if the user needs to collect and aggregate very specific subsets of data.

**Snippet 3: Dynamic Filtering with a Date Range (Advanced)**

In more advanced scenarios, especially when dealing with time-series data, you may need to access files based on a date range. We can combine glob patterns and Python’s date processing capabilities for this.

```python
from pyspark.sql import SparkSession
import datetime

def access_blob_data_date_range(container_name, base_path, start_date, end_date, file_type, storage_account):
    """
    Accesses blob data within a date range.
    """
    spark = SparkSession.builder.appName("BlobAccessDateRange").getOrCreate()
    all_dataframes = []

    start_dt = datetime.datetime.strptime(start_date,"%Y-%m-%d")
    end_dt = datetime.datetime.strptime(end_date, "%Y-%m-%d")
    current_dt = start_dt
    while current_dt <= end_dt:
        year = current_dt.strftime("%Y")
        month = current_dt.strftime("%m")
        day = current_dt.strftime("%d")
        subfolder = f"{base_path}/year={year}/month={month}/day={day}"
        file_pattern = f"wasbs://{container_name}@{storage_account}.blob.core.windows.net/{subfolder}/*.{file_type}"
        try:
            df = spark.read.format(file_type).load(file_pattern)
            all_dataframes.append(df)
        except Exception as e:
            print(f"Error accessing path {subfolder}, error: {e}")
        current_dt += datetime.timedelta(days=1)

    try:
        if len(all_dataframes) > 0:
          merged_df = all_dataframes[0]
          for i in range(1, len(all_dataframes)):
             merged_df = merged_df.union(all_dataframes[i])
          return merged_df
        else:
          print("No dataframes found")
          return None
    except Exception as e:
        print(f"Error merging dataframes, error: {e}")
        return None

    finally:
       spark.stop()

if __name__ == '__main__':
    container = "your-container"
    storage_account = "your-storage-account" #Replace
    base_folder = "" # or whatever base folder you need
    start_date = "2021-01-01"
    end_date = "2021-01-03"
    file_type = "csv"
    date_range_data = access_blob_data_date_range(container,base_folder,start_date, end_date, file_type, storage_account)
    if date_range_data:
        date_range_data.show(5)
```

This snippet takes a start and end date and iterates through them to dynamically generate the specific blob paths to access within that range. Again, the relevant data is read and merged. This approach is very efficient, especially when dealing with very large datasets or complex file structures as no recursive operations are being performed, and files are accessed only for the dates requested.

Regarding resources, I’d strongly advise reviewing the official PySpark documentation, particularly the sections on data sources and file handling. A good book to solidify your understanding of distributed computing is "Hadoop: The Definitive Guide" by Tom White; although it focuses on Hadoop, many concepts are directly applicable to PySpark. Also, explore "Learning Spark" by Jules Damji et al.; it's more Spark-specific and very useful. For a deep dive into Azure Storage, the official Azure documentation is invaluable, especially the part concerning blob storage and its API.

These examples and explanations cover the essence of what I’ve learned over time when working with Azure Blob Storage and PySpark. The key is to embrace pattern-based access rather than attempting recursive traversal. That's where the performance gains truly are. Remember to adapt these techniques to your particular directory structures and use cases. And, always, validate your data ingress process thoroughly.
