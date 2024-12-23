---
title: "How can I loop through multiple folders and subfolders using Pyspark in Azure Blob container (ADLS Gen2)?"
date: "2024-12-23"
id: "how-can-i-loop-through-multiple-folders-and-subfolders-using-pyspark-in-azure-blob-container-adls-gen2"
---

Let’s address that, shall we? Handling nested directory structures within an Azure Blob container using pyspark is a fairly common challenge, and I've certainly had my share of battles with it. I recall a past project where we were processing sensor data from a network of devices, each dumping into its own subfolder hierarchy. The sheer volume of data, and the inherent nesting, made simply listing files impractical. A purely recursive approach would’ve been a performance bottleneck given the scale. So, let's explore a more efficient and robust way to tackle this.

The core issue revolves around how pyspark interacts with the underlying storage abstraction provided by Hadoop and, specifically, how it handles directory listings. When you point pyspark to a top-level directory within your adls gen2 container, it *doesn't* automatically dive into subdirectories. It treats that initial directory path as the singular input path. We need to explicitly instruct it to recurse. The `hadoopPath` parameter and its wildcard functionality become our primary tools here.

A straightforward method, although sometimes less granular, involves using a wildcard to target all files within subdirectories. For example, if your storage structure looks like this: `/data/year=2023/month=01/day=01/file1.csv`, `/data/year=2023/month=01/day=02/file2.csv`, and so on, we can use a wildcard like `/data/*/*/*` to reach all files beneath the `/data` root.

However, this can become problematic if there are non-data-related files or subdirectories mixed in. It also doesn’t give you detailed control over traversal. More importantly, relying *solely* on wildcards means we're implicitly trusting that all subfolders follow the same depth and naming conventions, which is not always a given in real-world scenarios.

The most effective approach, in my experience, hinges on strategically using `spark.read.csv` (or parquet, json, whatever your file format might be) along with the underlying hadoop file system path. The key is to leverage the wildcards, and the `spark.sparkContext.hadoopFile` for file discovery. Let's look at specific code examples to clarify:

**Example 1: Simple Wildcard Approach**

This approach is suitable when your directory depth is consistent and well-defined. It's good for a quick, broad sweep of files.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("adls_wildcard").getOrCreate()

container_path = "abfss://<container_name>@<storage_account_name>.dfs.core.windows.net/data"
input_path = f"{container_path}/*/*/*" # three-level deep directory structure

df = spark.read.csv(input_path, header=True, inferSchema=True)
df.show()
```
In this snippet, we're leveraging the wildcard pattern to instruct spark to discover files within three nested layers under the 'data' folder. If your depth varies, you'll need to adjust the pattern accordingly. It's a fast and straightforward, if not always the most flexible way of getting your file paths. This approach works if the structure is predictable and the depth is consistent.

**Example 2: Dynamic Path Generation with `hadoopFile`**

This technique gives you greater control, it lets you dynamically construct file paths based on metadata. This is particularly important when you need to, for instance, only process a subset of files, or if you want to process files based on certain criteria.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("adls_dynamic_path").getOrCreate()
container_path = "abfss://<container_name>@<storage_account_name>.dfs.core.windows.net/data"

# using the hadoop file system object to list all directories within data
fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration)
path = spark._jvm.org.apache.hadoop.fs.Path(container_path)
sub_dirs = fs.listStatus(path)

all_files = []

for dir_status in sub_dirs:
    if dir_status.isDirectory():  # avoid listing files at this level
        sub_dir_path = dir_status.getPath().toString()
        year_dirs = fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(sub_dir_path))
        for year_dir in year_dirs:
            if year_dir.isDirectory():
                year_dir_path = year_dir.getPath().toString()
                month_dirs = fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(year_dir_path))
                for month_dir in month_dirs:
                  if month_dir.isDirectory():
                      month_dir_path = month_dir.getPath().toString()
                      file_statuses = fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(month_dir_path))
                      for file_status in file_statuses:
                        if not file_status.isDirectory():
                             all_files.append(file_status.getPath().toString())
df = spark.read.csv(all_files, header = True, inferSchema = True)
df.show()

```

Here, we directly interact with the Hadoop file system through the spark context's jvm bindings. This allows us to list directories iteratively and create a list of files, which we then provide as input to `spark.read.csv`. This provides fine-grained control. In this example, we are still working with a consistent 3-level structure, but the loop can be easily modified to suit more complex and varied structures. It illustrates the idea of dynamically discovering file paths rather than relying on static wildcard patterns.

**Example 3: Using a filter**

A more refined approach incorporates a filter using the `hadoopFile` to include only files that match a specific naming criteria, this is particularly helpful when dealing with diverse data sets.

```python
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name

spark = SparkSession.builder.appName("adls_filtered_path").getOrCreate()

container_path = "abfss://<container_name>@<storage_account_name>.dfs.core.windows.net/data"


fs = spark._jvm.org.apache.hadoop.fs.FileSystem.get(spark._jsc.hadoopConfiguration)
path = spark._jvm.org.apache.hadoop.fs.Path(container_path)
sub_dirs = fs.listStatus(path)

file_paths = []
for dir_status in sub_dirs:
    if dir_status.isDirectory():
       sub_dir_path = dir_status.getPath().toString()
       year_dirs = fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(sub_dir_path))
       for year_dir in year_dirs:
            if year_dir.isDirectory():
                year_dir_path = year_dir.getPath().toString()
                month_dirs = fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(year_dir_path))
                for month_dir in month_dirs:
                   if month_dir.isDirectory():
                      month_dir_path = month_dir.getPath().toString()
                      file_statuses = fs.listStatus(spark._jvm.org.apache.hadoop.fs.Path(month_dir_path))
                      for file_status in file_statuses:
                        if not file_status.isDirectory() and "sensor" in file_status.getPath().getName():
                             file_paths.append(file_status.getPath().toString())


df = spark.read.csv(file_paths, header = True, inferSchema = True)
df = df.withColumn("filename", input_file_name())
df.show()
```
This example builds on the previous one by incorporating a filter during the file path discovery phase. Here, we only consider files that contain 'sensor' in the filename. This offers much more precise control over which data gets loaded, a vital aspect of any production data processing pipeline. We're also incorporating `input_file_name()` to enrich the dataframe with the source filename, for audit or provenance reasons.

For in-depth information, I recommend consulting "Hadoop: The Definitive Guide" by Tom White. It provides a comprehensive overview of the Hadoop file system and related concepts. Another essential resource is the Apache Spark documentation itself, particularly the sections detailing the Spark Context and data source APIs, which provides the nitty-gritty on file handling. Additionally, while not a book, the documentation for the Azure SDK for python will give you insights into how the library interacts with the adls layer. Lastly, the paper “MapReduce: Simplified Data Processing on Large Clusters” by Dean and Ghemawat is also very relevant as it explains the foundational concept that pyspark is built on.

The key takeaway here isn’t just about the mechanics of using wildcard patterns or `hadoopFile`; it's about understanding *how* spark interacts with the underlying filesystem to effectively ingest data from your adls gen2 containers. Having that, and practicing with these different methods will make you more efficient at dealing with the complexities of real-world data pipelines.
