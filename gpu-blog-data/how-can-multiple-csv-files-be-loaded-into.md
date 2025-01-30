---
title: "How can multiple CSV files be loaded into a TensorFlow Federated dataset?"
date: "2025-01-30"
id: "how-can-multiple-csv-files-be-loaded-into"
---
Data aggregation across disparate sources is a fundamental challenge in federated learning, particularly when dealing with heterogeneous data formats like CSV files. Constructing a TensorFlow Federated (TFF) dataset from multiple CSV files requires a careful pipeline, as TFF expects data to be in a structured, client-partitioned format rather than as a collection of files. In my experience, working on a federated recommender system, I encountered this issue extensively; raw user interaction logs were provided as a series of daily CSV files, necessitating a robust solution for TFF compatibility.

The core problem lies in bridging the gap between file-based storage and TFF's expectation of a `tf.data.Dataset` per client. TFF uses datasets that are pre-partitioned, where each dataset represents the data belonging to a specific client in the federated system. The approach I’ve found most effective involves four key stages: 1) identifying client identifiers within the CSV files (either directly or through a mapping), 2) creating a function to read and parse a single CSV, producing a `tf.data.Dataset`, 3) grouping the files by client, and 4) generating a client-specific `tf.data.Dataset` by concatenating datasets from that client's CSV files.

Let's first consider the scenario where each CSV filename directly encodes the client ID, which simplifies the process significantly. For example, files named “client_a.csv”, “client_b.csv”, etc. In these cases, file parsing becomes simpler. The process is to create a parsing function:

```python
import tensorflow as tf
import pandas as pd
import os

def create_dataset_from_csv(filepath):
  """
  Reads a CSV file into a tf.data.Dataset. Assumes a header row exists.

  Args:
    filepath: The path to the CSV file.

  Returns:
    A tf.data.Dataset containing the CSV data.
  """
  
  try:
        df = pd.read_csv(filepath)
  except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None


  ds = tf.data.Dataset.from_tensor_slices(dict(df))
  return ds

# Example Usage for a single file
example_path = 'client_a.csv'
with open(example_path, 'w') as f:
  f.write("feature1,feature2,label\n")
  f.write("1,2,0\n")
  f.write("3,4,1\n")

single_client_ds = create_dataset_from_csv(example_path)

if single_client_ds:
  for element in single_client_ds.take(2):
    print(element)

os.remove(example_path)
```

This code snippet demonstrates the creation of a `tf.data.Dataset` from a single CSV file using pandas to read the CSV into a dataframe, then creating slices from the dictionary. Note that I have added error handling here to account for file issues, which was often a real-world issue for my data pipeline. The `take(2)` in the print section demonstrates how to access the contents for debugging. This `create_dataset_from_csv` function constitutes the core of the data loading process.

Now let's tackle a scenario where you have multiple CSV files per client, rather than a 1:1 relationship, i.e, multiple interaction logs for a given day or week.  In this case, we must first map each file to the client it belongs to. This is accomplished using glob to list all csv files, and regex to extract the client ID. Suppose your files are named like `client_a_day1.csv`, `client_a_day2.csv`, `client_b_day1.csv`, etc. Here is how one might load and partition the data:

```python
import glob
import re

def group_files_by_client(filepaths, pattern):
  """
  Groups CSV files by client ID using regex

  Args:
    filepaths: A list of file paths to CSV files.
    pattern: Regex pattern for extracting client IDs from file names

  Returns:
     A dict where keys are client IDs, and values are lists of file paths
     associated with that client.
  """

  client_files = {}

  for filepath in filepaths:
    match = re.search(pattern, filepath)
    if match:
      client_id = match.group(1)
      if client_id not in client_files:
        client_files[client_id] = []
      client_files[client_id].append(filepath)
  return client_files


# Example usage
file_pattern = 'client_([a-z]+)_day\d+.csv'
example_filepaths = ['client_a_day1.csv','client_a_day2.csv','client_b_day1.csv','client_c_day1.csv']

for filepath in example_filepaths:
    with open(filepath, 'w') as f:
      f.write("feature1,feature2,label\n")
      f.write("1,2,0\n")
      f.write("3,4,1\n")


client_mapping = group_files_by_client(example_filepaths, file_pattern)
print(client_mapping)

for filepath in example_filepaths:
  os.remove(filepath)
```

This function utilizes a regex pattern to identify and extract client IDs from filenames, and groups files accordingly.  The print statement shows how the mapping is created. Now to finalize the dataset creation, the `create_dataset_from_csv` function will be applied in conjunction with the `group_files_by_client` function to finalize the TFF dataset.

The next step involves creating a TFF compatible dataset from client-grouped files:

```python

def create_federated_dataset_from_csv_files(file_paths, client_id_pattern):
    """
     Generates a TFF dataset from file paths to csv files
     Args:
       file_paths: a list of full file paths to csv files
       client_id_pattern: Regex to extract client id from file path
     Returns:
      A list of tf.data.Datasets, one per client.
     """
    client_files = group_files_by_client(file_paths, client_id_pattern)
    client_datasets = []

    for client_id, file_paths in client_files.items():
        client_dataset = None
        for file_path in file_paths:
            single_file_dataset = create_dataset_from_csv(file_path)
            if single_file_dataset: # Add this check to ensure dataset was made
                if client_dataset is None:
                    client_dataset = single_file_dataset
                else:
                    client_dataset = client_dataset.concatenate(single_file_dataset)
        if client_dataset is not None:
            client_datasets.append(client_dataset)


    return client_datasets



# Example Usage
file_paths = ['client_a_day1.csv','client_a_day2.csv','client_b_day1.csv','client_c_day1.csv']
client_id_pattern = 'client_([a-z]+)_day\d+.csv'

for filepath in file_paths:
    with open(filepath, 'w') as f:
      f.write("feature1,feature2,label\n")
      f.write("1,2,0\n")
      f.write("3,4,1\n")
federated_datasets = create_federated_dataset_from_csv_files(file_paths, client_id_pattern)

if federated_datasets:
    print(f"Generated {len(federated_datasets)} client datasets.")
    for idx, dataset in enumerate(federated_datasets):
      print(f"Client {idx}:")
      for element in dataset.take(1):
        print(element)

for filepath in file_paths:
    os.remove(filepath)
```

This code defines the full pipeline, using `group_files_by_client` and `create_dataset_from_csv`. The dataset from each file belonging to a client is concatenated. A final check has been included to skip clients that failed dataset creation.  The example shows the results, one TFF compatible dataset per client, ready for use in the federated learning loop.

These examples are based on my experience handling federated tabular data. A robust federated learning system often requires additional considerations, such as data validation, error handling, and handling missing data. Furthermore, techniques like TFRecord and sharding can enhance scalability. For individuals seeking to deepen their knowledge, I suggest delving into the TensorFlow documentation on `tf.data.Dataset` and TFF’s `tff.simulation` modules. The pandas documentation offers a comprehensive overview of data manipulation tools useful for this type of processing. Finally, a deeper understanding of federated learning concepts and challenges is beneficial for effective data preprocessing, with multiple books on the topic available, and papers on federated learning from top ML conferences.
