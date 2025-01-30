---
title: "How can I load data from CSV files in TensorFlow 2.0?"
date: "2025-01-30"
id: "how-can-i-load-data-from-csv-files"
---
Working with structured data in TensorFlow often necessitates loading information from CSV files, a common format for datasets. I've repeatedly encountered scenarios requiring this, and TensorFlow 2.0 offers several effective methods. Understanding these options and their nuances is crucial for efficient data pipeline development. The primary goal is to transform the raw CSV into a `tf.data.Dataset`, the foundational data structure for TensorFlow training. This structure enables efficient batching, shuffling, and preprocessing of the data.

The most direct approach involves using `tf.data.experimental.make_csv_dataset`. This function streamlines the loading process by automatically handling file parsing and data type inference. It takes a list of file paths or a file pattern, a batch size, and a dictionary describing the column data types (if not inferable), among other parameters. I've found this method exceptionally helpful for quickly prototyping models and testing data loading strategies. It eliminates the boilerplate code typically associated with manual parsing and conversion.

Here's an example demonstrating the basic usage:

```python
import tensorflow as tf
import pandas as pd
import os
# Create a dummy csv file.
data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c'], 'col3': [1.1, 2.2, 3.3]}
df = pd.DataFrame(data)
df.to_csv("dummy.csv", index=False)
file_path = "dummy.csv"
```
```python
# Define column types if inference is insufficient
column_types = [tf.int32, tf.string, tf.float32]
feature_names = ['col1', 'col2', 'col3'] # Needed for data dictionary.

# Load CSV into a tf.data.Dataset
dataset = tf.data.experimental.make_csv_dataset(
    file_path,
    batch_size=2,
    column_names=feature_names,
    column_defaults=[0, "default", 0.0],
    label_cols=None # Assumes all columns are features
    num_epochs=1
    #dtype=column_types # Can also use this.
)
# Iterate and print the first batch of tensors to confirm data load.
for features in dataset.take(1):
  print(features)

# Delete dummy csv
os.remove(file_path)
```

In this example, `make_csv_dataset` automatically parses the CSV file specified by `file_path` creating a `tf.data.Dataset`.  `batch_size` determines the number of records returned in each batch.  `column_defaults` provides default values in case of missing data, and `column_names` is used to explicitly specify the names of the columns (otherwise assumed to be the first row of csv).  The dataset produces a dictionary where the keys are the feature names and values are `tf.Tensor` objects. I've learned that providing explicit `column_names` and defaults can improve code robustness, preventing errors from unexpected data entries. While I've used `label_cols=None`, this can be set to any column name or set of columns, which separates labels from the rest of the dataset for supervised learning. The `column_types` can be set using an argument to `tf.data.experimental.make_csv_dataset` rather than with `column_defaults` as demonstrated.

Another approach, more granular but providing greater control, involves manually parsing each line of the CSV using `tf.io.decode_csv` within a `tf.data.Dataset.map` operation.  This method is beneficial for scenarios requiring custom preprocessing or when the structure of the CSV isn't entirely uniform. It also permits greater control over handling missing values and string preprocessing. It mandates defining a custom parsing function to convert each CSV record into a tensor.

```python
# Create a dummy csv file.
data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c'], 'col3': [1.1, 2.2, 3.3]}
df = pd.DataFrame(data)
df.to_csv("dummy.csv", index=False)
file_path = "dummy.csv"

# Define data type for each column.
column_types = [tf.int32, tf.string, tf.float32]

def decode_line(line):
    parsed_line = tf.io.decode_csv(
        line,
        record_defaults = [0, "default", 0.0], # Required to make types match.
        field_delim = ','
    )

    # Transform a parsed list of tensors into the correct format.
    feature_dict = {
        'col1': parsed_line[0],
        'col2': parsed_line[1],
        'col3': parsed_line[2]
    }
    return feature_dict

# Read the CSV, parsing each line, and batched.
dataset_manual = tf.data.TextLineDataset(file_path).skip(1) # Skip header.
dataset_manual = dataset_manual.map(decode_line)
dataset_manual = dataset_manual.batch(2)

# Print the first batch.
for features in dataset_manual.take(1):
  print(features)

# Delete dummy csv
os.remove(file_path)

```

Here, `tf.data.TextLineDataset` reads each line from the CSV file as a string.  The `skip(1)` is added to avoid the header row being interpreted as a data record. The `decode_line` function uses `tf.io.decode_csv` to parse the strings, and the output is wrapped into a dictionary for ease of use.  The returned dictionary from `decode_line` will now be batchable, with tensors having the correct types for the downstream computation, where we see that column one and three are parsed as `tf.int32` and `tf.float32`, respectively, even if they were strings in the csv. This method, while more verbose than `make_csv_dataset`, provides greater control over how the data is processed from CSV to Tensors, particularly when dealing with more irregular or idiosyncratic data format issues. I tend to employ this whenever I encounter non-standard data formats or require a unique handling of missing data.

Another helpful approach is integrating Pandas for preprocessing before loading data into TensorFlow. Pandas offers powerful data manipulation tools. This method is convenient for tasks such as data cleaning, feature engineering, and handling complex data transformations.  However, it's crucial to be mindful of the memory usage, as Pandas loads the entire dataset into memory.

```python
import tensorflow as tf
import pandas as pd
import os
# Create a dummy csv file.
data = {'col1': [1, 2, 3], 'col2': ['a', 'b', 'c'], 'col3': [1.1, 2.2, 3.3]}
df = pd.DataFrame(data)
df.to_csv("dummy.csv", index=False)
file_path = "dummy.csv"

# Load CSV into a pandas DataFrame
df = pd.read_csv(file_path)

# Convert the DataFrame to a dictionary of tensors.
dataset_pandas = tf.data.Dataset.from_tensor_slices(dict(df))
dataset_pandas = dataset_pandas.batch(2)

# Print the first batch.
for features in dataset_pandas.take(1):
    print(features)
# Delete dummy csv
os.remove(file_path)
```

In this case, the CSV data is first loaded into a Pandas DataFrame. Once any transformations or preprocessing with Pandas are complete, the `tf.data.Dataset.from_tensor_slices` function converts the pandas DataFrame to a TensorFlow Dataset directly using a python dictionary. Each key corresponds to a column of the DataFrame, and its values are now converted to `tf.Tensor` objects that can be efficiently used in tensor operations. This pattern allows for full use of the preprocessing and data handling in pandas, as well as an efficient tensor format for model development.

In summary, choosing the appropriate method for loading CSV data depends largely on the data characteristics and the complexity of the required preprocessing. `tf.data.experimental.make_csv_dataset` works well for basic scenarios with well-formatted CSV files. `tf.data.TextLineDataset` combined with `tf.io.decode_csv` offers flexibility when manual parsing is needed. Lastly, integrating Pandas can prove beneficial when complex preprocessing is required before moving to TensorFlow. A thorough evaluation of these methods can significantly impact the efficiency and robustness of data pipelines when working with tabular data.

For further exploration, I recommend focusing on the official TensorFlow documentation for `tf.data.Dataset`, paying special attention to the `tf.data.experimental` namespace. The tutorials provided on the TensorFlow website are an excellent starting point for working with structured data.  Furthermore, delving into documentation on Pandas dataframes and their interaction with TensorFlow would be beneficial for integrating the two libraries effectively. Lastly, exploring methods of dealing with malformed or missing data in CSV files would be important, given the variety of challenges in the real-world.
