---
title: "How can I build a federated learning model from CSV files using tff.simulation.datasets.ClientData?"
date: "2025-01-30"
id: "how-can-i-build-a-federated-learning-model"
---
Federated learning, specifically with TensorFlow Federated (TFF), presents a unique challenge when starting with local CSV files instead of pre-packaged datasets. The `tff.simulation.datasets.ClientData` abstraction is designed for distributed datasets, not raw CSVs. Therefore, I've learned through multiple project iterations that a critical first step involves converting your CSV data into a structure that TFF can understand, representing each client's local data as a distinct dataset.

The fundamental requirement for federated learning with TFF is that data be organized as a collection of `tf.data.Dataset` objects, each representing the data belonging to a single client. The `ClientData` class, while often loaded with pre-packaged datasets, isn't inherently tied to any specific data source. It functions more as an interface that provides access to individual client's datasets. Constructing this structure from CSVs involves reading each CSV, processing it, and mapping it into the `tf.data.Dataset` format. A common scenario involves a single CSV per client or a structured directory of CSV files where each directory represents a client. The key is to create a function that can efficiently convert a single CSV into a `tf.data.Dataset`.

The process can be summarized as follows: 1) Determine a schema for your CSV data (column types and names). 2) Implement a function to read a single CSV file and return a `tf.data.Dataset`. This will involve reading the CSV using `tf.data.experimental.make_csv_dataset` or pandas with conversion to a `tf.data.Dataset` and parsing features and labels. 3) Implement a function to create a list of `tf.data.Dataset` from a list of CSV files. 4) Use this list of datasets to create an instance of your custom `ClientData` implementation.

Here's an example of converting CSV files into a usable `ClientData` class using TensorFlow:

**Example 1: Single CSV Per Client**

Assume we have CSV files in a directory like `data/client_1.csv`, `data/client_2.csv`, etc., where each CSV belongs to a specific client.

```python
import tensorflow as tf
import pandas as pd
import os
import numpy as np
from collections import OrderedDict
import tensorflow_federated as tff

def create_tf_dataset_from_csv(csv_path, features_spec, batch_size=32):
    """Creates a tf.data.Dataset from a CSV file."""
    dataset = tf.data.experimental.make_csv_dataset(
            csv_path,
            batch_size=batch_size,
            column_names=list(features_spec.keys()),
            column_defaults=list(features_spec.values()),
            num_epochs=1,
            shuffle=False
        )

    def _parse_dataset(features):
        features_data = OrderedDict()
        labels = features.pop("target")
        for key in features:
          features_data[key] = features[key]
        return (features_data, labels)
    return dataset.map(_parse_dataset)

class CSVClientData(tff.simulation.datasets.ClientData):
    """Implements ClientData for CSV files with one file per client."""

    def __init__(self, data_dir, features_spec, batch_size=32):
        self._client_ids = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".csv")]
        self._data_dir = data_dir
        self._features_spec = features_spec
        self._batch_size = batch_size

    def client_ids(self):
        return self._client_ids

    def create_tf_dataset_for_client(self, client_id):
         csv_path = os.path.join(self._data_dir, f"{client_id}.csv")
         return create_tf_dataset_from_csv(csv_path, self._features_spec, self._batch_size)

# Example Usage
features_spec = OrderedDict({
    "feature1": tf.float32,
    "feature2": tf.float32,
    "target": tf.int32
})

data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
# Create dummy csv files
for client in ['client_1', 'client_2', 'client_3']:
    df = pd.DataFrame(np.random.rand(100, 2), columns = ["feature1", "feature2"])
    df['target'] = np.random.randint(0, 2, df.shape[0])
    df.to_csv(os.path.join(data_dir, f'{client}.csv'), index=False)

client_data = CSVClientData(data_dir, features_spec)
sample_client = client_data.client_ids()[0]
sample_dataset = client_data.create_tf_dataset_for_client(sample_client)

example_batch = next(iter(sample_dataset))
print("First batch for client", sample_client, ":", example_batch)

```

This code defines the `CSVClientData` class, which inherits from `tff.simulation.datasets.ClientData`. The initializer collects all `.csv` file names as client ids.  The key function is `create_tf_dataset_for_client`, which reads a specific CSV from the input directory and uses `tf.data.experimental.make_csv_dataset` to build the `tf.data.Dataset`. It also includes a parse function, `_parse_dataset` that separates target labels from features. We use the `OrderedDict` to define the spec for our CSV. The example usage creates some sample files for demonstration.

**Example 2: Multiple CSVs per Client**

If each client has multiple CSV files, the `create_tf_dataset_for_client` function needs to be modified to concatenate those files.

```python
import tensorflow as tf
import pandas as pd
import os
import numpy as np
from collections import OrderedDict
import tensorflow_federated as tff


def create_tf_dataset_from_csv(csv_paths, features_spec, batch_size=32):
    """Creates a tf.data.Dataset from multiple CSV files."""
    datasets = [tf.data.experimental.make_csv_dataset(
            csv_path,
            batch_size=batch_size,
            column_names=list(features_spec.keys()),
            column_defaults=list(features_spec.values()),
            num_epochs=1,
            shuffle=False
        )
         for csv_path in csv_paths]

    def _parse_dataset(features):
        features_data = OrderedDict()
        labels = features.pop("target")
        for key in features:
            features_data[key] = features[key]
        return (features_data, labels)
    
    concatenated_dataset = datasets[0]
    for dataset in datasets[1:]:
      concatenated_dataset = concatenated_dataset.concatenate(dataset)

    return concatenated_dataset.map(_parse_dataset)

class CSVClientDataMulti(tff.simulation.datasets.ClientData):
    """Implements ClientData for CSV files, with multiple files per client."""

    def __init__(self, data_dir, features_spec, batch_size=32):
       self._data_dir = data_dir
       self._features_spec = features_spec
       self._batch_size = batch_size
       self._client_ids = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]


    def client_ids(self):
      return self._client_ids

    def create_tf_dataset_for_client(self, client_id):
        client_dir = os.path.join(self._data_dir, client_id)
        csv_paths = [os.path.join(client_dir, f) for f in os.listdir(client_dir) if f.endswith(".csv")]
        return create_tf_dataset_from_csv(csv_paths, self._features_spec, self._batch_size)


# Example Usage
features_spec = OrderedDict({
    "feature1": tf.float32,
    "feature2": tf.float32,
    "target": tf.int32
})

data_dir = "multi_data"
os.makedirs(data_dir, exist_ok=True)
#Create client folder and dummy files
for client in ['client_1', 'client_2', 'client_3']:
    client_dir = os.path.join(data_dir, client)
    os.makedirs(client_dir, exist_ok=True)
    for i in range(2):
        df = pd.DataFrame(np.random.rand(100, 2), columns = ["feature1", "feature2"])
        df['target'] = np.random.randint(0, 2, df.shape[0])
        df.to_csv(os.path.join(client_dir, f'data_{i}.csv'), index=False)


client_data_multi = CSVClientDataMulti(data_dir, features_spec)

sample_client = client_data_multi.client_ids()[0]
sample_dataset = client_data_multi.create_tf_dataset_for_client(sample_client)
example_batch = next(iter(sample_dataset))
print("First batch for client", sample_client, ":", example_batch)

```

This version reads all CSV files in a client's folder. It concatenates these datasets and then maps each batch to separate features from labels. It has been modified to accommodate multiple CSV files per client by first creating a list of datasets, and then concatenating them into a single dataset.

**Example 3: Using Pandas with conversion to tf.data.Dataset**

Another approach involves using Pandas to read CSV data before converting it into a `tf.data.Dataset`.

```python
import tensorflow as tf
import pandas as pd
import os
import numpy as np
from collections import OrderedDict
import tensorflow_federated as tff


def create_tf_dataset_from_pandas_csv(csv_path, features_spec, batch_size=32):
    """Creates a tf.data.Dataset from a CSV using Pandas."""
    df = pd.read_csv(csv_path)
    labels = df.pop('target').values
    features = df.to_dict('list')

    dataset = tf.data.Dataset.from_tensor_slices((features, labels)).batch(batch_size)
    def _parse_dataset(features, labels):
        features_data = OrderedDict()
        for key, values in features.items():
            features_data[key] = tf.convert_to_tensor(values)
        return (features_data, labels)


    return dataset.map(_parse_dataset)


class PandasCSVClientData(tff.simulation.datasets.ClientData):
    """Implements ClientData using Pandas."""

    def __init__(self, data_dir, features_spec, batch_size=32):
         self._client_ids = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.endswith(".csv")]
         self._data_dir = data_dir
         self._features_spec = features_spec
         self._batch_size = batch_size

    def client_ids(self):
      return self._client_ids

    def create_tf_dataset_for_client(self, client_id):
         csv_path = os.path.join(self._data_dir, f"{client_id}.csv")
         return create_tf_dataset_from_pandas_csv(csv_path, self._features_spec, self._batch_size)

# Example Usage
features_spec = OrderedDict({
    "feature1": tf.float32,
    "feature2": tf.float32,
    "target": tf.int32
})

data_dir = "pandas_data"
os.makedirs(data_dir, exist_ok=True)

for client in ['client_1', 'client_2', 'client_3']:
    df = pd.DataFrame(np.random.rand(100, 2), columns = ["feature1", "feature2"])
    df['target'] = np.random.randint(0, 2, df.shape[0])
    df.to_csv(os.path.join(data_dir, f'{client}.csv'), index=False)

client_data_pandas = PandasCSVClientData(data_dir, features_spec)
sample_client = client_data_pandas.client_ids()[0]
sample_dataset = client_data_pandas.create_tf_dataset_for_client(sample_client)
example_batch = next(iter(sample_dataset))
print("First batch for client", sample_client, ":", example_batch)

```

This approach uses Pandas to load the CSV into a DataFrame. It then separates the label from features and then converts it to tensor slices which are then mapped and batched. This method might be more convenient for certain data manipulation or preprocessing needs.

These examples provide a starting point for building a custom `ClientData` implementation from CSV files. They demonstrate the necessary steps for reading, parsing, and converting data from a CSV format into a `tf.data.Dataset`, which can then be used with TFF for federated learning.

For further study, I recommend exploring the official TensorFlow documentation on `tf.data.Dataset` for dataset creation and transformation, particularly `tf.data.experimental.make_csv_dataset` and `tf.data.Dataset.from_tensor_slices`, as well as the `tensorflow_federated` documentation relating to `tff.simulation.datasets.ClientData`. A deeper dive into Pandas documentation may prove useful for data processing. A thorough understanding of federated learning concepts and the overall architecture of TFF will also be necessary for developing an effective federated learning strategy with custom data sets.
