---
title: "How can private data from GCS be streamed to Google Colab TPUs?"
date: "2025-01-30"
id: "how-can-private-data-from-gcs-be-streamed"
---
Accessing private data stored in Google Cloud Storage (GCS) for processing on Google Colab's Tensor Processing Units (TPUs) requires careful consideration of security, efficiency, and the specific I/O limitations imposed by the TPU environment. Standard direct file access is insufficient; instead, we must leverage Google Cloud’s authentication mechanisms and specialized data loading strategies. Having deployed numerous machine learning pipelines on GCP, I've encountered this problem frequently. Here’s the breakdown of how I typically handle it.

The core challenge arises from the fact that Colab TPUs run in a separate, isolated environment. They do not have direct access to your personal GCS buckets or the credentials required to access them.  We bridge this gap by utilizing service account keys, which provide authorized access to your resources, and by employing cloud-optimized data formats and reading protocols that minimize I/O bottlenecks.

The first step is to create a service account in the Google Cloud Console with the necessary permissions to read the relevant GCS buckets.  Specifically, the service account needs the "Storage Object Viewer" role, or a custom role with equivalent permissions. Crucially, it is not advisable to grant broad project-level permissions; adhere to the principle of least privilege. Once the service account is created, download its JSON key file.  Treat this key file as sensitive information and avoid committing it to any publicly accessible repository.

On the Colab side, we utilize this key file to authenticate with Google Cloud. Instead of explicitly loading the key file as a string, we mount it as a secret. Colab provides functionality to manage these secrets securely, meaning the key file is not directly exposed in your Colab notebook.

Here's a code example illustrating how to mount a service account key as a Colab secret, and then utilize it to instantiate a GCS client:

```python
# This code should be run in a Google Colab Notebook
from google.colab import auth
from google.cloud import storage
import os

# Mount the service account key as a Colab secret named 'gcs_key'
# This step is manual, and must be done in Colab's Secrets section in the left panel
# Make sure the secret is correctly labeled 'gcs_key'
# No code is needed to fetch the key file; Colab handles this under the hood.


# Define a helper function to access the GCS client.
# This keeps the authentication logic contained
def get_gcs_client():
    key_path = os.path.join(os.environ['HOME'],'.secrets/gcs_key.json')
    client = storage.Client.from_service_account_json(key_path)
    return client

# Example usage: list buckets
client = get_gcs_client()

for bucket in client.list_buckets():
    print(f"Bucket Name: {bucket.name}")
```

In this snippet, the `os.environ['HOME']` path points to a secure location within the Colab environment, and the secret, which was specified as `gcs_key`, is mounted to that location as `gcs_key.json`. We then use the `storage.Client.from_service_account_json` method to establish a connection to GCS using this mounted credential. The `get_gcs_client` helper function encapsulates the authentication, making it easy to use repeatedly.  This avoids scattering authentication logic throughout the notebook. The subsequent loop demonstrates a basic listing of buckets as a validation step. Remember to replace `gcs_key` with the actual name of your Colab secret.

After establishing authenticated access, the next critical phase is efficient data streaming. Directly loading large datasets into Colab's limited RAM is infeasible, especially when dealing with TPUs.  Data must be streamed in batches, ideally directly to the TPU. The recommended approach is to use `tf.data.Dataset` in conjunction with Google Cloud Storage files formatted as TensorFlow Record (TFRecord) files.  TFRecord is a binary format optimized for TensorFlow, allowing for fast, parallel reading and parsing.

The initial step involves creating TFRecord files from your raw data within your GCS bucket. The details of this creation will vary based on the original data format (e.g. CSV, image files).  Once your data is converted, your GCS bucket will contain a set of these TFRecord files.

Here's the code to construct a `tf.data.Dataset` for TPU input from a set of TFRecord files in GCS:

```python
import tensorflow as tf
from google.cloud import storage
import os
def get_gcs_client():
    key_path = os.path.join(os.environ['HOME'],'.secrets/gcs_key.json')
    client = storage.Client.from_service_account_json(key_path)
    return client

def create_gcs_dataset(bucket_name, prefix, pattern='*.tfrecord', batch_size=32, shuffle=True, prefetch_buffer=tf.data.AUTOTUNE):
    """Creates a tf.data.Dataset reading from TFRecords on GCS"""
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    files = [f"gs://{bucket_name}/{blob.name}" for blob in blobs if pattern in blob.name]

    dataset = tf.data.TFRecordDataset(files)

    if shuffle:
      dataset = dataset.shuffle(buffer_size=10000) # Choose appropriate value

    # This function must match the content of the records. This is just an example
    def parse_example(record):
        features = {
            'feature1': tf.io.FixedLenFeature([], tf.float32),
            'feature2': tf.io.FixedLenFeature([], tf.int64),
        }
        parsed = tf.io.parse_single_example(record, features)
        return parsed['feature1'], parsed['feature2']


    dataset = dataset.map(parse_example)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_buffer)
    return dataset

# Example usage
bucket_name = 'your-bucket-name'
prefix = 'your-data-folder/'
batch_size=64
train_dataset = create_gcs_dataset(bucket_name,prefix, batch_size=batch_size)

# Example loop.
for features, labels in train_dataset.take(5):
  print(f'Shape of features in batch: {features.shape}')
  print(f'Shape of labels in batch: {labels.shape}')
```

In this example, `create_gcs_dataset` constructs the `tf.data.Dataset`. It begins by creating a GCS client (as done previously). It then lists all the blobs (files) in a specific bucket, filtering for those matching the provided `pattern` (e.g. '*.tfrecord'). It constructs a `TFRecordDataset` from the list of GCS paths.  Crucially, the `parse_example` function must be defined to correctly interpret the data stored in the TFRecord files. It defines the data types and structure within each record. The `batch` method groups records into batches of the specified `batch_size`. The `prefetch` operation allows for overlap between the training and data loading, keeping the TPU fed. The commented out example loop at the bottom demonstrates iterating through the dataset. Remember to replace the placeholder values for `bucket_name`, `prefix`, and update the `parse_example` function to reflect your specific data structure. The shape of data in each batch will be printed to illustrate correctness of loading.

Finally, to use this data on a TPU, the `tf.distribute.TPUStrategy` must be used.  The strategy distributes computation across all TPU cores, and integrates seamlessly with `tf.data.Dataset`. Here is an example of how a model could be trained on data streamed from GCS:

```python
import tensorflow as tf
import os
def get_gcs_client():
    key_path = os.path.join(os.environ['HOME'],'.secrets/gcs_key.json')
    client = storage.Client.from_service_account_json(key_path)
    return client

def create_gcs_dataset(bucket_name, prefix, pattern='*.tfrecord', batch_size=32, shuffle=True, prefetch_buffer=tf.data.AUTOTUNE):
    """Creates a tf.data.Dataset reading from TFRecords on GCS"""
    client = get_gcs_client()
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)
    files = [f"gs://{bucket_name}/{blob.name}" for blob in blobs if pattern in blob.name]

    dataset = tf.data.TFRecordDataset(files)

    if shuffle:
      dataset = dataset.shuffle(buffer_size=10000)

    def parse_example(record):
        features = {
            'feature1': tf.io.FixedLenFeature([], tf.float32),
            'feature2': tf.io.FixedLenFeature([], tf.int64),
        }
        parsed = tf.io.parse_single_example(record, features)
        return parsed['feature1'], parsed['feature2']

    dataset = dataset.map(parse_example)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(prefetch_buffer)
    return dataset

resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(resolver)
tf.tpu.experimental.initialize_tpu_system(resolver)
strategy = tf.distribute.TPUStrategy(resolver)


bucket_name = 'your-bucket-name'
prefix = 'your-data-folder/'
batch_size=128

with strategy.scope():
    model = tf.keras.Sequential([
      tf.keras.layers.Dense(10, activation='relu', input_shape=(1,)), # Simple example
      tf.keras.layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
    loss = tf.keras.losses.MeanSquaredError()

    model.compile(optimizer=optimizer, loss = loss)

train_dataset = create_gcs_dataset(bucket_name,prefix, batch_size=batch_size, shuffle = True)

model.fit(train_dataset, epochs=10)
```

In this final code block, we initialize and configure a TPU strategy by creating a `TPUClusterResolver` and subsequently connecting to the cluster. The model, optimizer, and loss are then created within the strategy's scope to ensure they are distributed correctly on the TPU cores. The training dataset is generated using the same `create_gcs_dataset` function as earlier. Finally, the model is trained by passing the dataset to the `fit` method.  Again, remember to replace the placeholders for your actual bucket name and data path.

For further exploration, I recommend the following: the official Google Cloud Storage documentation details access permissions and methods for data storage and retrieval. The TensorFlow documentation thoroughly explains the use of `tf.data.Dataset` and the TFRecord format, including examples for parsing and creating records, which are paramount for efficient data loading. Also, the TPU documentation offers detailed guidance on working with TPUs in Google Colab and integrating with various training pipelines. Pay special attention to data sharding and optimization techniques described in the TPU section.
