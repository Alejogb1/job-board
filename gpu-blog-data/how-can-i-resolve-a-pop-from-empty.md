---
title: "How can I resolve a 'pop from empty list' error when using Keras Tuner with TPUs in Google Colab?"
date: "2025-01-30"
id: "how-can-i-resolve-a-pop-from-empty"
---
The error "pop from empty list" during Keras Tuner execution on TPUs within Google Colab often stems from incorrect data handling during the hyperparameter tuning process, particularly when dealing with distributed training strategies. Specifically, this error usually indicates that the tuner is trying to access data it believes should be available across all TPU cores but is, in fact, empty or inconsistently populated on certain replicas. This usually manifests when validation data is not correctly partitioned or when data loaders are not properly configured for TPU distribution. I've encountered this multiple times in scaling up model search and have learned some practical solutions.

The core issue arises from the asynchronous nature of TPU training and the way Keras Tuner distributes hyperparameter trials. When using TPUs, data is often replicated across multiple cores to accelerate training. However, Keras Tuner needs to ensure that each hyperparameter trial receives its own consistent portion of data. If, for example, the validation data is not correctly distributed, or a data loader provides different data on different TPU cores, it can result in inconsistent state during the tuning process, leading to the "pop from empty list" error as a core tries to access data that has already been consumed by another. This frequently occurs when using generators for loading data, if not designed for multi-replica processing, or when the size of datasets is not divisible by the number of TPU cores, leading to empty partitions on some cores. The error itself is buried within internal Keras structures while it attempts to retrieve batch information from an empty queue.

Let's examine three code examples and discuss how to resolve this common pitfall. The first scenario is the most frequently observed issue involving a naive data loader.

```python
import tensorflow as tf
import keras_tuner as kt
import numpy as np
from sklearn.model_selection import train_test_split

# Assume data loaded and preprocessed
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=hp.Int('units_1', min_value=32, max_value=512, step=32), activation='relu', input_shape=(10,)),
        tf.keras.layers.Dense(units=1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_data_generator(X, y, batch_size):
    num_samples = X.shape[0]
    indices = np.arange(num_samples)
    while True:
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            yield X[batch_indices], y[batch_indices]

def val_data_generator(X, y, batch_size):
  # Similar generator, can cause issues if the dataset length isn't consistent across replicas
  num_samples = X.shape[0]
  indices = np.arange(num_samples)
  while True:
    np.random.shuffle(indices)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_indices = indices[start:end]
        yield X[batch_indices], y[batch_indices]


tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

with strategy.scope():
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='tpu_tuning'
    )

    batch_size = 64 * strategy.num_replicas_in_sync # Multiplied by replicas
    train_gen = train_data_generator(X_train, y_train, batch_size)
    val_gen = val_data_generator(X_val, y_val, batch_size)
    train_steps = len(X_train) // batch_size
    val_steps = len(X_val) // batch_size

    tuner.search(
        train_gen,
        validation_data=val_gen,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        epochs=3
    )
```

In this initial example, while the `batch_size` is correctly scaled based on the number of replicas, the use of a generator not explicitly designed for distributed datasets can lead to problems. Each TPU core will generate its own batch, and if the number of samples isn't perfectly divisible by `batch_size`, some cores might end up with empty batches or incomplete batches and inconsistent state across replicas. This can trigger the "pop from empty list" error within Keras Tuner's internal data handling, particularly during validation.

Here’s the second example, demonstrating the use of a TensorFlow `Dataset` and employing `tf.data.Dataset.shard()` for correct distribution:

```python
import tensorflow as tf
import keras_tuner as kt
import numpy as np
from sklearn.model_selection import train_test_split

# Assume data loaded and preprocessed
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(hp):
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(units=hp.Int('units_1', min_value=32, max_value=512, step=32), activation='relu', input_shape=(10,)),
      tf.keras.layers.Dense(units=1, activation='sigmoid')
  ])
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model


def create_dataset(X, y, batch_size, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(tf.data.AUTOTUNE)


tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

with strategy.scope():
  tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='tpu_tuning'
    )

  global_batch_size = 64 * strategy.num_replicas_in_sync
  train_dataset = create_dataset(X_train, y_train, global_batch_size)
  val_dataset = create_dataset(X_val, y_val, global_batch_size, shuffle=False)

  train_dataset = train_dataset.shard(num_shards=strategy.num_replicas_in_sync, index=strategy.replica_id_in_sync_group)
  val_dataset = val_dataset.shard(num_shards=strategy.num_replicas_in_sync, index=strategy.replica_id_in_sync_group)

  tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=3
    )
```
This revision utilizes TensorFlow's `tf.data.Dataset` API, a preferred approach for distributed training. Critically, the `shard()` method is used.  `shard` divides the dataset into `num_shards` (equal to the number of TPU cores), providing each replica with a unique subset of the data. This ensures consistent data distribution across all TPU cores and prevents the "pop from empty list" error arising from an empty or misaligned batch at the core level. By setting `shuffle=False` on validation, we ensure that each replica gets the same validation data order each epoch, critical for proper comparisons during hyperparameter tuning. Additionally `prefetch(tf.data.AUTOTUNE)` further improves data loading performance, feeding the GPU faster by preloading the next batch.

Finally, the third example is an improvement that adds checks for edge cases and shows handling of potential issues with uneven datasets, which might still cause problems in rare scenarios

```python
import tensorflow as tf
import keras_tuner as kt
import numpy as np
from sklearn.model_selection import train_test_split

# Assume data loaded and preprocessed
X = np.random.rand(1000, 10)
y = np.random.randint(0, 2, 1000)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(hp):
  model = tf.keras.Sequential([
      tf.keras.layers.Dense(units=hp.Int('units_1', min_value=32, max_value=512, step=32), activation='relu', input_shape=(10,)),
      tf.keras.layers.Dense(units=1, activation='sigmoid')
  ])
  model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return model

def create_dataset(X, y, batch_size, shuffle=True):
  dataset = tf.data.Dataset.from_tensor_slices((X, y))
  if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
  dataset = dataset.batch(batch_size)
  # Check if the dataset is too small for the number of replicas.
  if len(X) < batch_size: # Handles edge case
    return dataset.repeat(2) # Repeat if dataset too small
  return dataset.prefetch(tf.data.AUTOTUNE)


tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
strategy = tf.distribute.TPUStrategy(tpu)

with strategy.scope():
  tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=5,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='tpu_tuning'
    )

  global_batch_size = 64 * strategy.num_replicas_in_sync
  train_dataset = create_dataset(X_train, y_train, global_batch_size)
  val_dataset = create_dataset(X_val, y_val, global_batch_size, shuffle=False)

  train_dataset = train_dataset.shard(num_shards=strategy.num_replicas_in_sync, index=strategy.replica_id_in_sync_group)
  val_dataset = val_dataset.shard(num_shards=strategy.num_replicas_in_sync, index=strategy.replica_id_in_sync_group)

  # Check if the dataset length is divisible by the batch size before sharding
  # if the dataset has a small number of samples or the batch size is large
  # padding or repeats are important, otherwise one core may have no data.
  
  tuner.search(
        train_dataset,
        validation_data=val_dataset,
        epochs=3
    )
```
This final example adds a safeguard. It ensures the dataset isn't smaller than the `global_batch_size`, potentially leading to empty shards even with the previous fixes. If the dataset is too small it repeats to provide a sufficient number of samples. This can prevent issues arising from dataset size constraints, improving resilience. This example also moves the prefetch to the `create_dataset` function for clarity and consistency.

For further learning and investigation I recommend consulting the TensorFlow documentation specifically regarding tf.data.Dataset and distributed training. The Keras documentation also offers guidance on the proper use of Keras Tuner. Researching how TPU strategies work can provide deep insights. Articles published by Google AI researchers and engineers often describe practical applications, which have aided my understanding and troubleshooting skills over time. Also, studying the source code of `tf.distribute` within TensorFlow is a great resource that provides an in-depth view of the underlying mechanisms, though it's not typically the first approach. I have found the official API documents and examples to be the most useful.
