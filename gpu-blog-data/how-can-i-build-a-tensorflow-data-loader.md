---
title: "How can I build a TensorFlow data loader for dynamically expanding datasets?"
date: "2025-01-30"
id: "how-can-i-build-a-tensorflow-data-loader"
---
The key challenge in handling dynamically expanding datasets with TensorFlow lies in efficiently feeding data into the training loop as new samples become available, without needing to regenerate the entire dataset. This avoids bottlenecks inherent in loading all data into memory beforehand or recreating static `tf.data.Dataset` objects. My experience with large-scale image classification pipelines where data acquisition was a continuous process has highlighted the necessity of this approach.

The core strategy is to employ a data pipeline that defers loading individual data samples until they are actually needed. This can be achieved using `tf.data.Dataset` in conjunction with generators or a custom class inheriting from `tf.data.Dataset`’s abstract base class. Unlike pre-defined static datasets, this allows new data to be added to the underlying storage and automatically integrated into the training loop without requiring restarts. The fundamental component is an iterator-like object responsible for fetching, preprocessing, and yielding data.

Here are three distinct implementations that accomplish this:

**1. Using a Python Generator with `tf.data.Dataset.from_generator`**

This approach leverages the simplicity of Python generators. The generator function will need to manage fetching new data, potentially from a file system, database, or message queue. Each iteration of the generator should yield a single training example, which can be a tuple of (features, labels) or just a feature tensor if labels are handled separately. The `tf.data.Dataset.from_generator` method then takes this generator and constructs a TensorFlow dataset that dynamically draws data as required.

```python
import tensorflow as tf
import numpy as np
import time

def data_generator(num_iterations=100, initial_data_size=20):
  """
    Simulates a dynamically expanding dataset.

    Yields random data points and simulates dataset growth by pausing
    before adding more samples to the "storage".
    """
  data = np.random.rand(initial_data_size, 10).astype(np.float32)
  labels = np.random.randint(0, 2, size=initial_data_size).astype(np.int32)

  for i in range(num_iterations):
    index = i % len(data) # cycle through initial data
    yield data[index], labels[index]
    if i % 20 == 0 and i > 0 :  #Simulate adding new data every 20 iterations
      time.sleep(0.1) # Simulate delay of new data
      new_data = np.random.rand(5, 10).astype(np.float32)
      new_labels = np.random.randint(0, 2, size=5).astype(np.int32)
      data = np.concatenate([data, new_data], axis=0)
      labels = np.concatenate([labels, new_labels], axis=0)
      print(f"added {len(new_data)} new data points. Total data now: {len(data)}")


dataset = tf.data.Dataset.from_generator(
    data_generator,
    output_signature=(
        tf.TensorSpec(shape=(10,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

#Iterate through dataset
for features, label in dataset.take(50):
    print(f"Processed: Features shape {features.shape}, Label {label}")

```

In the above example, the `data_generator` function simulates a growing dataset by adding more data points at intervals. The `tf.data.Dataset.from_generator` creates a dataset from this dynamic generator. The `output_signature` is critical; it defines the shape and data type of the elements generated, allowing TensorFlow to optimize operations. This approach is straightforward to implement, but the generator itself can become complex when data is fetched from complex sources, like a data lake. Additionally, the entire logic for how data is retrieved is enclosed within the generator, which can be undesirable for organization in more complex projects. The output signature is explicitly declared, allowing TensorFlow to work on batched datasets.

**2. Implementing a Custom `tf.data.Dataset` Class**

This method provides more control and flexibility over the data loading pipeline by creating a custom class that inherits from `tf.data.Dataset`. This is beneficial when you need custom logic for dataset handling or when working with unconventional data sources. The key aspect is implementing the `_as_variant_tensor` and `_inputs` methods from `tf.data.Dataset`.

```python
import tensorflow as tf
import numpy as np
import time

class DynamicDataset(tf.data.Dataset):
    def __init__(self, initial_data_size=20):
        self.data = np.random.rand(initial_data_size, 10).astype(np.float32)
        self.labels = np.random.randint(0, 2, size=initial_data_size).astype(np.int32)
        self.index = 0

    def _inputs(self):
        return []

    def _as_variant_tensor(self):
        return tf.data.Dataset.from_tensor_slices((self.data, self.labels)).repeat(1).as_variant_tensor()

    def __iter__(self):
      self.index = 0
      return self

    def __next__(self):
      if self.index >= len(self.data):
          time.sleep(0.1) # Simulate delay of new data
          new_data = np.random.rand(5, 10).astype(np.float32)
          new_labels = np.random.randint(0, 2, size=5).astype(np.int32)
          self.data = np.concatenate([self.data, new_data], axis=0)
          self.labels = np.concatenate([self.labels, new_labels], axis=0)
          print(f"Added {len(new_data)} new data points. Total data now: {len(self.data)}")
          self.index = 0
      features = self.data[self.index]
      label = self.labels[self.index]
      self.index +=1
      return features, label


dynamic_dataset = DynamicDataset()
dataset = tf.data.Dataset.from_tensor_slices(list(dynamic_dataset)) #convert generator output to a tf dataset
dataset = dataset.unbatch() #necessary because from_tensor_slices output is a batch

for features, label in dataset.take(50):
    print(f"Processed: Features shape {features.shape}, Label {label}")

```

In this example, `DynamicDataset` manages its internal state and updates the data as requested within its iterator method `__next__`. The crucial part is implementing the `_as_variant_tensor`, which provides a way for TensorFlow to access the underlying tensor representing the data. Using this approach, one can maintain any internal state or logic needed to retrieve and manage the data. The benefit is better modularity and organization compared to the simple generator approach. However, it requires a more in-depth understanding of TensorFlow’s data API. It's also important to note, that the output from this generator is converted to a standard tf dataset using `tf.data.Dataset.from_tensor_slices()` and then unbatched. This step is necessary due to the structure of the output from this generator and the way tensorflow datasets handle generators that do not return tensorflow tensors directly.

**3. Leveraging `tf.data.experimental.AUTOTUNE` with a custom generator function**

This approach focuses on optimizing the asynchronous data loading by leveraging `AUTOTUNE` with a custom generator. This strategy is beneficial when I/O is a bottleneck for the training process. The key aspect here is to ensure that the data loading logic is implemented in a way that allows TensorFlow to overlap the data fetching with the training process.

```python
import tensorflow as tf
import numpy as np
import time


def custom_generator_function(num_iterations=100, initial_data_size=20):
  data = np.random.rand(initial_data_size, 10).astype(np.float32)
  labels = np.random.randint(0, 2, size=initial_data_size).astype(np.int32)

  for i in range(num_iterations):
    index = i % len(data) # cycle through initial data
    yield data[index], labels[index]
    if i % 20 == 0 and i > 0 :  #Simulate adding new data every 20 iterations
      time.sleep(0.1) # Simulate delay of new data
      new_data = np.random.rand(5, 10).astype(np.float32)
      new_labels = np.random.randint(0, 2, size=5).astype(np.int32)
      data = np.concatenate([data, new_data], axis=0)
      labels = np.concatenate([labels, new_labels], axis=0)
      print(f"added {len(new_data)} new data points. Total data now: {len(data)}")

dataset = tf.data.Dataset.from_generator(
    custom_generator_function,
    output_signature=(
        tf.TensorSpec(shape=(10,), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32)
    )
)

dataset = dataset.prefetch(tf.data.AUTOTUNE)

for features, label in dataset.take(50):
    print(f"Processed: Features shape {features.shape}, Label {label}")

```

The core difference in this approach is the addition of `.prefetch(tf.data.AUTOTUNE)`. This allows TensorFlow to asynchronously prepare the next batches of data while the current batch is being used in the training process. This helps keep the GPUs busy by avoiding bottlenecks caused by slow data loading. This technique can be combined with the other approaches shown previously.

**Resource Recommendations**

For further exploration of these concepts, I suggest studying the official TensorFlow documentation, particularly the sections on the `tf.data` module. There are multiple tutorials and examples provided on their official website. Look into tutorials detailing the use of `tf.data.Dataset.from_generator`, and pay special attention to the `output_signature` parameter. Understanding asynchronous data loading in TensorFlow and the use of `tf.data.AUTOTUNE` will be essential for optimizing your pipeline. Further, the Tensorflow API guide on custom datasets can be invaluable when building and understanding custom tf.data.Dataset classes. Various community examples and tutorials regarding data pipelines in TensorFlow will prove useful in practice.
