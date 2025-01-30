---
title: "Is there a TensorFlow Dataset operation equivalent to `timeseries_dataset_from_array`?"
date: "2025-01-30"
id: "is-there-a-tensorflow-dataset-operation-equivalent-to"
---
The question of direct equivalence between Keras' `timeseries_dataset_from_array` and a TensorFlow Dataset operation is nuanced; there isn't a single, identical replacement. The former provides a high-level, convenience-oriented method specifically tailored for time series data, handling windowing and batching implicitly. TensorFlow Datasets offer far greater flexibility and lower-level control, necessitating manual implementation of similar windowing and batching logic.

My experience building custom machine learning models, particularly those involving sequential data processing, has led me to utilize the flexible capabilities of `tf.data.Dataset` directly. While initially requiring more code, this approach grants significantly more control over data preprocessing and pipeline optimization. The core challenge lies in replicating the windowing functionality of `timeseries_dataset_from_array` using the Dataset API.

Here's a breakdown of the problem: `timeseries_dataset_from_array` essentially transforms a NumPy array (or similar) representing a time series into a batched dataset of subsequences. It defines a `sequence_length` (the length of each subsequence), a `sequence_stride` (the step size between subsequent subsequences), and a `sampling_rate` (how often data points are included within each subsequence). The key objective, therefore, is to replicate this windowing, stride, and sampling process using `tf.data` primitives. This can be achieved primarily through methods such as `window`, `flat_map`, and `batch` within the TensorFlow Dataset API.

Let's examine a concrete example, assuming a time series stored as a NumPy array:

```python
import tensorflow as tf
import numpy as np

# Sample time series data (100 data points)
time_series_data = np.arange(100, dtype=np.float32)

# Configuration parameters
sequence_length = 10
sequence_stride = 2
batch_size = 8

# Create a TensorFlow Dataset from the NumPy array
dataset = tf.data.Dataset.from_tensor_slices(time_series_data)

# Apply windowing and striding
windowed_dataset = dataset.window(
    size=sequence_length,
    shift=sequence_stride,
    drop_remainder=True
)

# Flatten the windowed dataset into a dataset of tensors
windowed_dataset = windowed_dataset.flat_map(
    lambda window: window.batch(sequence_length)
)

# Batch the dataset
batched_dataset = windowed_dataset.batch(batch_size)

# Example iteration
for batch in batched_dataset.take(2):
  print(batch.numpy())

```

In this example, `tf.data.Dataset.from_tensor_slices` creates a dataset where each element corresponds to a single value from the input array. The `window` method divides the dataset into overlapping sequences (windows) according to `sequence_length` and `sequence_stride`. The `drop_remainder=True` argument discards any incomplete windows at the end of the data. Subsequently, `flat_map` transforms each window into a fixed-size tensor, as each window was initially a nested dataset. The inner `batch(sequence_length)` within the flatmap collapses each window into a single tensor of size `sequence_length`. Finally, `batch` aggregates tensors into batches of a predetermined `batch_size`. This process effectively mimics `timeseries_dataset_from_array`.

Now, let's consider an example where we include labels or targets for our time series. In many real-world situations, a time series model will not just predict a time series future but rather attempt to predict another output based on the input time series. Let’s assume that the labels are just the next value in the original time series:

```python
import tensorflow as tf
import numpy as np

# Sample time series data and corresponding labels
time_series_data = np.arange(100, dtype=np.float32)
labels = np.arange(1, 101, dtype=np.float32) # Shifted by one

sequence_length = 10
sequence_stride = 2
batch_size = 8

# Create a TensorFlow Dataset from the data
data_dataset = tf.data.Dataset.from_tensor_slices(time_series_data)
label_dataset = tf.data.Dataset.from_tensor_slices(labels)

# Apply windowing with a shift of sequence_length and sequence_stride
data_windowed_dataset = data_dataset.window(
    size=sequence_length,
    shift=sequence_stride,
    drop_remainder=True
)

label_windowed_dataset = label_dataset.window(
    size=sequence_length,
    shift=sequence_stride,
    drop_remainder=True
)

# Flatten and batch each window into individual tensors
data_windowed_dataset = data_windowed_dataset.flat_map(lambda window: window.batch(sequence_length))
label_windowed_dataset = label_windowed_dataset.flat_map(lambda window: window.batch(sequence_length))

# Zip datasets
combined_dataset = tf.data.Dataset.zip((data_windowed_dataset,label_windowed_dataset))


# Batch the dataset
batched_dataset = combined_dataset.batch(batch_size)

# Example iteration
for features, label in batched_dataset.take(2):
    print("Features:")
    print(features.numpy())
    print("Labels:")
    print(label.numpy())
```

Here, we create separate datasets for the input time series and the labels. We apply the windowing and flattening logic independently to each dataset. Then, using `tf.data.Dataset.zip`, we merge the two windowed and flattened datasets into a single dataset containing pairs of input sequences and label sequences. The subsequent `batch` method then batches these pairs.

Finally, let's illustrate how to handle a sampling rate, which introduces gaps within the subsequences themselves:

```python
import tensorflow as tf
import numpy as np

# Sample time series data
time_series_data = np.arange(100, dtype=np.float32)

# Configuration parameters
sequence_length = 5
sequence_stride = 2
sampling_rate = 3 # sampling rate
batch_size = 8


# Create a TensorFlow Dataset from the NumPy array
dataset = tf.data.Dataset.from_tensor_slices(time_series_data)

# Create an index dataset
indices_dataset = tf.data.Dataset.range(time_series_data.shape[0])

#Apply windowing and striding
indices_windowed_dataset = indices_dataset.window(
    size = (sequence_length-1)*sampling_rate+1,
    shift = sequence_stride,
    drop_remainder = True
)

indices_windowed_dataset = indices_windowed_dataset.flat_map(
    lambda window : window.batch((sequence_length-1)*sampling_rate+1)
)

# Sample from the indices, and then from the main dataset
sampled_dataset = indices_windowed_dataset.map(
    lambda indices : tf.gather(dataset.as_numpy_iterator().get_next(),indices[::sampling_rate])
)

#Batch the dataset
batched_dataset = sampled_dataset.batch(batch_size)

for batch in batched_dataset.take(2):
    print(batch.numpy())
```
In this final scenario, we introduce `sampling_rate`. We begin by creating an index dataset of the same length as our time series. We apply windowing to the index dataset, adjusting the size of the window to account for the sampling rate. For example, given a `sequence_length` of 5, and a `sampling_rate` of 3, the required size of the index dataset becomes (5-1)*3+1 = 13. The idea here is that we will sample elements from the original dataset, according to the sampled indices, after creating these large windows. `tf.gather` is then used to select values from the original dataset at these specific indices, but only considering indices at a rate indicated by `sampling_rate`. This means that indices [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] become, through the stride operator, [0, 3, 6, 9, 12] and are then used to sample the original data.

In essence, while no single TensorFlow Dataset operation provides the precise API of `timeseries_dataset_from_array`, its functionality can be replicated and even surpassed using the primitives provided by the `tf.data` module. This manual approach gives increased flexibility to modify the data pipeline in more sophisticated ways than would have been possible with the black box `timeseries_dataset_from_array` operation.

For further study, I would recommend examining the TensorFlow documentation pages pertaining to `tf.data.Dataset`, specifically focusing on the `window`, `flat_map`, `batch`, `map`, and `zip` methods. Additionally, reviewing examples involving time series data processing within the TensorFlow tutorials will solidify the concepts. Books discussing deep learning with TensorFlow often contain dedicated sections explaining how to process time series using the `tf.data` API.
