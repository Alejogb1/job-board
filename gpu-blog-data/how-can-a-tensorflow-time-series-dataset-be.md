---
title: "How can a TensorFlow time series dataset be windowed into sequences?"
date: "2025-01-30"
id: "how-can-a-tensorflow-time-series-dataset-be"
---
Windowing time series data in TensorFlow is a fundamental preprocessing step for training sequence models like LSTMs or Transformers. It involves converting a long, continuous series into overlapping or non-overlapping sub-sequences of a fixed length, each suitable for individual input to a model. This process transforms the data from a single large tensor to a dataset of smaller, interconnected tensors, enabling models to learn temporal dependencies within the data.

The core principle is sliding a window of specified size across the time series data. The position of this window is determined by a defined step size or stride. For example, a window of 10 with a stride of 1 would result in many overlapping sequences, while a stride of 10 would yield non-overlapping sequences. Each resulting window then forms a new observation (or sample) in the transformed dataset.

TensorFlow provides the `tf.data.Dataset` API which incorporates robust functionality for efficient and scalable time series windowing. I frequently use this API in my work developing predictive maintenance systems, and have found it considerably simplifies the process, allowing me to focus more on model design and less on data wrangling.

Specifically, the `window()` method of a `tf.data.Dataset` object is the central tool. It accepts several parameters: `size` (the length of the window), `shift` (the stride), `stride` (the interval at which the window moves), `drop_remainder` (whether to discard the last window if it doesn't have sufficient elements). The `flat_map()` method is also critical, as the `window()` method initially creates a nested dataset, with each element representing a window, thus we use flat_map to unnest the dataset.

Let's illustrate with examples based on some common scenarios. Assume we have a simple synthetic time series, such as a sequence of temperature readings recorded at hourly intervals.

**Example 1: Non-overlapping Windows**

In this example, we want to create completely separate sequences of length 12, suitable for training a model to learn daily temperature patterns. This means the window size will be 12 (e.g. 12 hourly readings), and the shift and stride will be equal to the window size, resulting in no overlaps.

```python
import tensorflow as tf

# Simulate hourly temperature readings for 10 days
time_series = tf.range(10 * 24, dtype=tf.float32) # 240 data points


window_size = 12
shift = window_size
stride = window_size
dataset = tf.data.Dataset.from_tensor_slices(time_series)


windowed_dataset = dataset.window(window_size, shift=shift, stride=stride, drop_remainder=True)
windowed_dataset = windowed_dataset.flat_map(lambda window: window.batch(window_size))

# Verify shape of the first batch
for batch in windowed_dataset.take(1):
    print(batch.shape) # Output: (12,)
    print(batch) # Output: a tensor with 12 elements (e.g. tf.Tensor([0., 1., 2., ..., 11.], shape=(12,), dtype=float32))


# Print shape of whole dataset
print(list(windowed_dataset.as_numpy_iterator())[0].shape) # Output: (12,)
print(len(list(windowed_dataset.as_numpy_iterator()))) # Output: 20, since 240/12 = 20
```

Here, the `tf.range()` call creates our 240-element synthetic time series. The `from_tensor_slices` call creates a `tf.data.Dataset` object. Then, the `window()` method partitions the data into non-overlapping windows of size 12, the `drop_remainder=True` argument discarding any remaining data that does not form a complete window. The `flat_map()` method with `batch()` converts the nested dataset into a flat dataset where each sample is a tensor of length 12, ready for use as model inputs. The output shows the shape (12,) of the first batch, and its 12 elements. We also verify the output dataset structure by printing the shape and size of the dataset.

**Example 2: Overlapping Windows**

Now let's create overlapping windows, commonly used when more data is required or when a smoothing effect is desirable. We will keep a window size of 12, but a stride of 1, resulting in significantly more overlapping sequences. This provides the model with more samples and a better representation of transitions in the data.

```python
import tensorflow as tf

# Simulate hourly temperature readings for 10 days
time_series = tf.range(10 * 24, dtype=tf.float32) # 240 data points


window_size = 12
shift = 1
stride = 1
dataset = tf.data.Dataset.from_tensor_slices(time_series)

windowed_dataset = dataset.window(window_size, shift=shift, stride=stride, drop_remainder=True)
windowed_dataset = windowed_dataset.flat_map(lambda window: window.batch(window_size))

# Verify shape of the first batch
for batch in windowed_dataset.take(1):
    print(batch.shape) # Output: (12,)
    print(batch) # Output: a tensor with 12 elements (e.g. tf.Tensor([0., 1., 2., ..., 11.], shape=(12,), dtype=float32))


# Print shape of whole dataset
print(list(windowed_dataset.as_numpy_iterator())[0].shape) # Output: (12,)
print(len(list(windowed_dataset.as_numpy_iterator()))) # Output: 229, because 240 - window size +1 is 240 -12 +1 = 229

```

The only changes from the previous example are that the `shift` and `stride` are both set to 1. This change results in 229 windows, as each window shifts forward by one time step compared to the previous one. Consequently, the resulting dataset is much larger than the one in example one. The output is printed in the same fashion, verifying shape and size of the windowed data.

**Example 3: Creating Input/Target Pairs**

Often, time series models are trained to predict the next value in the sequence or to forecast a range of values into the future. This implies dividing each window into input features and target values. A common approach is to use all elements of the window except for the last one for input features, and the last element of the window for the target value.

```python
import tensorflow as tf

# Simulate hourly temperature readings for 10 days
time_series = tf.range(10 * 24, dtype=tf.float32) # 240 data points


window_size = 12
shift = 1
stride = 1
dataset = tf.data.Dataset.from_tensor_slices(time_series)

windowed_dataset = dataset.window(window_size, shift=shift, stride=stride, drop_remainder=True)
windowed_dataset = windowed_dataset.flat_map(lambda window: window.batch(window_size))

def split_window(window):
  input_features = window[:-1]
  target = window[-1:]
  return input_features, target

split_dataset = windowed_dataset.map(split_window)

# Verify shape of the first batch
for features, target in split_dataset.take(1):
    print(features.shape) # Output: (11,)
    print(target.shape) # Output: (1,)
    print(features) # Output: a tensor of 11 elements
    print(target) # Output: a tensor with one element


# Print shape of whole dataset
print(list(split_dataset.as_numpy_iterator())[0][0].shape) # Output: (11,)
print(list(split_dataset.as_numpy_iterator())[0][1].shape) # Output: (1,)
print(len(list(split_dataset.as_numpy_iterator())))  # Output: 229

```

The initial windowing remains the same as in example 2, producing overlapping windows. Then, a new function `split_window` splits each window into input features and the single target. The dataset is updated using the `map()` method, effectively converting it to a tuple of (features, target) elements. The output displays the respective shapes, illustrating the transformation from a single window to an input/output pair.

For further exploration of time series processing using TensorFlow, I would recommend consulting resources that extensively cover the `tf.data` API, specifically its methods for creating and transforming datasets. Additionally, studying examples that demonstrate the application of recurrent neural networks and Transformers to time series data within the TensorFlow ecosystem will deepen your understanding of the practical use of the windowing process. I suggest looking into the Keras documentation associated with each layer and the Tensorflow tutorials specific to sequence data modeling. Moreover, reading journal papers that discuss the latest techniques in time series forecasting and anomaly detection can help in designing complex and sophisticated models with appropriate data preprocessing strategies.
