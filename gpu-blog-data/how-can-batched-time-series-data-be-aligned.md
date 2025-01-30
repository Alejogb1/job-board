---
title: "How can batched time series data be aligned for TensorFlow/Keras using `timeseries_dataset_from_array` and `TimeseriesGenerator`?"
date: "2025-01-30"
id: "how-can-batched-time-series-data-be-aligned"
---
The primary challenge in processing batched time series data for machine learning lies in ensuring consistent alignment of input sequences and target variables across the batch. Incorrect alignment can introduce spurious correlations, degrade model performance, and prevent convergence. TensorFlow's `timeseries_dataset_from_array` and Keras' `TimeseriesGenerator` both offer solutions, but each operates differently and requires a distinct approach to alignment. I have frequently observed during my experience developing predictive models for sensor data that subtle differences in implementation can have significant ramifications.

`timeseries_dataset_from_array` from TensorFlow's `tf.keras.utils` is designed for direct construction of batched datasets from NumPy arrays. It achieves alignment implicitly through the specification of sequence length, `sequence_stride`, and `sampling_rate`, as well as whether or not we are using offset targets. The function generates slices of the input data, effectively sliding a window of a specified size across the time series. The resultant batches naturally align because the window movement is uniform for each time series within the batch; no further adjustments are needed if your dataset has a continuous nature. Crucially, the function operates directly on the source data arrays which helps streamline data loading and reduce pre-processing overhead.

The key parameters for `timeseries_dataset_from_array` relating to alignment are `sequence_length` (the length of each input sequence) and `sequence_stride` (the step size by which the window moves). For example, if `sequence_length` is 10 and `sequence_stride` is 1, then the first input sequence will be indices 0 to 9, the second 1 to 10 and so on. The alignment comes from the uniform window shifting. If `targets` is also an array and `target_offset` is defined the alignment of the inputs and outputs is done. Consider the case where you want to predict one step ahead, in that case target should be the same as inputs, but offset by `target_offset` which is 1. If target is not an offset and you don't use `targets` then the output of the generator will just be the input time series batch.

Below is a code example illustrating the basic usage of `timeseries_dataset_from_array` to generate aligned batches of time series data:

```python
import numpy as np
import tensorflow as tf

# Sample time series data (batch_size, timesteps, features)
data = np.array([
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
    [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]],
    [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]
], dtype=np.float32) # Shape (3, 6, 2)
data_targets = np.array([
    [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]],
    [[4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]],
    [[5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
], dtype=np.float32)

sequence_length = 3
sequence_stride = 1
sampling_rate = 1
batch_size = 2
target_offset = 1

dataset = tf.keras.utils.timeseries_dataset_from_array(
    data,
    targets=data_targets,
    sequence_length=sequence_length,
    sequence_stride=sequence_stride,
    sampling_rate=sampling_rate,
    batch_size=batch_size,
    target_offset = target_offset
)


for batch in dataset:
    inputs, targets = batch
    print("Input batch shape:", inputs.shape)
    print("Target batch shape:", targets.shape)
    print("Input batch:")
    print(inputs.numpy())
    print("Target batch:")
    print(targets.numpy())
    break
```

In this example, `data` contains three time series, each with 6 time steps and 2 features. We set `sequence_length` to 3 and `sequence_stride` to 1. The resulting dataset contains batches where each input sequence is a segment of length 3, and the target is the sequence one step later due to the `target_offset` parameter. Note that I used `targets` and `target_offset` to define an offset target, to demonstrate how this can be used. If you wanted to do auto-regression where you are predicting the next step, then the targets would be the input data offset by the target_offset. The loop prints the first batch to demonstrate the shape and that targets are offset as expected.

On the other hand, `TimeseriesGenerator` from Keras, offers an alternative approach. Instead of using the direct array access offered by `timeseries_dataset_from_array`, `TimeseriesGenerator` acts as a Python generator. It yields batches on demand. Alignment is achieved through the `length`, `sampling_rate`, and `stride` parameters (note that in this case `stride` defines how many timesteps between the start of one sub-sequence and the next). The internal mechanism to generate sequences is functionally equivalent to what is done by `timeseries_dataset_from_array` though, meaning both generator will produce similar results given equivalent parameters. The difference is how we iterate and generate our data batches.

Here is an example using `TimeseriesGenerator`:

```python
from keras.preprocessing.sequence import TimeseriesGenerator
import numpy as np

# Sample time series data (batch_size, timesteps, features)
data = np.array([
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
    [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11], [12, 13]],
    [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]]
], dtype=np.float32) # Shape (3, 6, 2)
data_targets = np.array([
    [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14]],
    [[4, 5], [6, 7], [8, 9], [10, 11], [12, 13], [14, 15]],
    [[5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]
], dtype=np.float32)

length = 3
sampling_rate = 1
stride = 1
batch_size = 2

generator = TimeseriesGenerator(
    data,
    data_targets,
    length=length,
    sampling_rate=sampling_rate,
    stride=stride,
    batch_size=batch_size
)

for i in range(len(generator)):
    inputs, targets = generator[i]
    print("Input batch shape:", inputs.shape)
    print("Target batch shape:", targets.shape)
    print("Input batch:")
    print(inputs)
    print("Target batch:")
    print(targets)
    break
```

Here, I use similar parameters as the `timeseries_dataset_from_array` example. Note that `length` is equivalent to `sequence_length`, and `stride` represents the `sequence_stride` from `timeseries_dataset_from_array`. Also `targets` is used to construct an offset target. Again the loop prints the first batch showing the correct input and target shapes as well as the batch contents. While both examples appear to do the same, note that the iteration differs: in the `timeseries_dataset_from_array` we iterate the generated dataset object directly, in the `TimeseriesGenerator` we need to access the batches using an index.

A more complex scenario might involve time series with irregular intervals, or missing values. Both functions handle missing values in the same way: they do not have built-in handling of missing data. Any missing values must be handled before passing the data to the generator. If you are dealing with irregular intervals, the functions do not explicitly support it. You will have to resample the irregular series to a regular one before using either function. The advantage of `timeseries_dataset_from_array` is that as a `tf.data.Dataset` object it can be used more easily with the TensorFlow data pipeline.

Below is an example demonstrating the functionality of `timeseries_dataset_from_array` in the context of a more complex input that requires processing before training:

```python
import numpy as np
import tensorflow as tf

# Sample time series data with different lengths (batch_size, timesteps, features)
data = [
    [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
    [[2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
    [[3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15,16]]
] # Shape (3, varied, 2)

# Determine max length and pad
max_length = max([len(seq) for seq in data])
padded_data = np.zeros((len(data),max_length,2))
for i, seq in enumerate(data):
    padded_data[i,0:len(seq),:] = np.array(seq)

sequence_length = 3
sequence_stride = 1
sampling_rate = 1
batch_size = 2
target_offset = 1

dataset = tf.keras.utils.timeseries_dataset_from_array(
    padded_data,
    targets=padded_data,
    sequence_length=sequence_length,
    sequence_stride=sequence_stride,
    sampling_rate=sampling_rate,
    batch_size=batch_size,
    target_offset = target_offset
)


for batch in dataset:
    inputs, targets = batch
    print("Input batch shape:", inputs.shape)
    print("Target batch shape:", targets.shape)
    print("Input batch:")
    print(inputs.numpy())
    print("Target batch:")
    print(targets.numpy())
    break
```

In this example, the input time series have different lengths, which is common in many applications. To prepare the data for `timeseries_dataset_from_array` (which expects an array with a fixed number of timesteps), I padded all series to the length of the longest series. The rest of the code uses the same parameters as before. This shows a case where data must be preprocessed to have a uniform shape. After that, it can be passed directly into `timeseries_dataset_from_array` for batch generation.

In summary, both `timeseries_dataset_from_array` and `TimeseriesGenerator` offer mechanisms to align batched time series data. The choice depends on the specific use case and the need for direct array access (`timeseries_dataset_from_array`) or a more general Python generator (`TimeseriesGenerator`). Alignment in both cases is managed through the consistent application of parameters such as `sequence_length`, `sequence_stride`, `sampling_rate`, `length`, `stride` and `target_offset`. In practice, I find that `timeseries_dataset_from_array` provides a streamlined approach when working directly with NumPy arrays and the TensorFlow ecosystem, especially with its seamless integration with the data pipeline. I recommend exploring TensorFlow's documentation on `tf.data` and Keras’ documentation on preprocessing for more information on specific data handling techniques when using these functions. You will also find research papers on time series analysis useful, which contain several approaches you can take to deal with the specificities of temporal data, such as missing data or irregular time series.
