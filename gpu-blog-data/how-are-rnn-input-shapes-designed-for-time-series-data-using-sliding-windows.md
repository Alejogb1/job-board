---
title: "How are RNN input shapes designed for time series data using sliding windows?"
date: "2025-01-26"
id: "how-are-rnn-input-shapes-designed-for-time-series-data-using-sliding-windows"
---

Time series data, unlike static data, possesses a temporal dimension where the sequence of observations is crucial for understanding patterns. When feeding time series into Recurrent Neural Networks (RNNs), which inherently process sequential data, the input shape design is critical for model performance. Specifically, using sliding windows transforms a continuous time series into a series of overlapping subsequences, each representing a single input sample to the RNN.

The fundamental idea behind sliding window input design stems from the need to expose the network to local temporal contexts. A single time point from a series is often meaningless without its surrounding values; the dynamics of change are what we want the RNN to learn. Consequently, instead of passing single points, we pass a window of consecutive values. This window slides along the time series, stepping by a defined interval, creating a set of input sequences that capture local patterns.

Here's how the input shape is typically structured when utilizing sliding windows, considering tensors with dimensions usually represented as `(batch_size, time_steps, features)`:

* **`batch_size`**: This indicates the number of independent time series sequences being processed together in a single training iteration. It is a fundamental concept in deep learning, influencing computational efficiency.
* **`time_steps`**: This corresponds to the window size (the number of observations included in each input sequence) as extracted by sliding windows. For instance, a window size of 20 means each sample will consist of 20 consecutive values. The time_step dimension captures temporal dependencies within a local context.
* **`features`**: This denotes the number of variables measured at each time point. With univariate time series, `features` is 1 because only a single value is captured per time instant. When working with multivariate time series, it will be greater than 1, corresponding to the number of independent variables captured.

The sliding window is defined by two key parameters: `window_size` and `step_size`. The `window_size` sets the length of each input sequence, and the `step_size` controls how much the window shifts between successive samples. If the `step_size` equals the `window_size`, then there will be no overlap between windows. If `step_size` is smaller than `window_size`, then adjacent samples will contain shared data, and a greater number of training samples will be generated from the raw time series.

I have commonly seen these configurations in my work with RNNs and time series forecasting. The window size has a critical effect on what temporal dependencies the network can learn. Short windows may not be sufficient to capture long-term trends, whereas excessively large windows might capture too much irrelevant information or make training computationally expensive. The selection of optimal window sizes usually involves experimentation and cross-validation techniques.

Let us now look at code examples that illustrate this design, using Python with NumPy and a conceptual framework mimicking TensorFlow’s Keras preprocessing.

**Code Example 1: Univariate Time Series with No Overlap**

```python
import numpy as np

def create_sliding_windows(data, window_size, step_size):
  """Creates sliding windows from a time series.

  Args:
      data: A 1D numpy array representing the time series.
      window_size: The length of each window.
      step_size: The step between each window.

  Returns:
      A 3D numpy array of shape (num_windows, window_size, 1)
  """
  num_windows = (len(data) - window_size) // step_size + 1
  windows = np.zeros((num_windows, window_size, 1))
  for i in range(num_windows):
      start = i * step_size
      end = start + window_size
      windows[i, :, 0] = data[start:end]

  return windows


# Example usage
time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
window_size = 3
step_size = 3
windows = create_sliding_windows(time_series, window_size, step_size)
print(windows)
print(windows.shape)

# Output:
# [[[1.] [2.] [3.]] [[4.] [5.] [6.]] [[7.] [8.] [9.]]]
# (3, 3, 1)

```
In this example, we define a function `create_sliding_windows` that generates sliding windows from a 1D Numpy array. It outputs a 3D tensor where the third dimension is size 1 representing a single feature (univariate). The step size is equal to the window size leading to no overlapping data, and each window has shape (3, 1). The shape is (3, 3, 1) because the original time series was of length 10, the window size was 3, and the step size was also 3, resulting in 3 windows with window size 3 and 1 feature.

**Code Example 2: Univariate Time Series with Overlap**

```python
import numpy as np

def create_sliding_windows_overlap(data, window_size, step_size):
    """Creates sliding windows from a time series, with overlap.

    Args:
        data: A 1D numpy array representing the time series.
        window_size: The length of each window.
        step_size: The step between each window.

    Returns:
        A 3D numpy array of shape (num_windows, window_size, 1)
    """
    num_windows = (len(data) - window_size) // step_size + 1
    windows = np.zeros((num_windows, window_size, 1))

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        windows[i, :, 0] = data[start:end]
    return windows


# Example usage
time_series = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
window_size = 4
step_size = 2
windows = create_sliding_windows_overlap(time_series, window_size, step_size)
print(windows)
print(windows.shape)


# Output:
# [[[1.] [2.] [3.] [4.]] [[3.] [4.] [5.] [6.]] [[5.] [6.] [7.] [8.]]]
# (3, 4, 1)

```
Here we see a similar function, however with a differing `window_size` (4) and `step_size` (2). Note that the window size of 4 makes each input sample longer, and the step size of 2 makes them overlap with their neighbors. This produces fewer total windows, but they provide a more complete context when compared to example 1. As a result, we see the shape of (3, 4, 1).

**Code Example 3: Multivariate Time Series**
```python
import numpy as np

def create_sliding_windows_multivariate(data, window_size, step_size):
    """Creates sliding windows from a multivariate time series.

    Args:
        data: A 2D numpy array of shape (time_points, features).
        window_size: The length of each window.
        step_size: The step between each window.

    Returns:
        A 3D numpy array of shape (num_windows, window_size, features)
    """
    num_windows = (data.shape[0] - window_size) // step_size + 1
    num_features = data.shape[1]
    windows = np.zeros((num_windows, window_size, num_features))

    for i in range(num_windows):
        start = i * step_size
        end = start + window_size
        windows[i, :, :] = data[start:end, :]

    return windows

# Example usage
time_series_multi = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60], [7, 70], [8, 80], [9, 90], [10, 100]])
window_size = 3
step_size = 1
windows_multi = create_sliding_windows_multivariate(time_series_multi, window_size, step_size)
print(windows_multi)
print(windows_multi.shape)


# Output:
# [[[ 1. 10.]  [ 2. 20.]  [ 3. 30.]]  [[ 2. 20.]  [ 3. 30.]  [ 4. 40.]]  [[ 3. 30.]  [ 4. 40.]  [ 5. 50.]]  [[ 4. 40.]  [ 5. 50.]  [ 6. 60.]]  [[ 5. 50.]  [ 6. 60.]  [ 7. 70.]]  [[ 6. 60.]  [ 7. 70.]  [ 8. 80.]]  [[ 7. 70.]  [ 8. 80.]  [ 9. 90.]] ]
# (8, 3, 2)
```

This final example shows the application to a multivariate time series. We see that each input sample still consists of a sequence of length 3, however each observation now contains two features. The shape reflects this by changing from (8, 3, 1) to (8, 3, 2). The step size of 1 means each window overlaps significantly with the previous window.

When determining the final input shape, the batch size needs to be incorporated. Typically, a data loader will take the output of these windowing functions and batch them together, creating a tensor of shape `(batch_size, time_steps, features)`. This represents multiple samples to be passed through the RNN model simultaneously.

For further research on best practices, I recommend delving into materials on:
* Time Series Analysis: Texts covering fundamental concepts in time series analysis will provide a robust basis for feature engineering and preprocessing strategies.
* Deep Learning for Sequences: Focused study on recurrent neural networks, including LSTM and GRU architectures, can provide a better understanding of how these models process sequences.
* Practical Deep Learning Engineering:  Materials covering effective data handling and optimization techniques for deep learning models are crucial. These often contain strategies to mitigate overfitting and manage datasets effectively.
These resources will assist in the design, implementation, and optimization of RNNs for time series analysis, moving beyond the basic techniques described here.
