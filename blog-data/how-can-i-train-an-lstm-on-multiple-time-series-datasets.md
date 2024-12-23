---
title: "How can I train an LSTM on multiple time series datasets?"
date: "2024-12-23"
id: "how-can-i-train-an-lstm-on-multiple-time-series-datasets"
---

Let’s tackle this challenge of training an LSTM on multiple time series datasets head-on; it’s a situation I’ve navigated quite a few times in my work. It’s a little more intricate than single series forecasting, but fundamentally, the key lies in how we structure and present the data to the LSTM. The core principle is that, while each time series may represent a different system or entity, the LSTM can learn patterns across them if the input structure is consistent.

The first hurdle is data preprocessing. We can't simply feed raw, disparate time series into the model. Often, the series will have different scales and ranges. Normalization is crucial here. A typical approach is to normalize each series independently before combining them, ensuring each contributes equally to the learning process. I've found that using a min-max scaler or a standard scaler works well. We're aiming for a consistent range, often between 0 and 1, or mean 0 and standard deviation 1, respectively. If the series are stationary, standard scaling is usually my preference, but min-max can handle non-stationary cases a bit more gracefully. I prefer to keep these pre-processing transforms separate for each individual time series, before combining them, as it reflects real world situations.

Another critical aspect is dealing with differing lengths of the time series. We can't have a tensor with variable lengths as input to an LSTM. I've generally used a combination of padding and masking. We pad the shorter time series with a predefined value, usually zero, to match the length of the longest series in your batch. Masking prevents the LSTM from considering padded values when learning. The model learns to ignore those padding values, so they don't skew the training.

Now let’s move to the architecture. The LSTM will need to handle multiple input features, one for each time series. So, rather than a single feature dimension, your input will have a higher dimensionality - the number of time series. The output, however, could be a single prediction for all time series or a prediction for each time series. It all depends on the specific goal of your model. If we are doing some sort of clustering, then one might have a single output. If we are doing forecasting for each series, then multiple outputs are needed.

Another method which is beneficial to consider involves introducing a time-invariant feature associated with each time series. I’ve found this crucial, especially in cases where we want the model to distinguish, say, individual assets among different stock time series. This could be a one-hot encoded vector or a simple numerical identifier assigned to each unique series. By concatenating this identifier with each time step’s data, we provide the LSTM with the context of which series it is currently processing. This helps the network in learning specific patterns which are linked to individual time series. In some cases, I have also added the type of time series as categorical data too, which can be highly beneficial in cases where there are multiple sources of data, or the data is collected by different instrumentation.

Now, let’s delve into some code examples. These snippets are in Python using TensorFlow, which is my framework of choice these days.

**Example 1: Independent Time Series Forecasting**

This demonstrates forecasting multiple time series with a single LSTM output layer which produces individual forecasts.

```python
import tensorflow as tf
import numpy as np

def create_dataset(series_list, sequence_length, stride=1):
  """Creates input/output sequence pairs for LSTM training."""
  x, y = [], []
  for series in series_list:
    for i in range(0, len(series) - sequence_length, stride):
        x.append(series[i : i + sequence_length])
        y.append(series[i + sequence_length])
  return np.array(x), np.array(y)

def lstm_model(input_shape, num_series):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=input_shape),
      tf.keras.layers.LSTM(64, return_sequences=False),
      tf.keras.layers.Dense(num_series) # Prediction for each series.
    ])
    return model

# Mock data: Three time series of varying lengths
series1 = np.sin(np.linspace(0, 10*np.pi, 100))
series2 = np.cos(np.linspace(0, 5*np.pi, 70)) * 0.5
series3 = np.tan(np.linspace(0, 2*np.pi, 120)) * 0.2

series_list = [series1, series2, series3]
sequence_length = 20

# Preprocessing (example, scale each series between 0 and 1)
min_values = np.array([np.min(series) for series in series_list])
max_values = np.array([np.max(series) for series in series_list])

normalized_series_list = [(series - min_val) / (max_val - min_val) for series, min_val, max_val in zip(series_list, min_values, max_values)]
padded_series_list = [tf.keras.utils.pad_sequences([series], maxlen=max([len(s) for s in normalized_series_list]), padding='post', dtype='float32')[0] for series in normalized_series_list]
num_series = len(padded_series_list)

padded_series_array = np.array(padded_series_list).T # Transpose for input to be (num_steps, num_series)

x, y = create_dataset(padded_series_array, sequence_length)

model = lstm_model(input_shape=(sequence_length, num_series), num_series=num_series)

model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=10, verbose=0)

print("Model trained successfully")
```

**Example 2: Time Series with Individual Identifiers**

Here we demonstrate the use of identifiers for each series, useful for situations where you need to teach the model individual pattern associated with the time series.

```python
import tensorflow as tf
import numpy as np

def create_dataset_with_ids(series_list, sequence_length, stride=1, ids=None):
  """Creates input/output sequence pairs with time-series ids."""
  x, y = [], []
  for idx, series in enumerate(series_list):
      for i in range(0, len(series) - sequence_length, stride):
          seq = series[i : i + sequence_length]
          if ids is not None:
             id_vector = np.full((sequence_length, len(ids)), 0, dtype=float)
             id_vector[:,idx] = 1
             seq = np.concatenate([np.array(seq).reshape(sequence_length, 1), id_vector], axis=-1)

          x.append(seq)
          y.append(series[i + sequence_length])
  return np.array(x), np.array(y)

def lstm_model_with_ids(input_shape, num_series):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=input_shape),
      tf.keras.layers.LSTM(64, return_sequences=False),
      tf.keras.layers.Dense(1)  # Output is a single prediction for the series of interest
    ])
    return model

# Mock data: Three time series of varying lengths
series1 = np.sin(np.linspace(0, 10*np.pi, 100))
series2 = np.cos(np.linspace(0, 5*np.pi, 70)) * 0.5
series3 = np.tan(np.linspace(0, 2*np.pi, 120)) * 0.2

series_list = [series1, series2, series3]
sequence_length = 20
ids = ['series1', 'series2', 'series3']

# Preprocessing
min_values = np.array([np.min(series) for series in series_list])
max_values = np.array([np.max(series) for series in series_list])
normalized_series_list = [(series - min_val) / (max_val - min_val) for series, min_val, max_val in zip(series_list, min_values, max_values)]
padded_series_list = [tf.keras.utils.pad_sequences([series], maxlen=max([len(s) for s in normalized_series_list]), padding='post', dtype='float32')[0] for series in normalized_series_list]

x, y = create_dataset_with_ids(padded_series_list, sequence_length, ids=ids)
num_series = len(ids)

model = lstm_model_with_ids(input_shape=(sequence_length, 1 + num_series), num_series=num_series)

model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=10, verbose=0)

print("Model trained successfully with individual identifiers.")
```

**Example 3: Using Masking**

Demonstrates the importance of masking when dealing with padded sequences.

```python
import tensorflow as tf
import numpy as np

def create_dataset_with_padding(series_list, sequence_length, stride=1):
  """Creates input/output sequence pairs for LSTM training."""
  x, y = [], []
  for series in series_list:
    for i in range(0, len(series) - sequence_length, stride):
        x.append(series[i : i + sequence_length])
        y.append(series[i + sequence_length])

  return np.array(x), np.array(y)

def lstm_model_with_masking(input_shape):
    model = tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=input_shape),
      tf.keras.layers.Masking(mask_value=0.0),
      tf.keras.layers.LSTM(64, return_sequences=False),
      tf.keras.layers.Dense(1)
    ])
    return model


# Mock data: Three time series of varying lengths
series1 = np.sin(np.linspace(0, 10*np.pi, 100))
series2 = np.cos(np.linspace(0, 5*np.pi, 70)) * 0.5
series3 = np.tan(np.linspace(0, 2*np.pi, 120)) * 0.2

series_list = [series1, series2, series3]
sequence_length = 20

# Preprocessing
min_values = np.array([np.min(series) for series in series_list])
max_values = np.array([np.max(series) for series in series_list])
normalized_series_list = [(series - min_val) / (max_val - min_val) for series, min_val, max_val in zip(series_list, min_values, max_values)]

padded_series_list = [tf.keras.utils.pad_sequences([series], maxlen=max([len(s) for s in normalized_series_list]), padding='post', dtype='float32')[0] for series in normalized_series_list]
num_series = len(padded_series_list)

padded_series_array = np.array(padded_series_list).T # Transpose for input to be (num_steps, num_series)

x, y = create_dataset_with_padding(padded_series_array, sequence_length)

model = lstm_model_with_masking(input_shape=(sequence_length, num_series))

model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=10, verbose=0)

print("Model trained successfully with masking.")
```

For deeper understanding, I recommend exploring some resources. "Deep Learning with Python" by François Chollet is a good starting point. For a rigorous academic perspective, "Deep Learning" by Goodfellow, Bengio, and Courville is invaluable. For time series specific methods, "Time Series Analysis" by James Hamilton offers strong theoretical background. When it comes to practical implementation, Tensorflow documentation, especially the sections on recurrent layers and data processing, is essential for working with these codes. This should provide a good basis on how to handle the nuances of training an LSTM on multiple time series.
