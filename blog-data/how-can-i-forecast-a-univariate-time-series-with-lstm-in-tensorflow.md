---
title: "How can I forecast a univariate time series with LSTM in TensorFlow?"
date: "2024-12-16"
id: "how-can-i-forecast-a-univariate-time-series-with-lstm-in-tensorflow"
---

Let’s tackle time series forecasting with lstms in TensorFlow, because, truthfully, it's something I’ve spent a decent chunk of my career refining. I remember one particularly challenging project back at 'Acme Analytics' where we had to predict daily stock prices based solely on historical trends—no exogenous factors, just pure time series data. The learning curve was steep, but we eventually landed on a robust system. The key, as with most complex tasks, lies in breaking it down into manageable steps.

At its core, forecasting a univariate time series using lstms involves leveraging the recurrent nature of these networks to understand temporal dependencies within your data. The "univariate" part signifies you are only dealing with a single sequence, unlike multivariate scenarios with multiple contributing sequences. Let's approach this methodically, focusing on preprocessing, model building, and finally, prediction.

First, consider preprocessing. Your time series data isn't usually in a state directly ingestible by an lstm. We need to transform it into a format that emphasizes temporal relationships. This generally involves creating 'windowed' sequences. Imagine your time series as a long sequence of data points: [x1, x2, x3, x4, x5, x6, ...]. We’ll need to split this into sequences of a fixed length (say, n), where each sequence is then paired with its next subsequent data point as a target. For instance, with n=3, you would have input sequences like [x1, x2, x3] with target x4, then [x2, x3, x4] with target x5, and so forth. This "sliding window" approach creates training samples that the lstm can then learn from. Furthermore, it is often crucial to normalize your data—scaling it, for example, between 0 and 1 using min-max scaling, or standardizing it to have zero mean and unit variance. This can prevent the model from becoming unstable during training and also improve convergence speed.

Now, for the model architecture itself, TensorFlow’s `tf.keras` api provides an elegant way to build such networks. The standard approach begins with an `lstm` layer. It's important to experiment with the number of units (nodes) in this layer to find what best fits your data. Too few, and the model might be underfitting, too many and it could overfit. Often, adding a second or even third `lstm` layer, with possibly some `dropout` in between for regularization, significantly boosts performance. Lastly, the output layer generally employs a dense (fully connected) layer, typically with a single node since we are outputting a single forecasted value. Activation function choices also matters; for a regression task like forecasting, a linear activation (i.e., none) is usually the most appropriate.

Let's get into some code examples. Remember, for detailed mathematical treatment, I highly recommend *“Deep Learning”* by Goodfellow, Bengio, and Courville or *“Time Series Analysis: Forecasting and Control”* by Box, Jenkins, Reinsel, and Ljung as starting points; these go beyond superficial explanations and dive deep into the theory behind these methods.

Here's a basic example in TensorFlow:

```python
import tensorflow as tf
import numpy as np

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# dummy time series data
time_series = np.sin(np.linspace(0, 10 * np.pi, 1000)) # Example
seq_length = 50

X, y = create_sequences(time_series, seq_length)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=64, input_shape=(X.shape[1], 1)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, verbose=0)  # Silence verbose output for brevity
forecast = model.predict(X[-1:].reshape(1, seq_length, 1))

print(f"Forecast: {forecast[0][0]:.4f}")
```

This script demonstrates basic sequence generation and a simple lstm network. The `create_sequences` function transforms the raw time series into input/output pairs, then the model is compiled and trained, and finally, a forecast is generated using the last data sequence as input. Remember that the loss function used here is ‘mse’ meaning “Mean Squared Error”, this is appropriate for regression tasks.

Here's a slightly more complex model, including regularization with dropout:

```python
import tensorflow as tf
import numpy as np

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


time_series = np.sin(np.linspace(0, 10 * np.pi, 1000))
seq_length = 50

X, y = create_sequences(time_series, seq_length)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=64, input_shape=(X.shape[1], 1), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(units=64),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, verbose=0)  # Silence verbose output for brevity
forecast = model.predict(X[-1:].reshape(1, seq_length, 1))

print(f"Forecast: {forecast[0][0]:.4f}")

```

This example includes a second lstm layer and `dropout` layers for improved robustness. Note the use of `return_sequences=True` in the first lstm layer, which means the output is a sequence allowing it to be fed to the second lstm layer. You also need to consider the input shape parameter during the first lstm layer.

Finally, an example demonstrating scaling using MinMaxScaler:

```python
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Sample data
time_series = np.sin(np.linspace(0, 10 * np.pi, 1000))
seq_length = 50

# Scaling
scaler = MinMaxScaler()
scaled_time_series = scaler.fit_transform(time_series.reshape(-1, 1))

X, y = create_sequences(scaled_time_series, seq_length)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=64, input_shape=(X.shape[1], 1)),
    tf.keras.layers.Dense(1)
])


model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=20, verbose=0) # Silence verbose output for brevity

# Predict
last_sequence = scaled_time_series[-seq_length:].reshape(1, seq_length, 1) # Get scaled version of last sequence
forecast_scaled = model.predict(last_sequence)
forecast = scaler.inverse_transform(forecast_scaled)[0][0] # Inverse transform back to original scale

print(f"Forecast: {forecast:.4f}")


```

This example demonstrates scaling our time-series data using `MinMaxScaler`. Note the inverse transformation after forecasting so that we are dealing with the original scale of the data. It's a step often overlooked which might give nonsensical results if we leave it out.

Forecasting with lstms requires careful tuning and a solid understanding of time series analysis, but it is an extremely valuable tool once properly mastered. There are many other important considerations, such as choosing the correct window size, optimizing the hyperparameters, and, for real-world data, dealing with anomalies and missing values. Experimentation is critical, and these examples should give you a solid foundation to begin your exploration into time series forecasting using TensorFlow and lstms. Remember, the devil is in the details, so always double-check your data preprocessing steps and model architecture.
