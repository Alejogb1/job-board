---
title: "How do you forecast univariate time series with LSTM in TensorFlow?"
date: "2024-12-16"
id: "how-do-you-forecast-univariate-time-series-with-lstm-in-tensorflow"
---

Alright, let's talk about forecasting univariate time series using lstms in tensorflow. This is a problem I've tackled more times than I can count, especially back when I was deeply involved in optimizing supply chain inventory for a retail outfit. It wasn’t just about predicting trends; it was about nailing the peaks and troughs in demand to minimize waste and maximize efficiency. So, trust me when I say this approach, when implemented meticulously, can deliver substantial results.

Forecasting time series data, particularly univariate time series, with long short-term memory (lstm) networks in tensorflow hinges on a structured methodology. The key is to understand that an lstm isn’t a magic bullet; it’s a powerful tool that needs to be wielded correctly. The first challenge is prepping your data. This isn't just about loading a csv and chucking it into the model; that’s a recipe for underperformance. It involves a few critical steps. Firstly, you need to decide on a lookback window – how many preceding time steps will the lstm consider when making a forecast? A good starting point is usually to experiment with different window sizes. Too small, and the model might miss crucial longer-term patterns. Too large, and you might introduce noise. Normalization is absolutely essential. Typically, scaling the data to the range [0, 1] or standardizing it to have zero mean and unit variance makes the network more stable during training. After that, creating sequences is the next hurdle. You take the normalized time series and break it up into overlapping windows of your chosen size, along with corresponding target values – i.e., the next value in the sequence you’re aiming to predict.

The architecture of the lstm network itself should be deliberately considered. For univariate forecasting, I typically start with a relatively simple stacked lstm setup. This consists of one or more lstm layers followed by a dense layer to map the lstm output to the single forecast value. Adding a dropout layer between lstm layers is a good practice to prevent overfitting. This will prevent the model from memorizing the training data instead of actually learning the underlying patterns. The choice of activation function in the final layer usually depends on the data range. For scaled data like [0, 1], a sigmoid function might be suitable. Otherwise, a linear activation (or no activation at all) is usually a good starting point, especially if your target range isn't constrained.

Then, there's the training process. The choice of optimizer and loss function are paramount. Adam is a decent general-purpose optimizer, while mean squared error (mse) is a typical loss function for regression tasks like this. Hyperparameter tuning, which can be time-consuming, is key here. You’ll want to experiment with the number of lstm units, the number of layers, the dropout rate, the learning rate of the optimizer, and, of course, the training epochs and batch size. This iterative fine-tuning phase directly influences the final performance of your model. Splitting the data into training and validation sets is imperative, and it would be prudent to include a final test set for proper evaluation of the model’s generalizability once you’ve chosen the optimal hyperparameters. Be cautious of overfitting; monitoring validation set performance during training is a critical part of achieving optimal results.

Finally, forecasting with your trained model is pretty straightforward. You take the last window of the time series, pass it through the network, get the prediction, and use that prediction to create the new window for the next step, continuing the rolling forecast as far as you need.

Let's get into some code examples to illustrate the points above.

```python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, lookback):
  xs, ys = [], []
  for i in range(len(data)-lookback-1):
    v = data[i:(i+lookback)]
    xs.append(v)
    ys.append(data[i+lookback])
  return np.array(xs), np.array(ys)

# example data: a synthetic timeseries for illustration
data = np.sin(np.linspace(0, 10*np.pi, 200)) + np.random.normal(0, 0.1, 200)
lookback = 10
# scale data
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1,1))
data = data.flatten()


X, y = create_sequences(data, lookback)
X = X.reshape(X.shape[0], X.shape[1], 1)

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0) # verbose=0 suppresses training log outputs
# Now, lets predict the next value.
last_window = data[-lookback:].reshape(1,lookback,1)
predicted_value = model.predict(last_window)
print(scaler.inverse_transform(predicted_value))


```

This initial snippet demonstrates the core logic behind creating sequences, building the basic lstm model, and a fundamental prediction.

Next, here’s an example incorporating a rolling prediction function.

```python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

def create_sequences(data, lookback):
    xs, ys = [], []
    for i in range(len(data) - lookback - 1):
        v = data[i:(i+lookback)]
        xs.append(v)
        ys.append(data[i+lookback])
    return np.array(xs), np.array(ys)

def lstm_model(lookback):
    return tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

def rolling_forecast(model, data, lookback, forecast_steps, scaler):
    forecasts = []
    current_window = data[-lookback:].reshape(1, lookback, 1)
    for _ in range(forecast_steps):
        predicted_value = model.predict(current_window)
        forecasts.append(scaler.inverse_transform(predicted_value)[0,0])
        # Append the prediction to the end of the current window and remove the oldest value.
        current_window = np.concatenate((current_window[:,1:,:], predicted_value.reshape(1,1,1)), axis=1)
    return forecasts

# example data: a synthetic timeseries for illustration
data = np.sin(np.linspace(0, 10*np.pi, 200)) + np.random.normal(0, 0.1, 200)
lookback = 10
forecast_steps = 5 # how many steps we want to forecast ahead
# scale data
scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1,1))
data = data.flatten()

X, y = create_sequences(data, lookback)
X = X.reshape(X.shape[0], X.shape[1], 1)

model = lstm_model(lookback)

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=100, verbose=0)


forecasted_values = rolling_forecast(model, data, lookback, forecast_steps, scaler)
print(forecasted_values)
```

This shows how to do a rolling forecast, by sequentially taking the last window, making a prediction, then appending that prediction to the end and discarding the oldest data point. This continues for the number of steps you specify.

Finally, a more complete example, including data split and evaluation.

```python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def create_sequences(data, lookback):
    xs, ys = [], []
    for i in range(len(data) - lookback - 1):
        v = data[i:(i+lookback)]
        xs.append(v)
        ys.append(data[i+lookback])
    return np.array(xs), np.array(ys)

def lstm_model(lookback):
  return tf.keras.Sequential([
        tf.keras.layers.LSTM(50, activation='relu', input_shape=(lookback, 1), return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(50, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

def rolling_forecast(model, data, lookback, forecast_steps, scaler):
    forecasts = []
    current_window = data[-lookback:].reshape(1, lookback, 1)
    for _ in range(forecast_steps):
        predicted_value = model.predict(current_window)
        forecasts.append(scaler.inverse_transform(predicted_value)[0,0])
        current_window = np.concatenate((current_window[:,1:,:], predicted_value.reshape(1,1,1)), axis=1)
    return forecasts

# example data: a synthetic timeseries for illustration
data = np.sin(np.linspace(0, 10*np.pi, 500)) + np.random.normal(0, 0.1, 500)
lookback = 10
forecast_steps = 5 # how many steps we want to forecast ahead

scaler = MinMaxScaler()
data = scaler.fit_transform(data.reshape(-1,1))
data = data.flatten()


X, y = create_sequences(data, lookback)
X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, shuffle=False)
model = lstm_model(lookback)
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=100, verbose=0)
predicted = model.predict(X_test)
predicted_unscaled = scaler.inverse_transform(predicted)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1,1))

mse = mean_squared_error(y_test_unscaled, predicted_unscaled)
print(f'MSE: {mse}')
forecast_data = data
forecasted_values = rolling_forecast(model, forecast_data, lookback, forecast_steps, scaler)
print(f"Forecasted values: {forecasted_values}")

```

This example now splits the data and computes and prints mean square error on the test set, providing a full working example of the training process.

For deeper understanding, I recommend exploring these resources: "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville for a solid theoretical base, and “Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow” by Aurélien Géron for a more pragmatic implementation-focused guide. The original lstm paper, by Hochreiter and Schmidhuber, is also essential, for understanding the details behind lstm operation. "Time Series Analysis: Forecasting and Control" by George E.P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Greta M. Ljung is also a great resource for foundational understanding of time series analysis. The material from these resources will undoubtedly reinforce your understanding and help you avoid common pitfalls when employing lstms for time series forecasting. Remember, patience and iterative experimentation are key when you're optimizing these models, and paying careful attention to the underlying mathematics, data, and training methodology are vital for success.
