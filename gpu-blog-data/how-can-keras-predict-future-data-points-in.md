---
title: "How can Keras predict future data points in a time series?"
date: "2025-01-30"
id: "how-can-keras-predict-future-data-points-in"
---
Predicting future data points in a time series using Keras typically involves framing the problem as a supervised learning task. Specifically, we transform the sequential nature of the time series into input-output pairs, leveraging past observations to predict subsequent values. This shift from a temporal sequence to a structured dataset allows us to apply powerful neural network models available within Keras. This isn't inherent to the data itself, but rather a necessary transformation before applying learning algorithms.

The fundamental approach revolves around creating lagged features. Given a time series *X* = {*x<sub>1</sub>, x<sub>2</sub>, ..., x<sub>n</sub>*}, where *x<sub>i</sub>* represents an observation at time *i*, we construct a dataset where the input is a sequence of past values and the output is a future value. For instance, if I'm using a lag of three, my input might be {*x<sub>t-3</sub>*, *x<sub>t-2</sub>*, *x<sub>t-1</sub>*}, and the target is *x<sub>t</sub>*. This 'sliding window' technique is crucial. We can then use these lagged sequences to train various models, with recurrent neural networks (RNNs), and especially Long Short-Term Memory (LSTM) networks, being particularly effective given their ability to model temporal dependencies. Convolutional neural networks (CNNs) can also work when properly designed, particularly in combination with time-delay embedding. After training, these models predict values one step, or multiple steps ahead, using the recent historical data.

Here are three different code examples demonstrating this process with Keras and Python, each focusing on different levels of complexity and architectural choices.

**Example 1: Simple LSTM for One-Step-Ahead Prediction**

This example demonstrates a basic LSTM model for predicting the next value in the time series. I use a simple synthetic dataset with a single sine wave, split into training and testing sets using a 70/30 ratio. The key steps are data preparation, model definition, and then training. This is often my starting point for new time series.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. Generate Synthetic Time Series
np.random.seed(42)
time = np.linspace(0, 100, 500)
data = np.sin(time) + np.random.normal(0, 0.1, 500)
data = data.reshape(-1, 1)

# 2. Data Preprocessing (Min-Max Scaling)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 3. Create Lagged Sequences (window_size is our lookback)
window_size = 10
X, y = [], []
for i in range(len(scaled_data) - window_size - 1):
    X.append(scaled_data[i : (i + window_size)])
    y.append(scaled_data[i + window_size])
X = np.array(X)
y = np.array(y)

# 4. Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# 5. Build the LSTM Model
model = keras.Sequential([
    keras.layers.LSTM(50, activation='relu', input_shape=(window_size, 1)),
    keras.layers.Dense(1)
])

# 6. Model Compilation and Training
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, verbose=0)

# 7. Prediction
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
y_test = scaler.inverse_transform(y_test)

# 8. Print evaluation metric
loss = model.evaluate(X_test, y_test)
print(f"Test Mean Squared Error: {loss}")
```

In this code, I start by creating a simple sinusoidal time series with added noise. Then, I scale the data between 0 and 1 using MinMaxScaler which is almost always important when working with time-series. Following this is the critical step of creating the lagged sequences, where a sequence of 10 past values predict the next value. I then split the data into train and test sets using sklearn. The LSTM model itself is relatively simple: a single LSTM layer followed by a dense layer. I compile the model with mean squared error loss and the adam optimizer, and fit it. After training, I predict the future values using the test set and unscale the data to the original scale to properly evaluate it. Finally, the script prints evaluation metric to access the performance. I often prefer to use mean squared error since it is usually intuitive when comparing different models.

**Example 2: Multi-Step Prediction with a Convolutional Layer**

This example uses a CNN in conjunction with a recurrent layer, specifically designed for multi-step prediction. The idea here is that the CNN can extract local features from the lagged data, providing a more robust input to the LSTM layer. I modify the synthetic time series to be slightly more complex using higher frequency components. The crucial difference in comparison with the first example is the output layer which now produces an output of length equal to *prediction_horizon*.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. Generate Synthetic Time Series
np.random.seed(42)
time = np.linspace(0, 100, 500)
data = np.sin(time) + np.sin(time*2) + np.random.normal(0, 0.2, 500)
data = data.reshape(-1, 1)

# 2. Data Preprocessing (Min-Max Scaling)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 3. Create Lagged Sequences (window_size is our lookback)
window_size = 15
prediction_horizon = 5
X, y = [], []
for i in range(len(scaled_data) - window_size - prediction_horizon):
    X.append(scaled_data[i : (i + window_size)])
    y.append(scaled_data[i + window_size : (i+window_size+prediction_horizon)])
X = np.array(X)
y = np.array(y)

# 4. Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# 5. Build the CNN-LSTM Model
model = keras.Sequential([
    keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(window_size, 1)),
    keras.layers.MaxPooling1D(pool_size=2),
    keras.layers.LSTM(50, activation='relu'),
    keras.layers.Dense(prediction_horizon)
])

# 6. Model Compilation and Training
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=50, verbose=0)

# 7. Prediction
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
y_test = scaler.inverse_transform(y_test)

# 8. Print evaluation metric
loss = model.evaluate(X_test, y_test)
print(f"Test Mean Squared Error: {loss}")
```

Here, the time series is modified to be a combination of multiple sinusoids with some noise. The prediction task is more complex as well: we now need to predict not one but several points in the future, using prediction horizon=5. The major change is the introduction of a `Conv1D` layer followed by `MaxPooling1D` to extract important features. This sequence of layers improves feature extraction from the lagged data, and often boosts performance. The final dense layer output now needs to have a dimension equal to the prediction horizon. The training and evaluation are the same as before, but now the test mean squared error will be based on the accuracy of multi-step prediction.

**Example 3: Stateful LSTM for Long Sequences**

This example demonstrates a stateful LSTM model, which is appropriate when dealing with extremely long sequences with dependencies that span larger time frames. Stateful LSTMs maintain internal states across batches, making them useful when dependencies extend beyond a batch size. This example will focus on more complicated data: a multivariate time series. I include two additional variables besides the main target variable.

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. Generate Multivariate Time Series
np.random.seed(42)
time = np.linspace(0, 100, 500)
data = np.column_stack([
    np.sin(time) + np.random.normal(0, 0.1, 500),
    np.cos(time*0.7) + np.random.normal(0, 0.05, 500),
    np.sin(time*1.2) + np.cos(time*0.5) + np.random.normal(0, 0.15, 500)
])


# 2. Data Preprocessing (Min-Max Scaling)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# 3. Create Lagged Sequences (window_size is our lookback)
window_size = 20
X, y = [], []
for i in range(len(scaled_data) - window_size - 1):
    X.append(scaled_data[i : (i + window_size)])
    y.append(scaled_data[i + window_size,0]) #only target value is predicted
X = np.array(X)
y = np.array(y)

# 4. Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

# 5. Reshape for stateful LSTM
batch_size = 32
X_train = X_train.reshape(X_train.shape[0] // batch_size, batch_size, window_size, 3)
X_test = X_test.reshape(X_test.shape[0] // batch_size, batch_size, window_size, 3)
y_train = y_train.reshape(y_train.shape[0] // batch_size, batch_size)
y_test = y_test.reshape(y_test.shape[0] // batch_size, batch_size)


# 6. Build the Stateful LSTM Model
model = keras.Sequential([
    keras.layers.LSTM(50, batch_input_shape=(batch_size, window_size, 3), stateful=True, return_sequences=True),
    keras.layers.LSTM(50, stateful=True),
    keras.layers.Dense(1)
])

# 7. Model Compilation and Training
model.compile(optimizer='adam', loss='mse')

for epoch in range(50): #explicit state reset during training
    model.fit(X_train, y_train, epochs=1, batch_size=batch_size, shuffle=False, verbose=0)
    model.reset_states()


# 8. Prediction
predicted = model.predict(X_test, batch_size=batch_size)
predicted = predicted.flatten() #remove dimensionality
y_test = y_test.flatten()

# 9. Invert Scaling
dummy_scaler = MinMaxScaler() # we need to fit it to the 0th dimension data
dummy_scaler.fit(data[:,0].reshape(-1,1))
predicted = dummy_scaler.inverse_transform(predicted.reshape(-1,1))
y_test = dummy_scaler.inverse_transform(y_test.reshape(-1,1))

# 10. Print Evaluation Metric
loss = tf.keras.metrics.mean_squared_error(y_test, predicted)
print(f"Test Mean Squared Error: {loss.numpy().mean()}")

```
In this code, the dataset is changed to a multivariate time series containing three dimensions. To leverage stateful LSTM layers, the data reshaping step is critical. The batch size parameter during creation of the data, needs to match the `batch_input_shape` of the `LSTM` layer. Crucially, during training we need to manually reset the states after each epoch. This ensures that the internal states are cleared before beginning a new epoch. Also, the prediction process needs to specify batch size and flatten the output. The evaluation metric is calculated similarly to previous examples.

For further exploration, I recommend consulting academic texts on time series analysis, including works focusing on statistical approaches like ARIMA and Exponential Smoothing. Resources detailing neural network architectures, particularly RNNs and LSTMs, are equally valuable. Furthermore, exploring the Keras documentation on recurrent layers and model configuration will provide much deeper insight. Practical hands-on experience using multiple datasets is invaluable as well.
