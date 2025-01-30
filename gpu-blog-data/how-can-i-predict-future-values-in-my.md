---
title: "How can I predict future values in my model?"
date: "2025-01-30"
id: "how-can-i-predict-future-values-in-my"
---
In the realm of predictive modeling, accurately anticipating future values hinges on understanding the nuances of time series data and selecting appropriate forecasting techniques. Many practitioners, when first confronted with this challenge, might gravitate towards purely regression-based approaches, overlooking the inherent temporal dependencies present in sequential data. My own experience building a demand forecasting system for a large retail chain highlighted the critical difference between models designed for static inputs and those tailored for dynamic time series.

Predicting future values is fundamentally about extrapolating trends and patterns observed in historical data. Time series data, unlike static datasets, are indexed by time, making the order of observations crucial. Simple regression models, while effective for predicting outputs based on independent inputs, often fail to capture the autocorrelative nature of time series; that is, how past values influence future ones. These models typically treat each data point as independent from others, ignoring the fact that, for example, today’s sales are likely correlated with yesterday’s.

Forecasting methodologies therefore necessitate models that explicitly incorporate temporal dependencies. Broadly, these methods fall under two categories: statistical models and machine learning-based models. Statistical models, such as ARIMA (Autoregressive Integrated Moving Average) and its variants (SARIMA for seasonality), rely on identifying patterns like trends, seasonality, and autocorrelation within the time series data. These models mathematically describe how the current value relates to previous values and forecast error terms. In contrast, machine learning models, including recurrent neural networks (RNNs) like LSTMs (Long Short-Term Memory) and GRUs (Gated Recurrent Units), learn these patterns through training on historical data, without explicit mathematical formulation.

Choosing the right approach depends on the characteristics of the data and the desired level of accuracy and interpretability. Statistical models are often easier to interpret and computationally cheaper to train, but might struggle with highly non-linear or complex patterns. Conversely, machine learning models are more flexible and can handle complex relationships but require more computational resources and often act as “black boxes,” making understanding their decision-making less transparent.

The following code examples demonstrate how different methods are applied in Python, a common tool used for predictive modeling, illustrating their practical implementation.

**Example 1: ARIMA Model**

This example demonstrates the implementation of an ARIMA model using the `statsmodels` library. ARIMA models are suitable for time series data that exhibit stationarity (constant statistical properties over time) or can be made stationary through differencing.

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

# Sample time series data
data = pd.Series([20, 22, 25, 23, 28, 30, 29, 33, 36, 34, 39, 41])

# Plot autocorrelation and partial autocorrelation functions to guide parameter selection
plot_acf(data, lags=5)
plot_pacf(data, lags=5)
plt.show()

# Defining model parameters: (p, d, q) (p = autoregressive order, d = degree of differencing, q = moving average order)
# Parameters are selected based on inspection of the ACF and PACF plots. Here I've chosen parameters
# based on the assumption that these plots showed strong correlation with lag=1.

model = ARIMA(data, order=(1, 1, 1))
model_fit = model.fit()

# Forecast the next three values.
forecast = model_fit.forecast(steps=3)

print(forecast)
```

**Commentary:**

Initially, I load a sample time series as a pandas series.  The autocorrelation (ACF) and partial autocorrelation (PACF) plots are useful here for choosing suitable `p` and `q` parameters, which specify the orders of the autoregressive and moving average components, respectively. The `d` parameter represents the degree of differencing required to make the data stationary.  These are inspected to choose suitable values. Differencing the series allows for more robust model building by ensuring stationarity which is a critical assumption of the model. The `ARIMA` class fits the model to the historical data. Finally, the `forecast()` method generates predictions for the specified number of future steps. The output here would be an array showing these forecasted values.

**Example 2: LSTM Model with Keras**

This code demonstrates building and training an LSTM recurrent neural network using Keras, a high-level API in TensorFlow. LSTMs are capable of learning complex long-term dependencies in sequential data.

```python
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# Sample time series data
data = pd.Series([20, 22, 25, 23, 28, 30, 29, 33, 36, 34, 39, 41])
data = np.array(data).reshape(-1, 1)

# Scaling the data to the range [0,1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Function to create input sequences
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Sequence length of 3
seq_length = 3
X, y = create_sequences(scaled_data, seq_length)

# Building the LSTM Model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Training the model
model.fit(X, y, epochs=100, verbose=0)

# Forecasting the next three values
last_sequence = scaled_data[-seq_length:].reshape(1, seq_length, 1)
forecast_scaled = []
for _ in range(3):
  next_val_scaled = model.predict(last_sequence)
  forecast_scaled.append(next_val_scaled[0,0])
  last_sequence = np.append(last_sequence[:,1:,:], next_val_scaled.reshape(1,1,1), axis=1)

forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1,1))

print(forecast)
```

**Commentary:**

Here, the data undergoes scaling using `MinMaxScaler` to ensure inputs are within a normalized range, which generally helps with model convergence during training. The `create_sequences` function transforms the time series into a supervised learning format, where each input sequence corresponds to a single future target value. An LSTM layer learns temporal patterns within the input sequences, and a final dense layer predicts the next value. The model is trained using mean squared error (MSE) as the loss function, common for regression problems. Note the reshaping that is needed to generate new predictions, since the input must be a 3D tensor.  The generated predictions need to be inverse scaled to return them to the original data scale.

**Example 3: Time Series Decomposition with Prophet**

This snippet utilizes Prophet, a library developed by Facebook, which is well-suited for time series with strong seasonality and trend. Prophet decomposes the time series into trend, seasonality, and holiday components.

```python
import pandas as pd
from prophet import Prophet

# Sample time series data
data = pd.DataFrame({
    'ds': pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05', '2024-01-06',
                         '2024-01-07', '2024-01-08', '2024-01-09', '2024-01-10', '2024-01-11', '2024-01-12']),
    'y': [20, 22, 25, 23, 28, 30, 29, 33, 36, 34, 39, 41]
})

# Prophet Model instantiation
model = Prophet()

# Fitting the model to the historical data
model.fit(data)

# Creating a dataframe with future dates. 
future = model.make_future_dataframe(periods=3)

# Forecast generation
forecast = model.predict(future)

print(forecast[['ds', 'yhat']][-3:])
```

**Commentary:**

Prophet expects a pandas DataFrame with two columns: ‘ds’ representing dates and ‘y’ representing the time series values.  The model is instantiated, fit to the historical data, and then used to create a dataframe with future dates. The `predict` method then returns the forecasts including the original history. The output shows the final 3 rows of the data frame. The `yhat` column is the forecasted value. Prophet’s strength lies in its ability to identify and incorporate seasonality as well as trends in the time series, making it a practical choice for real-world data with periodic patterns.

For further exploration and to deepen your understanding, I recommend consulting the documentation for the `statsmodels`, `tensorflow`, and `prophet` libraries, as well as textbooks on time series analysis. Academic articles on forecasting methods, specifically those focused on the particular nuances of your application area, can also be valuable. The book "Forecasting: Principles and Practice" by Hyndman and Athanasopoulos provides a comprehensive overview of forecasting techniques.  These resources can help guide the development of robust and accurate forecasting models.
