---
title: "How can LSTMs be used to forecast non-stationary time series?"
date: "2024-12-23"
id: "how-can-lstms-be-used-to-forecast-non-stationary-time-series"
---

Alright, let's tackle this. I've spent a fair bit of time wrestling with non-stationary time series, particularly when trying to squeeze out accurate predictions from systems that just refuse to behave nicely. It’s a common hurdle, and long short-term memory networks (LSTMs) can be incredibly powerful tools, but they're not a silver bullet. It's crucial to understand their strengths, weaknesses, and, most importantly, how to properly prepare your data to make them effective.

Firstly, the core challenge with non-stationary time series is that their statistical properties change over time. Think of, say, stock market prices: their mean, variance, and autocorrelation patterns shift constantly. LSTMs, by nature, are designed to capture temporal dependencies. They have internal memory cells that can retain information over long sequences, making them a great candidate for handling complex temporal dynamics. However, if the underlying distribution of your data shifts dramatically, an LSTM trained on one portion of the series may perform poorly on a different period.

The initial step, and this is something I learned the hard way after a few painful prediction model failures, is preprocessing. You can't just throw raw non-stationary data at an LSTM and hope for the best. Here are some strategies, all of which I've used with varying degrees of success:

1.  **Differencing:** This is a classic time series technique. We subtract the current data point from the previous one (or a point lagged by more than one step) to stabilize the mean. You can apply differencing multiple times until you achieve stationarity. For instance, a first-order difference transforms a series *y(t)* into *y(t) - y(t-1)*. While this can help an LSTM, it also changes the interpretation of the model's output—you're predicting the *change* in the series, not the value itself. The final prediction will need to be transformed back to the original scale.

2.  **Detrending:** If your series has a clear trend, that trend may overwhelm the LSTM, making it difficult to pick out subtle, shorter-term patterns. The removal can be done by fitting a regression model (linear or polynomial) to the trend and then subtracting this from the original series. Alternatively, more robust approaches like the Hodrick-Prescott filter or moving average smoothing can also be employed.

3.  **Rolling Statistics:** Rather than using raw time series data, we can feed the LSTM rolling statistics (like moving averages, standard deviations, or skew) computed over a specific window. These features adapt over time and might be better representations for the LSTM to latch onto than raw values, especially if the raw data is incredibly noisy.

4.  **Data Transformation:** Applying mathematical transformations, such as logarithm or Box-Cox transformation, can stabilize variance in a series if needed.

Here is a crucial insight: It’s rare that only one preprocessing technique will fully resolve non-stationarity; often, a combination of approaches is required.

Let me share some illustrative code snippets in python (using `numpy` and `pandas`) that shows how to use these techniques:

```python
import numpy as np
import pandas as pd

# Example data - Non stationary time series
np.random.seed(42)
t = np.arange(100)
data = 5*t + 2*np.sin(t/2) + np.random.normal(0,5,100) # with some trend

ts = pd.Series(data)
# 1. First Order Differencing
diff_ts = ts.diff().dropna()
print("Differenced Data:", diff_ts.head())


# 2. Detrending with linear regression
from sklearn.linear_model import LinearRegression
X = t.reshape(-1, 1)
model = LinearRegression()
model.fit(X, ts)
trend = model.predict(X)
detrended_ts = ts - trend
print("Detrended Data:", detrended_ts.head())

# 3. Rolling statistics
window_size = 10
rolling_mean = ts.rolling(window=window_size).mean().dropna()
rolling_std = ts.rolling(window=window_size).std().dropna()

print("Rolling Mean:", rolling_mean.head())
print("Rolling Std:", rolling_std.head())
```

Now, these transformations aren't magic. They help, but careful selection of the appropriate technique and their hyperparameters needs some experimentation and evaluation of the forecast performance. For instance, when using rolling statistics, selecting appropriate window sizes is crucial, and for differencing, it’s important to select the right lag. It's usually not a one-time process but a part of an iterative experiment cycle.

Now, beyond preprocessing, the LSTM architecture itself might need tweaking. For example, you could consider:

1.  **Windowed Input:** We don’t typically feed an LSTM the entire time series as a single input sequence. Instead, we divide it into overlapping or non-overlapping windows. The window size and the stride (how much the window shifts for each input sequence) become hyperparameters that can significantly affect the model’s performance.

2.  **Attention Mechanism:** Incorporating an attention mechanism allows the LSTM to weigh the importance of different parts of the sequence during the forecasting process. This can be particularly useful if certain time periods have more predictive value than others.

3.  **Encoder-Decoder Architecture:** If the forecast horizon is relatively long, an encoder-decoder setup might be beneficial. The encoder compresses the input sequence, and the decoder generates the prediction. This allows the model to focus on the overall trend and not get lost in the immediate past.

Let's expand on code to include windowed input creation and some example architecture. Using `tensorflow/keras` this time.

```python
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


# Example data - Non stationary time series
np.random.seed(42)
t = np.arange(100)
data = 5*t + 2*np.sin(t/2) + np.random.normal(0,5,100)
ts = pd.Series(data)


# 1. Windowing the data
def create_windows(data, window_size):
    windows = []
    for i in range(len(data) - window_size):
        windows.append(data[i:i + window_size])
    return np.array(windows)

window_size = 10
windows = create_windows(ts.values, window_size)
X = windows[:, :-1]  # all but last element
y = windows[:, -1]   # last element

X = X.reshape((X.shape[0], X.shape[1], 1)) # reshaping to 3d array for lstm

# 2. Defining a basic LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

model.fit(X, y, epochs=20, verbose =0) # train model, verbose=0 to silence output

print("Model created and fitted")

# Sample prediction using a new sequence
test_seq = ts.values[-window_size:].reshape(1, window_size, 1)
prediction = model.predict(test_seq)
print("Prediction using new seq:", prediction)

```

Lastly, proper evaluation is critical. Just looking at root mean squared error (RMSE) or mean absolute error (MAE) on a single holdout set might not be sufficient. Using time series cross-validation techniques like rolling forward validation, where you predict the next section using historical data, can provide a more realistic evaluation of performance.

For in-depth understanding, I recommend looking at Hyndman and Athanasopoulos' "Forecasting: Principles and Practice." It provides a strong statistical foundation for time series analysis. For LSTMs specifically, “Deep Learning” by Goodfellow, Bengio, and Courville includes excellent sections on recurrent neural networks and their variants. And to really hone in on practical time series forecasting with deep learning, searching for research papers on methods specific to non-stationary time series using LSTMs on sites like arXiv is advisable. A lot of cutting-edge techniques are often detailed in research papers before making their way into more popular books.

So, in closing, while LSTMs are a strong contender for forecasting non-stationary time series, they require significant data preprocessing, thoughtful architecture design, and rigorous evaluation. They're not magic, but with the right approach, they can provide quite powerful predictive capabilities.
