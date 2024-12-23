---
title: "How can I forecast a univariate time series 20/30 days ahead using tensorflow LSTM?"
date: "2024-12-23"
id: "how-can-i-forecast-a-univariate-time-series-2030-days-ahead-using-tensorflow-lstm"
---

, let’s tackle forecasting univariate time series with LSTMs in tensorflow, focusing on that 20-30 day horizon you’ve specified. I’ve been down this road myself several times, and it’s never as straightforward as just plugging in data and hoping for the best. The key, from my experience, is in careful preparation and understanding the subtleties of your input data.

First off, thinking solely about the LSTM architecture is jumping the gun. Before that, we need to deal with the time series data itself. We're talking univariate here, which simplifies some things, but not all. A common issue I see repeatedly is the lack of appropriate preprocessing. Time series data often isn’t stationary; that is, the statistical properties (mean, variance) change over time. This can massively impact your model’s ability to predict the future accurately. So, think about things like differencing, where you subtract the value at time 't' from time 't+1', for example, or other more robust methods detailed in *Time Series Analysis and Its Applications* by Robert H. Shumway and David S. Stoffer – a solid, practical resource, by the way.

Then, we have the data scaling. This can profoundly impact the training of your LSTM. LSTMs, being gradient-based models, can struggle with inputs on drastically different scales. Standardizing (zero mean, unit variance) or min-max scaling your data is usually a crucial initial step. I’ve seen networks that completely fail without it, and others that show noticeable improvement when done correctly.

Now, let's assume we've taken care of preprocessing and have scaled data ready for input. The way we structure data for an LSTM is also essential. An LSTM doesn’t take the raw time series directly. Instead, it expects sequences. We need to convert our single time series into sequences of a certain length (the lookback window) to predict the next value. This is sometimes called creating lagged features. The lookback period significantly affects performance, and what works for one series might not work for another. There is no golden rule here, other than experiment and pay careful attention to evaluation metrics.

Regarding the architecture of the LSTM itself, consider it as having multiple layers, and a number of hidden units. It’s tempting to add layers and units, but sometimes less is more. Overfitting, where the network memorizes the training data rather than learning generalizable patterns, is a constant threat, especially with relatively shorter time series, or noisy data. Regularization techniques such as dropout can be useful, or early stopping – monitoring validation loss during training and stopping when performance starts to degrade. These concepts are covered with much more mathematical detail in *Deep Learning* by Ian Goodfellow, Yoshua Bengio, and Aaron Courville – a fantastic and essential resource for anyone working in deep learning.

Now, for the crucial 20-30 day forecasting requirement, we cannot simply predict the next day in a loop. That approach can quickly accumulate errors over such a length of prediction. We will typically implement an autoregressive, or recursive approach where the model predicts the next day based on the sequence, and then we feed that prediction back in to the sequence to predict another day, and so on. Another alternative strategy is sequence-to-sequence forecasting. In this case, the last *n* time steps are fed in as input, and the model will output the next *m* steps as the prediction. While sequence-to-sequence is more complex to implement, it can lead to superior performance when long-term dependencies exist in the time series.

Here's some code to illustrate the windowing of time series data and building a basic LSTM model (using tensorflow/keras). This will form the foundation from which we build up:

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length -1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Sample Time series data
data = np.sin(np.linspace(0, 10*np.pi, 500)) # synthetic data
seq_length = 20 # lookback window length

X, y = create_sequences(data, seq_length)
X = X.reshape(X.shape[0], X.shape[1], 1) # reshaping for LSTM

model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, verbose=0)
```

That snippet takes some example time series data, creates sequences, and trains a very basic LSTM. You'll want to adjust things like `seq_length`, number of LSTM units (50 here), and epochs to suit your dataset.

Here's an example of iterative, or recursive prediction, that assumes the `model` from above has been trained:

```python
def predict_future_iterative(model, last_sequence, n_days):
    predicted_values = []
    current_sequence = last_sequence.copy()
    for _ in range(n_days):
      next_prediction = model.predict(current_sequence.reshape(1, seq_length, 1), verbose=0)[0][0]
      predicted_values.append(next_prediction)
      current_sequence = np.concatenate((current_sequence[1:], [next_prediction]))
    return predicted_values


last_sequence = X[-1] # Taking last seen sequence from training data
forecast = predict_future_iterative(model, last_sequence, 30) # predict 30 steps ahead
print("Forecast:", forecast)
```

Here's how the sequence-to-sequence approach might look like, using `n` as input lookback, and `m` as the number of future timesteps to predict. Note this is a little more involved in terms of training than the iterative approach:

```python
def create_seq_to_seq_data(data, n, m):
    xs, ys = [], []
    for i in range(len(data) - n - m - 1):
        x = data[i : i+n]
        y = data[i+n : i+n+m]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

n = 20
m = 30

X_seq_to_seq, y_seq_to_seq = create_seq_to_seq_data(data, n, m)
X_seq_to_seq = X_seq_to_seq.reshape(X_seq_to_seq.shape[0], X_seq_to_seq.shape[1], 1)
y_seq_to_seq = y_seq_to_seq.reshape(y_seq_to_seq.shape[0], y_seq_to_seq.shape[1])

model_seq_to_seq = Sequential([
    LSTM(50, activation='relu', input_shape=(n,1)),
    Dense(m)
])

model_seq_to_seq.compile(optimizer='adam', loss='mse')
model_seq_to_seq.fit(X_seq_to_seq, y_seq_to_seq, epochs=50, verbose=0)

last_input_seq = X_seq_to_seq[-1] # last sequence
forecast_seq_to_seq = model_seq_to_seq.predict(last_input_seq.reshape(1,n,1), verbose=0)[0]
print("Sequence-to-sequence forecast:", forecast_seq_to_seq)
```

Both of these examples illustrate the mechanics of generating forecasts, however you will still need to preprocess your data and evaluate the prediction in the context of your specific problem.

Important, and I cannot stress this enough, you must evaluate your model thoroughly on a separate hold-out set, and use sensible metrics for your task. Mean absolute error (mae), root mean squared error (rmse) or mean absolute percentage error (mape) are all viable metrics, but some are more appropriate than others depending on the error distribution of your predictions. The *Forecasting: Principles and Practice* book by Rob J Hyndman and George Athanasopoulos, has an excellent chapter on choosing the appropriate evaluation method.

Finally, do keep an eye out for over- or under-fitting, particularly when tuning the length of your input sequence and the model architecture. Regularization and early stopping techniques are invaluable in this effort, along with cross-validation if you have sufficient training data. Building a good forecasting model isn't about having the most complex network; it is much more about understanding the nuances of your specific dataset, and implementing the methods that are most appropriate to extract useful patterns.
