---
title: "How to Create an ML model with tensorflow that predicts values at any given time range at hourly intervals?"
date: "2024-12-15"
id: "how-to-create-an-ml-model-with-tensorflow-that-predicts-values-at-any-given-time-range-at-hourly-intervals"
---

so, you're looking to build an ml model, specifically with tensorflow, that can predict values at hourly intervals across a time range. i’ve been there, trust me. this kind of time series prediction can be a real rabbit hole, but let's break it down into digestible pieces. 

first things first, the core of your problem is making predictions based on temporal patterns. we’re not dealing with static data here; the order and time deltas between your data points are crucial. because of this, the first thing that comes to mind is some flavour of recurrent neural network, rnn or the newer transformer models since they excel at catching these dependencies. i remember a project, back in my university days, where i was trying to forecast stock prices using a basic feedforward network and it was, well, a complete disaster. the model just couldn't handle the sequential nature of the stock market data. i learned the hard way that the architecture matters a lot in time series problems, this was before transformers were widely available. 

ok so let's talk about the data. for something like an hourly interval forecast, you need to have your data arranged in time series format. the classic example is data having timestamps and the value you are predicting. let’s say you have a dataset where you have hourly recorded temperature. you will have to build a time series with the history of the values plus the future values to predict for an specific period. and you will have to process this data to prepare it for the model. you will want to include the time stamps as part of your training but you will want to convert the date into numerical data. you may use unix timestamps or you could even encode them by means of sin and cosine encodings so you can get a numerical feature that also keeps the periodic nature of time. this is very important because otherwise your model may have trouble learning the weekly or daily patterns that might exist.

here's a small piece of python code to show how you can pre-process your time series data and split it into training and testing data:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def create_time_series_dataset(data, lookback, forecast_horizon):
    xs, ys = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        v = data[i:(i + lookback)]
        xs.append(v)
        ys.append(data[i + lookback : i + lookback + forecast_horizon])
    return np.array(xs), np.array(ys)


def prepare_data(df, value_col, timestmap_col, lookback, forecast_horizon):
    df[timestmap_col] = pd.to_datetime(df[timestmap_col])
    df = df.set_index(timestmap_col)
    data = df[value_col].values
    xs, ys = create_time_series_dataset(data, lookback, forecast_horizon)
    x_train, x_test, y_train, y_test = train_test_split(xs, ys, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

#example
data = {'timestamp': pd.date_range(start='2023-01-01', periods=24*365, freq='H'),
        'temperature': np.random.normal(20, 5, 24*365)}
df = pd.DataFrame(data)

lookback_window = 24*7 #one week in hours
forecast_window = 24 #one day in hours

x_train, x_test, y_train, y_test = prepare_data(df, 'temperature', 'timestamp', lookback_window, forecast_window)

print (f"shape of training data: {x_train.shape}, {y_train.shape}")
print (f"shape of testing data: {x_test.shape}, {y_test.shape}")
```

in the above example i have added how to create a basic time series, as you can see a `lookback` variable of 24\*7 (one week), and a `forecast` window of 24 hours to predict a day ahead based on a week data. and how to split it into training and testing datasets using the sklearn library. this is a basic example but it sets the structure for the next steps.

now regarding the model itself, let’s talk tensorflow. i'd recommend starting with a simple lstm (long short-term memory) network. it's a type of rnn that's pretty good at handling long-range dependencies in sequences, a crucial advantage in time series. a simple lstm with maybe one or two layers would be a good starting point before you go more sophisticated. the good thing is that in tensorflow it's pretty simple to build an lstm:

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def create_lstm_model(input_shape, output_shape):
    model = Sequential()
    model.add(LSTM(units=64, activation='relu', input_shape=input_shape, return_sequences = True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(units=output_shape))
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


# assuming x_train is of shape (number of samples, lookback, number of features)
input_shape = (x_train.shape[1], x_train.shape[2] if len(x_train.shape) > 2 else 1)
output_shape = y_train.shape[1]

model = create_lstm_model(input_shape, output_shape)

model.summary()
```

in the example above i added a model function using keras, a high level api of tensorflow. as you can see we add two `lstm` layers with some `dropout` between them to avoid overfitting. i added a dense layer at the end with the `output_shape`, usually the number of steps in the future that we want to predict. then we compile with the adam optimizer and a loss like `mse` (mean squared error) because it's a regression problem, for classification you would change both the loss and the activation of the last layer. the summary method is very helpful since it shows you the layers and the parameters you are training. also, the input shape depends on your data. note that the example assumes that it may have more than one feature, but in our example we only have one feature: temperature.

then you have the training loop. you want to train it with batches of data, and you also want to validate your results and monitor the loss and metrics. tensorflow has a nice feature called tensorboard that can help you with this. i remember, one time i was training a model and i had a bug in the data loader, the tensorboard helped me track the error because the loss was not decreasing, it was in fact increasing. if i didn't use tensorboard i would probably have spent days trying to debug that issue. and it turned out to be a very silly one.

```python
from tensorflow.keras.callbacks import TensorBoard

# train the model
epochs = 50
batch_size = 32

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, 
        validation_data=(x_test, y_test), callbacks=[tensorboard_callback])
```

in this part we define the number of epochs (number of times the model see all the data), and the batch size (number of examples it sees each training step) and also i create a callback for tensorboard to visualize the training process. then we call fit in the model with the training and the test datasets with the callbacks. after the training is complete you can see the results in tensorboard by going to the folder where the log is stored and opening a terminal: `tensorboard --logdir=./logs` it will give you an address in your localhost.

now, some extra tips based on my past experiences:

*   **feature engineering:** don't just throw the raw time data at the model. consider creating features like day of the week, hour of the day, and so on as mentioned before (periodic encoding using sin and cosine is often very helpful). this can give the model more context.
*   **normalization:** scale your input data. this helps the model train faster and more stably. standard techniques like min-max scaling or z-score normalization work well.
*   **hyperparameter tuning:** don't expect to nail the model architecture and hyperparameters on the first try. you'll need to experiment with different numbers of layers, units, learning rates, and so on. this can be an iterative process, for this there are many useful libraries that could help you like keras tuner.
*   **regularization:** use techniques like dropout to prevent overfitting. overfitting means your model learns the training data too well and it performs poorly on the test data.
*   **error analysis:** when your model makes bad predictions, analyze those cases. this might reveal patterns in your data you weren't aware of. i once found that a daily pattern was very noticeable in the residual errors of my predictions, it was kind of funny because i missed that when i analysed the data initially.

for further reading, i would recommend these books:
*   "deep learning with python" by francois chollet: this book covers the deep learning fundamentals and also includes keras usage. it's very well explained and easy to understand. 
*   "forecasting: principles and practice" by hyndman and athanasopoulos: this is a book focusing on forecasting techniques in general, it's useful for understanding time series data better.
*   "hands-on machine learning with scikit-learn, keras & tensorflow" by aurélien géron: very practical book that covers many machine learning techniques using scikit-learn, keras and tensorflow.

building a good time series prediction model is often a mix of experimentation and understanding the specific nuances of your data. there is no such thing as the "best" model, but a good methodology of data understanding, experimenting and analysing results is crucial. don't be discouraged if things don't work perfectly the first few times, we've all been there. and remember the most important thing is always to learn from your mistakes.
