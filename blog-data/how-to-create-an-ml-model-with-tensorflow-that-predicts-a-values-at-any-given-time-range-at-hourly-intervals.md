---
title: "How to Create an ML model with tensorflow that predicts a values at any given time range at hourly intervals?"
date: "2024-12-14"
id: "how-to-create-an-ml-model-with-tensorflow-that-predicts-a-values-at-any-given-time-range-at-hourly-intervals"
---

alright, so you're looking to build a machine learning model using tensorflow that spits out predictions at hourly intervals, across a given time range. i've been down this road a few times, it's a pretty common scenario when dealing with time series data. i'll walk you through my approach, the challenges i've bumped into, and the general stuff you need to think about.

first off, let's be clear: "predicting values at any given time range at hourly intervals" can mean a few things, but i'm assuming you want a model that, given some historical data, can predict values for *future* hours, not just interpolate between existing data points.

here's the core idea. you're dealing with a time series problem so recurrent neural networks, specifically lstms or grus, are your best bet. these networks are designed to handle sequential data like time series. they maintain internal states that capture temporal dependencies, allowing them to remember the past and predict the future. that's the crux of it.

i can recall a project i worked on a few years ago. it was related to predicting energy consumption patterns for a small factory. we had hourly measurements of power consumption and the usual stuff, temperature, humidity, and a bunch of other readings. my team initially tried simpler models, like linear regression, but the prediction error was horrendous. the cyclical patterns in the consumption data and all the correlations were really not captured by those models, it felt like trying to nail jelly to the wall. that experience made me realize the power of lstms for sequences like this. they can learn the repeating patterns, they are pretty good at that.

now let's get into some code examples. i'll focus on tensorflow, and you should have a dataset structured as timeseries where the 'x' is some sort of features and 'y' are the target values for each time point, usually a pandas dataframe with a date time index.

```python
import tensorflow as tf
import numpy as np
import pandas as pd

# assume a pandas dataframe called 'data' with a datetime index and 'target' column

def create_sequences(data, sequence_length, prediction_horizon):
    x, y = [], []
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
      x.append(data.iloc[i:i+sequence_length].values)
      y.append(data['target'].iloc[i+sequence_length:i+sequence_length+prediction_horizon].values)
    return np.array(x), np.array(y)

sequence_length = 24 # 24 hours is one day, how many past hours to consider
prediction_horizon = 1 # predict the next hour
x, y = create_sequences(data, sequence_length, prediction_horizon)

train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=50, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])),
    tf.keras.layers.Dense(units=prediction_horizon)
])

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, epochs=50, batch_size=32)

loss = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}')
```

this first snippet is basic, preparing the sequences, and building a vanilla lstm. the `create_sequences` function is very important it does a sliding window approach making training data. the `sequence_length` is how many hours you want the model to see before making a prediction, this will depend on the underlying seasonality of the data. and `prediction_horizon` is the number of hours in the future you want to predict. for the purpose of this response i picked 1, which is predict the next single hour, but you can change that. make sure the input data `data` has a column called `target` which is what we want to predict.

remember lstms have a parameter called `units` that determines how many internal nodes they have, you might need to experiment with this. the `activation` function should be `relu` in my opinion, it usually works. this code does not deal with any preprocessing so data normalization and standardisation may be important to improve model performance. also, i am predicting the next single value, but you can also predict a sequence of future values, which i will show in the next snippet.

now, let's say you want to predict not just one hour into the future, but a few hours. you could do a modified `create_sequences` function like above, but for the sake of demonstration, i'll show you another approach which is predicting sequentially. this method is where your lstm model is asked to predict one time point, and you give the output back as an input into the network as a sequence. this is a recurrent process.

```python
import tensorflow as tf
import numpy as np
import pandas as pd

def create_sequences(data, sequence_length, prediction_horizon):
    x, y = [], []
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
      x.append(data.iloc[i:i+sequence_length].values)
      y.append(data['target'].iloc[i+sequence_length:i+sequence_length+prediction_horizon].values)
    return np.array(x), np.array(y)

sequence_length = 24 # 24 hours
prediction_horizon = 3 # predict next 3 hours

x, y = create_sequences(data, sequence_length, prediction_horizon)

train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)

model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=50, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])),
    tf.keras.layers.Dense(units=1)
])

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train[:, 0].reshape(-1, 1), epochs=50, batch_size=32)

def predict_future(model, input_sequence, future_steps):
    predicted_sequence = []
    current_sequence = input_sequence.copy()

    for _ in range(future_steps):
        prediction = model.predict(current_sequence.reshape(1, sequence_length, -1))[0][0]
        predicted_sequence.append(prediction)
        current_sequence = np.concatenate((current_sequence[1:], np.array([prediction]).reshape(1,1)), axis=0)

    return np.array(predicted_sequence)


start_index = 0 # pick the last sequence available in the test dataset
input_data = x_test[start_index]
future_predictions = predict_future(model, input_data, prediction_horizon)
print(f"the predicted values are: {future_predictions}")
```

i've updated the model to output a single value, which is the next predicted hour and added a `predict_future` function. this function loops for `future_steps` times, feeding the predicted value back into the model, this is a very simple method but it works for a proof of concept. it uses the last sequence of the testing dataset as an example of `input_data` and will make a prediction for the next `prediction_horizon` hours. note that this approach could accumulate errors because you are feeding the model it's own predictions, so don't expect very accurate results for big `prediction_horizon` values.

one of the hard parts in these time series models is hyperparameter tuning, you will have to test different values for sequence_length, lstm units, optimizers, learning rates, and number of epochs. in all of my experience the correct data is always more important than the model's architecture, garbage in, garbage out. so make sure the data is clean, correctly structured, and correctly preprocessed and then worry about the model's architecture. in our energy consumption example, a lot of the effort was not in creating the models, it was in cleaning up the inconsistent data feeds and making sure we had high quality data coming into the models.

if you are dealing with seasonality, and it is apparent in your data, you might want to add some positional encoding. this is where you encode the position of each timestamp of your input data. positional encoding can help capture underlying periodic patterns, the paper "attention is all you need" explains that in detail. they used a sinusoids encoding for sequence information. for time series i like to use the day of the week, the hour of the day, and the day of the month. here is an example of how to do it, using pandas:

```python
import tensorflow as tf
import numpy as np
import pandas as pd

# assuming a pandas dataframe called 'data' with a datetime index

def create_time_features(data):
    data = data.copy()
    data['hour'] = data.index.hour
    data['day_of_week'] = data.index.dayofweek
    data['day_of_month'] = data.index.day
    data['month'] = data.index.month
    data = pd.get_dummies(data, columns=['hour', 'day_of_week', 'day_of_month','month']) # one hot encoding
    return data

data_with_features = create_time_features(data)


def create_sequences(data, sequence_length, prediction_horizon):
    x, y = [], []
    for i in range(len(data) - sequence_length - prediction_horizon + 1):
      x.append(data.iloc[i:i+sequence_length].values)
      y.append(data['target'].iloc[i+sequence_length:i+sequence_length+prediction_horizon].values)
    return np.array(x), np.array(y)

sequence_length = 24 # 24 hours
prediction_horizon = 1 # predict next single hour
x, y = create_sequences(data_with_features, sequence_length, prediction_horizon)


train_size = int(len(x) * 0.8)
x_train, x_test = x[:train_size], x[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)


model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(units=50, activation='relu', input_shape=(x_train.shape[1], x_train.shape[2])),
    tf.keras.layers.Dense(units=prediction_horizon)
])

model.compile(optimizer='adam', loss='mse')

model.fit(x_train, y_train, epochs=50, batch_size=32)

loss = model.evaluate(x_test, y_test)
print(f'Test loss: {loss}')
```

notice how the input to the lstm is the original features of the data, plus the generated categorical time features.  i did a one hot encoding because it works better for categories rather than having just the integer representation. the data `data` must have a datetime index, if not you will have to convert it.

for more details on different time series models, the book "forecasting: principles and practice" is a great resource. also, you can explore the paper "long-term recurrent convolutional networks for time series prediction" if you want more advanced architectures, it's a paper about a variation of lstms that incorporate convolutional layers.

and one important detail, always, always, always use a validation set, this snippet just has training and testing set, which will not show you the overall performance of the model for data that was not seen during training and testing. for the sake of demonstration this is enough but it's not good practice in real life.

i hope these examples and my experience help you on your quest. it can be a bit tricky at first but once you get the hang of it, you'll find that this is a very common pattern. good luck, and remember, if you can not get it to work at first, there's no shame in using a bigger hammer.. i meant, a bigger neural network.. haha!
