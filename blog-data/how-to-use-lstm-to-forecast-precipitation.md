---
title: "How to Use LSTM to forecast Precipitation?"
date: "2024-12-14"
id: "how-to-use-lstm-to-forecast-precipitation"
---

so, you're looking into using lstms for precipitation forecasting, huh? been there, done that. it's a classic problem, and let me tell you, it’s not always straightforward. i've spent a good chunk of my career knee-deep in time series data, specifically atmospheric stuff, and lstms can be powerful, but they also come with their own set of quirks.

first off, let's talk about why lstms are even considered for this. traditional time series models like arima often struggle with non-linear relationships, which, guess what? weather patterns are full of. lstms, on the other hand, are a type of recurrent neural network capable of learning complex temporal dependencies. they can essentially remember what happened in the past and use that to predict the future, which is crucial for precipitation, which tends to have inertia. if it rained yesterday, there is a higher chance it will rain today (usually), compared if it was completely dry for a week.

so, where did i first stumble upon this problem? well, back in the day, when i was doing some consulting work for a small agricultural company, they needed better predictions for irrigation planning. existing weather forecasts were just too coarse, and they needed something tailored to their specific location, also the forecast data they had was not the best quality. that's when i decided to give lstms a shot and it was a mix of total failure and slow gradual success.

the core of using lstms for precipitation is structuring your data. you will need sequences of past weather data as input and future precipitation as output. here's a basic conceptual example of how your input data might look:

```
[[temp_t-5, humidity_t-5, pressure_t-5, wind_t-5],
 [temp_t-4, humidity_t-4, pressure_t-4, wind_t-4],
 [temp_t-3, humidity_t-3, pressure_t-3, wind_t-3],
 [temp_t-2, humidity_t-2, pressure_t-2, wind_t-2],
 [temp_t-1, humidity_t-1, pressure_t-1, wind_t-1]]
```
and your corresponding output:

```
[precip_t]
```

notice that `temp_t-x` represents the temperature `x` time steps in the past, where time step depends if your data is hourly, daily, or another scale. it might contain other features like humidity, pressure, and wind at the same past time steps. the `precip_t` is the precipitation we want to forecast. this is a very very simple example. i personally had much more granular data, including soil temperature, evaporation rates, and several wind components for my model.

this is your input sequence. lstms ingest time-series data in sequences. we need to craft these sequences. i personally like a sliding window approach where you create sequences of say, 5 days, and then move one day ahead and create a new sequence. and you need a output which is precipitation in your case for the next day. or whatever you choose to forecast. a code snippet that exemplifies that in python:

```python
import numpy as np

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length), :-1] # all columns except last which is precipitation
        y = data[i + seq_length, -1] # last column, precipitation
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# example usage
data = np.random.rand(100, 5) # example 100 days data with 4 features + 1 precipitation
seq_length = 7
xs, ys = create_sequences(data, seq_length)
print(f"shape of input sequences: {xs.shape}")
print(f"shape of output: {ys.shape}")
```

note how in the function `create_sequences` we are using a `seq_length` parameter, that controls the length of the time sequence and how we create the output. it might sound simple, but choosing the correct length is an experiment in itself. there is no one size fits all.

now, let's get to the model architecture. lstms are commonly used in combination with dense layers. i found a very simple architecture to be useful in the past, it was a single lstm layer followed by a dense layer to get the precipitation output. this is of course the most simple setup you can use and in my experience it is a good starting point. my early models looked something like this, in keras:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1)) # output is one value (precipitation)
    model.compile(optimizer='adam', loss='mse')
    return model

# example usage
input_shape = (seq_length, 4) # time_steps, features
model = build_lstm_model(input_shape)
model.summary()
```

important details: the `input_shape` parameter is critical. make sure it is compatible with your data. also the activation function can have an impact. try relu or tanh. mse which stands for mean squared error, might be an acceptable loss function. but it depends if you are doing regression or classification. in the early days, my initial attempts, believe or not, predicted negative rain, yeah i know, not very useful. we had to use a few adjustments and tweaks. sometimes the problem with machine learning is that they will "learn" what you tell them to, even if it is nonsense.

the devil, as they say, is in the details. there are so many things you'll need to take care of. for instance, feature scaling. before feeding data to your lstm, you need to scale it. some features have big scales and some have small. without scaling, training will not be effective. i've seen that firsthand. i usually standardize my features using mean and standard deviation. sklearn’s `standardscaler` class comes in handy for that. also, you will notice that your model might just output a very low precipitation, something like 0.0001. why? because it is trying to minimize the mean squared error, that means, if the values of precipitation are low for your dataset, minimizing the error will usually lead to low predictions. we have to be mindful of that. it is the "everything is a nail" problem with lstms, also known as the "everything looks like a low value" problem.

another critical part is data. the amount and quality of your data greatly influence the performance of your model. my initial dataset was small and very inconsistent. i had to go back and invest quite some time in cleaning the data and looking for better sources. and this is a recurring theme in any machine learning project. garbage in, garbage out.

as an example, if you are trying to predict rain with hourly resolution you will see less rain compared if you predict for a whole day. this is important when building the dataset, since many things like temperature and wind might change multiple times during a single day.

as for resources, instead of sending you random links, i recommend reading deep learning by ian goodfellow, this is a very comprehensive book in the field. also, the time series analysis book by james d. hamilton is a very solid choice, if you want to dive deeper into the statistical models. also, if you want to learn more about time series, i recommend "forecasting principles and practice" by hyndman and athanasopoulos, this one is freely available online. these will give a solid base to understand the problem.

finally, evaluating your lstm is vital. standard metrics like mean absolute error, mean squared error and r2 can help. however, sometimes the metrics don’t tell the whole story. for instance, if the model is constantly underestimating rainfall, that might be a bigger problem than the mean square error tells you. so, i always evaluate the results from a practical perspective. what we want is useful rain forecast, that’s the main objective. remember your goal. is not to achieve perfect metrics. this is a mistake i did a few times.

and well, that is the gist of it. using lstms for precipitation is not like baking a cake. you need to iterate, experiment and most importantly, understand your data. you will eventually get there. and remember that if your model outputs a weird result, it is probably because of the data or how you are structuring the data itself. and finally, don't forget to have some good coffee nearby.

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# create dummy data
np.random.seed(42)
data = np.random.rand(500, 5)
seq_length = 24 # lets use 24 hours as an example

# prepare the data
xs, ys = create_sequences(data, seq_length)

# scale the input data
scaler_x = StandardScaler()
xs_reshaped = xs.reshape(-1, xs.shape[-1])
xs_scaled = scaler_x.fit_transform(xs_reshaped)
xs = xs_scaled.reshape(xs.shape)

# scale the output
scaler_y = StandardScaler()
ys = ys.reshape(-1, 1)
ys_scaled = scaler_y.fit_transform(ys)
ys = ys_scaled.flatten()

# split the dataset
xs_train, xs_test, ys_train, ys_test = train_test_split(xs, ys, test_size=0.2, shuffle=False)

# build model
input_shape = (seq_length, 4)
model = build_lstm_model(input_shape)

# train
model.fit(xs_train, ys_train, epochs=10, batch_size=32, verbose=0)

# predict
y_pred_scaled = model.predict(xs_test)

# inverse scale the prediction
y_pred = scaler_y.inverse_transform(y_pred_scaled)

# print example
print(f"predicted precipitation for 10 samples: {y_pred[:10].flatten()}")
```
