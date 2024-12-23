---
title: "How to forecast univariate time series with LSTM in TensorFlow?"
date: "2024-12-23"
id: "how-to-forecast-univariate-time-series-with-lstm-in-tensorflow"
---

Let's dive straight into this, shall we? I've tackled my fair share of time series forecasting projects over the years, and the challenge of accurately predicting future values from past data, especially with the added complexity of long-term dependencies, is something I've seen evolve considerably. LSTMs (Long Short-Term Memory networks) within TensorFlow have proven to be remarkably powerful tools, and I'd like to walk you through how I typically approach these kinds of problems, focusing on univariate (single variable) time series.

First, it’s essential to understand that time series forecasting, even with powerful techniques like LSTMs, isn't simply about plugging data into a model and hoping for the best. It's a meticulous process that requires careful preprocessing, thoughtful model construction, and rigorous evaluation. I remember a project where we were tasked with predicting daily energy consumption for a large manufacturing plant. We started with a basic regression model, and it quickly became clear that we were missing something – the models failed to capture the seasonality and dependencies across multiple days. That's when we shifted to using LSTMs.

Before we get to code, let's discuss key preparation steps. Preprocessing your time series data is paramount. Standard scaling (also known as z-score normalization) is my go-to method, though min-max scaling works well too, especially if your data is bounded. This brings the data into a uniform range, which makes the training process significantly easier and faster for the neural network. A time series is inherently sequential, so we can't just randomly shuffle the data like with traditional classification or regression tasks. Instead, I create sequences of a fixed length (a look-back window), where each sequence becomes a training instance, and the target is typically the next value in the time series, effectively a one-step-ahead prediction. The creation of such sequence data from the raw time series is crucial for feeding into an LSTM.

Now, let's get into the actual TensorFlow implementation. I'll break it down into a few manageable code snippets with explanations.

**Snippet 1: Data Preprocessing and Sequence Generation**

```python
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def preprocess_data(raw_data, seq_length):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(raw_data.reshape(-1, 1)).flatten()
    x, y = create_sequences(scaled_data, seq_length)
    return x, y, scaler

# Example usage with dummy data
raw_data = np.array(range(100), dtype=float) #Dummy Data
seq_length = 10
x, y, scaler = preprocess_data(raw_data, seq_length)
print("Shape of X:", x.shape)
print("Shape of y:", y.shape)

```

In this snippet, `create_sequences` is responsible for forming the input-output pairs from your time series, taking a raw data input and generating sequences of a specified length along with the corresponding target values. I implement scaling using the `StandardScaler` from scikit-learn within `preprocess_data`. It scales your input time series data, making training much more stable and faster for the lstm model. I return also the scaler to reverse the normalization after prediction. It’s important to note that the shapes of the resulting arrays (x,y) are structured to match the inputs the LSTM layers will require, that is, a sequence length dimension, number of features(which here is just one), and number of samples.

**Snippet 2: LSTM Model Construction**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_lstm_model(seq_length, units=50, dropout_rate=0.2):
    model = Sequential()
    model.add(LSTM(units, activation='relu', input_shape=(seq_length, 1)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))  # Single output for the next value in the series
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Model initialization and compiling
model = build_lstm_model(seq_length=seq_length)
model.summary()

```

Here, we define the `build_lstm_model` function. It creates a Sequential model with an LSTM layer followed by a dropout layer to help prevent overfitting, and finally a dense layer for the output. I'm using 'relu' as the activation function in the LSTM layer. Other activation functions like 'tanh' are viable alternatives, but I often start with relu. The model uses an adam optimizer which generally works very well. The loss function being mean squared error (mse), is standard for regression tasks. The `model.summary()` provides a glimpse into the model architecture, which I find particularly useful when working with complex networks.

**Snippet 3: Training and Prediction**

```python
def train_and_predict(model, x_train, y_train, x_test, scaler, epochs=20, batch_size=32):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
    predicted_scaled = model.predict(x_test)
    predicted = scaler.inverse_transform(predicted_scaled).flatten()
    return predicted

# Splitting data, training, and making predictions
split_index = int(0.8 * len(x)) # 80% for train and 20% for testing.
x_train, x_test = x[:split_index], x[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted = train_and_predict(model, x_train, y_train, x_test, scaler)

print("Predictions:", predicted)

```

The `train_and_predict` function is where the magic happens. We fit the model to the training data (x_train, y_train), make predictions on the test data (x_test) and then, crucially, reverse the scaling to bring the predictions back to the original data's scale. Then it's just a matter of printing the results. The crucial step here is reshaping the arrays before passing to the model, adding an extra dimension which is needed for keras.

Beyond this basic framework, there are several avenues for improvement. You might explore stacking multiple LSTM layers to create more complex models. Different optimization algorithms could improve convergence speed or model accuracy. Hyperparameter tuning, often via techniques like grid search or random search, is essential for finding the ideal model parameters and number of units for a specific problem. Don't underestimate the importance of evaluating the model not just with mse, but with metrics that more closely align with your objective, such as mean absolute percentage error (mape). Further, you might need to consider stationarity in your time series data; that is, your data should be constant in mean and variance over time, if not, there are transformations, such as differencing, that you could apply to the time series.

For further study, I highly recommend 'Forecasting: Principles and Practice' by Rob J Hyndman and George Athanasopoulos. For a deeper dive into neural networks, 'Deep Learning' by Ian Goodfellow, Yoshua Bengio, and Aaron Courville is the standard. These resources provide a solid theoretical and practical foundation for tackling advanced time series forecasting challenges.

Ultimately, successful time series forecasting is an iterative process. It’s about understanding your data, carefully designing your model, and meticulously evaluating the results. The code snippets here are a starting point, but they will get you moving in the right direction, and from there it's a matter of experimentation, refinement, and continuous learning.
