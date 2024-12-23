---
title: "How can RNNs predict future values using multiple features?"
date: "2024-12-23"
id: "how-can-rnns-predict-future-values-using-multiple-features"
---

Okay, let's tackle this. It's a problem I've seen crop up in quite a few projects, from predicting network traffic to anticipating stock prices. The crux of it lies in how we feed multi-dimensional time-series data into a recurrent neural network (rnn) effectively. It's not just about chucking all the features in and hoping for the best; a bit of finesse is required. Let's break it down.

The fundamental challenge when predicting future values with an rnn using multiple features is how to represent and process the input sequence. An rnn, at its core, is designed to handle sequential data where the output at a given time step is dependent on past inputs. When we introduce multiple features, we aren't just dealing with a single series of values anymore, but rather a set of parallel time series. So, each time step has multiple values associated with it, each representing a different feature. We need to carefully arrange this data so the rnn can learn the underlying relationships and dependencies within *and* across these features.

My experience with this goes back to a project involving predicting energy consumption in a data center. We had a whole host of variables – temperature, humidity, cpu load, network traffic, you name it. Simply concatenating all those features at each timestamp and presenting that as a single input vector to the rnn resulted in pretty poor performance initially. The network didn't seem to be capturing the distinct characteristics of each feature. The solution, as it often does in deep learning, involved a nuanced approach to data preprocessing and network architecture.

The first crucial step is **data normalization**. We can’t feed raw data into our rnn without preprocessing each feature individually. Features may have different scales. Consider network traffic measured in gigabytes per second and temperature in degrees Celsius. If left as is, the network may unfairly give more 'weight' to features with larger values. Standardizing each feature to have a mean of 0 and a standard deviation of 1 is a common approach. This ensures that no single feature dominates during the training phase. Another approach, min-max scaling, transforms values into a range of 0 and 1, which can be beneficial depending on your dataset characteristics. Critically, any normalization should be performed on the *training* data, with the mean and standard deviation of the training set used to transform test data.

Now, let’s discuss **input reshaping**. You can think of your input as a tensor of shape `[number_of_samples, timesteps, number_of_features]`. The 'samples' represent individual time-series segments in your dataset. 'Timesteps' is the length of the historical data you feed into the rnn at a time. And 'features' is the number of parallel time series that you’re dealing with. For instance, if you are using 10 timesteps of 3 different features to predict the 11th timestep, the input tensor might have dimensions of, say, [1000, 10, 3], where 1000 is the number of training sequences, 10 is the number of timesteps used to make a prediction, and 3 is the number of features being used.

Let's look at some code snippets using python with keras/tensorflow to make this a bit more concrete.

**Snippet 1: Data Reshaping and Normalization**

```python
import numpy as np
from sklearn.preprocessing import StandardScaler

def prepare_data(data, sequence_length):
    """
    Prepares time-series data for an rnn.

    Args:
    data: A numpy array with shape [number_of_samples, number_of_features].
    sequence_length: The length of the input sequence.

    Returns:
    A tuple containing input data, output data, and scalers.
    """
    num_samples = data.shape[0] - sequence_length
    num_features = data.shape[1]

    X = np.zeros((num_samples, sequence_length, num_features))
    y = np.zeros((num_samples, num_features))
    scalers = []

    for i in range(num_features):
        scaler = StandardScaler()
        scaled_feature = scaler.fit_transform(data[:,i].reshape(-1,1)).flatten()
        scalers.append(scaler)

        for j in range(num_samples):
            X[j, :, i] = scaled_feature[j:j+sequence_length]
            y[j, i] = scaled_feature[j+sequence_length]

    return X, y, scalers

# Example usage:
data = np.random.rand(1000, 3) # 1000 time steps, 3 features
sequence_length = 20

X, y, scalers = prepare_data(data, sequence_length)

print(f"Input data shape: {X.shape}") # Output: Input data shape: (980, 20, 3)
print(f"Output data shape: {y.shape}") # Output: Output data shape: (980, 3)
```

This script showcases the core idea: iterate through each feature, scale it independently, and reshape it into input sequences and corresponding outputs. The `StandardScaler` is just one possibility; `MinMaxScaler` or other scalers from `sklearn.preprocessing` could be used as well.

Now let's talk about the rnn architecture itself. We often use lstm (long short-term memory) or gru (gated recurrent unit) layers instead of vanilla rnn layers due to their capability to capture long-term dependencies.

**Snippet 2: Constructing the RNN**

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

def build_rnn_model(input_shape, num_features):
    """Builds a simple LSTM-based model for multi-feature prediction.
      Args:
        input_shape: Shape of the input data (timesteps, features).
        num_features: The number of features in the input.

      Returns:
        A Keras Sequential model
    """

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='relu', return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(num_features))
    model.compile(optimizer='adam', loss='mse')
    return model

#Example usage
input_shape = (sequence_length, num_features)
model = build_rnn_model(input_shape, num_features)
model.summary()
```

This builds a basic lstm model. Notice how the `input_shape` parameter dictates the time steps and features. The key point here is that the final dense layer has a number of nodes corresponding to the number of output features that the model needs to predict. The `return_sequences` parameter is essential to stack lstm layers.

Finally, **making predictions** and, crucially, **reversing the scaling** is vital. We don't want to present scaled outputs as our predictions, we want real-world values.

**Snippet 3: Prediction and Inverse Scaling**

```python
def predict_future(model, last_sequence, scalers):
    """Predicts the next value using the trained model.

      Args:
        model: Trained Keras model.
        last_sequence: The last sequence of data used to generate the prediction.
        scalers: List of fitted scalers from the prepare_data function

      Returns:
          The predicted output (numpy array with one element for each feature).
    """
    last_sequence = last_sequence.reshape((1, last_sequence.shape[0], last_sequence.shape[1]))
    prediction = model.predict(last_sequence)

    # Inverse transform the scaled prediction.
    original_prediction = []
    for i, scaler in enumerate(scalers):
       original_prediction.append(scaler.inverse_transform(prediction[0,i].reshape(1, -1))[0,0])
    return np.array(original_prediction)

# Example of making a prediction:
last_sequence = X[-1] # Use the last training sequence for demonstration
predicted_value = predict_future(model, last_sequence, scalers)

print(f"Predicted value: {predicted_value}")
```

The key here is using the same scalers (fitted on the training set) to reverse the scaling and recover our original units.

For anyone wishing to delve deeper, I’d highly recommend looking into *Deep Learning* by Goodfellow, Bengio, and Courville. It's a comprehensive resource for understanding the fundamentals of deep learning, including recurrent neural networks. Another excellent resource is *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Géron which provides practical implementations of various machine learning algorithms, including time-series analysis using rnn's. Furthermore, research papers published by scholars such as Hochreiter and Schmidhuber would provide a good deep dive into specific architectures such as lstm’s and gru’s.

In summary, predicting future values using rnn's and multiple features involves careful data preparation including normalization and proper reshaping of input, choosing suitable rnn architecture (typically lstm or gru), and scaling back the predictions to original values, and as ever, careful testing of your model with proper cross validation is essential to make sure that the results are generalizable and aren't suffering from overfitting. Each project might need slight modifications, but this should give you solid starting point.
