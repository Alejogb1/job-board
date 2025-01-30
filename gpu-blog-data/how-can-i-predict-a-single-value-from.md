---
title: "How can I predict a single value from multiple parallel multivariate time series using LSTMs in Keras?"
date: "2025-01-30"
id: "how-can-i-predict-a-single-value-from"
---
Parallel multivariate time series, where multiple sequences of observations are recorded simultaneously, present a distinct challenge for predictive modeling. Unlike single-sequence time series analysis, these data structures involve intricate dependencies *within* each series and *between* different series. Successfully extracting these relationships, crucial for accurate single-value prediction, requires careful architectural design and training considerations when utilizing Long Short-Term Memory (LSTM) networks within Keras.

Fundamentally, the approach involves processing each multivariate time series in parallel through individual LSTM layers or sub-networks, followed by a mechanism to combine the learned features into a single representation. This unified feature vector is then projected to the final scalar output. This two-stage process allows the model to learn both the temporal characteristics of each individual time series, along with any shared or correlated dynamics across the set.

The core architecture hinges on how to fuse these parallel streams of temporal feature representations. A naive solution of directly concatenating the LSTM outputs can lead to overly complex and high-dimensional spaces, risking overfitting. Instead, we need a more refined aggregation technique, with possibilities ranging from simple averaging or weighted aggregation to complex attention mechanisms. The best choice depends on the nature of the data and anticipated relationships.

Consider a hypothetical scenario: I worked on a project involving predicting the overall health status of industrial machinery based on simultaneous sensor readings. Here, each machine emitted various types of sensor data (temperature, pressure, vibration, etc.) over time. Each of these sensors formed an individual time series, which together make a single multivariate time series for each machine. The task was to predict a single 'health score' at the end of a given observation window, providing a single value representing the overall condition.

My initial approach used a relatively straightforward method utilizing a combination of individual LSTMs and a simple average pooling operation. The basic framework was to have an LSTM layer for each individual time series within a multivariate sequence. Each LSTM processed one of the input streams producing a representation (the last hidden state). Those resulting hidden states from each parallel LSTM are then averaged together element-wise, resulting in a single representation. Finally, a dense layer mapped this single representation to the desired single scalar prediction output.

Below is the Keras code example:

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, AveragePooling1D, Layer
from tensorflow.keras.models import Model
import numpy as np

def create_parallel_lstm_average_model(input_shape, num_series, lstm_units, dense_units):
    inputs = []
    lstm_outputs = []

    for i in range(num_series):
        input_tensor = Input(shape=(input_shape[1], 1), name=f"input_{i}") # Expects (timesteps, features) format
        lstm_layer = LSTM(lstm_units, return_sequences=False)(input_tensor)
        inputs.append(input_tensor)
        lstm_outputs.append(lstm_layer)

    # Manually Average:
    merged = tf.keras.layers.Average()(lstm_outputs)


    output = Dense(dense_units, activation='relu')(merged)
    output = Dense(1)(output)

    model = Model(inputs=inputs, outputs=output)
    return model

# Example usage:
input_shape = (20, 1) # 20 time steps, single feature per time series
num_series = 3       # 3 parallel time series
lstm_units = 64
dense_units = 32

model = create_parallel_lstm_average_model(input_shape, num_series, lstm_units, dense_units)
model.compile(optimizer='adam', loss='mse')
model.summary()

# Dummy data
X = [np.random.rand(100, 20, 1) for _ in range(num_series)]  # 100 samples, 20 time steps, single feature per series
y = np.random.rand(100, 1) # Target single output

model.fit(X, y, epochs=10, verbose=0)

test_X = [np.random.rand(5, 20, 1) for _ in range(num_series)]
prediction = model.predict(test_X)
print("prediction: ", prediction)


```
This code creates a model accepting a list of time series input tensors. Each series is processed by a separate LSTM layer, and their outputs are averaged prior to further processing.

While simple averaging can be effective in some cases, sometimes a weighted average is necessary. If certain time series are more predictive of the target, assigning them higher weights can improve performance. The model needs a way to learn those weights. This can be implemented through a learned dense layer which will generate individual weights for each time series.

Here is the code example demonstrating a weighted sum of the LSTM outputs:

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Layer, Multiply
from tensorflow.keras.models import Model
import numpy as np

class WeightedSum(Layer):
  def __init__(self, num_series, **kwargs):
    super(WeightedSum, self).__init__(**kwargs)
    self.num_series = num_series
    self.weight_layer = None

  def build(self, input_shape):
        self.weight_layer = self.add_weight(name='weights',
                                          shape=(self.num_series,),
                                          initializer='uniform',
                                          trainable=True)

  def call(self, lstm_outputs):
    weighted_outputs = [Multiply()([lstm_outputs[i], self.weight_layer[i]]) for i in range(self.num_series)]
    return tf.keras.backend.sum(tf.stack(weighted_outputs, axis=1), axis=1)


def create_parallel_lstm_weighted_model(input_shape, num_series, lstm_units, dense_units):
    inputs = []
    lstm_outputs = []

    for i in range(num_series):
        input_tensor = Input(shape=(input_shape[1], 1), name=f"input_{i}")
        lstm_layer = LSTM(lstm_units, return_sequences=False)(input_tensor)
        inputs.append(input_tensor)
        lstm_outputs.append(lstm_layer)

    merged = WeightedSum(num_series)(lstm_outputs)

    output = Dense(dense_units, activation='relu')(merged)
    output = Dense(1)(output)

    model = Model(inputs=inputs, outputs=output)
    return model

# Example usage:
input_shape = (20, 1) # 20 time steps, single feature per time series
num_series = 3       # 3 parallel time series
lstm_units = 64
dense_units = 32

model = create_parallel_lstm_weighted_model(input_shape, num_series, lstm_units, dense_units)
model.compile(optimizer='adam', loss='mse')
model.summary()

# Dummy data
X = [np.random.rand(100, 20, 1) for _ in range(num_series)]  # 100 samples, 20 time steps, single feature per series
y = np.random.rand(100, 1) # Target single output

model.fit(X, y, epochs=10, verbose=0)

test_X = [np.random.rand(5, 20, 1) for _ in range(num_series)]
prediction = model.predict(test_X)
print("prediction: ", prediction)

```
In this example, a custom Keras Layer, `WeightedSum`, is defined to learn aggregation weights. Each LSTM output is multiplied by its corresponding weight before summing.

Finally, complex correlations or non-linear relationships between the outputs of the parallel LSTMs can be handled by using an additional network to combine the LSTM outputs. Rather than using a custom averaging or weight layer, the outputs are combined via a final MLP (Multi Layer Perceptron).

Here is the code example showing fusion by MLP:

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.models import Model
import numpy as np


def create_parallel_lstm_mlp_model(input_shape, num_series, lstm_units, dense_units):
    inputs = []
    lstm_outputs = []

    for i in range(num_series):
        input_tensor = Input(shape=(input_shape[1], 1), name=f"input_{i}")
        lstm_layer = LSTM(lstm_units, return_sequences=False)(input_tensor)
        inputs.append(input_tensor)
        lstm_outputs.append(lstm_layer)

    merged = Concatenate()(lstm_outputs)

    # MLP for fusion
    output = Dense(dense_units, activation='relu')(merged)
    output = Dense(dense_units, activation='relu')(output)

    output = Dense(1)(output)

    model = Model(inputs=inputs, outputs=output)
    return model

# Example usage:
input_shape = (20, 1) # 20 time steps, single feature per time series
num_series = 3       # 3 parallel time series
lstm_units = 64
dense_units = 32

model = create_parallel_lstm_mlp_model(input_shape, num_series, lstm_units, dense_units)
model.compile(optimizer='adam', loss='mse')
model.summary()

# Dummy data
X = [np.random.rand(100, 20, 1) for _ in range(num_series)]  # 100 samples, 20 time steps, single feature per series
y = np.random.rand(100, 1) # Target single output

model.fit(X, y, epochs=10, verbose=0)

test_X = [np.random.rand(5, 20, 1) for _ in range(num_series)]
prediction = model.predict(test_X)
print("prediction: ", prediction)

```
In this example, the individual LSTM outputs are concatenated, which is then fed to a dense layer, followed by another dense layer. The final layer provides the desired single scalar output.

The success of each of these methods hinges on data characteristics. Average pooling works well if all series contribute similarly; weighted pooling provides more flexibility; and finally, an MLP based fusion is often beneficial when non-linear or complex interactions are expected between series. For further exploration in this area, consider studying literature on attention mechanisms, specifically those adapted for multi-input scenarios. Resources from machine learning conferences focusing on sequence modeling would prove useful. Look into various case studies to observe diverse approaches for combining parallel streams.
Experimentation is key; therefore, a careful hyperparameter selection based on the specific dataset is often necessary for optimal performance.
