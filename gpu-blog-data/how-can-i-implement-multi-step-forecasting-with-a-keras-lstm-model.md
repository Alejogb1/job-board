---
title: "How can I implement multi-step forecasting with a Keras LSTM model?"
date: "2025-01-26"
id: "how-can-i-implement-multi-step-forecasting-with-a-keras-lstm-model"
---

Multi-step forecasting with Long Short-Term Memory (LSTM) networks in Keras necessitates a departure from the standard single-step prediction paradigm. Instead of forecasting a single future value, you're aiming to predict a sequence of future values, which requires architectural and data-handling modifications. My experience in time series modeling for power grid load forecasting highlighted the critical importance of this, as day-ahead predictions relied on hourly multi-step outputs.

The primary challenge lies in aligning the input data with the desired output sequence. For a single-step forecast, your input at time *t* maps to the output at time *t+1*. With multi-step, you must adjust the target to include values *t+1*, *t+2*, ..., *t+n*, where *n* is the forecast horizon (the number of steps you're trying to predict). This involves reshaping the training data accordingly. This is not just about changing shape, but changing how training proceeds and predictions are generated.

There are two predominant strategies for multi-step forecasting using LSTMs: direct multi-step and recursive multi-step. Direct multi-step involves training separate models, or a single model with multiple output nodes, to predict each future time step. Recursive multi-step, conversely, feeds predicted values from the previous steps back into the model to generate subsequent predictions. Both methods have their own use cases and trade-offs, largely surrounding stability and ease of implementation.

Let’s examine how each approach is practically implemented in Keras, using a simplified example where the input sequence length is 20 time steps, and the forecast horizon is 5 steps. I'll assume the presence of time series data, `data`, with a shape of `(number_of_samples, sequence_length)`.

**Example 1: Direct Multi-step Forecasting with a Single LSTM Layer**

In direct forecasting with a single model, we must ensure the output layer has enough units for all predictions. The structure is: `input sequence -> LSTM layer -> output layer`.

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Hypothetical data generation (replace with your actual data)
data = np.random.rand(1000, 20) # 1000 samples, 20 time steps each
forecast_horizon = 5

# Prepare training data and targets
def create_multistep_data(data, sequence_length, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length:i+sequence_length+forecast_horizon])
    return np.array(X), np.array(y)

X, y = create_multistep_data(data, 20, forecast_horizon)

# Model definition
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(20, 1)))  # Single LSTM layer with 50 units
model.add(Dense(forecast_horizon)) # Output layer for 5 predicted values
model.compile(optimizer='adam', loss='mse')

# Reshape input data to match expected input shape (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train model (replace with your training data)
model.fit(X, y, epochs=10, verbose=0)

# Make a prediction on new input
new_input = np.random.rand(1, 20)
new_input = new_input.reshape((new_input.shape[0], new_input.shape[1], 1))
predicted_sequence = model.predict(new_input)

print("Predicted Sequence:", predicted_sequence)
```

Here, the `create_multistep_data` function transforms your time series data into overlapping input-output pairs. The LSTM layer takes a sequence of 20 time steps and returns 50 hidden states. The output `Dense` layer then generates the 5 predicted steps directly. Crucially, the final dense layer has an output dimension equal to the `forecast_horizon`. Note that for the sake of simplicity I've used random data and set training epochs low. Also, reshaping the input data is critical, since the LSTM expects a three-dimensional tensor in the format (samples, timesteps, features), even when the data only has a single feature.

**Example 2: Direct Multi-step Forecasting using Multiple Output LSTMs**

In this variant, we modify the model to use multiple LSTM output heads, which can sometimes provide better model control compared to a single output layer.

```python
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.layers import concatenate

# Hypothetical data generation (replace with your actual data)
data = np.random.rand(1000, 20) # 1000 samples, 20 time steps each
forecast_horizon = 5

# Prepare training data and targets
def create_multistep_data(data, sequence_length, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - sequence_length - forecast_horizon + 1):
        X.append(data[i:i+sequence_length])
        y.append(data[i+sequence_length:i+sequence_length+forecast_horizon])
    return np.array(X), np.array(y)

X, y = create_multistep_data(data, 20, forecast_horizon)

# Model definition
input_layer = Input(shape=(20, 1))
lstm_layer = LSTM(50, activation='relu')(input_layer)

output_layers = []
for i in range(forecast_horizon):
    output = Dense(1)(lstm_layer) #  Predict one value at each step
    output_layers.append(output)

merged_output = concatenate(output_layers)
model = Model(inputs=input_layer, outputs=merged_output)

model.compile(optimizer='adam', loss='mse')


# Reshape input data to match expected input shape (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train model (replace with your training data)
model.fit(X, y, epochs=10, verbose=0)

# Make a prediction on new input
new_input = np.random.rand(1, 20)
new_input = new_input.reshape((new_input.shape[0], new_input.shape[1], 1))
predicted_sequence = model.predict(new_input)

print("Predicted Sequence:", predicted_sequence)
```

Here, the architecture is a bit more complex.  Instead of a single `Dense` output, the  LSTM outputs a hidden state, and that hidden state is used to drive `forecast_horizon` separate `Dense` output layers. Finally, we concatenate all of these individual predictions to create the final multi-step output. This can sometimes help the model learn more granular relationships between the input and output sequences since each output has its own direct connection to the LSTM hidden state. This is not always better than a single output, but it does offer a different way to structure the output layer for specific data cases.

**Example 3: Recursive Multi-step Forecasting**

Recursive multi-step forecasting uses a single output, feeding the prediction back to the input iteratively to produce the full forecast sequence. This approach can propagate errors, as each subsequent step relies on earlier predictions, but avoids training a model with a large output layer.

```python
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Hypothetical data generation (replace with your actual data)
data = np.random.rand(1000, 20) # 1000 samples, 20 time steps each
forecast_horizon = 5

# Prepare training data and targets
def create_singlestep_data(data, sequence_length):
  X, y = [], []
  for i in range(len(data) - sequence_length - 1):
    X.append(data[i:i + sequence_length])
    y.append(data[i + sequence_length])
  return np.array(X), np.array(y)

X, y = create_singlestep_data(data, 20)

# Model definition
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(20, 1))) # Single LSTM layer with 50 units
model.add(Dense(1)) # Output layer with a single prediction
model.compile(optimizer='adam', loss='mse')

# Reshape input data to match expected input shape (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Train model (replace with your training data)
model.fit(X, y, epochs=10, verbose=0)

# Make multi-step predictions recursively
def make_recursive_prediction(model, input_seq, steps):
  predictions = []
  current_seq = input_seq
  for _ in range(steps):
      current_seq = current_seq.reshape((1, current_seq.shape[0], 1))
      next_pred = model.predict(current_seq)
      predictions.append(next_pred[0][0])
      current_seq = np.append(current_seq[0][1:],next_pred[0][0]) # Update input sequence
  return np.array(predictions)


new_input = np.random.rand(20)
predicted_sequence = make_recursive_prediction(model, new_input, forecast_horizon)
print("Predicted Sequence:", predicted_sequence)
```

Here, the model is trained for single-step predictions only. A new function `make_recursive_prediction` is used to generate multi-step forecasts. It iteratively makes predictions and feeds them back into the input sequence, discarding the oldest element from the sequence to maintain a consistent sequence length. Notice the difference in `create_singlestep_data`; because we are now feeding previous *predictions* back in, we train to forecast *one* step ahead, not a sequence. This dramatically changes how the model learns. This strategy avoids complex output layers but can accumulate errors.

For resources, I’d recommend exploring documentation from libraries such as Keras and TensorFlow.  Additionally, time series analysis books and tutorials often contain sections dedicated to forecasting techniques which can be highly useful in understanding the conceptual underpinnings of multi-step forecasting.  Finally, studying academic papers, specifically on recurrent neural networks and time series prediction, will allow you to explore state-of-the-art techniques and advancements in this space. These papers also frequently discuss both the theoretical aspects and practical implications of these methodologies.
