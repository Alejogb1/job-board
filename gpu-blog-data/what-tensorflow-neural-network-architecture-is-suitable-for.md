---
title: "What TensorFlow neural network architecture is suitable for this task?"
date: "2025-01-30"
id: "what-tensorflow-neural-network-architecture-is-suitable-for"
---
Given a time series dataset of hourly electricity consumption for a single household over several years, and the objective of forecasting consumption for the next 24 hours, a recurrent neural network (RNN), specifically a Long Short-Term Memory (LSTM) network, offers a robust architecture. Traditional feedforward networks struggle with sequential data where temporal dependencies are critical, making them unsuitable for time series forecasting. The sequential nature of consumption patterns – where past usage heavily influences future demand – necessitates an architecture that can learn and retain information across time steps.

The choice of an LSTM over other RNN variants stems from its ability to mitigate the vanishing gradient problem prevalent in vanilla RNNs. This issue makes learning long-range dependencies difficult, which is crucial in this electricity consumption scenario. An LSTM achieves this through a cell state and specialized gates: input, forget, and output gates. These gates regulate the flow of information, selectively adding, removing, or retaining information from the cell state, enabling the network to maintain long-term memory effectively.

For this task, we wouldn’t utilize a simple single-layer LSTM; instead, we should employ a stacked LSTM network. Stacking multiple LSTM layers increases the model's capacity to learn complex temporal patterns. A single layer might be limited in extracting intricate dependencies present in electricity consumption data. The first LSTM layer can identify basic patterns like daily cycles, while subsequent layers can focus on more subtle seasonal variations or anomalous behavior.

The input data for our model will consist of a sequence of hourly consumption values. We will pre-process this data by scaling it, usually through standardization (subtracting the mean and dividing by the standard deviation) or min-max scaling (scaling values between 0 and 1). This preprocessing step ensures the model converges faster and prevents features with larger magnitudes from dominating the learning process. The target variable will be the electricity consumption for the next 24 hours. Consequently, we frame this as a supervised learning problem, where we create input sequences of a certain length (the lookback window) and corresponding output sequences (the next 24 hours).

Now, let's explore code examples. The first demonstrates how to create the time-series sequences from raw data:

```python
import numpy as np

def create_sequences(data, lookback, forecast_horizon):
    """
    Generates input and output sequences for time series forecasting.

    Args:
      data: Numpy array of time series data.
      lookback: The length of the input sequence (lookback window).
      forecast_horizon: The length of the output sequence.

    Returns:
      A tuple containing input sequences (X) and output sequences (y).
    """
    X, y = [], []
    for i in range(len(data) - lookback - forecast_horizon + 1):
        X.append(data[i:(i + lookback)])
        y.append(data[(i + lookback):(i + lookback + forecast_horizon)])
    return np.array(X), np.array(y)

# Example usage with dummy data
dummy_data = np.arange(100).astype(float)
lookback_window = 48
forecast_window = 24
X, y = create_sequences(dummy_data, lookback_window, forecast_window)
print(f"Shape of Input Sequences (X): {X.shape}")
print(f"Shape of Output Sequences (y): {y.shape}")
```

This code demonstrates the core logic of constructing input/output pairs for our LSTM. Given a time-series of 100 values, a `lookback_window` of 48, and a `forecast_window` of 24, it creates 29 training examples. Each input sequence in X has a length of 48 representing past hours and each corresponding output sequence in y has a length of 24 representing the next 24 hours. The function efficiently creates all the sequences without explicit iteration over the entire dataset.

The second code snippet shows a basic TensorFlow model construction using the `tf.keras.layers` API for a stacked LSTM architecture:

```python
import tensorflow as tf

def create_lstm_model(lookback, forecast_horizon, num_features, lstm_units, dropout_rate, optimizer, learning_rate):
    """
    Constructs a stacked LSTM model for time series forecasting.

    Args:
      lookback: The length of the input sequence.
      forecast_horizon: The length of the output sequence.
      num_features: The number of features in each time step.
      lstm_units: The number of LSTM units in each layer.
      dropout_rate: Dropout rate to prevent overfitting.
      optimizer: The optimizer to use, e.g., 'adam'.
      learning_rate: The learning rate.

    Returns:
      A compiled TensorFlow Keras model.
    """
    model = tf.keras.models.Sequential([
      tf.keras.layers.Input(shape=(lookback, num_features)),
      tf.keras.layers.LSTM(lstm_units, return_sequences=True),
      tf.keras.layers.Dropout(dropout_rate),
      tf.keras.layers.LSTM(lstm_units),
      tf.keras.layers.Dropout(dropout_rate),
      tf.keras.layers.Dense(forecast_horizon)
    ])
    model.compile(optimizer=optimizer, loss='mse') # Using mean squared error
    return model

# Example Usage
num_features = 1 # Only electricity consumption for now
lstm_units = 50
dropout_rate = 0.2
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model = create_lstm_model(lookback_window, forecast_window, num_features, lstm_units, dropout_rate, optimizer, learning_rate)
model.summary()
```

This code creates a sequential Keras model consisting of two LSTM layers and two dropout layers to mitigate overfitting. It takes the sequence length (`lookback`), the forecast length, and the number of input features (one in our case - electricity consumption) as parameters. Crucially, the first LSTM layer is set to `return_sequences=True`, ensuring its output remains a sequence, which is then consumed by the subsequent LSTM layer. The final dense layer outputs the predicted values for the next 24 hours. The model utilizes the Adam optimizer and Mean Squared Error (MSE) as the loss function, suitable for regression tasks. The `model.summary()` call provides a detailed overview of the model's architecture and parameters.

The final example showcases how we can train the created model and make predictions:

```python
# Assuming X and y were created from previous code and split into train/test sets.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the input data to include the features dimension
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], num_features))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], num_features))

# Training the model
epochs = 50 # Adjust as needed
batch_size = 32
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=0) # verbose=0 suppresses the output of epoch loss

# Making predictions
predictions = model.predict(X_test)

print(f"Shape of test set predictions:{predictions.shape}")

# Now the predictions can be evaluated based on a set of metrics.
```

Here, after partitioning the dataset into train and test sets, we adjust the input to accommodate the feature dimension required by the `LSTM` layers in our model. We train the model for a set number of epochs using a batch size of 32 and validate the model during training. The verbose=0 argument suppresses the usual epoch-by-epoch output, which is helpful for cleaner logs. After training, `model.predict` is used to produce forecast for our test dataset. The shape of predictions is `(number of testing examples, 24)`.

For further information, consider exploring resources dedicated to time series forecasting using RNNs. Textbooks on Deep Learning often contain detailed explanations of RNN architectures. Books specifically tailored to practical machine learning with time-series data would provide invaluable guidance. Publications and documentation from TensorFlow are essential for mastering the intricacies of the framework, and comprehensive resources covering time-series analysis are paramount to understand the characteristics of the data. Exploring advanced preprocessing techniques, further experimentation with model architectures and parameters (such as the number of LSTM units and layers), and investigating more sophisticated metrics for model evaluation will further enhance performance of this forecast method.
