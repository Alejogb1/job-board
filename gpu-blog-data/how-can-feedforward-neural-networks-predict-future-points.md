---
title: "How can feedforward neural networks predict future points?"
date: "2025-01-30"
id: "how-can-feedforward-neural-networks-predict-future-points"
---
Predicting future points using feedforward neural networks involves a critical conceptual shift from static classification or regression to time-series forecasting, requiring adaptation of both the input data and the network's training methodology. The fundamental challenge lies in capturing temporal dependencies inherent in sequential data using a network architecture traditionally designed for independent inputs.

My experience building predictive models in a high-throughput sensor data environment has reinforced the understanding that, while feedforward networks are not natively equipped to handle sequences as recurrent neural networks (RNNs) are, they can be effectively utilized for time-series forecasting through a technique known as 'windowing' or 'lagging'. This method transforms sequential data into a series of overlapping or non-overlapping windows of historical values, which serve as static inputs for the feedforward network. The network, then, learns to predict the next value (or values) based on this historical context. This, essentially, converts a temporal problem into a spatial one.

The crucial step is preparing the input data. Rather than feeding raw sequential values directly, we create input vectors that represent a ‘snapshot’ of the past *n* time steps, where *n* is the selected window size. For example, consider a time series: [10, 12, 15, 13, 16, 18, 20, 22]. If we choose a window size of 3, we would construct input-output pairs like this:

Input: [10, 12, 15] Output: [13]
Input: [12, 15, 13] Output: [16]
Input: [15, 13, 16] Output: [18]
Input: [13, 16, 18] Output: [20]
Input: [16, 18, 20] Output: [22]

The output is essentially the value immediately following the windowed input. The feedforward network learns a mapping from this 'historical' input vector to the predicted future value. This simple transformation enables us to use standard feedforward architectures, such as multi-layer perceptrons (MLPs), for predictive modeling.

The choice of window size, *n*, is a critical hyperparameter and significantly impacts the model’s performance. Too small a window might not capture sufficient historical context, while too large a window might introduce irrelevant noise and increase the number of model parameters, leading to overfitting. We often experiment with different window sizes and utilize a validation set to evaluate their effectiveness. Additionally, the network's depth and width also play a role and should be tuned for each specific application.

Moreover, data normalization and scaling are crucial preprocessing steps. Given that the network's activation functions typically operate within specific ranges, applying techniques like min-max scaling or standardization can significantly improve convergence speed and prediction accuracy.

Here are three code examples to illustrate these concepts using Python and libraries commonly employed in machine learning.

**Example 1: Simple Windowing and MLP Training with NumPy and scikit-learn.**

```python
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample Time Series Data
data = np.array([10, 12, 15, 13, 16, 18, 20, 22, 25, 23, 26, 28, 30, 32])

def create_windows(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

window_size = 3
X, y = create_windows(data, window_size)

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the MLP model
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Make Predictions
predictions = model.predict(X_test_scaled)

print("Predictions:", predictions)
```
*Commentary:* This example demonstrates the fundamental process. The `create_windows` function generates lagged input features. Data is then split, scaled using standardization, and finally, an MLP regressor is trained to predict future values. This illustrates the core concept: transforming a time-series problem into a regression problem through data preprocessing.

**Example 2: Prediction of Multiple Steps Ahead.**

```python
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Sample Time Series Data
data = np.array([10, 12, 15, 13, 16, 18, 20, 22, 25, 23, 26, 28, 30, 32])

def create_windows_multi_step(data, window_size, steps_ahead):
    X = []
    y = []
    for i in range(len(data) - window_size - steps_ahead + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+steps_ahead])
    return np.array(X), np.array(y)

window_size = 3
steps_ahead = 2
X, y = create_windows_multi_step(data, window_size, steps_ahead)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the MLP model
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Make predictions
predictions = model.predict(X_test_scaled)

print("Predictions:", predictions)
```
*Commentary:* This example extends the previous one to predict multiple steps ahead. The `create_windows_multi_step` function is modified to produce multiple outputs, effectively transforming the problem into multi-output regression. The network is still trained using standard methods, but now predicts a vector of future values. This highlights the adaptability of the method to handle a broader set of forecasting requirements.

**Example 3: Using a sliding window for online prediction.**

```python
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Sample Time Series Data (Assume new data arrives online)
data = np.array([10, 12, 15, 13, 16, 18, 20, 22, 25, 23, 26, 28, 30, 32])
window_size = 3

# Initialize and Train Model (Use data up to some point)
train_data = data[:10]
def create_windows(data, window_size):
    X = []
    y = []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)
X_train, y_train = create_windows(train_data, window_size)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
model = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
model.fit(X_train_scaled, y_train)

# Online prediction loop
for t in range(10,len(data)):
  current_window = data[t-window_size:t]
  current_window_scaled = scaler.transform(current_window.reshape(1,-1))
  prediction = model.predict(current_window_scaled)
  print(f"Time {t}: Window: {current_window}, Predicted: {prediction[0]:.2f}, Actual: {data[t]}")
```

*Commentary:* This example demonstrates how to use a sliding window in an 'online' scenario. Instead of a fixed training/test split, we imagine new data arriving in real time. We train the model initially, and then, at each new time step, we create a new window of data, scale it, and predict the next value. This showcases a practical approach to deploying these models in streaming data environments.

In conclusion, predicting future points using feedforward neural networks is achievable through careful data preprocessing, specifically by creating lagged windows. While these networks lack the native temporal memory of recurrent architectures, this windowing technique enables them to effectively model short-term dependencies within time-series data. Careful selection of the window size, data scaling techniques, and network parameters remain crucial for achieving good prediction performance. Furthermore, these techniques can be extended to multi-step prediction or online prediction scenarios.

For further exploration of this topic, I recommend exploring resources on time series forecasting and feature engineering, specifically examining the concepts of lagging, windowing, and different data scaling approaches. Books and research papers focusing on non-recurrent neural networks for time series also provide valuable insights. It is also beneficial to familiarize oneself with the capabilities of common machine learning libraries such as scikit-learn and TensorFlow or PyTorch which are heavily used in this domain.
