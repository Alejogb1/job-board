---
title: "Why does LSTM univariate time series forecasting exhibit high accuracy on testing data but low accuracy on unseen data?"
date: "2025-01-30"
id: "why-does-lstm-univariate-time-series-forecasting-exhibit"
---
Univariate time series forecasting with Long Short-Term Memory (LSTM) networks often demonstrates a perplexing discrepancy: high accuracy on a designated test set but significantly lower performance when applied to truly unseen data. This disparity arises primarily from a confluence of factors related to data characteristics, model training, and evaluation procedures. I've repeatedly encountered this issue in my work developing predictive models for energy consumption, and the underlying cause is rarely a single point of failure but a combination of subtle effects.

Firstly, the inherent nature of time series data makes it particularly susceptible to overfitting, even within the confines of a seemingly well-defined training and testing regime. Unlike independent and identically distributed (i.i.d.) data, time series data exhibit temporal dependencies and serial correlations. These autocorrelations mean that subsequent data points are often influenced by prior observations. During training, an LSTM network can become excessively attuned to these specific temporal patterns within the training set, effectively memorizing sequences rather than generalizing underlying causal relationships. When the model encounters new data with slightly different temporal structures, its predictive performance suffers. The test set is frequently drawn from the same data distribution as the training data, albeit a different subset, which allows for better performance.

Secondly, improper data preprocessing can contribute to the issue. Techniques such as scaling or normalization are typically applied to the entire dataset before partitioning it into training, validation, and test sets. This approach creates data leakage. Specifically, the scaling parameters (e.g., mean, standard deviation) derived from the entire dataset influence both training and testing data. During inference with unseen data, the network relies on these scaling parameters which are not representative of future data distributions. Furthermore, if the time series exhibits non-stationarity—its statistical properties change over time—the scaling process may further exacerbate the problem because one set of scaling values will be unsuitable across diverse temporal segments. The test set, being derived from the same overall distribution as training data, benefits from these training-derived scaling parameters and will yield overly optimistic results.

Thirdly, the optimization process during training can contribute. An LSTM network can become trapped in a local minima of the loss function, especially when dealing with complex time series. This locally optimal model, though appearing accurate on the test set, may not generalize well. The inherent bias embedded in the specific training data, coupled with the optimization process, causes the model to develop shortcuts and to overfit. The test set, closely related to the training data, may not reveal these shortcuts. This highlights the crucial importance of a proper validation set and model regularization.

Fourthly, the evaluation metric itself may offer a misleading perception of performance. Metrics such as Mean Squared Error (MSE) or Root Mean Squared Error (RMSE) can be skewed by outliers. Even if a model performs poorly on most data, a few extremely accurate predictions can artificially lower the overall error rate. This can happen even if the model has not grasped the underlying patterns and can not apply them to unseen data. The evaluation metrics themselves can be overly optimistic.

Here are three code examples illustrating this problem and its potential solutions:

**Example 1: Basic LSTM Model with Data Leakage**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Generate synthetic time series data
np.random.seed(42)
data = np.sin(np.linspace(0, 10 * np.pi, 300)).reshape(-1,1) + np.random.normal(0,0.2,300).reshape(-1,1)
# Data preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequential data for LSTM input
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
      x = data[i:(i + seq_length), 0]
      y = data[i + seq_length, 0]
      xs.append(x)
      ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 20
X, y = create_sequences(scaled_data, seq_length)

# Split into training and test sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=0)

# Evaluate on test data
mse_test = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MSE: {mse_test:.4f}") # Often quite low

# Simulate unseen data
unseen_data = np.sin(np.linspace(10 * np.pi, 15 * np.pi, 100)).reshape(-1,1)  + np.random.normal(0,0.3,100).reshape(-1,1)
scaled_unseen = scaler.transform(unseen_data)
X_unseen, y_unseen = create_sequences(scaled_unseen,seq_length)
X_unseen = np.reshape(X_unseen, (X_unseen.shape[0], X_unseen.shape[1], 1))
mse_unseen = model.evaluate(X_unseen, y_unseen, verbose=0)
print(f"Unseen Data MSE: {mse_unseen:.4f}") # Often significantly higher

```
In this example, we observe the data leakage because the `MinMaxScaler` is fitted on the whole dataset prior to train/test split. The model also experiences a reduced performance when predicting the ‘unseen data’ at the end.

**Example 2: Correct Scaling and Validation**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error

# Generate synthetic time series data
np.random.seed(42)
data = np.sin(np.linspace(0, 10 * np.pi, 300)).reshape(-1,1) + np.random.normal(0,0.2,300).reshape(-1,1)

# Split into train and test sets BEFORE SCALING
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Scale separately
scaler_train = MinMaxScaler()
scaled_train = scaler_train.fit_transform(train_data)

scaler_test = MinMaxScaler()
scaled_test = scaler_test.fit_transform(test_data)

# Create sequential data for LSTM input
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length), 0]
        y = data[i + seq_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 20
X_train, y_train = create_sequences(scaled_train, seq_length)
X_test, y_test = create_sequences(scaled_test, seq_length)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# Define the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=100, verbose=0)


# Evaluate on test data
y_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse_test:.4f}")

# Simulate unseen data
unseen_data = np.sin(np.linspace(10 * np.pi, 15 * np.pi, 100)).reshape(-1,1) + np.random.normal(0,0.3,100).reshape(-1,1)
scaler_unseen = MinMaxScaler()
scaled_unseen = scaler_unseen.fit_transform(unseen_data)
X_unseen, y_unseen = create_sequences(scaled_unseen,seq_length)
X_unseen = np.reshape(X_unseen, (X_unseen.shape[0], X_unseen.shape[1], 1))
y_pred_unseen = model.predict(X_unseen)
mse_unseen = mean_squared_error(y_unseen, y_pred_unseen)
print(f"Unseen Data MSE: {mse_unseen:.4f}")
```
Here, scaling is performed separately on the training set and test set to avoid data leakage.

**Example 3: Early Stopping and Regularization**

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error

# Generate synthetic time series data
np.random.seed(42)
data = np.sin(np.linspace(0, 10 * np.pi, 300)).reshape(-1,1) + np.random.normal(0,0.2,300).reshape(-1,1)

# Split into train and test sets BEFORE SCALING
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Scale separately
scaler_train = MinMaxScaler()
scaled_train = scaler_train.fit_transform(train_data)

scaler_test = MinMaxScaler()
scaled_test = scaler_test.fit_transform(test_data)

# Create sequential data for LSTM input
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length), 0]
        y = data[i + seq_length, 0]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 20
X_train, y_train = create_sequences(scaled_train, seq_length)
X_test, y_test = create_sequences(scaled_test, seq_length)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Split train into train and validation set
X_train, X_val, y_train, y_val = train_test_split(X_train,y_train, test_size = 0.2, random_state = 42)

# Define the LSTM model with regularization
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(seq_length, 1), return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Implement early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
model.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val), verbose=0, callbacks = [early_stopping])


# Evaluate on test data
y_pred = model.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse_test:.4f}")

# Simulate unseen data
unseen_data = np.sin(np.linspace(10 * np.pi, 15 * np.pi, 100)).reshape(-1,1) + np.random.normal(0,0.3,100).reshape(-1,1)
scaler_unseen = MinMaxScaler()
scaled_unseen = scaler_unseen.fit_transform(unseen_data)
X_unseen, y_unseen = create_sequences(scaled_unseen,seq_length)
X_unseen = np.reshape(X_unseen, (X_unseen.shape[0], X_unseen.shape[1], 1))
y_pred_unseen = model.predict(X_unseen)
mse_unseen = mean_squared_error(y_unseen, y_pred_unseen)
print(f"Unseen Data MSE: {mse_unseen:.4f}")
```
In this example, I've incorporated a validation set, early stopping, and dropout layers for regularization. These measures help mitigate overfitting. However, the performance of the unseen data may still be reduced compared with the test set, since the unseen data is inherently different from training data.

For deeper exploration, I recommend consulting resources focusing on time series analysis techniques. These include books on statistical forecasting, machine learning for time series, and practical guides on building robust LSTM models. Further knowledge may be gained from academic papers discussing the subtleties of recurrent neural networks applied to sequential data. Specifically, resources that discuss regularization strategies, proper cross-validation schemes for time series, and the perils of data leakage would be highly beneficial. Additionally, texts explaining techniques for handling non-stationary time series, particularly methods involving detrending and differencing, will be useful.
