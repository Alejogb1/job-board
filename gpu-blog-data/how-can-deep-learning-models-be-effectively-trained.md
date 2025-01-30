---
title: "How can deep learning models be effectively trained on time series data using cross-validation and feature/lag creation?"
date: "2025-01-30"
id: "how-can-deep-learning-models-be-effectively-trained"
---
Time series data presents unique challenges for deep learning, primarily due to its inherent temporal dependencies and the need to respect the order of observations during both training and evaluation. Conventional cross-validation, which randomly splits data, can introduce data leakage and provide an overly optimistic performance estimate. My experience developing predictive models for industrial equipment failure highlighted the critical need for adapted cross-validation techniques and proper feature engineering.

The core issue with applying standard cross-validation to time series stems from the temporal dependencies between data points. In time series, data at time ‘t’ is inherently related to data at times ‘t-1’, ‘t-2’, and so on. Randomly splitting the data into folds for cross-validation ignores this dependency. Imagine having training data from one period influencing your validation data from a preceding period. The model learns from the future to predict the past, which is unrealistic. This leads to models that look good in cross-validation but fail spectacularly on out-of-sample data.

To address this, we utilize techniques like “time series split” or "walk-forward validation.” The fundamental idea is to respect the temporal ordering of the data. Instead of random splits, we create folds such that each validation set occurs *after* its corresponding training set. Consider the simplest case: we use data from periods 1-N for training, then evaluate on period N+1. Subsequently, we use 1-N+1 for training and evaluate on period N+2, and so forth. This process mimics the real-world application where the model is trained on historical data to predict future outcomes. The specifics of how the training and validation sets are carved out depend on the use-case. For instance, I've used rolling windows, where each training window has a fixed size and shifts forward in time, and expanding windows, where training data increases with each iteration. Rolling windows work well if a limited past is most informative, while expanding windows suit scenarios where more history is always better.

In terms of deep learning architectures, Recurrent Neural Networks (RNNs), especially Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs), are commonly employed due to their ability to model sequential data. Convolutional Neural Networks (CNNs) can also be used, often with the addition of time-distributed layers, particularly when there are localized temporal patterns. Regardless of the specific model, feature engineering plays a critical role in successful time series forecasting.

Feature engineering in time series primarily involves creating "lagged" features – values of the target variable or other features from past time steps. These lags provide the model with access to the historical context that it would otherwise not have. In addition to lags, rolling statistical features, like moving averages and standard deviations, over different time windows are frequently useful. These summarise recent behavior and often capture temporal trends and volatility. Finally, features related to time, such as hour of day, day of week, or seasonality indicators can capture cyclicality in data. These additional features provide essential signals for the deep learning model.

Here are code examples illustrating this:

**Example 1: Time Series Split with Lag Creation (Python using scikit-learn and pandas)**

```python
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

def create_lags(df, target_col, n_lags):
    """Creates lagged features for a given column."""
    for lag in range(1, n_lags + 1):
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    return df

# Sample time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
data = {'value': np.random.rand(100)}
df = pd.DataFrame(data, index=dates)

# Create lagged features
n_lags = 3
df = create_lags(df, 'value', n_lags)
df = df.dropna() # Remove rows with NaNs from shifting.

# Time Series Split (3 splits)
tscv = TimeSeriesSplit(n_splits=3)
for train_index, test_index in tscv.split(df):
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]
    print("Train set shape:", train_df.shape, "Test set shape:", test_df.shape)
    # Perform training here on train_df, and testing on test_df
```

This code snippet first demonstrates lag creation using a custom function that shifts the target column to create lagged features. Following this, it shows how to use `TimeSeriesSplit` to split the data. Each split respects time order; test data always follows its corresponding training data. The output of the print statement demonstrates the shape of each data split.

**Example 2: Rolling Window Feature Creation and LSTM with Time Series Split (Python using pandas, TensorFlow/Keras)**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, InputLayer

def create_rolling_features(df, target_col, window_size):
    """Creates rolling average and std features."""
    df[f'{target_col}_rolling_mean'] = df[target_col].rolling(window_size).mean()
    df[f'{target_col}_rolling_std'] = df[target_col].rolling(window_size).std()
    return df

# Sample time series data (same as previous example)
dates = pd.date_range('2023-01-01', periods=100, freq='D')
data = {'value': np.random.rand(100)}
df = pd.DataFrame(data, index=dates)

#Create rolling window features
window_size = 7
df = create_rolling_features(df, 'value', window_size)
df = df.dropna() # Remove rows with NaNs from rolling features

# Reshape for LSTM input (assuming a single feature)
X = df[['value_rolling_mean','value_rolling_std']].values # use rolling features
y = df['value'].values
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Time Series Split (3 splits)
tscv = TimeSeriesSplit(n_splits=3)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # LSTM model definition (simplified)
    model = Sequential([
        InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=2, batch_size=16, verbose = 0) #Reduced epochs for clarity
    loss = model.evaluate(X_test, y_test, verbose = 0)
    print(f"Test MSE: {loss}")
```

This example demonstrates rolling feature creation, combining it with an LSTM model. It first calculates the rolling mean and standard deviation using a custom function with pandas functionality. The data is then reshaped to fit the LSTM's input requirements (samples, time steps, features). Then it proceeds to perform time series splitting and training a simplified LSTM model, showing a basic train and test loop.

**Example 3: Feature Creation with Multiple Lagged Variables and GRU with Time Series Split (Python using pandas, TensorFlow/Keras)**

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, InputLayer

def create_lags_multivar(df, feature_cols, n_lags):
    """Creates lagged features for multiple columns."""
    for col in feature_cols:
        for lag in range(1, n_lags + 1):
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df

# Sample Multivariate time series data
dates = pd.date_range('2023-01-01', periods=100, freq='D')
data = {'target': np.random.rand(100), 'feature1': np.random.rand(100), 'feature2': np.random.rand(100)}
df = pd.DataFrame(data, index=dates)


# Lagged features for multiple columns
n_lags = 2
feature_cols = ['feature1', 'feature2']
df = create_lags_multivar(df, feature_cols, n_lags)
df = df.dropna() # Drop any nan rows.

# Prepare data
X = df[['feature1_lag_1', 'feature1_lag_2', 'feature2_lag_1', 'feature2_lag_2']].values
y = df['target'].values
X = X.reshape((X.shape[0], 1, X.shape[1])) # Reshape to 3D tensor (samples, timesteps, features)

# Time Series Split (3 splits)
tscv = TimeSeriesSplit(n_splits=3)

for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # GRU model definition
    model = Sequential([
        InputLayer(input_shape=(X_train.shape[1], X_train.shape[2])),
        GRU(50),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=2, batch_size=16, verbose = 0) #Reduced epochs for clarity
    loss = model.evaluate(X_test, y_test, verbose = 0)
    print(f"Test MSE: {loss}")
```

This third example extends the previous ones by showing how to create lagged features for *multiple* input features. It uses a custom function to generate these lags. Following this, it demonstrates an implementation with a GRU (Gated Recurrent Unit) network, and the rest of the structure mirrors previous examples. The key difference here is that the input `X` contains lagged values for 'feature1' and 'feature2', demonstrating how to leverage multiple input features in a time series context.

For further exploration, I recommend focusing on resources that discuss time series analysis and forecasting in detail. Books on applied time series analysis often dedicate chapters to feature engineering and appropriate validation methods. In terms of deep learning, research papers focusing on RNNs, LSTMs, and GRUs within the context of time series provide crucial insights. Tutorials and case studies on time series with frameworks like TensorFlow and PyTorch are very helpful for implementing practical solutions. Finally, it’s beneficial to explore papers relating specifically to time series cross-validation methods and their effects on model generalization. A careful balance of theory and practical experimentation will significantly increase effectiveness when applying deep learning models to time series data.
