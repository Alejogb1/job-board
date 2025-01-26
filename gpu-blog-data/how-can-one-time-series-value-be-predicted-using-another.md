---
title: "How can one time series value be predicted using another?"
date: "2025-01-26"
id: "how-can-one-time-series-value-be-predicted-using-another"
---

Predicting a target time series value using another related time series is a common task in many domains, from finance to environmental science. The core challenge lies in capturing the underlying temporal dependencies and correlations between the two series. I've personally encountered this problem numerous times while developing forecasting models for retail demand, where promotions (one time series) significantly influence sales (the other time series). The key to success is not just correlation, but understanding the *lagged* effects one series might have on another.  Direct, contemporaneous correlation can exist, but often the relationship isn’t so straightforward. The influence might manifest a few time steps later.

The most effective approach usually involves employing lagged features of the predictor time series within a predictive model. This concept introduces the idea that the value of the predictor series at some time *t-n* (where *n* is the lag) has an effect on the target series at time *t*. The selection of the appropriate lag values is critical, and varies greatly depending on the specific nature of the time series and the process under study. It's rare to find a single lag that works for all time points; usually, a combination of lags yields the best predictions. We are looking to capture a dynamic relationship, not a static one.

The predictive model itself can range from simple linear regression to more complex algorithms such as Recurrent Neural Networks (RNNs), particularly LSTMs (Long Short-Term Memory Networks) or GRUs (Gated Recurrent Units), or even traditional time series models like VAR (Vector Autoregression). The choice depends on the complexity of the relationship, available data, and computational resources. If the relationship is roughly linear, and the time series doesn't exhibit significant non-stationarity, linear regression with lagged features can be a surprisingly good starting point. For non-linear relationships or where long-term dependencies are prominent, deep learning models become more appropriate.

Let me illustrate with some simplified scenarios and code examples using Python, primarily leveraging the `pandas` and `scikit-learn` libraries for demonstration. The core will always involve creating those lagged features.

**Example 1: Linear Regression with a Single Lag**

Imagine we have a 'promotions' time series (0 or 1 indicating absence or presence of promotion, respectively) and a 'sales' time series. A simple hypothesis is that a promotion today will have an immediate impact on sales. This can be modeled using linear regression with a single, same-day lag.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample Data (replace with your actual data)
data = {'promotions': [0, 1, 0, 0, 1, 0, 1, 0],
        'sales': [100, 150, 110, 105, 160, 115, 170, 120]}
df = pd.DataFrame(data)

# Create lagged feature
df['promotions_lag1'] = df['promotions'].shift(1)
df = df.dropna() #remove the row with a null value created by the shift

# Prepare data
X = df[['promotions_lag1']]
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

In this example, we created a new feature named 'promotions_lag1' which holds the promotions value from the previous time step. We then used this as our sole predictor in a linear regression model. Note that while this approach captures immediate impact, it might be insufficient if the impact of promotions is delayed. This simple example serves to demonstrate the generation of the lagged feature.

**Example 2: Linear Regression with Multiple Lags**

Now let’s consider that promotions influence sales over several days. We can add multiple lagged features to reflect this prolonged effect.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample Data
data = {'promotions': [0, 1, 0, 0, 1, 0, 1, 0,1,1,0],
        'sales': [100, 120, 140, 120, 160, 150, 180, 140,190,200,150]}
df = pd.DataFrame(data)

# Create multiple lagged features
df['promotions_lag1'] = df['promotions'].shift(1)
df['promotions_lag2'] = df['promotions'].shift(2)
df = df.dropna()

# Prepare data
X = df[['promotions_lag1', 'promotions_lag2']]
y = df['sales']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

This expands upon the previous example by adding ‘promotions_lag2,’ representing the promotions two days prior. This provides more information to the linear regression model and potentially improves its ability to capture the dynamics of the relationship. It's a simple extension, but the concept is crucial; the art lies in selecting the correct lags based on the specific domain knowledge.

**Example 3: LSTM Network with Multiple Lags**

For time series displaying non-linearities or long-term dependencies, linear regression is often inadequate. Here's an example utilizing an LSTM network with multiple lagged features. Please note that this example provides a basic implementation and may require parameter tuning for real-world scenarios.

```python
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

# Sample Data
data = {'promotions': [0, 1, 0, 0, 1, 0, 1, 0,1,1,0,1,0,0,1,0,1,0,1,0],
        'sales': [100, 120, 140, 120, 160, 150, 180, 140,190,200,150,180,170,140,180,150,190,160,200,170]}
df = pd.DataFrame(data)

# Create multiple lagged features
df['promotions_lag1'] = df['promotions'].shift(1)
df['promotions_lag2'] = df['promotions'].shift(2)
df = df.dropna()

# Prepare data
X = df[['promotions_lag1', 'promotions_lag2']].values
y = df['sales'].values.reshape(-1, 1)


#Scale the data
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
X = scaler_x.fit_transform(X)
y = scaler_y.fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)


X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))



# Build the LSTM model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, activation='relu', input_shape=(1, X_train.shape[2])),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')


# Train the model
model.fit(X_train, y_train, epochs=100, verbose = 0) #reduce epochs as needed for speed


# Make predictions
y_pred = model.predict(X_test)

# Invert Scaling
y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)


# Evaluate
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
```

This example shows how to incorporate lagged features into a deep learning model. We reformatted the data to fit the LSTM’s requirements, created and trained the model, and then converted predictions back to their original scale using inverse transform. Note the inclusion of data scaling, which is beneficial when training neural networks.

**Resource Recommendations**

For a deeper understanding of time series analysis and prediction, I would recommend exploring texts on time series analysis, specifically those discussing autoregressive and moving average models, as well as more advanced methods like ARIMA and state-space models. Texts focused on deep learning with time series can provide the necessary insights to implement networks like LSTMs and GRUs. Additionally, familiarizing yourself with the documentation of Python libraries such as Pandas, Scikit-learn, and TensorFlow is essential for any practical implementation.  A strong statistical foundation is important in selecting and interpreting model results. The core concepts of stationarity, autocorrelation, and partial autocorrelation are all important to comprehend.
