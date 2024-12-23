---
title: "How can a random forest algorithm predict the next time step?"
date: "2024-12-23"
id: "how-can-a-random-forest-algorithm-predict-the-next-time-step"
---

,  Predicting future time steps using random forests is a task I've certainly encountered a few times, particularly in early projects involving predictive maintenance. It's a problem that initially feels like forcing a square peg into a round hole because random forests aren’t inherently designed for time-series forecasting. They're more classification and regression beasts, working best when data points are independent. However, with the correct feature engineering and a specific kind of setup, they can be quite effective, though they are not the optimal choice. So, here’s how we approach using them for this type of prediction, drawing on what I’ve learned from past iterations of building such systems.

The key is transforming our time-series data into a format that a random forest can actually use. We do this by essentially creating a supervised learning problem, where the target variable is the future time step we're trying to predict. We build feature vectors by extracting relevant information from previous time steps. In essence, we’re not directly feeding the time series into the forest, we're feeding in *lagged values* and other engineered features that reflect patterns in the time series. It's crucial to understand this shift in how we're framing the problem. We're no longer predicting “what happens next in a sequence”, but rather “predicting the next value based on a set of previous observed values”.

Think of it like this: rather than trying to guess the next number in a sequence like 1, 3, 5, 7, we are creating a dataset where one example would be: features (1, 3, 5), target variable (7). Then we teach the forest to find this mapping. This also highlights one major limitation, if the underlying process is not autoregressive (where current value depends on previous values), it simply won't work.

Now, let’s examine how we construct those feature vectors. We primarily use lagged variables. A lag of 1 simply means the value at t-1; lag 2 means t-2, and so on. Deciding how many lags to include is an art, partly guided by domain knowledge and partly through cross-validation. I've found that starting with a handful (maybe 3-5) and then adjusting from there, depending on the performance, often works quite well.

Beyond just lags, you might also consider including simple moving averages, rolling standard deviations, or even differences between lags. The more complex these extracted features become, the better the random forest is at capturing subtleties in the time series. However, it also increases the risk of overfitting, and we need to keep that at bay. This is where cross-validation and regularization strategies come in.

Here's a basic example in python using `scikit-learn` to demonstrate lagged feature construction and the prediction process:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def create_lagged_features(data, lags):
  df = pd.DataFrame(data)
  for lag in range(1, lags+1):
    df[f'lag_{lag}'] = df[0].shift(lag)
  df = df.dropna()
  return df

# Sample time series data
time_series = np.array([10, 12, 15, 13, 16, 18, 20, 22, 21, 23, 25, 27, 26])

# Create lagged features (lags of 3)
lags_to_use = 3
lagged_data = create_lagged_features(time_series, lags_to_use)
X = lagged_data.iloc[:, 1:].values
y = lagged_data.iloc[:, 0].values

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# Train a random forest regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluate
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root mean squared error: {rmse:.2f}")

# Prepare data for forecasting the next step
last_window = X[-1].reshape(1,-1) #last observation to forecast next step
next_prediction = rf_model.predict(last_window)[0]
print(f"Predicted next time step: {next_prediction:.2f}")
```
This snippet generates lagged features for a given time series data and then uses a random forest regressor to predict the next value.

Now, consider this scenario: you want to include not just simple lags, but also moving averages to smooth out the data, potentially capturing some longer-term trends. Here’s how you could adapt the feature engineering part of the code:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def create_advanced_features(data, lags, window_size):
    df = pd.DataFrame(data, columns=['value'])
    for lag in range(1, lags + 1):
        df[f'lag_{lag}'] = df['value'].shift(lag)
    df['moving_average'] = df['value'].rolling(window=window_size).mean()
    df = df.dropna()
    return df

# Sample time series data (same as before)
time_series = np.array([10, 12, 15, 13, 16, 18, 20, 22, 21, 23, 25, 27, 26])

# Create features with lags and moving averages
lags_to_use = 3
window_size = 3
lagged_data = create_advanced_features(time_series, lags_to_use, window_size)
X = lagged_data.iloc[:, 1:].values
y = lagged_data.iloc[:, 0].values

# Split the data, train the model, and make a prediction as previously done
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root mean squared error: {rmse:.2f}")
last_window = X[-1].reshape(1,-1)
next_prediction = rf_model.predict(last_window)[0]
print(f"Predicted next time step: {next_prediction:.2f}")
```
Here, the code now constructs a feature that calculates the moving average, which adds extra context to the random forest model.

Finally, let's consider a situation where we want to use a difference of lags, i.e. the difference between the current and lagged value. This helps in capturing changes in the signal, or in other words, its *derivative*. This can often be more useful than the raw signal depending on the properties of your data:

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def create_differential_features(data, lags):
    df = pd.DataFrame(data, columns=['value'])
    for lag in range(1, lags + 1):
      df[f'diff_lag_{lag}'] = df['value'] - df['value'].shift(lag)

    df = df.dropna()
    return df

# Sample time series data (same as before)
time_series = np.array([10, 12, 15, 13, 16, 18, 20, 22, 21, 23, 25, 27, 26])


# Create features with lag differences
lags_to_use = 3
lagged_data = create_differential_features(time_series, lags_to_use)
X = lagged_data.iloc[:, 1:].values
y = lagged_data.iloc[:, 0].values

# Split the data, train the model, and make a prediction as previously done
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred = rf_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root mean squared error: {rmse:.2f}")
last_window = X[-1].reshape(1,-1)
next_prediction = rf_model.predict(last_window)[0]
print(f"Predicted next time step: {next_prediction:.2f}")
```

In essence, you will see how to take the series data, convert it into lagged features by taking the original value and creating columns with shifted data (or other engineered features) and train a regressor on them, after which you can predict the future.

It's worth noting that random forests are inherently point predictors; they predict a single value at each time step. If you need interval predictions (confidence intervals), you would have to explore other techniques or use a method like quantile regression with a random forest. Also, for very long time series or those exhibiting complex non-linear behaviors, methods like recurrent neural networks, particularly those with long short-term memory (lstm) units, are typically superior. So, while random forests can work, it's crucial to keep their limitations in mind and select the right tool for the job. For those looking to delve deeper, I’d recommend starting with "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman, especially the chapter on regression trees. For more on time-series analysis in general, "Time Series Analysis" by James D. Hamilton is excellent. This combination will give you both the necessary theoretical understanding and practical tools to approach these types of problems.
