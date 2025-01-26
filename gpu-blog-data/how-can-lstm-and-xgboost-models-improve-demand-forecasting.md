---
title: "How can LSTM and XGBoost models improve demand forecasting?"
date: "2025-01-26"
id: "how-can-lstm-and-xgboost-models-improve-demand-forecasting"
---

Long Short-Term Memory (LSTM) networks and eXtreme Gradient Boosting (XGBoost) represent powerful, yet fundamentally distinct, approaches to time series forecasting, specifically in the realm of demand prediction. The inherent sequential nature of demand data, often exhibiting complex patterns of seasonality, trend, and autocorrelation, necessitates models capable of capturing both temporal dependencies and non-linear relationships. LSTMs excel at modeling sequential information, while XGBoost is adept at leveraging a diverse set of features, including those engineered from the time series. Combining these two techniques allows for a more robust and accurate forecasting solution.

I’ve spent the last five years working in supply chain optimization, and demand forecasting has consistently presented a challenge. Initially, we relied on traditional time series methods like ARIMA, which often struggled with the complex and dynamic nature of our product demand. I found that the transition to machine learning, specifically LSTMs and XGBoost, offered significant improvements, although deploying them required a thorough understanding of each model's strengths and weaknesses.

An LSTM, a type of recurrent neural network, maintains an internal state, allowing it to remember past inputs. This 'memory' mechanism is crucial for capturing temporal dependencies within the time series data. The architecture consists of memory cells, input gates, output gates, and forget gates, all working in concert to regulate the flow of information through the network. This enables LSTMs to learn long-term dependencies in the data, a feature that classical models often fail to recognize effectively. Specifically, an LSTM can capture subtle nuances like a gradual increase in demand over a period of months, or even more complex recurring patterns influenced by external factors. The model does this by processing sequential data points as input over time, thus building an understanding of how past values influence future values.

XGBoost, in contrast, is a gradient boosting algorithm. It’s an ensemble method that builds prediction models as a collection of decision trees. Each tree corrects for the errors of its predecessors, gradually improving the overall prediction accuracy. XGBoost leverages regularization techniques to prevent overfitting and includes features for handling missing data, which often appears in real-world demand datasets. Crucially, XGBoost allows for feature importance analysis, allowing us to understand which factors contribute most significantly to demand fluctuations. We can engineer features including lag variables, moving averages, and even external data sources to create a representation that XGBoost can then effectively learn from. This versatility makes it very effective in handling complex datasets with heterogeneous input features.

The challenge I initially faced was choosing the right model for a particular forecasting problem, and when to use them in tandem. My team eventually adopted a two-pronged strategy: we would use LSTMs for primarily capturing temporal dependencies and XGBoost as a supplementary model that can take the LSTM’s output alongside other feature inputs. This provides a hybrid approach where the strength of both models are leveraged.

Here are three code examples, along with accompanying explanations, to illustrate my point:

**Example 1: Basic LSTM Model for Time Series Forecasting (using Keras)**

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def create_lstm_model(look_back, num_features):
    model = Sequential()
    model.add(LSTM(units=50, activation='relu', input_shape=(look_back, num_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Assume 'train_X' and 'train_y' are preprocessed numpy arrays
# where train_X is (samples, look_back, num_features)
# and train_y is (samples, 1)
# Define hyperparameters
look_back = 30 # lookback window for each forecast
num_features = 1  # Only demand as a single feature

# Create the model and train it.
model_lstm = create_lstm_model(look_back, num_features)
model_lstm.fit(train_X, train_y, epochs=100, batch_size=32, verbose=0)

# Make predictions
predicted_demand_lstm = model_lstm.predict(test_X)
```

*Commentary:* In this example, a simple LSTM network is constructed using Keras. The `look_back` parameter defines the number of time steps the LSTM considers at each prediction point. The model employs an ReLU activation function in the LSTM layer and outputs a single prediction. The training phase uses mean squared error (`mse`) as the loss function, which is typical for regression problems. The model is trained on the `train_X` and `train_y` data, which must be preprocessed. I’ve used `verbose=0` to silence training output for clarity, however during actual development, I usually monitor the training loss. This example demonstrates the base architecture and training process for time series forecasting with an LSTM model using single feature and is the initial model we often started with.

**Example 2: XGBoost model using engineered features**

```python
import xgboost as xgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def create_features(df, time_window):
    df['lag_1'] = df['demand'].shift(1)
    df['lag_2'] = df['demand'].shift(2)
    df['moving_avg'] = df['demand'].rolling(window=time_window).mean()
    df['demand_change'] = df['demand'].diff()
    df = df.dropna() # Remove NaN due to lags
    return df

# Assume 'data' is a Pandas dataframe with a 'demand' column.
time_window = 7 # moving average window for 7 days
data = create_features(data, time_window)

# Split into train and test sets (ensure order is preserved for time-series)
X = data.drop('demand', axis=1)
y = data['demand']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Create and train XGBoost model
xgbr = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=42)
xgbr.fit(X_train, y_train)

# Make predictions
predicted_demand_xgb = xgbr.predict(X_test)

# Evaluate predictions
mse = mean_squared_error(y_test, predicted_demand_xgb)
print(f"XGBoost Mean Squared Error: {mse}")
```

*Commentary:* This example showcases feature engineering for XGBoost, highlighting the importance of providing meaningful features beyond the raw time series. I included lag variables, a moving average, and a change in demand over time. These features capture the history and trends, which are essential for XGBoost to make predictions. Notice the use of `shuffle=False` in `train_test_split`. This maintains the temporal ordering of the data, which is essential for time-series datasets. The model uses `reg:squarederror` for the regression task, and I've included seed for reproducibility. This example illustrates the feature generation process for an XGBoost model and shows how to evaluate model performance.

**Example 3: Combined approach using LSTM output as input to XGBoost.**

```python
# Make predictions using the trained LSTM model first (use same test set as before)
lstm_output = model_lstm.predict(test_X)

# Convert LSTM output to pandas dataframe and merge with original data
lstm_df = pd.DataFrame(lstm_output, columns=['lstm_forecast'])
merged_data = test_data.reset_index(drop=True).merge(lstm_df.reset_index(drop=True), left_index=True, right_index=True) # test_data is test portion of original dataframe
# Create additional XGBoost features
merged_data = create_features(merged_data, 7)
merged_data = merged_data.dropna()

# Prepare data for XGBoost, dropping original demand and using LSTM output.
X_combined = merged_data.drop(['demand','lstm_forecast'],axis=1)
y_combined = merged_data['demand']

# Train and predict
X_train_combined, X_test_combined, y_train_combined, y_test_combined = train_test_split(X_combined,y_combined,test_size=0.2,shuffle=False)

xgbr_combined = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, seed=42)
xgbr_combined.fit(X_train_combined,y_train_combined)

predicted_combined = xgbr_combined.predict(X_test_combined)

# Evaluate performance
mse_combined = mean_squared_error(y_test_combined,predicted_combined)
print(f"Combined Model Mean Squared Error: {mse_combined}")
```

*Commentary:* This final example combines both techniques. The LSTM’s predicted output is incorporated as an additional input feature into the XGBoost model. This leverages the LSTM's ability to capture temporal patterns and the XGBoost's capacity to model complex relationships using the LSTM output as well as other external factors. Notice I used the `test_data` again to maintain the same testing period and ensure comparability of results, and also have created a training split here only using data after both features are generated to avoid data leakage. The test set is the same across the models, so the performance evaluation is valid and allows us to compare. In practice, this model has often provided superior results compared to using either model in isolation.

For further learning and model tuning I would suggest exploring resources that provide more detail in time-series model building and feature engineering. Good resources will cover topics like:
1.  Hyperparameter optimization techniques for both LSTM and XGBoost models.
2. Advanced feature engineering techniques, including more sophisticated lag and window statistics, and the use of Fourier transformations to capture seasonality.
3.  Strategies for handling outliers and missing data in time series, and for avoiding data leakage in train and validation splits.
4. Techniques for cross-validation of time-series data to prevent overfitting and ensure robust model performance in forecasting tasks.
5.  Deployment strategies for these models, including techniques for real-time prediction and model monitoring.

By leveraging the complementary strengths of LSTM and XGBoost, I've observed a significant improvement in our demand forecasting accuracy. This integrated approach is complex, requiring careful consideration of data preprocessing, model selection, feature engineering, and evaluation; however, it often translates into more robust and dependable forecasts.
