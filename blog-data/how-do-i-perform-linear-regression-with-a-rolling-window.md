---
title: "How do I perform linear regression with a rolling window?"
date: "2024-12-23"
id: "how-do-i-perform-linear-regression-with-a-rolling-window"
---

,  I've actually had to implement rolling window linear regression more times than I care to remember, usually in situations involving time-series data where the underlying relationships shift over time. There isn’t one magic bullet, but understanding the core concepts and variations is key to getting it working reliably.

Essentially, what we're aiming for is to apply linear regression repeatedly, but each time on a subset of the data defined by a sliding or rolling window. This allows us to capture localized trends rather than assuming a single, static relationship across the entire dataset. A standard linear regression assumes a constant relationship, but a time-series could have different relationships at different times.

The straightforward way is to iterate through your dataset, extract the data within the window, and compute the regression for that specific window. We're not talking about something overly complex, but the details of the implementation can impact performance.

Let’s consider a scenario I encountered working with high-frequency stock data back in my quant days. We needed to model the relationship between two assets, but that relationship wasn’t constant. It would vary depending on market conditions, intraday patterns, and a host of other factors. A global regression on the entire dataset was not giving us good predictions. This is where using a rolling window linear regression became essential. We achieved a much more sensitive prediction by doing so.

Here's how I’d approach it, breaking down the process with a Python example using `numpy` and `scikit-learn`, tools I’ve come to rely on over the years:

```python
import numpy as np
from sklearn.linear_model import LinearRegression

def rolling_window_linear_regression(x, y, window_size):
    """
    Performs linear regression with a rolling window.

    Args:
        x (np.array): Independent variable data.
        y (np.array): Dependent variable data.
        window_size (int): Size of the rolling window.

    Returns:
        list: A list of regression coefficients (slopes and intercepts)
              for each window. Returns an empty list if invalid input
              or data length is smaller than window size.
    """

    if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(window_size, int):
        return []

    if x.size == 0 or y.size == 0 or x.size != y.size or window_size <= 0 or x.size < window_size:
      return []


    coefficients = []
    for i in range(x.size - window_size + 1):
        x_window = x[i:i + window_size].reshape(-1, 1)
        y_window = y[i:i + window_size]

        model = LinearRegression()
        model.fit(x_window, y_window)
        coefficients.append((model.coef_[0], model.intercept_)) # Store slope and intercept

    return coefficients


# Example Usage:
x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_data = np.array([2, 4, 5, 4, 5, 7, 8, 9, 11, 12])
window = 3
results = rolling_window_linear_regression(x_data, y_data, window)

if results:
    for index, (slope, intercept) in enumerate(results):
      print(f"Window {index+1}: Slope = {slope:.2f}, Intercept = {intercept:.2f}")
else:
    print("Invalid input provided.")
```

This first code snippet is the most basic implementation. The key part is the loop where we extract `x_window` and `y_window` based on the window size and current iteration. The `reshape(-1, 1)` is crucial because `scikit-learn`'s linear regression expects the independent variable to be a 2D array, where the second dimension is the number of features. In this case, since there's only one predictor variable, the shape needs to be changed to `(n_samples, 1)`.

Now, let's consider another scenario, maybe in the context of sensor data processing. Imagine you have readings from a temperature sensor, and you’re correlating it with ambient light levels. You might find that their relationship changes during the day, so a simple rolling window approach is the correct way to analyze it. Often, we are not only interested in regression but also in prediction. Here's an example showing how to incorporate prediction using the calculated coefficients.

```python
import numpy as np
from sklearn.linear_model import LinearRegression

def rolling_window_regression_and_prediction(x, y, window_size, x_predict):
  """
    Performs linear regression with a rolling window and provides predictions
    for specified input x_predict.

    Args:
      x (np.array): Independent variable data.
      y (np.array): Dependent variable data.
      window_size (int): Size of the rolling window.
      x_predict (np.array): Independent variable values for prediction.

    Returns:
      list: A list of tuples, where each tuple is a list of predictions
            associated with the calculated linear regressions. Returns empty
            list if invalid input.
    """
  if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray) or not isinstance(x_predict, np.ndarray) or not isinstance(window_size, int):
    return []

  if x.size == 0 or y.size == 0 or x.size != y.size or window_size <= 0 or x.size < window_size or x_predict.size == 0:
    return []


  predictions = []
  for i in range(x.size - window_size + 1):
    x_window = x[i:i + window_size].reshape(-1, 1)
    y_window = y[i:i + window_size]

    model = LinearRegression()
    model.fit(x_window, y_window)

    current_predictions = model.predict(x_predict.reshape(-1,1))
    predictions.append(current_predictions)

  return predictions

# Example Usage:
x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_data = np.array([2, 4, 5, 4, 5, 7, 8, 9, 11, 12])
x_pred = np.array([12, 13, 14])
window = 3
results = rolling_window_regression_and_prediction(x_data, y_data, window, x_pred)

if results:
    for index, pred_set in enumerate(results):
        print(f"Window {index+1} Predictions: {pred_set}")
else:
  print("Invalid Input.")
```

In this example, after fitting the model on each window, we use the fitted model to predict values based on the provided `x_predict`. This illustrates how the rolling window approach is not just for finding the coefficients; it’s also to derive predictions that are contextualized within a specific time frame (window).

Finally, let's think about cases where we don't have equally spaced data, as is common in financial analysis and IoT data. Time stamps need to be considered, and we can incorporate a method of calculating window sizes based on time instead of sequence index.

```python
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def rolling_window_time_based_regression(data, x_col, y_col, time_col, window_duration):
    """
    Performs time-based rolling window linear regression.

    Args:
        data (pd.DataFrame): DataFrame with x, y, and time columns.
        x_col (str): Name of the independent variable column.
        y_col (str): Name of the dependent variable column.
        time_col (str): Name of the time column.
        window_duration (pd.Timedelta): Duration of the rolling window.

    Returns:
       list: A list of tuples, where each tuple contains the timestamp and the
       regression coefficients for the given time frame. Returns an empty list
       if invalid input.
    """
    if not isinstance(data, pd.DataFrame) or not isinstance(x_col, str) or not isinstance(y_col, str) or not isinstance(time_col, str) or not isinstance(window_duration, pd.Timedelta):
        return []

    if data.empty or x_col not in data.columns or y_col not in data.columns or time_col not in data.columns:
      return []

    if len(data[x_col]) < 2 or len(data[y_col]) < 2 or len(data[time_col]) < 2:
      return []

    # Ensure time is pandas datetime
    data[time_col] = pd.to_datetime(data[time_col])
    results = []

    for i in range(len(data)):
        start_time = data[time_col].iloc[i]
        end_time = start_time + window_duration

        window_data = data[(data[time_col] >= start_time) & (data[time_col] < end_time)]
        if len(window_data) < 2:
          continue
        x_window = window_data[x_col].values.reshape(-1, 1)
        y_window = window_data[y_col].values

        model = LinearRegression()
        model.fit(x_window, y_window)
        results.append((start_time, model.coef_[0], model.intercept_))

    return results


# Example Usage:
data = pd.DataFrame({
    'time': pd.to_datetime(['2024-01-01 00:00:00', '2024-01-01 00:15:00', '2024-01-01 00:30:00',
                           '2024-01-01 00:45:00', '2024-01-01 01:00:00', '2024-01-01 01:15:00']),
    'x': [1, 2, 3, 4, 5, 6],
    'y': [2, 4, 5, 4, 5, 7]
})
window_duration = pd.Timedelta(minutes=30)
results = rolling_window_time_based_regression(data, 'x', 'y', 'time', window_duration)


if results:
    for timestamp, slope, intercept in results:
        print(f"Time: {timestamp}, Slope: {slope:.2f}, Intercept: {intercept:.2f}")
else:
  print("Invalid input provided.")
```

This third example utilizes the Pandas library and explicitly uses time stamps.  We're now using a `pd.Timedelta` for window size definition, which gives us more flexibility compared to a fixed window size based on index. This is vital when dealing with irregularly sampled data.

For a deep dive into these topics, I'd recommend exploring these resources: *“Time Series Analysis”* by James D. Hamilton, a comprehensive reference that delves into the theoretical underpinnings of time series. For a practical approach using Python, “*Python for Data Analysis*” by Wes McKinney, the creator of Pandas, is also highly valuable. And finally, for regression specifically, *“The Elements of Statistical Learning*” by Hastie, Tibshirani, and Friedman covers the mathematical and practical components of regression extensively.

Remember to always validate results with appropriate metrics specific to your problem domain, and thoroughly test your implementations with realistic datasets, rather than synthetic datasets. These implementations I showed you are just starting points. You'll likely need to adjust these approaches to fit your own unique requirements.
