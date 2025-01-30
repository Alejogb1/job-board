---
title: "Why does linear regression fail to produce a line in time series forecasting?"
date: "2025-01-30"
id: "why-does-linear-regression-fail-to-produce-a"
---
Time series data inherently violates the fundamental assumption of independence required by ordinary linear regression, rendering it unsuitable for accurate forecasting. In my experience developing predictive models for high-frequency stock trading at Quantify Analytics, I routinely encountered this limitation firsthand. Specifically, linear regression, while excellent at modeling relationships between variables where each observation is considered independent, struggles to capture the temporal dependencies crucial to understanding sequential data. This stems from its core mechanism.

Linear regression seeks to model a dependent variable (the target) as a weighted sum of independent predictor variables, with a primary assumption that the error terms are uncorrelated. In a time series, each data point is inherently related to its preceding and subsequent values. The price of a stock at time *t* is not independent of its price at *t-1*. This autocorrelation, or dependence of a variable on its past values, violates this independence assumption, leading to biased and often unreliable results when applying linear regression directly. Linear regression models are trained to generalize to previously unseen *independent* data; in time series data, new observations are not independent, but highly correlated with past observations.

Furthermore, linear regression is a stationary model, meaning that it assumes that the relationships between variables remain constant over time. In time series, this is rarely the case. Trends, seasonality, and other non-stationary patterns introduce variations that a static linear model cannot accommodate. For example, a simple linear regression trained on historical sales data might fit well during one particular month but fail dramatically during a holiday season. The model treats these temporal changes as noise, rather than as meaningful patterns, resulting in significant forecast errors. Attempting to use features like 'time-of-day' or 'day-of-week' doesn't address the inherent dependency; these become fixed regressors and do not consider the dynamics of the series itself. A simple linear regression will, in effect, flatten out any non-stationary patterns it encounters.

Let's consider several code examples to illustrate this point. These examples are intentionally simple to focus on the core problem rather than complex time series libraries.

**Example 1: Simple Linear Regression on Autocorrelated Data**

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate autocorrelated data (a simple AR(1) process)
np.random.seed(42)
n_points = 100
phi = 0.8  # Autocorrelation coefficient
errors = np.random.normal(0, 0.2, n_points)
data = np.zeros(n_points)
data[0] = errors[0]
for i in range(1, n_points):
    data[i] = phi * data[i-1] + errors[i]

# Prepare the data for linear regression (using a lagged value)
X = data[:-1].reshape(-1, 1)  # Use previous value as predictor
y = data[1:]  # Values to predict

# Train linear regression
model = LinearRegression()
model.fit(X, y)

# Predict and evaluate (omitted here for brevity)
predicted = model.predict(X)

print(f"Regression coefficient: {model.coef_[0]:.2f}") #Outputting the slope coefficient
```

This code creates a time series exhibiting autocorrelation where the current value is partially dependent on the previous value (AR(1) process). We then attempt to predict each data point based solely on its immediately preceding data point using linear regression. The model *will* fit a line; this is the point. However, the model's performance is fundamentally limited by its inability to understand the temporal relationships beyond the single lag term. The regression coefficient will, unsurprisingly, be close to the autocorrelation coefficient *phi*, but the model doesn't learn the *process* that generated the data.  This is further illustrated if one were to plot the predicted values vs. the actual values - they would tend to follow the general trend, but with considerable deviations due to the model ignoring the inherent dynamics. The model simply calculates a slope and intercept that best fits the relationship between each point and the one previous. This is not a forecasting model, it is a model that relates consecutive data points.

**Example 2: Linear Regression with Time as a Feature**

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt #added for visualization

# Generate data with a trend and some noise
np.random.seed(42)
n_points = 100
time = np.arange(n_points)
trend = 0.5 * time
noise = np.random.normal(0, 5, n_points)
data = trend + noise

# Prepare the data for linear regression (time as a feature)
X = time.reshape(-1, 1)
y = data

# Train linear regression
model = LinearRegression()
model.fit(X, y)

#Predict and Plot
predicted = model.predict(X)

plt.scatter(X,y, label="Actual Values")
plt.plot(X, predicted, label="Linear Regression", color="red")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Linear Regression with Time Feature")
plt.show()

print(f"Regression coefficient: {model.coef_[0]:.2f}") #Outputting the slope coefficient
```

This example demonstrates a common, but ultimately inadequate, attempt to address time series data. Here, we create a data set with an upward linear trend, then introduce Gaussian noise. We subsequently attempt to fit a linear regression model using time as the sole predictor. The model *will* capture the trend, evidenced by a positive slope. However, the model doesn't incorporate any of the sequential information beyond time. It will make similar predictions regardless of what the immediate previous values were. If one were to zoom in on this trend, they would still see the data oscillating away from the trend due to noise. The model does not understand the *sequential* dependence of each point, and simply aims to minimize the residual error between actual values and the linear trend over time. The model ignores that values at time t-1, t-2, etc, are related to the value at time t. It is therefore a poor predictor of future values, where past data will become *future* data.

**Example 3: Linear Regression on Seasonal Data**

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt  #added for visualization

# Generate data with a seasonal pattern
np.random.seed(42)
n_points = 100
time = np.arange(n_points)
seasonal_component = 10*np.sin(2 * np.pi * time / 24)  #24 hour cycle
noise = np.random.normal(0, 2, n_points)
data = seasonal_component + noise

# Prepare the data for linear regression
X = time.reshape(-1, 1)
y = data

# Train linear regression
model = LinearRegression()
model.fit(X,y)

#Predict and Plot
predicted = model.predict(X)

plt.scatter(X,y, label="Actual Values")
plt.plot(X, predicted, label="Linear Regression", color="red")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Value")
plt.title("Linear Regression with Seasonal Data")
plt.show()


print(f"Regression coefficient: {model.coef_[0]:.2f}") #Outputting the slope coefficient

```

This final example introduces a seasonal component and Gaussian noise. Applying linear regression with time as the predictor leads to a relatively flat line, which essentially averages out the seasonal variations, resulting in a near-zero slope. The linear regression model, by design, attempts to fit the best linear trend and, in doing so, essentially ignores the dominant seasonal pattern and inherent correlations between data points across time. Again, the model makes no attempt to learn the sequential dynamics of the series; it seeks the single best line that describes the overall data, failing entirely to recognize the sinusoidal pattern present.  The lack of model fit and the regression coefficient's near-zero value demonstrates that linear regression completely ignores the underlying pattern. It will be unable to predict even one data point ahead.

In summary, the core problem lies in the violation of independence and stationarity assumptions. Linear regression simply doesn't possess the structure to understand that each data point in a time series is not just a value, but also has a relationship to its neighbors and follows some pattern over time. It is fundamentally a model for cross-sectional, rather than sequential, data. Its attempt to fit a line to data that follows its own dynamic will always lead to misrepresentation.

For those interested in further exploration, I would recommend researching the following topics: autoregressive models (AR), moving average models (MA), autoregressive moving average models (ARMA), autoregressive integrated moving average models (ARIMA), exponential smoothing techniques, and state-space models like Kalman filters. These methodologies explicitly model the temporal dependencies present in time series data, thus avoiding the shortcomings of linear regression in this context. Additionally, examining the concepts of stationarity, autocorrelation functions, and partial autocorrelation functions will deepen one's understanding of the problem domain. Understanding these more advanced techniques is crucial when addressing time series forecasting.
