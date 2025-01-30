---
title: "How can ARIMA forecasts be constrained to an interval in Python?"
date: "2025-01-30"
id: "how-can-arima-forecasts-be-constrained-to-an"
---
The inherent nature of ARIMA models often leads to forecasts that, while statistically sound, can sometimes violate real-world constraints. A common challenge is ensuring that forecasts remain within a reasonable interval, preventing illogical predictions like negative sales figures or physically impossible outputs. I've encountered this issue frequently in predictive maintenance scenarios where a failure rate can never be less than zero, and during demand forecasting where supply chain capacity establishes a clear upper bound.

The standard ARIMA implementation in packages like `statsmodels` doesn't inherently offer interval constraints. To achieve this, we must modify the forecasting process, typically by post-processing the generated predictions. This involves two main approaches: truncation and transformation. Truncation is simpler; any forecast exceeding the upper limit is set to that limit, and likewise, any forecast below the lower limit is set to the lower bound. Transformation, on the other hand, aims to modify the data or model in a way that inherently leads to constrained forecasts. While transformation often provides more robust results, it requires a deeper understanding of the data and model properties and can be more computationally expensive.

Truncation provides a quick fix, particularly when constraints are narrow, and the ARIMA model’s raw forecasts tend to adhere to the given bounds except for a few outliers. It's a pragmatic approach that ensures feasibility of the results, especially in cases when accuracy outside of the bounds isn’t the primary concern. Let's explore a Python implementation:

```python
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def constrained_arima_forecast(data, order, lower_bound, upper_bound, steps):
    """
    Forecasts using an ARIMA model and truncates to ensure forecasts stay within bounds.
    """
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)

    # Truncate the forecast
    truncated_forecast = forecast.clip(lower=lower_bound, upper=upper_bound)

    return truncated_forecast

# Example Usage:
data = pd.Series([25, 28, 32, 35, 39, 42, 46, 49, 52, 55])
order = (1, 1, 1) # Example order: (p, d, q)
lower_bound = 20
upper_bound = 60
steps = 5
constrained_forecast = constrained_arima_forecast(data, order, lower_bound, upper_bound, steps)
print("Truncated Forecast:", constrained_forecast)
```
This first example demonstrates a basic implementation. The function `constrained_arima_forecast` takes the time series data, ARIMA order, lower and upper bounds, and number of forecast steps. We use `statsmodels.tsa.arima.model.ARIMA` for model fitting and forecasting. The key operation is `forecast.clip(lower=lower_bound, upper=upper_bound)`, which ensures that each element of the forecast is within the specified bounds. This method is simple and computationally cheap, but it abruptly adjusts the forecast, potentially distorting the predicted trends when the model generates forecasts that are frequently out of bounds.

A more refined approach involves data transformations. The Box-Cox transformation, specifically, can be useful when the data exhibits non-constant variance and when a hard lower bound of zero is required. This method can transform the data to ensure the forecasts never go below zero in the original scale. However, if both a lower *and* upper bound are needed in the original scale, this method becomes more complex, requiring an inverse transformation that is sensitive to the specific parameter of the Box-Cox method. For simplicity, we will stick to a Log transformation (a special case of Box-Cox) in the following example, which ensures positivity, particularly useful if the data itself doesn't have a zero value:

```python
import numpy as np

def log_transform_constrained_arima(data, order, upper_bound, steps):
    """
    Forecasts using ARIMA on log-transformed data, then transforms back and clips at the upper bound.
    Note: The lower bound will always be positive in the original scale with this log transform
    """
    # Transform data (ensure data is positive)
    log_data = np.log(data)

    model = ARIMA(log_data, order=order)
    model_fit = model.fit()
    log_forecast = model_fit.forecast(steps=steps)

    # Inverse transform forecast and clip
    forecast = np.exp(log_forecast)
    truncated_forecast = forecast.clip(upper=upper_bound)

    return truncated_forecast


# Example Usage:
data = pd.Series([25, 28, 32, 35, 39, 42, 46, 49, 52, 55])
order = (1, 1, 1)
upper_bound = 60
steps = 5

transformed_forecast = log_transform_constrained_arima(data, order, upper_bound, steps)
print("Log Transformed Forecast:", transformed_forecast)
```

In the second example, we apply a logarithmic transformation to the data using `np.log()`. We then fit the ARIMA model using the transformed data. The forecasted values are obtained, exponentiated to reverse the log transform (using `np.exp()`), and finally, clipped to ensure the forecast remains below the `upper_bound`. It's important that the raw input data should be strictly positive for the log transformation to work. If the data might include zero or negative values, a more complex transform like `log(data + 1)` should be applied before the fitting process. Using transformations can lead to forecasts that adhere better to realistic constrains, particularly the positive constraint, but may make forecasting less accurate than non-transformed data, requiring careful model selection and parameter tuning.

Implementing an approach that is not just truncation or simple log transformation, but directly uses a constrained optimization algorithm while fitting the ARIMA model could lead to more robust results. Packages like `scipy` offer constrained optimization methods that could potentially integrate with the ARIMA modeling process. However, this approach can lead to complex custom implementations and computational expenses. As such, this goes beyond what can be explained in a simple example, but here's the conceptual idea in the last example by implementing constraints in the optimization process.

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize

def constrained_arima_optimization(data, order, lower_bound, upper_bound, steps):
    """
    Uses constrained optimization to fit the ARIMA and returns forecast.
    *Note: This demonstrates an alternative concept; the implementation in practice requires careful integration
    with statsmodels parameter estimation.*
    """
    # Dummy function to simulate ARIMA's objective function, needs correct parameter mapping
    def objective_function(params, data):
        # Simulate ARIMA forecast for demonstration.
        # Actual Objective will be log-likelihood
        dummy_model = ARIMA(data, order = order) # Need to find a way to set the parameter
        dummy_fit = dummy_model.fit()
        forecast = dummy_fit.forecast(steps = steps)
        return np.sum(np.square(forecast)) # Minimizes the sum of squared error as an example objective
    
    # Define bounds for optimization, for each parameter in the ARIMA model
    # This will need careful extraction from the model
    initial_params = np.ones(order[0] + order[2] + 1 ) * 0.5 # Placeholder for real parameter vector
    bounds = [(None, None)]* len(initial_params)  # Example: no bound on AR and MA parameters; need to set correct one
    
    # Define constraints for forecast values using SciPy
    def constraint(params, data):
        # Simulate ARIMA forecast for demonstration.
        # Actual constraint calculation requires full model fitting with params
        dummy_model = ARIMA(data, order = order) # Need to find a way to set the parameter
        dummy_fit = dummy_model.fit()
        forecast = dummy_fit.forecast(steps = steps)
        return np.array([np.min(forecast) - lower_bound, upper_bound - np.max(forecast)])

    
    # Fit the constrained model by optimizing the objective with the defined bounds and constraints
    res = minimize(objective_function, initial_params, args = (data,), method='trust-constr',
        bounds=bounds, constraints = ({'type': 'ineq', 'fun': constraint, 'args': (data,) }))

    # Extract the parameters, retrain and return forecast
    dummy_model = ARIMA(data, order = order) # Need to find a way to set the parameter
    dummy_fit = dummy_model.fit()
    forecast = dummy_fit.forecast(steps = steps) # This is NOT the forecast constrained by the optimization. 
    # A more complex re-implementation is required to pass the parameters.
    return forecast

# Example Usage:
data = pd.Series([25, 28, 32, 35, 39, 42, 46, 49, 52, 55])
order = (1, 1, 1)
lower_bound = 20
upper_bound = 60
steps = 5
constrained_forecast_optimized = constrained_arima_optimization(data, order, lower_bound, upper_bound, steps)
print("Optimized Constrained Forecast:", constrained_forecast_optimized)

```
This third example showcases the *concept* of implementing constraints through optimization. It uses `scipy.optimize.minimize` with the `trust-constr` method, which allows for incorporating both bounds and constraints during the optimization process. However, a crucial caveat is that the example uses a *dummy* function for both the objective and the constraint since directly interfacing the optimization process with statsmodel's ARIMA parameter estimation is complex and out of the scope for a simple illustration. A real implementation requires a detailed understanding of the ARIMA likelihood function, parameter extraction, and how it integrates with `scipy.optimize`. This method can provide the most robust approach to ensure the predictions meet constraints, however, it is the most complex to implement and computationally expensive.

For further study, I recommend focusing on the *statsmodels* documentation for in-depth understanding of ARIMA model parameters and fitting procedures. A deep dive into *scipy.optimize* will be valuable when implementing constrained optimization. Furthermore, books on time series analysis that cover Box-Cox transformations in detail would offer a theoretical foundation for informed decision-making. Reviewing case studies on time series forecasting where such constraints are imposed could provide practical insight. Remember, the choice of method is not just a matter of technique, but also a balance between performance, accuracy, and the specific characteristics of your data and the nature of your constraints.
