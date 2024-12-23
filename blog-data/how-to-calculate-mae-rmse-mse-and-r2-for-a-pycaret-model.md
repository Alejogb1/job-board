---
title: "How to calculate MAE, RMSE, MSE, and R^2 for a PyCaret model?"
date: "2024-12-23"
id: "how-to-calculate-mae-rmse-mse-and-r2-for-a-pycaret-model"
---

Alright, let's tackle this. From my experience, evaluating model performance isn't just about hitting the 'fit' button and calling it a day. Understanding the nuances of different error metrics is crucial, especially when dealing with predictive models. You mentioned wanting to know how to calculate mean absolute error (mae), root mean squared error (rmse), mean squared error (mse), and r-squared (r²) specifically within the pycaret framework. I've seen this need surface quite a few times in production environments, so let me break it down, showing you some code snippets that should clarify the process, rather than relying just on pycaret's built-in functions which, while convenient, often hide the underlying mechanics.

Before diving into code, let's establish what each metric represents. **Mean squared error (mse)** gives us the average of the squares of the errors (the differences between predicted and actual values). It penalizes larger errors more heavily due to the squaring operation. **Root mean squared error (rmse)** is simply the square root of the mse, putting the error on the same scale as the target variable, which makes it easier to interpret. **Mean absolute error (mae)** provides the average magnitude of the errors, without considering their direction, making it robust to outliers. Finally, **r-squared (r²)** measures the proportion of the variance in the dependent variable that is predictable from the independent variables. It essentially tells us how well the model fits the data, but keep in mind its limitations when comparing models across differing datasets.

Now, about pycaret. While it offers a `evaluate_model` function and methods to pull out these metrics, sometimes you need a deeper dive, especially when debugging or customizing the evaluation. One common scenario I faced was a discrepancy in the metrics being reported by different segments of a large dataset, and it required us to calculate these metrics manually for various subsets to pinpoint the issue. So, let's not just rely on `evaluate_model`; let's do this from first principles.

Here’s how you can calculate these using numpy and scikit-learn after you have trained a pycaret model. This assumes you've already setup your pycaret environment and have a trained model, which, for demonstration purposes, we will refer to as `best_model`.

First, let's establish the data. We need both the predicted values and the actual target values. Pycaret makes this straightforward:

```python
import numpy as np
import pandas as pd
from pycaret.regression import *
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Assuming setup has been run and 'best_model' has been created.
# For the sake of this example, let's simulate the training process with some dummy data
data = pd.DataFrame({'feature1': np.random.rand(100), 'feature2': np.random.rand(100), 'target': np.random.rand(100)})
s = setup(data, target = 'target', verbose = False)
best_model = compare_models(sort = 'rmse', verbose = False, n_select=1)

predictions = predict_model(best_model)
predicted_values = predictions['prediction_label'].values
actual_values = predictions['target'].values
```

Now that we have our predicted and actual values, let's write functions to calculate our metrics:

```python
def calculate_mse(actual, predicted):
  """Calculates mean squared error."""
  return np.mean((np.array(predicted) - np.array(actual)) ** 2)

def calculate_rmse(actual, predicted):
  """Calculates root mean squared error."""
  return np.sqrt(calculate_mse(actual, predicted))

def calculate_mae(actual, predicted):
  """Calculates mean absolute error."""
  return np.mean(np.abs(np.array(predicted) - np.array(actual)))

def calculate_r2(actual, predicted):
  """Calculates R-squared."""
  ss_res = np.sum((np.array(actual) - np.array(predicted)) ** 2)
  ss_tot = np.sum((np.array(actual) - np.mean(actual)) ** 2)
  return 1 - (ss_res / ss_tot)

# Calculate using custom functions:
mse = calculate_mse(actual_values, predicted_values)
rmse = calculate_rmse(actual_values, predicted_values)
mae = calculate_mae(actual_values, predicted_values)
r2 = calculate_r2(actual_values, predicted_values)

print(f"Custom MSE: {mse:.4f}")
print(f"Custom RMSE: {rmse:.4f}")
print(f"Custom MAE: {mae:.4f}")
print(f"Custom R²: {r2:.4f}")
```
This gives you complete control and the ability to investigate the computation step by step if needed.

Furthermore, you could compare these calculated values against those provided by sklearn’s metrics library, as an additional level of validation:

```python
# Calculate using sklearn's functions:
sklearn_mse = mean_squared_error(actual_values, predicted_values)
sklearn_rmse = np.sqrt(sklearn_mse) #sklearn doesn't have an explicit rmse metric, so we take the square root of mse
sklearn_mae = mean_absolute_error(actual_values, predicted_values)
sklearn_r2 = r2_score(actual_values, predicted_values)

print(f"Sklearn MSE: {sklearn_mse:.4f}")
print(f"Sklearn RMSE: {sklearn_rmse:.4f}")
print(f"Sklearn MAE: {sklearn_mae:.4f}")
print(f"Sklearn R²: {sklearn_r2:.4f}")
```

In this snippet, I'm leveraging `mean_squared_error`, `mean_absolute_error`, and `r2_score` from scikit-learn. As you can see, the custom and sklearn values should be consistent, reinforcing the correct implementation of the error metrics.

Now, regarding resources, I highly recommend diving into *“The Elements of Statistical Learning”* by Hastie, Tibshirani, and Friedman. It offers a rigorous theoretical grounding on these concepts, among many others. Also, *“Pattern Recognition and Machine Learning”* by Christopher Bishop is an excellent companion for understanding the statistical foundations of these metrics. For a more practical perspective, and not specific to pycaret, “*Applied Predictive Modeling*” by Kuhn and Johnson presents a comprehensive approach to model evaluation that goes beyond just these metrics.

Understanding these metrics and how to calculate them, not just through high-level functions, gives you a powerful tool for debugging and improving the performance of your models. I find that having a grasp on both the theoretical and practical aspects allows me to navigate complex scenarios with more confidence. As you build more complex systems, you'll find it's not just about the libraries, but truly understanding what’s happening under the hood.
