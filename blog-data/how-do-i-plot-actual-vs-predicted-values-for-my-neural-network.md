---
title: "How do I plot actual vs predicted values for my neural network?"
date: "2024-12-23"
id: "how-do-i-plot-actual-vs-predicted-values-for-my-neural-network"
---

Alright, let's dive into plotting actual versus predicted values for a neural network, a task I've tackled quite a few times over the years, both in academic settings and during my time at various tech firms. This isn't just about visualizing results; it's crucial for understanding model performance, identifying potential biases, and debugging training issues. We need to move beyond just loss values and get our hands dirty with the actual output data.

The process essentially boils down to taking your network’s predictions and juxtaposing them against the true, observed values from your dataset. This simple plot can illuminate a variety of issues: consistent over or under-prediction, areas of high variance, and the overall fit of your model to the data. It’s a fundamental diagnostic tool, much more informative than just aggregate statistics.

First things first, I’ve always found it helpful to structure my data appropriately for this process. Specifically, we need to ensure we have paired `y_true` (actual) and `y_pred` (predicted) values ready for plotting. Let's move to the practical part. Here is a code snippet using `matplotlib`, a staple in the data visualization toolbox:

```python
import matplotlib.pyplot as plt
import numpy as np

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted"):
    """
    Plots actual versus predicted values.

    Args:
        y_true (np.array): Array of actual values.
        y_pred (np.array): Array of predicted values.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.grid(True)
    # Add a reference line for perfect prediction
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
    plt.legend()
    plt.show()

# Example usage:
# Generate sample data. This would normally be the output of your model
y_true_sample = np.linspace(0, 10, 100)
y_pred_sample = y_true_sample + np.random.normal(0, 1, 100)  # Add some noise to simulate predictions
plot_actual_vs_predicted(y_true_sample, y_pred_sample, title="Example Plot")
```

This snippet does the following: it takes your `y_true` and `y_pred` arrays as input, creates a scatter plot, labels the axes, adds a title, includes grid lines, and adds a red dashed reference line. The reference line represents perfect prediction (where actual equals predicted) and is particularly useful for seeing how your predictions are distributed around this ideal. The addition of the legend makes the plot more readable. I usually generate an example dataset with a bit of random noise to make it more realistic for my use.

One thing I’ve observed is that when dealing with regression tasks, it's crucial to understand the scale of your target variable. It’s common to use standardization or normalization during model training, and you need to reverse this process before plotting your values. Let’s look at this in action. Suppose that your target variable was scaled using a standard scalar during pre-processing, you’d have to reverse this to be able to plot meaningful values against the actuals.

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

def inverse_transform_and_plot(scaler, y_true_scaled, y_pred_scaled, title="Transformed Actual vs Predicted"):
    """
    Inversely transforms scaled y_true and y_pred, then plots them.
    Args:
        scaler (StandardScaler or similar): Fitted scaler object.
        y_true_scaled (np.array): Scaled array of actual values.
        y_pred_scaled (np.array): Scaled array of predicted values.
        title (str): Title of the plot.
    """
    y_true_unscaled = scaler.inverse_transform(y_true_scaled.reshape(-1, 1))
    y_pred_unscaled = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    plot_actual_vs_predicted(y_true_unscaled.flatten(), y_pred_unscaled.flatten(), title=title)

#Example usage:
# Create sample data and apply a StandardScaler:
y_true_original = np.linspace(100, 500, 100)
scaler = StandardScaler()
y_true_scaled = scaler.fit_transform(y_true_original.reshape(-1, 1))
y_pred_scaled = y_true_scaled + np.random.normal(0, 0.2, 100)
inverse_transform_and_plot(scaler, y_true_scaled, y_pred_scaled, title="Unscaled Values Plot")
```

Here, we use the `StandardScaler` from scikit-learn to simulate pre-processing of our target variables. The important part is the `inverse_transform` method on the trained scaler instance, which we apply to both `y_true_scaled` and `y_pred_scaled` before plotting. This ensures that we’re visualizing the actual values we care about and not their scaled versions. Ignoring this step could lead to misinterpretations of your plot and thus your model's performance.

Let's now consider a more complex scenario. In time series forecasting, you often have a sequence of values, and you might want to plot actual versus predicted values over time. Here’s a snippet illustrating how that can be approached.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_timeseries_actual_vs_predicted(y_true, y_pred, time_index, title="Timeseries Actual vs Predicted"):
  """
  Plots actual versus predicted values over time.
  Args:
      y_true (np.array or pd.Series): Actual values.
      y_pred (np.array or pd.Series): Predicted values.
      time_index (np.array or pd.Series): Time index values.
      title (str): Title of the plot
  """
  plt.figure(figsize=(12, 6))
  plt.plot(time_index, y_true, label="Actual", marker='o', linestyle='-')
  plt.plot(time_index, y_pred, label="Predicted", marker='x', linestyle='--')
  plt.xlabel("Time")
  plt.ylabel("Value")
  plt.title(title)
  plt.legend()
  plt.grid(True)
  plt.show()

# Example usage:
time_index_sample = pd.date_range(start='2023-01-01', periods=100, freq='D')
y_true_sample_ts = np.sin(np.linspace(0, 10*np.pi, 100)) + 2
y_pred_sample_ts = y_true_sample_ts + np.random.normal(0, 0.3, 100)

plot_timeseries_actual_vs_predicted(y_true_sample_ts, y_pred_sample_ts, time_index_sample, "Time Series Example")
```
This snippet illustrates that we need to take care when plotting time series data. We use a date range for the x axis, and plot the actuals and predictions against this time index to accurately visualize model performance over time. This approach is invaluable when evaluating models trained on time-dependent data.

In my experience, for further reading and a more rigorous understanding, I highly recommend consulting "Pattern Recognition and Machine Learning" by Christopher Bishop for its deep dives into machine learning concepts and "Deep Learning" by Goodfellow, Bengio, and Courville for a comprehensive overview of neural networks. Additionally, “Hands-On Machine Learning with Scikit-Learn, Keras & Tensorflow” by Aurélien Géron is a practical, hands-on reference that would be beneficial.

These plots are more than just visual aids; they're tools for understanding. They provide a direct insight into how your model performs on individual instances of your dataset, and that’s information you simply cannot get from aggregate statistics alone. From detecting bias in prediction to debugging data pipeline issues, these visual inspections have often saved me hours of unnecessary investigation, so be sure to use them effectively.
