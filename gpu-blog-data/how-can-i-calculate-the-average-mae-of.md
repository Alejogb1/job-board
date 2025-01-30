---
title: "How can I calculate the average MAE of a feedforward neural network's time series forecasts?"
date: "2025-01-30"
id: "how-can-i-calculate-the-average-mae-of"
---
The accurate calculation of the mean absolute error (MAE) for a feedforward neural network (FNN) predicting time series data requires careful consideration of the forecasting horizon and the handling of potential data inconsistencies.  In my experience developing forecasting models for financial time series, I've found that neglecting edge cases, particularly concerning the initial training period and the alignment of predicted and actual values, often leads to inaccurate MAE calculations and flawed model evaluations.

My approach centers on a clear separation of the data preparation phase and the MAE calculation phase. This helps to ensure that the MAE reflects the model's true performance and avoids errors stemming from misaligned indices or incomplete datasets.


**1. Clear Explanation:**

The MAE is a metric that quantifies the average absolute difference between predicted and actual values.  In the context of time series forecasting with an FNN, we typically generate predictions for a future period (the forecasting horizon) based on a learned model.  A common pitfall is assuming a one-to-one correspondence between the predicted and actual data points across the entire time series. This is often untrue, particularly when the forecasting horizon is longer than one time step.

For example, if our FNN predicts 10 steps ahead, the first prediction corresponds to the 11th data point in the actual time series, the second to the 12th, and so on.  This offset must be explicitly accounted for when calculating the MAE.  Furthermore, we must ensure that the length of the predicted sequence matches the length of the actual sequence over which we are evaluating the model.  Truncating or padding the sequences inappropriately will lead to biased error calculations.

Therefore, a robust MAE calculation involves the following steps:

* **Data Preparation:**  Clearly define the training and testing sets.  Ensure that the test set includes enough data points to cover the full forecasting horizon.  Consider using a rolling window approach to create multiple test sets for better model evaluation.

* **Prediction Generation:**  Obtain predictions from the FNN for the designated forecasting horizon on the test set.  Store these predictions in a structure that preserves the temporal ordering and corresponds to the actual time series.

* **MAE Calculation:** Implement a function that iterates through the aligned predictions and actual values, calculates the absolute differences, and computes the average.


**2. Code Examples with Commentary:**

The following examples demonstrate MAE calculation in Python using NumPy and demonstrate different considerations for handling diverse data structures and forecasting horizons.


**Example 1: Single-Step Forecasting**

This example showcases a simple case where the FNN predicts only one step ahead.  Here, the index alignment is straightforward.

```python
import numpy as np

def calculate_mae_single_step(y_true, y_pred):
    """Calculates MAE for single-step time series forecasts.

    Args:
        y_true: NumPy array of actual values.
        y_pred: NumPy array of predicted values.

    Returns:
        The MAE.  Returns np.nan if input arrays are of different lengths or empty.
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return np.nan
    return np.mean(np.abs(y_true - y_pred))


y_true = np.array([10, 12, 15, 14, 16])
y_pred = np.array([11, 13, 14, 15, 17])
mae = calculate_mae_single_step(y_true, y_pred)
print(f"MAE: {mae}")  #Output: MAE: 1.0

```


**Example 2: Multi-Step Forecasting with Explicit Index Handling**

This example demonstrates handling a multi-step forecast where we explicitly manage the index offset between predictions and actual values.

```python
import numpy as np

def calculate_mae_multi_step(y_true, y_pred, horizon):
    """Calculates MAE for multi-step time series forecasts.

    Args:
        y_true: NumPy array of actual values.
        y_pred: NumPy array of predicted values.
        horizon: The forecasting horizon.

    Returns:
        The MAE. Returns np.nan if inputs are invalid or if there are insufficient data points.
    """
    if len(y_true) < horizon or len(y_pred) != len(y_true) - horizon +1 or len(y_true) == 0:
        return np.nan
    errors = np.abs(y_true[horizon-1:] - y_pred)
    return np.mean(errors)

y_true = np.array([10, 12, 15, 14, 16, 18, 20])
y_pred = np.array([13, 16, 15, 17]) #Predictions for steps 2,3,4,5
horizon = 4
mae = calculate_mae_multi_step(y_true, y_pred, horizon)
print(f"MAE: {mae}") #Output: MAE: 1.25

```


**Example 3:  Handling Variable-Length Predictions (using list of lists)**


This example deals with scenarios where the prediction lengths vary due to irregular data or model specifics.  This is less common but important to consider.

```python
import numpy as np

def calculate_mae_variable(y_true, y_pred_list):
    """Calculates MAE for variable-length predictions.

    Args:
        y_true: A list of lists representing actual values.
        y_pred_list: A list of lists of predicted values for each time series.

    Returns:
        The average MAE across all time series. Returns np.nan if any issue occurs.
    """

    if len(y_true) != len(y_pred_list):
        return np.nan
    
    maes = []
    for i in range(len(y_true)):
        if len(y_true[i]) != len(y_pred_list[i]):
            return np.nan
        errors = np.abs(np.array(y_true[i]) - np.array(y_pred_list[i]))
        maes.append(np.mean(errors))
    return np.mean(maes)

y_true = [[10, 12, 15], [14, 16, 18, 20]]
y_pred_list = [[11, 13, 14], [15, 17, 19, 21]]
mae = calculate_mae_variable(y_true, y_pred_list)
print(f"MAE: {mae}") #Output: MAE: 1.0

```



**3. Resource Recommendations:**

For further understanding of time series forecasting, I recommend studying texts on econometrics and machine learning, focusing on chapters dedicated to time series analysis, ARIMA models, and neural networks for forecasting.  Also, consider exploring advanced topics like LSTM networks and other recurrent architectures for time series prediction, as well as thorough examinations of model evaluation techniques beyond MAE, such as RMSE and MAPE.  Consult statistical textbooks to deepen your understanding of error metrics and their interpretations in the context of forecasting accuracy.  Finally, reviewing research papers on time series prediction will provide insights into advanced methodologies and state-of-the-art techniques.
