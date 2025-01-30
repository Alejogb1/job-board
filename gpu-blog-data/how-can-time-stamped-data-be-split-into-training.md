---
title: "How can time-stamped data be split into training and testing sets?"
date: "2025-01-30"
id: "how-can-time-stamped-data-be-split-into-training"
---
Time-stamped data, unlike randomly sampled data, exhibits inherent temporal dependencies which must be respected when creating training and testing splits. Disregarding this dependency leads to inflated performance metrics during evaluation and poor generalization of the model in real-world scenarios. The core challenge lies in preventing data leakage from future time periods into the training set.

Specifically, the goal isn't to create splits with uniformly distributed data points across time. Rather, it's to simulate how the model would behave in a production environment where it encounters unseen data from the future. Consequently, the training set should consist of historical data, while the testing set should consist of more recent, chronologically subsequent data. Three primary techniques facilitate this chronological separation: simple chronological splitting, rolling window expansion, and gap-based splitting, each applicable under different circumstances.

**1. Simple Chronological Split**

The most straightforward approach involves dividing the data at a single point in time. All records preceding that point are placed into the training set; all records following it are placed into the test set. This method assumes a clear demarcation between historical and future behavior.

```python
import pandas as pd

def simple_split(data, split_date):
    """Splits a time-series dataframe into training and testing sets based on a date.

    Args:
        data (pd.DataFrame): DataFrame with a 'timestamp' column.
        split_date (str): Date string used to divide the dataset (YYYY-MM-DD).

    Returns:
        tuple: (training DataFrame, testing DataFrame).
    """
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    train_data = data[data['timestamp'] < split_date]
    test_data = data[data['timestamp'] >= split_date]
    return train_data, test_data

# Example Usage:
data = pd.DataFrame({
    'timestamp': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-15', '2023-01-20', '2023-01-25']),
    'value': [10, 12, 15, 13, 18, 20]
})

train_set, test_set = simple_split(data, '2023-01-15')
print("Training Set:\n", train_set)
print("\nTesting Set:\n", test_set)
```
This function `simple_split` first converts the 'timestamp' column to the datetime data type. It then performs a direct split based on the specified `split_date`. Any data before this date is included in the training set, while data on or after is placed in the testing set.  This method is efficient for datasets with a single, distinct temporal cutoff. Its primary limitation arises when the data exhibits significant non-stationarity—changes in the statistical properties of data over time—which the model needs to adjust to. In this case, the simple split may cause the model to be biased due to a lack of more recent training data.

**2. Rolling Window Expansion**

To mitigate issues with non-stationarity and to provide the model with a more current perspective, we can use rolling window techniques for training. This means incrementing the size of the training set while maintaining a fixed, non-overlapping testing window. This approach simulates the iterative nature of model deployment, where the model is progressively re-trained with increasing amounts of data.

```python
import pandas as pd

def rolling_window_split(data, window_size, step_size):
    """Splits a time-series dataframe into training and testing sets using a rolling window.

    Args:
        data (pd.DataFrame): DataFrame with a 'timestamp' column.
        window_size (int): The number of data points to use in the training set.
        step_size (int): How many data points to advance the window for each split.

    Yields:
        tuple: (training DataFrame, testing DataFrame) for each iteration.
    """
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    data = data.sort_values(by='timestamp').reset_index(drop=True)
    num_samples = len(data)
    start = 0
    while start + window_size < num_samples:
        train_end = start + window_size
        test_end = min(start + window_size + step_size, num_samples)
        train_data = data.iloc[start:train_end]
        test_data = data.iloc[train_end:test_end]
        yield train_data, test_data
        start += step_size


# Example usage:
data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-15', '2023-01-20', '2023-01-25', '2023-01-30', '2023-02-05', '2023-02-10', '2023-02-15']),
        'value': [10, 12, 15, 13, 18, 20, 22, 25, 28, 30]
    })

window_size = 5
step_size = 2

for i, (train, test) in enumerate(rolling_window_split(data, window_size, step_size)):
    print(f"Split {i+1}:")
    print("Training Set:\n", train)
    print("Testing Set:\n", test, "\n")
```

The `rolling_window_split` function incorporates a window of fixed size (`window_size`) for training data and then advances this window by a step size (`step_size`), creating distinct training-testing pairs with expanding training sets.  The test set always follows the training set in time.  This approach effectively simulates scenarios where the model is continually retrained on new data. The advantage is the inclusion of more recent data as the training window shifts, which allows the model to adapt to changing data distributions. The choice of `window_size` and `step_size` is determined by data characteristics and computational constraints; smaller steps lead to more robust performance evaluation at the cost of additional computation.

**3. Gap-Based Split**

In certain applications, a period of time needs to be explicitly excluded from training and testing.  This gap is introduced to simulate situations where the data immediately following the training period might be unavailable or unreliable.  This gap-based approach is critical in scenarios where future data is anticipated to be significantly different from historical trends and where immediate predictions after the training period would be too speculative.

```python
import pandas as pd
import numpy as np

def gap_split(data, train_end, gap_duration, test_duration):
    """Splits a time-series dataframe into training and testing sets with a gap.

    Args:
        data (pd.DataFrame): DataFrame with a 'timestamp' column.
        train_end (str): The last date of training data (YYYY-MM-DD).
        gap_duration (str): Duration of the gap between training and testing (e.g., '5D', '1M').
        test_duration (str): Duration of the testing set (e.g., '10D', '2M').

    Returns:
        tuple: (training DataFrame, testing DataFrame).
    """
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    train_end_date = pd.to_datetime(train_end)
    gap_duration_delta = pd.Timedelta(gap_duration)
    test_duration_delta = pd.Timedelta(test_duration)

    test_start = train_end_date + gap_duration_delta
    test_end = test_start + test_duration_delta

    train_data = data[data['timestamp'] <= train_end_date]
    test_data = data[(data['timestamp'] >= test_start) & (data['timestamp'] < test_end)]

    return train_data, test_data

# Example usage:
data = pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-05', '2023-01-10', '2023-01-15', '2023-01-20', '2023-01-25', '2023-01-30', '2023-02-05', '2023-02-10', '2023-02-15', '2023-02-20']),
        'value': [10, 12, 15, 13, 18, 20, 22, 25, 28, 30, 32]
    })
train_end_date = '2023-01-25'
gap_duration = '5D'
test_duration = '10D'

train_set, test_set = gap_split(data, train_end_date, gap_duration, test_duration)
print("Training Set:\n", train_set)
print("\nTesting Set:\n", test_set)
```

The `gap_split` function explicitly calculates the training window's end, adds a specified gap duration to determine the start of the testing window and specifies the total test duration.  Data before and up to the `train_end` are used for training, while those occurring within the determined `test_start` and `test_end` are allocated for testing. This method is necessary when data transitions need to be explicitly considered or when a delay in data availability is part of the simulation.

Implementing these temporal splitting techniques is essential for evaluating the real-world performance of predictive models using time-stamped data. The simple chronological split serves as a baseline for the initial performance of models. Rolling window expansion is crucial when models need to be adaptable to evolving data characteristics. Gap-based splitting provides a simulation more congruent with situations where a time window needs to be left out of the model's learning or evaluation. Selecting the appropriate splitting technique depends on the characteristics of your data and the objectives of the model.

For more detailed information on time series analysis and splitting techniques, refer to statistical textbooks and online courses focused on time-series data. The documentation of libraries specializing in time series forecasting often provides additional guidelines on these techniques. Publications focused on machine learning for forecasting can also offer theoretical insights into the different splitting approaches.
