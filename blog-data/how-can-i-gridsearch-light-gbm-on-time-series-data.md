---
title: "How can I gridsearch Light GBM on time series data?"
date: "2024-12-23"
id: "how-can-i-gridsearch-light-gbm-on-time-series-data"
---

Alright, let’s tackle this. Gridding hyperparameters for LightGBM on time series data is definitely a challenge that requires a somewhat nuanced approach compared to typical classification or regression tasks. Over the years, I've seen plenty of folks, myself included initially, fall into traps that lead to suboptimal models. The core issue, as you're probably sensing, is the temporal dependence inherent in time series. Blindly splitting data without considering the time axis will almost certainly lead to overly optimistic performance metrics during cross-validation. I've personally experienced this in a past project predicting retail sales where a carelessly created gridsearch gave me wonderful results on paper, but promptly failed in a production environment, serving as a harsh lesson in avoiding temporal leakage.

So, what's the proper way to gridsearch LightGBM (or any model for that matter) with time series data? The short answer is to use a time-aware cross-validation technique. Instead of random splits, we need to split the data sequentially, preserving the temporal order. Think of it as creating validation sets that are always *in the future* relative to the training sets. This simulates the real-world scenario where you're forecasting forward, not evaluating on data from the past.

There are several cross-validation strategies suitable for time series, but a fairly common and robust one is a rolling window approach. We'll build this into the grid search. The core idea behind rolling window is that we select a fixed size window to be the training set and a following window to be the validation set, and then repeat this process by moving each window forward to cover the entire time series data.

Here's a breakdown of how to structure that for lightgbm, using python's `scikit-learn` and `lightgbm` packages:

**1.  Time-Aware Splitting Function:**

First, we need a function that generates the indices for our time-aware training/validation splits. Here’s a basic implementation I've used before, using the rolling window approach:

```python
import numpy as np

def time_based_cross_validation(data_len, window_size, step_size):
    """
    Generates time-based indices for training and validation sets using a rolling window.

    Args:
        data_len (int): Total length of the time series data.
        window_size (int): The size of the training set window.
        step_size (int): The size of the validation window and the amount to shift forward on each iteration.

    Yields:
       tuple: A tuple containing (train_indices, validation_indices).
    """
    for i in range(0, data_len - window_size - step_size + 1, step_size):
        train_indices = np.arange(i, i + window_size)
        val_indices = np.arange(i + window_size, i + window_size + step_size)
        yield (train_indices, val_indices)


```

This function `time_based_cross_validation` is a generator function, producing tuples with train and validation indices based on the given time series length, window size and step size. This function produces sequential training and validation sets, meaning the validation window is always after the training window. Using a step-size equal to the window size creates non-overlapping folds.

**2.  Grid Search Implementation:**

Next, we'll incorporate this into a grid search using LightGBM:

```python
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pandas as pd


def time_series_grid_search(data, features, target, params_grid, window_size, step_size):
    """
    Performs a time-series aware grid search on LightGBM.

    Args:
        data (pandas.DataFrame): Input dataframe, including features and target.
        features (list): A list of feature column names.
        target (str): The target column name.
        params_grid (dict): A dictionary of LightGBM hyperparameters to grid search.
        window_size (int): Size of the training window.
        step_size (int): Size of the validation window and step size.

    Returns:
        dict: Dictionary of best hyperparameters and their corresponding average metric.
    """
    best_score = float('inf')
    best_params = None
    data_len = len(data)

    for params in ParameterGrid(params_grid):
        fold_scores = []
        for train_indices, val_indices in time_based_cross_validation(data_len, window_size, step_size):
            train_data = data.iloc[train_indices]
            val_data = data.iloc[val_indices]

            X_train, y_train = train_data[features], train_data[target]
            X_val, y_val = val_data[features], val_data[target]

            model = lgb.LGBMRegressor(**params, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_val)
            score = mean_squared_error(y_val, y_pred)
            fold_scores.append(score)


        avg_score = np.mean(fold_scores)
        if avg_score < best_score:
           best_score = avg_score
           best_params = params
    return {"best_params": best_params, "best_score": best_score}


from sklearn.model_selection import ParameterGrid
if __name__ == '__main__':
    # Generating dummy time series data for example
    date_rng = pd.date_range(start='2020-01-01', end='2021-01-01', freq='D')
    data = pd.DataFrame({'date': date_rng, 'feature_1': np.random.rand(len(date_rng)), 'feature_2': np.random.rand(len(date_rng)),
                         'target': np.random.rand(len(date_rng))}).set_index('date')

    features = ['feature_1', 'feature_2']
    target = 'target'

    params_grid = {
    'num_leaves': [31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'max_depth': [-1, 5, 10]
}
    window_size= 100
    step_size = 20

    best_result = time_series_grid_search(data, features, target, params_grid, window_size, step_size)
    print(f"Best Parameters: {best_result['best_params']}")
    print(f"Best Score: {best_result['best_score']}")

```

Here we’re iterating over parameter combinations generated by `ParameterGrid` from scikit-learn. For each combination, we use our `time_based_cross_validation` function to get our time-aware cross validation folds. We then train and evaluate a LightGBM model on each fold and record the performance before averaging the scores, finally selecting the set of parameters that produce the best average score. This setup avoids leakage and provides a better estimate of the model's performance on unseen future data.

**3. Incorporating Early Stopping:**

For extra stability and speed, I've often found it helpful to add early stopping. LightGBM has built-in early stopping capabilities that further improve performance. I often find my models improve faster, use less resources and generalize better using early stopping. The trick is to enable this in each fold for each hyperparameter combination.

```python
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid


def time_series_grid_search_early_stopping(data, features, target, params_grid, window_size, step_size, early_stopping_rounds):
    """
    Performs a time-series aware grid search on LightGBM with early stopping.

    Args:
        data (pandas.DataFrame): Input dataframe, including features and target.
        features (list): A list of feature column names.
        target (str): The target column name.
        params_grid (dict): A dictionary of LightGBM hyperparameters to grid search.
        window_size (int): Size of the training window.
        step_size (int): Size of the validation window and step size.
        early_stopping_rounds (int): The number of rounds of early stopping to use.

    Returns:
        dict: Dictionary of best hyperparameters and their corresponding average metric.
    """

    best_score = float('inf')
    best_params = None
    data_len = len(data)

    for params in ParameterGrid(params_grid):
        fold_scores = []
        for train_indices, val_indices in time_based_cross_validation(data_len, window_size, step_size):
            train_data = data.iloc[train_indices]
            val_data = data.iloc[val_indices]
            X_train, y_train = train_data[features], train_data[target]
            X_val, y_val = val_data[features], val_data[target]
            model = lgb.LGBMRegressor(**params, random_state=42)

            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
                      eval_metric='mse', callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)])
            y_pred = model.predict(X_val)
            score = mean_squared_error(y_val, y_pred)
            fold_scores.append(score)


        avg_score = np.mean(fold_scores)
        if avg_score < best_score:
            best_score = avg_score
            best_params = params
    return {"best_params": best_params, "best_score": best_score}


if __name__ == '__main__':
    # Generating dummy time series data for example
    date_rng = pd.date_range(start='2020-01-01', end='2021-01-01', freq='D')
    data = pd.DataFrame({'date': date_rng, 'feature_1': np.random.rand(len(date_rng)), 'feature_2': np.random.rand(len(date_rng)),
                         'target': np.random.rand(len(date_rng))}).set_index('date')

    features = ['feature_1', 'feature_2']
    target = 'target'
    params_grid = {
    'num_leaves': [31, 63],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [50, 100, 200],
    'max_depth': [-1, 5, 10]
}

    window_size= 100
    step_size = 20
    early_stopping_rounds = 10
    best_result = time_series_grid_search_early_stopping(data, features, target, params_grid, window_size, step_size, early_stopping_rounds)
    print(f"Best Parameters: {best_result['best_params']}")
    print(f"Best Score: {best_result['best_score']}")
```

We add `eval_set=[(X_val, y_val)]`, `eval_metric='mse'`, and `callbacks=[lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False)]` to the model’s fit method to get early stopping behavior, which will make our models train a lot faster.

**A few final thoughts**:

*   **Parameter Space:**  The hyperparameter grid you select will significantly impact performance. Start with a relatively coarse grid and then refine it around the best-performing regions. Consider adding regularization parameters like `lambda_l1` and `lambda_l2`.
*   **Feature Engineering:** Don't underestimate the importance of good features. Lagged variables, rolling statistics, and other time-aware feature transformations can greatly boost model accuracy.
*   **Time Series Specific Metrics**: consider using time series specific error metrics like mean absolute scaled error (MASE) instead of mean squared error if it suits your time series application.

**Resource Recommendations**:

For further depth, I highly recommend the following materials:

*   **"Forecasting: Principles and Practice" by Rob J Hyndman and George Athanasopoulos:** An excellent, practical guide to time series forecasting. It covers a broad range of models and validation techniques.

*   **"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron:** While not exclusively about time series, this book has a great chapter on time series, cross validation, and model tuning techniques, and is an indispensable resource for anyone working with machine learning models in Python.

*   **LightGBM Documentation:** Always refer to the official LightGBM documentation for the most up-to-date information on parameters and usage. This also gives important insights into the model internals.

This approach should steer you clear of common pitfalls and provide a robust framework for optimizing your LightGBM models on time series data. Remember that this is an iterative process. Don't be afraid to experiment and adapt these techniques to the specific nuances of your dataset.
