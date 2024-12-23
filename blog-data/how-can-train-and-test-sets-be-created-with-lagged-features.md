---
title: "How can train and test sets be created with lagged features?"
date: "2024-12-23"
id: "how-can-train-and-test-sets-be-created-with-lagged-features"
---

Okay, let’s tackle lagged feature creation for training and testing sets. It's a common challenge, and I've certainly had my share of encounters with it over the years, particularly when dealing with time-series data, which, let’s face it, tends to be a large chunk of what we all deal with. Getting this process correct is fundamental for reliable predictive models. The key difficulty isn't just about adding a lag, but ensuring that the temporal ordering is maintained across the training and testing data, and that we avoid any 'data leakage'.

Essentially, the concept of a lagged feature revolves around using a value at a previous time point as an input for the model at the current time point. Imagine you're predicting sales for tomorrow; a lagged feature could be the sales from yesterday or the average sales from the past week. The tricky bit arrives when we need to maintain this lagged structure when splitting data into training and testing sets. Simply shuffling data or using a random sample split will obliterate the time dependency and make our evaluation useless. The temporal component must be respected and preserved.

Let me elaborate on this with a bit of my past experience. Back when I was working on a demand forecasting project for a retail chain, we used daily sales data to predict future demand. Ignoring the correct temporal splits was the initial blunder. We ended up with a model that appeared fantastic on paper, but once it hit production, it was practically useless. The reason, as we painfully discovered, was that our test set was, in essence, bleeding information from the future into the training process. We had to rewrite the whole pipeline to manage lagged features meticulously. From that experience onwards, I developed a more structured approach, which I’ll detail here.

The central concern here is that our training set should only contain information from prior to what's available in our test set. Otherwise, we are inadvertently using future information to predict the past. It's illogical, and it invalidates our entire modeling endeavor.

Here's the breakdown of how to properly construct these sets with lagged features, along with some code snippets in python using pandas, since that's probably what most are familiar with.

**1. Data Preparation and Lag Generation**

First, you’ll have your time series data, presumably organized chronologically. Let's assume you have a pandas dataframe with a ‘date’ column and a ‘sales’ column. The first step involves generating the lagged features. We can create these features using the pandas `shift()` function. Here's a simple example:

```python
import pandas as pd

def create_lagged_features(df, lag_periods, feature_col):
  """Creates lagged features for a given column."""
  for lag in lag_periods:
    df[f'{feature_col}_lag_{lag}'] = df[feature_col].shift(lag)
  return df

# Example data (replace with your actual data)
data = {'date': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
        'sales': [100, 110, 120, 130, 140]}
df = pd.DataFrame(data)
df = df.set_index('date')

lag_periods = [1, 2, 3] # Example lags
df = create_lagged_features(df, lag_periods, 'sales')
print(df)
```

In this snippet, the `create_lagged_features` function will create columns like `sales_lag_1`, `sales_lag_2`, and `sales_lag_3`, representing sales one, two, and three days ago respectively. Note that for the earliest rows, some lagged values will be `NaN` because there aren't any previous values in the dataset for that time point. This `NaN` occurrence is natural and needs to be considered when preparing data for a model.

**2. Maintaining Temporal Integrity: The Split**

The crucial aspect here is the split. You must avoid random shuffling. Instead, you split based on time. Typically, you choose a date to act as a cutoff, with everything before it becoming your training set, and everything after as the testing set.

```python
def train_test_split_temporal(df, split_date):
  """Splits data into training and testing sets based on a date."""
  train_df = df[df.index < pd.to_datetime(split_date)]
  test_df = df[df.index >= pd.to_datetime(split_date)]
  return train_df, test_df

split_date = '2023-01-04' # Example split date
train_df, test_df = train_test_split_temporal(df, split_date)

print("Training Set:")
print(train_df)
print("\nTesting Set:")
print(test_df)
```

Here, we simply partition the dataframe based on the split date. No random shuffling or random sampling is employed. This ensures that the testing dataset will never contain data used within the training set, preventing the potential issue of "future leakage". It should also be noted that this method is not only applicable to the last portion of the dataset. You can pick multiple portions as your testing set, such as every fourth day, which could be used for a more robust evaluation of your model. However, the main principle remains: do not use data in the future of a given point in the training data when testing the model at that point.

**3. Handling Missing Values and Training**

Finally, you need to address those `NaN` values which resulted from the lag operation. These are typically at the beginning of the training set. A common method is to either drop rows with `NaN` values, or fill them with an appropriate strategy, such as zero, or the mean, or a median. It’s critical that you **never** fill `NaN` values using information from your test dataset; this would defeat the entire purpose of splitting data correctly.

```python
def prepare_for_training(train_df, test_df, fill_method='ffill'):
    """Prepares train and test sets, filling missing values only within each set."""
    train_df_filled = train_df.copy()
    test_df_filled = test_df.copy()

    if fill_method == 'ffill':
       train_df_filled.fillna(method='ffill', inplace=True)
       test_df_filled.fillna(method='ffill', inplace=True)
    elif fill_method == 'zero':
        train_df_filled.fillna(0, inplace=True)
        test_df_filled.fillna(0, inplace=True)
    elif fill_method == 'mean':
        train_df_filled.fillna(train_df_filled.mean(), inplace=True)
        test_df_filled.fillna(test_df_filled.mean(), inplace=True)
    elif fill_method == 'median':
         train_df_filled.fillna(train_df_filled.median(), inplace=True)
         test_df_filled.fillna(test_df_filled.median(), inplace=True)
    
    train_df_filled.dropna(inplace=True) # Drop remaining if using ffill, or from other issues
    test_df_filled.dropna(inplace=True)
    
    X_train = train_df_filled.drop('sales', axis=1)
    y_train = train_df_filled['sales']
    X_test = test_df_filled.drop('sales', axis=1)
    y_test = test_df_filled['sales']
    
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = prepare_for_training(train_df, test_df, fill_method='ffill')
print("\nX_train:")
print(X_train)
print("\ny_train:")
print(y_train)
print("\nX_test:")
print(X_test)
print("\ny_test:")
print(y_test)
```

The `prepare_for_training` function handles this, using `ffill` (forward fill) as the default. It is vital that we fill `NaN` values *within* each set only, to maintain the proper split. It will fill missing values based on previous values, zeros, or the mean of the values *within the same training or test set*. Once the missing data is dealt with, we're ready to use `X_train`, `y_train`, `X_test`, and `y_test` for model training and evaluation.

**Further Reading**

For a more in-depth understanding, I would highly recommend the following resources:

*   **"Forecasting: Principles and Practice" by Rob J Hyndman and George Athanasopoulos:** This is an excellent and freely available online book covering time series forecasting extensively. It goes into detail on many forecasting techniques, including the creation and use of lagged features, and explains the concepts of training and testing in a time-series context. It's an absolute must-read.

*   **"Time Series Analysis" by James D. Hamilton:** A more mathematically oriented and rigorous treatment of time series analysis. It’s considered a standard textbook in econometrics and provides a comprehensive theoretical framework.

*   **"Python for Data Analysis" by Wes McKinney:** If you're working with Python, as I demonstrated in my examples, this book by the creator of pandas is a must-have. It is very thorough and provides a solid base for working with data using pandas.

The key takeaway here is that building robust predictive models with lagged features goes hand-in-hand with respecting the time-based nature of the data. By ensuring proper data splitting and avoiding any kind of information leakage, we build a foundation for trustworthy model evaluations and reliable predictions.
