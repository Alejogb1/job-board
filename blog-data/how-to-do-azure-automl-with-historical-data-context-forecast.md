---
title: "How to do Azure AutoML with historical data context forecast?"
date: "2024-12-15"
id: "how-to-do-azure-automl-with-historical-data-context-forecast"
---

let's tackle this azure automl with historical context forecasting thing. i've seen this pop up more times than i care to count, and it's a bit more nuanced than just throwing data at the machine. the core issue isn't just predicting future values, it's making those predictions *aware* of past trends and patterns. it's like trying to predict the weather tomorrow by only looking at today's temperature - sure it's *something*, but you’re missing a huge chunk of useful information, all that past data is valuable context.

so, my past experience? well, back in the day, i was working on a project to forecast energy consumption for a small city. at first, we just used basic time series models. the predictions were… let's just say less than stellar. they'd miss those big peak demands and underestimate the baseline usage during holidays. it felt like we were trying to predict stock prices by throwing darts at a board. we needed a more intelligent way to inject the past information, the historical context. we had mountains of historical data but couldn't effectively tell the model what to look for. that’s when we started experimenting with what you're asking about here, effectively using historical context in azure automl.

first up, remember, azure automl isn't a magical black box, it needs data in a particular shape. if you have multiple historical series, treat them as individual time series. you don't want to blend a bunch of different energy usage from different areas with each other and then expect it to work effectively. so, if the context is *other* time series impacting the main time series you want to forecast, make sure these are properly aligned on the time axis. basically, every record has the timestamp for the forecasted series and all the context data available at that time. this means, you should be mindful of the data granularity, both the target variable (the thing you want to predict) and the features from the historical data. you need to ensure that they all share the same time frequency.

now, how do we actually get this in the right format for automl?  let me show you a simple example. suppose you have two time series: `target_series` and `context_series`. let’s use pandas for handling this:

```python
import pandas as pd

# Sample data (replace with your data)
data = {
    'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
    'target_series': [10, 12, 15, 13, 16],
    'context_series': [5, 7, 6, 8, 9]
}

df = pd.DataFrame(data)
df = df.set_index('timestamp')
print(df)
```
this snippet demonstrates how to put the target and context series in a pandas dataframe with the timestamp as the index. this way all time series are aligned.

the critical part comes when configuring automl. you'll need to tell it which column contains the timestamp, which one is the target and which columns are features. remember to make sure these are all numerical values if you expect automl to perform any sensible computations.

let's get into the actual azure automl code configuration:
```python
from azure.ml import automl, Input, Output
from azure.identity import DefaultAzureCredential
from azure.ml import MLClient, automl, Input, Output
from azure.core.exceptions import ResourceNotFoundError
from azure.ml.entities import Dataset, ManagedIdentityConfiguration

# set up connection to workspace using default azure credential
credential = DefaultAzureCredential()

# workspace configurations
subscription_id = "<your subscription id>"
resource_group = "<your resource group>"
workspace_name = "<your workspace name>"

# get a handle to the workspace
try:
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)
except ResourceNotFoundError:
    print(
        "could not find workspace. Please provide valid subscription_id, resource_group, and workspace_name."
    )

# data configuration
# load data with pandas and prepare dataframe as previously shown

input_dataset_name = "your_data_set_name"
input_data = Input(type="uri_file", path="your_local_csv_file.csv")

try:
    input_dataset = ml_client.data.get(name=input_dataset_name, version="1")
except ResourceNotFoundError:
    input_dataset = Dataset(name=input_dataset_name, path=input_data, description="input dataset")
    ml_client.data.create_or_update(input_dataset)
    input_dataset = ml_client.data.get(name=input_dataset_name, version="1")
    print("data created")

# configuration of training parameters
target_column_name = "target_series"
time_column_name = "timestamp"
features_columns_names = ["context_series"]  # additional columns to use as features

# set experiment settings
experiment_timeout_minutes = 60
experiment_settings = automl.training.AutoMLSettings(
    task_type="forecasting",
    primary_metric="normalized_root_mean_squared_error",
    experiment_timeout_minutes=experiment_timeout_minutes,
    forecasting_parameters=automl.training.ForecastingParameters(
        time_column_name=time_column_name,
        time_series_id_column_names=None,  # if each series has no id it must be None
        forecast_horizon=7,
    )
)

# create job definition for training experiment
automl_job = automl.automl_job(
    inputs={"training_data": input_dataset},
    outputs={"best_model": Output(type="mlflow_model")},
    training_settings=experiment_settings,
    compute= "your_cluster_name", # put your cluster name here
    target_column=target_column_name,
)
# run job
returned_job = ml_client.jobs.create_or_update(automl_job)
print(f"job started: {returned_job.name}")

```
the main thing here is the `forecasting_parameters`. note i've set `time_column_name` and the `forecast_horizon`. also `time_series_id_column_names` should be None if you are not dealing with multiple series. you'll also notice how you pass your dataframe as an input dataset using `ml_client.data.create_or_update` and then pass the dataset to the automl job in the `inputs` dictionary with key `training_data`.

another thing that could help, instead of just feeding the raw historical data, consider engineered features. for instance, lagged values can be incredibly effective. this means, instead of just using the raw `context_series` you create new columns that represent the same variable a few steps back in time, this could greatly help model understand the temporal dependencies between your data. it is the models way to "remember" the past. you can add some rolling statistics as well, such as moving average and standard deviation of your variables. here is a quick pandas example:

```python
import pandas as pd

# sample dataframe (replace with your data)
data = {
    'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07']),
    'context_series': [5, 7, 6, 8, 9, 11, 10],
    'target_series': [10, 12, 15, 13, 16, 17, 19]
}
df = pd.DataFrame(data)
df = df.set_index('timestamp')

# create lagged features
for i in range(1, 4): # we will create 3 lagged features
  df[f'context_series_lag_{i}'] = df['context_series'].shift(i)

# create rolling window features
df['context_series_rolling_mean'] = df['context_series'].rolling(window=3).mean()
df['context_series_rolling_std'] = df['context_series'].rolling(window=3).std()
print(df)
```

here i've created lagged features (using `.shift()`) and rolling window statistics (using `.rolling()` with a window of size 3).  you'd need to carefully consider what makes sense for your specific data. some more complex feature engineering steps could involve discrete Fourier transforms, or wavelet decompositions if your data requires this level of complexity, this could capture seasonality or other periodicities in the data.

now, something to keep in mind when dealing with historical data, you must split the data into training, validation and test sets. since we are dealing with time series you should *not* shuffle the data. you should keep the temporal order. the validation and test sets must be after the train set so you ensure that the model will not see data from the future, which is very important for time series modeling. another thing, if your time series has missing data, you must handle it, either by imputing the missing values or dropping the records.

regarding resources, i'd recommend looking at "forecasting: principles and practice" by hyndman and athanasopoulos. it's a pretty solid resource for understanding time series forecasting. for more specific azure automl stuff, the microsoft documentation is usually quite comprehensive, just search the specific classes and functions.

and one more thing, be very careful to specify the number of time periods you want to forecast with `forecast_horizon` , it is a common mistake to leave it undefined. it is also a very important parameter. you must understand what you need to forecast in terms of the time horizon.

so, what was the outcome of my energy usage prediction project, with the historical data? well, the improvement was significant. instead of just making guesses, the model actually learned from past patterns and could predict those peak loads with more accuracy. we didn't have a perfect system, but it was significantly better than throwing darts. we even created some dashboards that would show the time series along the prediction, it looked beautiful. it was almost like… we could see the future… ok i'm just kidding. but it did feel like a huge step in the right direction.

in summary, you need to shape your data appropriately, properly configure your automl parameters, consider feature engineering, understand your train/test split and make sure the temporal order is respected. remember to consider the specifics of your problem, don't just copy/paste code. hopefully this helps clear up how to use historical context in your azure automl forecasting project.
