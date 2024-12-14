---
title: "What is a Simple way to create "lags" when testing time series models?"
date: "2024-12-14"
id: "what-is-a-simple-way-to-create-lags-when-testing-time-series-models"
---

here we go.

so, you're looking to inject some artificial lag into your time series data, huh? i get it. testing these models can be a pain, especially when trying to see how they react to delays or data inconsistencies. i've been there, staring at graphs that refuse to cooperate, so let me share what worked for me.

the core idea is to manipulate the timestamps associated with your data points. you're essentially shifting them in time to simulate a delay. it sounds simple, and it is when you break it down. no fancy math or deep algorithms are needed for this part.

let's start with the simplest method: a fixed lag. think of it as dragging all your data points a certain duration backward in time. this is useful for assessing how a model performs when the input data is systematically delayed. imagine you have a model predicting stock prices. if your input data has a one-hour lag, how will the model’s accuracy be affected? this approach simulates that scenario.

i remember having a rough time with this back in the day, maybe around 2015-2016 when i was playing around with simple neural nets for some weather pattern prediction. i had all this hourly data and the model was performing too well, i felt it was basically cheating because it was getting information a bit before the ‘real’ time. by adding a fixed lag and some random errors, i got a much more real scenario and identified critical problems in how the input layers were structured, so i had to re-do the feature engineering completely.

here's a python snippet using pandas to do exactly that:

```python
import pandas as pd
import numpy as np

def add_fixed_lag(df, time_column, lag_duration):
    df[time_column] = df[time_column] - pd.Timedelta(lag_duration)
    return df

#example:
data = {'time': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00']),
        'value': [10, 12, 15]}
df = pd.DataFrame(data)

lagged_df = add_fixed_lag(df, 'time', '1 hour')
print(lagged_df)

```

in this example, the `add_fixed_lag` function subtracts a specified `lag_duration` from your time column. pretty straightforward stuff. you pass your dataframe, the name of the column that holds the time information, and the lag you want, which can be a string like ‘1 hour’, ‘30 minutes’, or even ‘2 days’. the result is a new dataframe where the timestamps are all shifted by that amount.

then we move to a more realistic scenario: variable lag. this method introduces delays that fluctuate over time. they could be randomly distributed or follow a specific pattern. this can be a better representation of real-world conditions, where delays are rarely constant. for instance, maybe sensors reporting data can be flaky or network traffic is not uniform, thus generating fluctuating delays in data arrival times.

i had a particularly difficult case of this when i was trying to create a model that predicted sensor readings in a distributed industrial system, probably in 2018. i had very different devices from different manufacturers, and some were really unreliable, which meant that not all readings were arriving at the same time. this was a mess. some would be there on time, others would lag by a couple of seconds, and some would just disappear. when i applied a fixed lag the model was working flawlessly, but then, when i tried to use the real data in the real industrial system, it was almost completely useless. that forced me to test my models with this variability. this was a good test to evaluate my model's robustness.

here is some code that implements a variable lag:

```python
import pandas as pd
import numpy as np

def add_variable_lag(df, time_column, max_lag_duration):
    lag_values = np.random.uniform(0, max_lag_duration.total_seconds(), size=len(df))
    df[time_column] = df[time_column] - pd.to_timedelta(lag_values, unit='s')
    return df

#example:
data = {'time': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00']),
        'value': [10, 12, 15]}
df = pd.DataFrame(data)

lagged_df = add_variable_lag(df, 'time', pd.Timedelta('10 seconds'))
print(lagged_df)
```

in this snippet, `add_variable_lag` calculates a random lag between zero and the value passed as `max_lag_duration`, so each time point is delayed by a different amount. you pass in your dataframe, the name of the timestamp column, and the maximum duration of the lag, specified as a timedelata, like, in the example, 10 seconds.

finally, we can get a bit more sophisticated by simulating a rolling lag. this approach delays data points depending on some predefined pattern or based on another column in your data. for example, you could lag each data point by a duration that is proportional to another sensor reading. this is super useful when testing models in complex environments. for example imagine you have a network of sensors, and you have a reference sensor from where you know the delay. you can use that sensor data as the parameter for simulating delays in the other sensors.

this sort of problem was my headache around 2019-2020 when i was looking into some distributed sensor networks for environmental monitoring. one of the sensors was always delayed because its power unit was not the same as the other sensors. by analyzing the power usage of that sensor, i was able to create a lag that was based on that power consumption. it was a weird situation, to say the least. why would a power unit cause delay? well the power unit was used to transmit data, so when the power was low, the transmission would happen with delay because the communication unit was in a power saving mode.

here’s how you could achieve this in code:

```python
import pandas as pd
import numpy as np

def add_rolling_lag(df, time_column, lag_column, lag_multiplier):
    df['lag_seconds'] = df[lag_column] * lag_multiplier
    df[time_column] = df[time_column] - pd.to_timedelta(df['lag_seconds'], unit='s')
    df = df.drop(columns=['lag_seconds'])
    return df

# example:
data = {'time': pd.to_datetime(['2024-01-01 10:00:00', '2024-01-01 11:00:00', '2024-01-01 12:00:00']),
        'value': [10, 12, 15],
         'lag_reference' : [1, 2, 0.5]}
df = pd.DataFrame(data)

lagged_df = add_rolling_lag(df, 'time', 'lag_reference', 5)
print(lagged_df)

```

here, the `add_rolling_lag` function takes a lag reference column, multiplies it by a value `lag_multiplier` to determine how much to delay each time point. so, each timestamp’s delay is now dependent on the value in the ‘lag\_reference’ column and the `lag_multiplier` provided. we create a temporary column for the lag, use it to change the timestamps and then remove it, not to mess with the rest of the data.

about resources? well, i don't usually drop links, so instead, let's go for books and papers. for time series stuff in general, anything from box and jenkins is a classic (i.e., "time series analysis: forecasting and control"). for the nitty-gritty of time manipulations in pandas, the pandas documentation itself is your best bet. there is no secret sauce or magic, is all in the documentation. look for the `datetime` and `timedelta` objects and how to operate with them. and finally, for dealing with issues in distributed data systems, consider looking into academic papers on topics like distributed consensus and time synchronization. there is a deep rabbit hole to follow there.

to finish, remember that this type of problem is very context specific. choose the method that best reflects the data inconsistencies you expect. start simple and iterate, and don't be afraid to experiment. oh, and before i forget: why don't scientists trust atoms? because they make up everything! alright, good luck with your models, and remember: always test under the worst circumstances, that is how you learn.
