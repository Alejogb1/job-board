---
title: "Why am I getting a Mismatch in the number of cases despite there are the same number of cases sktime?"
date: "2024-12-14"
id: "why-am-i-getting-a-mismatch-in-the-number-of-cases-despite-there-are-the-same-number-of-cases-sktime"
---

alright, so you're banging your head against a wall with sktime and getting a mismatch in the number of cases, even though you're certain you have the same number of cases. i've been there, it's a classic gotcha that can trip up even the most seasoned time series wranglers. let's break this down, and i'll share some battle scars from my own encounters with this particular beast.

first off, let's talk about what sktime means by "cases." in time series land, "cases" usually refer to individual time series instances. think of it like this: if you’re tracking stock prices, each stock is a case. if you're measuring temperatures in different cities, each city's temperature readings are a case. sktime expects your data to be structured in a way where each case is identifiable and separate. this is where things can get tricky, because data doesn't always arrive in that format.

the most common culprit, in my experience, is that sktime expects your data to be in a pandas dataframe where each row is a case, and each column is a feature of that case. sometimes, though, you might have your data in a format where cases are grouped within a single dataframe, or even in a nested structure, which will completely confuse sktime. this mismatch causes the internal case counter to go haywire, hence the different number.

the first time this happened to me was about five years back. i was working on a project predicting energy consumption for different buildings on a campus. we had data for each building, but instead of having separate rows per building in the dataframe, each building's entire timeseries was stored as a series within a single row. it looked something like this:

```python
import pandas as pd

data = {'building_id': [1, 2, 3],
        'energy_data': [
            pd.Series([10, 12, 15, 14, 17]),  # building 1
            pd.Series([8, 9, 11, 10, 12]),   # building 2
            pd.Series([15, 16, 18, 17, 20])  # building 3
        ]}

df = pd.DataFrame(data)

print(df)
```
this structure looks ok, but if you pass this to a sktime function expecting each row to be a case, it will assume we have one case with a time series for the `building_id` column and another timeseries for `energy_data`, which is of course wrong.

sktime's transformers and estimators operate on the premise that they can iterate over rows which are cases. when i first encountered this error, i was convinced i was going insane, as every manual check told me the counts were equal. it was a good couple of hours of debugging before i understood the issue.

the fix, for this type of data, is to unnest the data so every timeseries belongs to a different row. a better dataframe format would be something like this

```python
import pandas as pd

data = {'building_id': [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
        'time': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'energy_data': [10, 12, 15, 14, 17, 8, 9, 11, 10, 12, 15, 16, 18, 17, 20]
        }

df = pd.DataFrame(data)
print(df)

```

in essence, instead of having lists or pandas series as the column values, we have individual time-stamped datapoints, with an associated case identifier (in this case `building_id`). this is the long format that sktime prefers for many use cases.
this can be converted into a time series dataframe with sktime's functions.

the second most common issue i’ve seen is related to how sktime handles different data types within the same dataframe. let’s say you have numerical features as well as time series data within the same structure. sktime might interpret the numerical columns as cases if you don’t explicitly tell it which columns represent the time series.
for example:

```python
import pandas as pd
import numpy as np

data = {'case_id': [1, 2, 3],
        'feature1': [10, 20, 30],
        'time_series_1': [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])],
         'time_series_2': [np.array([10, 11, 12]), np.array([13, 14, 15]), np.array([16, 17, 18])]
        }

df = pd.DataFrame(data)
print(df)

```

in this structure `feature1` might be mis-interpreted as a time-series column by sktime, even though it's meant to be a numerical feature associated with each case. in such a scenario, if you were expecting three cases, sktime might end up considering four.

to mitigate this, make sure you're feeding sktime *only* the time series columns.  and when doing that use `from_nested_to_3d_numpy` to convert your pandas df to a 3d numpy array which sktime expects. this ensures that you have the dimensions correct.

```python
import pandas as pd
import numpy as np
from sktime.utils.data_processing import from_nested_to_3d_numpy

data = {'case_id': [1, 2, 3],
        'feature1': [10, 20, 30],
        'time_series_1': [np.array([1, 2, 3]), np.array([4, 5, 6]), np.array([7, 8, 9])],
        'time_series_2': [np.array([10, 11, 12]), np.array([13, 14, 15]), np.array([16, 17, 18])]
        }
df = pd.DataFrame(data)
# Select the columns containing time series data
time_series_cols = ['time_series_1', 'time_series_2']

# Convert the selected columns to a 3d numpy array

ts_array = from_nested_to_3d_numpy(df[time_series_cols])
print(ts_array.shape)

```

finally, one additional problem i’ve seen is when dealing with unevenly sampled time series. if different cases have different time lengths, sktime can get confused, especially when applying operations that require consistent dimensions. in this situation, it might appear that there are fewer or more cases depending on how sktime handles that internally.  you would be better of standardising the length of each series before attempting to use sktime’s functions.

one time I was working on some network data where different machines had different amounts of data available, sktime kept returning different number of instances. the reason was that the estimator, behind the scenes, was dropping cases with insufficient datapoints as part of some implicit checks and I wasn't aware of this preprocessing step.
it's an odd feeling to see the case counts differ, despite having checked the data with my own eyes countless of times, it can make you feel like the data is haunted (or maybe the code is...).

so, here’s a quick recap of how to avoid this gotcha.

1.  **check your dataframe structure:** make sure each row is a case and that the data structure is of the correct form, long format is usually preferable.
2.  **explicitly select time series columns:** when dealing with multiple columns, make sure to explicitly provide sktime with only the correct time series columns. using `from_nested_to_3d_numpy` or similarly helpful functions to avoid errors.
3. **standardise time series lengths:** if you are dealing with time series of varying lengths, ensure these are consistently shaped or take these inconsistencies into account in the sktime pipeline.

for further reading, i recommend checking out the following:
*   **"time series analysis" by james hamilton**: a deep dive into the mathematical foundations of time series, it covers the underlying statistical theory and will provide a much deeper understanding.
*   **"forecasting: principles and practice" by hyndman and athanasopoulos**: this is a more hands-on guide to practical forecasting methods including some good insights in time series data handling.
* **the sktime documentation** is also quite good.

i hope this helps you get unstuck. data wrangling with time series data can be a real test of patience, but keep at it, and you'll get there.
