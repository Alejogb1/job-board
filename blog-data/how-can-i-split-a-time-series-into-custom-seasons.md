---
title: "How can I split a time series into custom seasons?"
date: "2024-12-23"
id: "how-can-i-split-a-time-series-into-custom-seasons"
---

, let’s tackle this. It's not uncommon to encounter the need for custom seasonality in time series analysis, especially when dealing with data that doesn't conform to standard calendar-based patterns. In fact, I remember a project back in '15 involving environmental sensor data where we had to account for shifts in seasonal patterns influenced by specific weather anomalies; something off-the-shelf tools simply wouldn't handle. We needed to break away from traditional monthly or quarterly segments.

The core challenge isn't simply partitioning a time series—it’s *meaningfully* segmenting it according to underlying patterns, even if those patterns are unique to the specific data set. I've found that the most effective approaches involve combining a clear understanding of the data with a pragmatic strategy. Let’s dive into how we can achieve this.

Essentially, splitting a time series into custom seasons relies on identifying the breakpoints that mark the transition from one season to another. These breakpoints aren’t always obvious and might require a combination of data exploration and potentially domain knowledge. I would normally look at several areas: first, descriptive statistics over small time windows (mean, variance, skew, and kurtosis); second, visual plots of the data; and third, any available external data that might be influencing these seasons. The goal here is to find *features* that change drastically at season boundaries.

Once you identify these boundary points, several techniques can be employed to formalize this segmentation process. Let's look at some examples using python, a very common tool for these types of tasks. We'll use `pandas` for handling time series data and `numpy` for numerical operations.

**Example 1: Using Fixed Date Ranges**

This method works best when you have pre-defined knowledge of the start and end dates of your custom seasons. Think of scenarios where seasons are defined by specific project milestones, promotional periods, or known recurring events.

```python
import pandas as pd
import numpy as np

# Sample time series data (replace with your own data)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.random.rand(len(dates)) * 100  # Example values
data = pd.DataFrame({'date': dates, 'value': values})
data = data.set_index('date')

# Define custom season boundaries as date strings
season_boundaries = {
    'Season 1': ('2023-01-01', '2023-03-31'),
    'Season 2': ('2023-04-01', '2023-06-30'),
    'Season 3': ('2023-07-01', '2023-09-30'),
    'Season 4': ('2023-10-01', '2023-12-31')
}

# Function to label each data point with the relevant season
def assign_season(date):
    for season_name, (start_date, end_date) in season_boundaries.items():
        if pd.to_datetime(start_date) <= date <= pd.to_datetime(end_date):
            return season_name
    return None  #Handles cases that dont fall into defined seasons.

# Create a new column for custom seasons
data['season'] = data.index.map(assign_season)

print(data.head())
print(data.groupby('season').describe())
```

In this first snippet, I’ve set up a very straightforward way to slice the data based on pre-defined dates. This approach is effective when you have external information that defines the seasons. Notice the `assign_season` function iterates through predefined bounds to assign a 'season' column. This makes grouping and analysis very easy, as I show in the `groupby('season').describe()` output.

**Example 2: Using a Data-Driven Approach with Thresholds**

Now, consider the case where we don't have pre-determined date ranges. In this scenario, the data itself should inform the season boundaries. This often requires identifying a characteristic in the data (for example, a moving average) that indicates changes between periods. I find that a threshold-based approach, where I define trigger points, often works well.

```python
import pandas as pd
import numpy as np

# Sample time series data (replace with your own data)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.random.rand(len(dates)) * 100
values[100:130] += 50 # Simulating a spike
values[250:280] -= 30 # Simulating a drop
data = pd.DataFrame({'date': dates, 'value': values})
data = data.set_index('date')

# Calculate a moving average (adjust window as needed)
data['moving_average'] = data['value'].rolling(window=30).mean()

# Define thresholds for season transitions
threshold_up = 55
threshold_down = 35

# Function to identify season transitions based on thresholds
def find_seasons_threshold(data, threshold_up, threshold_down):
    seasons = []
    current_season = 1
    start_date = data.index[0]

    for i in range(1, len(data)):
        if data['moving_average'].iloc[i] > threshold_up and data['moving_average'].iloc[i-1] <= threshold_up:
            seasons.append((start_date, data.index[i-1], current_season))
            start_date = data.index[i]
            current_season+=1
        elif data['moving_average'].iloc[i] < threshold_down and data['moving_average'].iloc[i-1] >= threshold_down:
            seasons.append((start_date, data.index[i-1], current_season))
            start_date = data.index[i]
            current_season+=1


    seasons.append((start_date, data.index[-1], current_season)) # Capture the last season

    return seasons


# Find season breaks
season_breaks=find_seasons_threshold(data,threshold_up, threshold_down)

# Assign seasons to each data point
def assign_seasons_by_threshold(date, season_breaks):
    for start, end, season in season_breaks:
        if start<=date<=end:
            return season
    return None

data['season'] = data.index.map(lambda date: assign_seasons_by_threshold(date, season_breaks))

print(data.head())
print(data.groupby('season').describe())

```

Here, we use a moving average and predefined thresholds. The `find_seasons_threshold` function uses the change of the moving average over the thresholds to create start and end points for the seasons. The rest of the code is quite similar to the prior example, except instead of a predefined date, we're using a function output. This approach is dynamic and adapts to changes within your data. Note that choosing the right moving average window and thresholds is critical; there will often be a trial-and-error component to this.

**Example 3: An Even More Advanced Option: Change Point Detection**

Finally, for situations where you need to be even more sophisticated in identifying season breaks, you can turn to algorithms designed for change point detection. This method is particularly useful when transitions aren't easily captured by simple thresholding. Change point detection can be based on statistical properties or machine learning models.

```python
import pandas as pd
import numpy as np
import ruptures as rpt # Change point detection

# Sample time series data (replace with your own data)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
values = np.random.rand(len(dates)) * 100
values[100:130] += 50 # Simulating a spike
values[250:280] -= 30 # Simulating a drop
data = pd.DataFrame({'date': dates, 'value': values})
data = data.set_index('date')

# Perform change point detection (using Pelt algorithm as example)
algo = rpt.Pelt(model="rbf").fit(data['value'].values)
result = algo.predict(pen=10)  # pen is the penalty for adding changepoints, needs tuning
change_points = [data.index[i] for i in result[:-1]]


# Create season boundaries
def get_season_boundaries(change_points, index):
    season_boundaries = []
    start_date = index[0]
    for change_point in change_points:
        season_boundaries.append((start_date,change_point ))
        start_date = change_point
    season_boundaries.append((start_date, index[-1] )) # Last season
    return season_boundaries


#Assign seasons based on detected change points
season_boundaries = get_season_boundaries(change_points, data.index)
def assign_seasons_change_points(date, season_boundaries):
    for i, (start, end) in enumerate(season_boundaries):
        if start <=date <=end:
            return i + 1 #start index at one
    return None #Catch any edge cases

data['season'] = data.index.map(lambda date : assign_seasons_change_points(date, season_boundaries))


print(data.head())
print(data.groupby('season').describe())
```

In this final example, I've used the `ruptures` library, which offers algorithms specifically designed to locate change points in time series. The `Pelt` algorithm identifies changes in the statistical properties of the data, which become our season boundaries.  The result is an array of dates where "significant" changes happen in the series.

**Resources and Further Study:**

For deeper exploration, I’d recommend the following:

*   **"Time Series Analysis and Its Applications" by Robert H. Shumway and David S. Stoffer:** This is a classic textbook covering various time series analysis techniques. It will provide a solid foundation on many of the concepts discussed here, especially if you want to understand the statistical basis of approaches like change point detection.

*  **"Forecasting: Principles and Practice" by Rob J Hyndman and George Athanasopoulos:** An excellent, freely available online resource that covers various time series analysis and forecasting methods, including handling seasonality.

*  **Scikit-learn documentation:** While scikit-learn doesn’t directly handle all the above needs, its core models and preprocessing methods are essential.

In summary, splitting a time series into custom seasons is not a one-size-fits-all problem. The best approach depends heavily on your data and available knowledge. Start with simple methods like fixed date ranges, graduate to threshold-based approaches, and consider change point detection techniques when needed. Remember to validate your results using domain expertise or by testing against different model performance metrics.
