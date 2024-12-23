---
title: "outlier detection in time series algorithm?"
date: "2024-12-13"
id: "outlier-detection-in-time-series-algorithm"
---

 so you’re asking about outlier detection in time series data yeah I’ve been there done that got the t-shirt and the battle scars frankly This is one of those problems that seems simple on paper but in the real world its a messy beast I'm gonna dump some of my knowledge here for you maybe it'll help

First off I’ve seen way too many people jump straight to complex machine learning models for this like they’re trying to nuke a mosquito with a bazooka You don’t always need that especially not when you’re first starting out and trying to just understand the baseline of your data I did that too in one of my early jobs building some monitoring tools the boss saw some paper on anomaly detection neural network something and i got tasked to implement it we burned through some GPUs for weeks just to get some results it wasn't really worth it hindsight 2020 though

Let's talk some simple approaches that are surprisingly effective I swear they often get overlooked in favor of the fancy stuff the kind they teach in fancy ML courses

**The Moving Average Approach**

 so you got your time series data right You’re thinking values changing over time I like to think about like a stock ticker or maybe some sensor reading from some hardware thing This approach just looks at the average of the data in a sliding window basically It’s super simple and it works great for catching spikes or dips If a value is significantly different from the moving average you mark it as an outlier

```python
import numpy as np

def moving_average_outlier(data, window_size, threshold):
    """Detects outliers using a moving average.

    Args:
        data (np.array): Time series data
        window_size (int): Size of the moving average window
        threshold (float): Threshold for outlier detection

    Returns:
        np.array: Boolean array indicating outlier status (True = outlier)
    """
    weights = np.repeat(1.0, window_size)/window_size
    moving_average = np.convolve(data, weights, 'valid')
    outlier_flags = np.zeros(len(data), dtype=bool)
    start_idx = window_size // 2
    for i in range(start_idx, len(data) - start_idx):
        if abs(data[i] - moving_average[i - start_idx]) > threshold:
            outlier_flags[i] = True
    return outlier_flags

# Example Usage
data = np.array([1, 2, 3, 10, 4, 5, 6, 1, 2, 100, 3, 4, 5, 6])
window = 3
thresh = 3 # just try and adjust to your needs
is_outlier = moving_average_outlier(data, window, thresh)

print(f"Data: {data}")
print(f"Outliers:{is_outlier}")


```

The core idea here is to take a rolling average and if a data point is too far from its corresponding moving average its considered an outlier The `threshold` variable lets you control how sensitive to outliers this method is and the `window_size` is how much back data you want to see to define the moving average The code above is a simple implementation you might want to explore for more optimized versions

This worked pretty well when i was dealing with some server traffic logs It helped us pinpoint some unusual spikes in requests that pointed to an issue with one of our internal systems

**The Standard Deviation Approach**

Moving on another easy win is looking at how much a data point deviates from the average standard deviation is your friend here It measures the spread of data around the mean if a value is several standard deviations away from the mean its probably an outlier Again this is a simple robust approach

```python
import numpy as np
def std_dev_outlier(data, threshold):
    """Detects outliers using the standard deviation.

    Args:
        data (np.array): Time series data
        threshold (float): Threshold for outlier detection (number of std devs)

    Returns:
        np.array: Boolean array indicating outlier status (True = outlier)
    """
    mean = np.mean(data)
    std_dev = np.std(data)
    outlier_flags = np.abs(data - mean) > threshold * std_dev
    return outlier_flags

# Example Usage
data = np.array([10, 12, 11, 13, 12, 15, 11, 10, 14, 30, 12, 13, 11, 10, 15])
thresh = 2
is_outlier = std_dev_outlier(data, thresh)

print(f"Data: {data}")
print(f"Outliers: {is_outlier}")

```

So this code computes the mean and standard deviation then it compares each data point to the mean in terms of standard deviations The `threshold` variable is how many standard deviations away you consider a data point to be an outlier Adjust to your data and see if you get any result

I remember using this when trying to detect faulty sensors in a factory monitoring system a sensor giving way to high or low values compared to the norm was an obvious outlier the standard deviation method just made it official

Now before you go wild with this keep this in mind this method is sensitive to data with trends over time as a trending data can be flagged as an outlier when it's actually part of the data behavior So its only useful when data has a somewhat stationary behavior but let's be real it works on most data if you have a good threshold value

**Interquartile Range (IQR) Approach**

 this is the last one you’ll thank me later This is similar to the standard deviation method but it focuses on the spread of the data by percentiles using quartiles It's more robust to outliers within the data compared to standard deviation based approach as it does not calculate the mean

```python
import numpy as np

def iqr_outlier(data, threshold):
    """Detects outliers using the IQR method

    Args:
        data (np.array): Time series data
        threshold (float): Threshold for outlier detection (times the IQR)

    Returns:
        np.array: Boolean array indicating outlier status (True = outlier)
    """
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    outlier_flags = (data < lower_bound) | (data > upper_bound)
    return outlier_flags

# Example Usage
data = np.array([10, 12, 11, 13, 12, 15, 11, 10, 14, 30, 12, 13, 11, 10, 15])
threshold = 1.5 # you can try other values
is_outlier = iqr_outlier(data, threshold)

print(f"Data: {data}")
print(f"Outliers: {is_outlier}")
```

Here we calculate the first quartile (25th percentile) and third quartile (75th percentile) the IQR is the difference between these two then we set lower and upper bounds and any data point outside of these bounds is marked as outlier and we used 1.5 as a default threshold here this depends on the data and how strict you need to be

I had this one project for analyzing some marketing data it had several campaigns running at same time so there was a lot of noise the standard deviation approach flagged a lot of valid data as outliers but using the IQR method it found real outliers and gave a good overview what was really important

Now I'm sure some of you are looking to implement these methods on your data so I’m gonna make a recommendation here don’t just copy and paste this code always try to understand what is really going under the hood its part of getting better as a developer
You are not just copy-pasting the code in production you know

**A Final Piece of Advice**

Look I’ve seen this a million times people get caught up in trying to find the perfect algorithm for outliers there isn’t one It depends on your data and your goals Sometimes the simple methods are great and all you need The key is to start simple and iterate and understand your data

Also these are just methods to _detect_ outliers the _cause_ of the outlier is a totally different question that’s what you’ll want to focus on next

As for reading material I recommend that you read *Practical Time Series Analysis: Prediction with Statistics and Machine Learning* by Aileen Nielsen that book is really good for practical real life scenarios and also *Time Series Analysis* by James D Hamilton this one is more of a textbook but its really useful for understanding the theory of the models it’s important to understand what these methods mean in statistical terms as well

Oh and one more thing before I finish what do you call an array of numbers that is not in order? A chaotic list

Anyway good luck out there and happy coding
