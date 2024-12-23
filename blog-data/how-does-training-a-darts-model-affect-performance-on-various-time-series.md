---
title: "How does training a darts model affect performance on various time series?"
date: "2024-12-23"
id: "how-does-training-a-darts-model-affect-performance-on-various-time-series"
---

, let’s dive into this. I've spent quite a bit of time tweaking time series models, particularly with the darts library, and it’s a nuanced area where understanding the interaction between training data characteristics and model performance is crucial. From my experience, just throwing data at a model, even a well-designed one from darts, isn't always a recipe for success. Performance on various time series, after training, is heavily influenced by several factors related to the training dataset and the model itself.

The first crucial aspect is the *similarity between the training time series and the target time series*. This might sound obvious, but it’s deeper than just having similar ranges. For instance, if your training set primarily consists of relatively stable time series with clear seasonal patterns, and you then attempt to predict a highly volatile time series with erratic jumps and no clear seasonality, you're setting up the model for failure. The model, having learned specific patterns from the training data, won't generalize well to a completely different statistical distribution. I recall a project where we tried to forecast energy consumption for a new industrial plant using models trained on historical data from similar but much smaller facilities. The performance on the new plant's data was abysmal until we incorporated a representative time series from a pilot setup that more accurately reflected the new plant's consumption patterns.

Another vital point is the *quantity and quality of the training data*. More data isn't always better; more *relevant* data is what truly matters. If your training set is riddled with anomalies or contains a lot of noise, the model will inevitably learn to incorporate this noise into its predictions. This can lead to overfitting, where the model performs well on the training data but poorly on unseen data. I once had a situation with stock market data where a model was trained on a dataset with erroneous pricing entries. The result was a model that fit the training data perfectly, including those errors, but failed spectacularly when applied to real-time market feeds. Clean, curated, and representative training data is paramount. We often used smoothing techniques, anomaly detection, and sometimes even manual review to cleanse the training data before feeding it to our darts models.

Then there's the *length of the training time series*. A short training sequence might not be sufficient for the model to learn underlying patterns, especially if those patterns span multiple cycles, be it daily, weekly, or yearly. The model might struggle to understand the complete picture. On the flip side, excessively long training sequences can be computationally expensive and may not contribute much to model performance beyond a certain point, potentially introducing the phenomenon of 'model forgetfulness' where older patterns are given less weight. We would typically use a sliding window approach to experiment with the training sequence length and observe how it impacts accuracy metrics on the validation set, ensuring that the model learns a generalizable pattern and doesn't just memorize the training sequence.

Now, regarding specific code snippets, let's illustrate these points with some practical examples using darts.

**Snippet 1: Demonstrating Data Similarity Impact**

This first snippet shows two very basic exponential smoothing models. The first is trained on a relatively smooth sine wave, and the second is trained on the same sine wave with added gaussian noise. The code demonstrates how the model trained on the noisy data performs poorly when trying to predict the smooth signal.

```python
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.metrics import mape
import matplotlib.pyplot as plt

# Generate smooth time series
t = np.arange(0, 100, 0.1)
smooth_series = np.sin(t/5)

# Generate noisy time series
noisy_series = smooth_series + np.random.normal(0,0.5, len(t))

# Create TimeSeries objects
smooth_ts = TimeSeries.from_values(smooth_series)
noisy_ts = TimeSeries.from_values(noisy_series)

# Split into train and validation
train_smooth, val_smooth = smooth_ts.split_before(int(len(t)*0.8))
train_noisy, val_noisy = noisy_ts.split_before(int(len(t)*0.8))

# Train on smooth series and predict validation set
model_smooth = ExponentialSmoothing()
model_smooth.fit(train_smooth)
pred_smooth = model_smooth.predict(len(val_smooth))
mape_smooth = mape(val_smooth, pred_smooth)


# Train on noisy series and predict validation set of smooth series
model_noisy = ExponentialSmoothing()
model_noisy.fit(train_noisy)
pred_noisy = model_noisy.predict(len(val_smooth))
mape_noisy = mape(val_smooth, pred_noisy)


print(f"MAPE of model trained on smooth data {mape_smooth:.2f}")
print(f"MAPE of model trained on noisy data {mape_noisy:.2f}")


plt.figure(figsize=(10, 6))
plt.plot(smooth_ts.values() , label = "Original Time Series")
plt.plot(pred_smooth.values(),label = "Prediction from model trained on smooth data")
plt.plot(pred_noisy.values(),label = "Prediction from model trained on noisy data")
plt.title('Prediction performance of models trained on smooth vs. noisy data')
plt.legend()
plt.show()

```

**Snippet 2: Demonstrating Training Data Length Impact**

Here, we generate a simple sine wave with a yearly cycle. We then train two ARIMA models; the first trained on just the first few years, the second on a significantly longer time span. The snippet visualizes predictions of a future year. The model trained on less data struggles because it hasn’t seen the full annual cycle.

```python
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import ARIMA
from darts.metrics import mape
import matplotlib.pyplot as plt

# Generate a time series with a yearly cycle
t = np.arange(0, 10 * 365, 1) # 10 years of daily data
series = np.sin(2 * np.pi * t / 365)


# Create TimeSeries object
ts = TimeSeries.from_values(series)


# Split into train and test sets - 2,5, and 1 year to test
train_short = ts[:2*365]
train_medium = ts[:5*365]
test_set = ts[9*365:]


# Train with a small data set and make predictions
model_short = ARIMA(p=5, d=1, q=2)
model_short.fit(train_short)
pred_short = model_short.predict(len(test_set))

# Train with a medium data set and make predictions
model_medium = ARIMA(p=5, d=1, q=2)
model_medium.fit(train_medium)
pred_medium = model_medium.predict(len(test_set))

mape_short = mape(test_set, pred_short)
mape_medium = mape(test_set, pred_medium)


print(f"MAPE of model trained on 2 years data {mape_short:.2f}")
print(f"MAPE of model trained on 5 years data {mape_medium:.2f}")



plt.figure(figsize=(10, 6))
plt.plot(test_set.values(), label = "Original Time Series")
plt.plot(pred_short.values(), label = "Model trained on 2 years")
plt.plot(pred_medium.values(),label = "Model trained on 5 years")
plt.title('Prediction performance with short and medium training set')
plt.legend()
plt.show()
```

**Snippet 3: Demonstrating Data Quality Impact**

This last snippet uses a simple linear trend as a starting point. We create one training set with an outlier that is well outside the trend, then train a second series with a more smooth line. The prediction will suffer greatly with the noise, but perform better where the data is clean.

```python
import numpy as np
import pandas as pd
from darts import TimeSeries
from darts.models import RegressionModel
from darts.metrics import mape
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate a linear trend
t = np.arange(0, 100, 1)
smooth_series = t*1.5 + 10

# Generate noisy time series with an outlier
noisy_series = smooth_series.copy()
noisy_series[50] += 40


# Create TimeSeries objects
smooth_ts = TimeSeries.from_values(smooth_series)
noisy_ts = TimeSeries.from_values(noisy_series)


# Split into train and validation
train_smooth, val_smooth = smooth_ts.split_before(80)
train_noisy, val_noisy = noisy_ts.split_before(80)


# Train a linear regression on a smooth training set and predict
model_smooth = RegressionModel(LinearRegression())
model_smooth.fit(train_smooth)
pred_smooth = model_smooth.predict(len(val_smooth))
mape_smooth = mape(val_smooth, pred_smooth)

# Train a linear regression on noisy training set and predict
model_noisy = RegressionModel(LinearRegression())
model_noisy.fit(train_noisy)
pred_noisy = model_noisy.predict(len(val_smooth))
mape_noisy = mape(val_smooth, pred_noisy)



print(f"MAPE of model trained on clean data {mape_smooth:.2f}")
print(f"MAPE of model trained on noisy data {mape_noisy:.2f}")

plt.figure(figsize=(10, 6))
plt.plot(smooth_ts.values(), label = "Original Time Series")
plt.plot(pred_smooth.values(), label = "Prediction from model trained on smooth data")
plt.plot(pred_noisy.values(), label = "Prediction from model trained on noisy data")
plt.title('Prediction performance on clean vs. noisy training data')
plt.legend()
plt.show()
```

For further reading, I highly recommend "Time Series Analysis: Forecasting and Control" by George E.P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, and Jun Chen. It's a classic text that covers the fundamentals of time series modeling. Another great resource for a more practical approach, albeit not solely focused on darts, is "Forecasting: Principles and Practice" by Rob J Hyndman and George Athanasopoulos. These books provide a solid theoretical background and real-world insights that are critical for understanding how the training dataset impacts the final model performance.

In summary, achieving good performance on diverse time series using darts, or any time series modeling framework for that matter, requires careful consideration of the training data's similarity to the target data, the quality of the data (absence of noise, outliers), and the length of the training series. It's often an iterative process involving data exploration, preprocessing, and model evaluation, rather than a single straightforward approach.
