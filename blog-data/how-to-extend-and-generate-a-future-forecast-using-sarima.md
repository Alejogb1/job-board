---
title: "How to extend and generate a future forecast using SARIMA?"
date: "2024-12-15"
id: "how-to-extend-and-generate-a-future-forecast-using-sarima"
---

alright, so you're looking into extending a time series forecast using sarima, right? i've been down that road more times than i care to remember, and let me tell you, it can get pretty hairy if you don't watch your step. let's break this down, no bs.

first off, sarima (seasonal autoregressive integrated moving average) is a pretty solid choice for forecasting when you've got that seasonality going on in your data. it's not magic, though. you still gotta know what you're doing. i've seen too many folks just throw a sarima model at data and pray it works. spoiler: it rarely does without a good bit of tweaking.

when i first started messing with this, i remember banging my head against the wall with a sales forecasting project for a small online retailer. their sales data had a clear weekly pattern – weekends were bonkers, weekdays were chill. the simple arima models i tried were just plain awful because they couldn't handle the seasonal swing. that's when i got properly introduced to sarima and its glorious (and occasionally frustrating) ways.

the core idea here is that sarima expands upon arima by including additional terms to account for the seasonality. where an arima model has p, d, and q parameters, which represent the autoregressive (ar) order, the differencing order, and the moving average (ma) order, respectively. a sarima model goes a step further and uses p, d, and q and then the parameters for the seasonality, usually called p_seasonal, d_seasonal, and q_seasonal and a seasonality period, usually called 's'. these are sometimes written like sarima(p,d,q)(p_seasonal,d_seasonal,q_seasonal)_s.

basically, you model both the non-seasonal and seasonal parts of your time series data.

now, generating the future forecast part. the key thing here isn't just fitting the model, but making sure it's stable and that the parameters you've found make sense. overfit the training data and your predictions for the future are going to look terrible, not much better than a random guess. believe me, i've seen the aftermath of overly confident models.

so, let's talk about how i usually tackle this. first, data prep is huge. clean that data, remove outliers if necessary, and make sure you handle any missing values gracefully. i'm a fan of using forward fills or interpolation in some cases, depending on how the gaps occurred in my data.

next up, figuring out the (p, d, q) and (p_seasonal, d_seasonal, q_seasonal, s) parameters. this is the part where things get a bit artful and it's why i call this part 'tuning the radio'. there are tools like auto arima functions, but i've found them not always be the best choice. they are useful to get a general sense of where to start, but never blindly accept their output. some resources, such as "time series analysis" by james d. hamilton and "forecasting: principles and practice" by hyndman and athanasopoulos, provide great insights into how to interpret autocorrelation and partial autocorrelation plots (acf and pacf) to estimate these parameters manually. it is an old school technique but its essential.

once you have a model that seems alright, you need to actually generate the forecast. this involves feeding in your historical data, letting the model learn the patterns, and then using the learned parameters to predict into the future. the number of steps ahead you forecast is up to you, but be realistic about the uncertainty; long-term predictions are always less accurate. i’ve had some projects where i needed to predict just one week ahead and others where i needed to predict several months.

let me give you a python example using the `statsmodels` library:

```python
import pandas as pd
from statsmodels.tsa.statespace.sarimax import sarimax
from statsmodels.datasets import co2
import matplotlib.pyplot as plt

# load data
data = co2.load_pandas().data
data = data['co2'].interpolate().ffill()

# split the data into train/test
train = data[:-30]
test = data[-30:]

# (p, d, q)(p_seasonal, d_seasonal, q_seasonal)_s
model = sarimax(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()

# forecast into future
forecast_steps = len(test)
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean
forecast_ci = forecast.conf_int()

# plotting results
plt.figure(figsize=(12,6))
plt.plot(train.index, train, label='train',color='blue')
plt.plot(test.index, test, label='test',color='green')
plt.plot(test.index, forecast_mean, label='forecast',color='red')
plt.fill_between(test.index,forecast_ci.iloc[:,0],forecast_ci.iloc[:,1], color='pink',alpha=0.2)
plt.legend()
plt.show()

```

this snippet loads some data, interpolates some values, splits into train and test, and fits the model, then gets the forecast and displays it. remember to install statsmodels, pandas and matplotlib, `pip install statsmodels pandas matplotlib`.

this part is usually pretty straightforward. it's all the steps before this that require most of the experience, and a lot of testing. if your model is not fit correctly the forecasting step will not give good results.

validation is super important. using your test data (or even better a separate validation set) you can check how well the forecasts align with actual historical values. i always use multiple metrics here, not just one. mean squared error (mse), root mean squared error (rmse), mean absolute error (mae), mean absolute percentage error (mape), and several others all bring different perspectives on how well your model is working.

here's a simple example of how to validate:

```python
import pandas as pd
from statsmodels.tsa.statespace.sarimax import sarimax
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.datasets import co2
import numpy as np

# load data
data = co2.load_pandas().data
data = data['co2'].interpolate().ffill()

# split the data into train/test
train = data[:-30]
test = data[-30:]


# (p, d, q)(p_seasonal, d_seasonal, q_seasonal)_s
model = sarimax(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
model_fit = model.fit()


forecast_steps = len(test)
forecast = model_fit.get_forecast(steps=forecast_steps)
forecast_mean = forecast.predicted_mean

# validation
mse = mean_squared_error(test, forecast_mean)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test, forecast_mean)


print(f"mse: {mse}")
print(f"rmse: {rmse}")
print(f"mae: {mae}")

```

this snippet is not very different from the previous one, it just performs validation on the forecasting, instead of displaying it on a graph.

something that i've learned the hard way is not to ignore the residuals of your model. they can tell you a lot about what the model is missing. if they show patterns, especially autocorrelation, it might mean that your model is not capturing all the data. this usually means you need to adjust your parameters or maybe even consider a different modeling approach entirely. you should study the diagnostic plots that `statsmodels` provides, they are very useful. also, you can google 'sarima diagnostics' to find very good explanations.

another problem i faced was when data was not stationary, or even worse when it had seasonality that was not constant through time. i ended up using other models for those situations or even transforming the data and modeling the transformed data, this topic is way too complex to discuss it here in just a few paragraphs, but keep it in mind as something you might have to deal with in real situations.

in conclusion, extending a forecast with sarima takes more than just running a few lines of code. it's about deeply understanding your data, carefully selecting your model parameters, and rigorously validating your results. oh, and if your prediction is wildly off, don’t blame the model, it's just doing what you told it to do. it is the model you built that probably had some assumptions that did not hold water. it is like blaming a hammer when you miss a nail, it is you who is misusing the hammer. remember a good model is not a perfect model, it's a model that's fit for purpose. and that includes the purpose of extending a forecast into the future.
