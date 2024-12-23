---
title: "durbin watson statistic in r explanation?"
date: "2024-12-13"
id: "durbin-watson-statistic-in-r-explanation"
---

 so you want to know about the Durbin-Watson statistic in R and why you might use it right? I get it I've been there staring at residuals like they hold the secrets to the universe so lets dive in

First off Durbin-Watson its a test for autocorrelation specifically in the residuals of a regression model Think of it like this you fit a model and you get these errors the stuff your model couldn't explain Durbin-Watson checks if those errors are hanging out with each other in a way they shouldn't you know if they are correlated over time or space or whatever your independent variable is

Now why do we care if the errors are correlated? Well its a big deal violates one of the fundamental assumptions of linear regression and makes the whole thing suspect you're assuming the residuals are independent but if they are showing trends that is trouble your model might be underestimating the uncertainty or worse it could be giving you entirely bogus results So you need Durbin-Watson to tell you if you're in trouble with this correlation business I remember one time I was building a time series model for predicting website traffic I had this beautiful looking model but the Durbin-Watson was screaming I spent a day just tearing my hair out before I figured out the seasonality was killing me I needed to take account of that trend or the model was garbage This led me to start looking into seasonal ARIMA models

The test itself gives you a value between 0 and 4 the sweet spot where there is not an issue is right around 2 Values much lower than 2 tend to suggest positive autocorrelation the errors are related in a way that if one is positive the next is more likely to be positive too and vice versa Values much higher than 2 suggest negative autocorrelation so an error being positive makes the next error more likely to be negative I know this feels confusing to read but it is important to understand why it works like that

So How do you do this in R lets fire up some code right

```R
# Example data let's use something easy to see in case you want to try to play around with it

set.seed(42)
x <- 1:100
y <- 2*x + rnorm(100,0,10)  # a good model
bad_y <- 2*x + cumsum(rnorm(100,0,5))  # Bad model errors are correlated because of cumsum

# Fit the linear models
model_good <- lm(y ~ x)
model_bad <- lm(bad_y ~ x)

# Calculate Durbin-Watson for each model
dw_good <- lmtest::dwtest(model_good)
dw_bad <- lmtest::dwtest(model_bad)

# Output the results
print("Durbin-Watson for the well-behaved model:")
print(dw_good)
print("Durbin-Watson for the badly behaved model:")
print(dw_bad)
```

So in this example code we use `lmtest::dwtest()` a very common package you want to get familiar with if you are going to be dealing with these kinds of analyses and the results you should be able to notice right away that the 'dw' value is near 2 for the 'good' model and far away for the 'bad' one

The first model I created should not be showing major autocorrelation and the Durbin Watson test confirms that the second one it is pretty bad so if you get a similar value I would be very concerned and you should investigate further your model might have serious issues as mine did when I was working on that website traffic problem

But you know that Durbin Watson is not a magic bullet it assumes that the errors are normally distributed and that the relationship between the variables is actually linear You know assumptions assumptions right they always come back to haunt you And also it only checks for first order autocorrelation if the error is related to the error from the last observation it is not going to find more complex dependencies this is key to understand its limitations its a good first check not the whole deal.

Also something very common for those new to the subject is that they see that the Durbin Watson statistic is a 'p value' and they will be like 'its not significant' and they don't go deeper into it because they don't understand that you actually want to see if the value is near to 2 not that its significant like a hypothesis test would be you need to understand that you are not testing a hypothesis you are testing the error structure of your model so treat it as such

Now lets go to another example this one is going to be a bit more complex as sometimes errors might be caused by more complex factors

```R
# More complex example with time series data
set.seed(123)
time_series <- 1:200
# Generate a time series with some autocorrelation
errors <- arima.sim(n = 200, model = list(ar = c(0.8)))
noisy_time_series <- 2 * time_series + errors
# Fit model
model_ts <- lm(noisy_time_series ~ time_series)
# Check for autocorrelation
dw_ts <- lmtest::dwtest(model_ts)
print("Durbin-Watson for the time series data:")
print(dw_ts)

# Attempt to fix autocorrelation by using lag
lag_noisy_time_series <- lag(noisy_time_series)
model_ts_lag <- lm(noisy_time_series ~ time_series + lag_noisy_time_series)
dw_ts_lag <- lmtest::dwtest(model_ts_lag[-1,])
print("Durbin-Watson after attempting to add a lag:")
print(dw_ts_lag)

```

So in this code snippet I create an artificial time series that has a pretty heavy autocorrelation component just so you can see what a bad one looks like. I use `arima.sim` which generates data according to an Autoregressive model and you can see that after generating a model with those correlated errors that the Durbin-Watson test goes low.
Then to correct this we try adding a lag of the noisy time series as an independent variable this tries to model that autocorrelation and if you check again the Durbin Watson should be improved significantly with this measure now it is closer to 2 and therefore we are closer to a good solution.

I used to get so frustrated with autocorrelation because it was messing with the models I made I remember one case where I was doing a study on the effectiveness of different marketing strategies and my model was completely giving me weird correlations after I adjusted for the autocorrelation all of my results shifted its was wild I even learned to be suspicious of good results because you should be able to back your findings up and not just rely on some good numbers

One of the most frustrating parts of this for most people is that there is no one way to fix these problems it is like a choose your own adventure book you get a bad reading you try something and see how it goes there is a lot of experimenting

And now the last example this one might be a bit more 'techy' and I know you guys love these type of issues this time we check for autocorrelation using a more complex dataset

```R
# Example with a real-world dataset (using data from the `tsibble` package)
if(!require(tsibble)){
  install.packages("tsibble")
  library(tsibble)
}

data(gafa_stock)
# Filter for a single company
aapl_data <- gafa_stock %>%
    filter(Symbol == "AAPL")

# Convert the data to a tsibble
aapl_ts <- as_tsibble(aapl_data, index=Date)

# Fit a simple model
model_aapl <- lm(Close ~ index(aapl_ts), data = aapl_ts)
# Check autocorrelation in residuals
dw_aapl <- lmtest::dwtest(model_aapl)
print("Durbin-Watson for the Apple Stock example")
print(dw_aapl)

# Another approach using time series models to see the difference
library(forecast)
model_ts_aapl <- auto.arima(aapl_ts$Close)
residuals_ts_aapl<- residuals(model_ts_aapl)
dw_ts_aapl <- lmtest::dwtest(residuals_ts_aapl ~1)
print("Durbin-Watson after trying auto.arima")
print(dw_ts_aapl)

```

Here we use a real world stock dataset from the `tsibble` package this is a common practice when dealing with financial data we filter the data for AAPL stock data and try to model the closing price with the index which is the date of the reading then we do a Durbin-Watson and see that the result is indeed very low which suggests some time related autocorrelation then to try to solve it we use `auto.arima` which automatically selects the best ARIMA model for our time series we then get the residuals and do a Durbin-Watson test again and the value this time should be closer to 2 which shows we were able to model the autocorrelation more accurately.

So basically thats all there is to know about Durbin-Watson a tool to verify if your residuals are well behaved and help you understand if your regression model has problems remember if that dw value is near 2 you are fine if it's far away from 2 your model residuals are most likely correlated in some way and you should investigate your model more

There is much more depth to all of this you should also learn about other tests such as the Breusch-Godfrey test to check more complex forms of autocorrelation and partial autocorrelation functions if you want to really understand what is going on in your time series this will also help you learn to solve these issues in more sophisticated ways

If you want to dig deeper into this I highly recommend checking out "Time Series Analysis and Its Applications" by Robert H Shumway and David S Stoffer it is an amazing book that will help you understand all these topics it goes super deep into the math and intuitions behind these concepts and it has lots of examples using R and another book that is more focused on linear regression and its assumptions is "Applied Linear Statistical Models" by Michael H Kutner Christopher J Nachtsheim John Neter and William Li this is a great foundational book if you are still a bit new to the topic

And a little joke before we wrap this up why did the regression model break up with the residuals? Because they just couldn't find a constant relationship haha get it

good luck with your modeling and if you have more questions come back anytime
