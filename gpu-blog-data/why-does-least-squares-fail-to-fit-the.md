---
title: "Why does least squares fail to fit the data?"
date: "2025-01-30"
id: "why-does-least-squares-fail-to-fit-the"
---
The failure of least squares regression to accurately model data stems fundamentally from its inherent assumptions about the data and error distributions, assumptions that are often violated in real-world datasets. Specifically, the ordinary least squares (OLS) method relies on several key premises: that the relationship between independent and dependent variables is linear; that the errors (residuals) are normally distributed with a mean of zero; that these errors exhibit homoscedasticity (constant variance); and that the errors are independent of each other and the predictors. When these assumptions are significantly breached, the resulting regression model may provide a poor fit, biased parameter estimates, and unreliable predictions. My experience in developing predictive models for environmental sensor data has consistently highlighted these limitations. I've observed seemingly robust models failing dramatically when confronted with outliers, non-linear trends, or data exhibiting time-series autocorrelation.

The most common reason for least squares failure is the violation of the linearity assumption. OLS regression attempts to fit a straight line (or hyperplane in higher dimensions) to the data. If the true relationship is non-linear – for instance, quadratic, exponential, or logarithmic – the linear model will inevitably deviate significantly from the observed data, particularly at the extremes. This deviation manifests as large residuals and a poor overall fit. For instance, consider the problem of modeling the growth rate of a population based on resource availability. Initially, growth might be nearly exponential, which is a distinctly non-linear relationship. Forcing a linear model in such a scenario would result in gross underestimation at lower resource levels and overestimation at higher levels. Even transformations to the input and output variables might only provide a marginal improvement if the underlying relationship is inherently non-linear.

Another frequent cause of OLS inadequacy is the presence of outliers. The least squares algorithm is sensitive to extreme data points. Since it attempts to minimize the *sum of squared errors*, outliers – that is, observations far removed from the general trend – exert a disproportionately large influence on the regression line, pulling it away from the general direction of the bulk of the data. In my work with financial market data, I regularly encountered this issue. Anomalous price movements, stemming from significant external events, would act as outliers, causing a linear model trained on typical data to perform poorly during volatile periods. Instead of capturing the trend of regular price movement, the model often overcompensated for outlier positions, leading to inaccurate predictions during such events.

Furthermore, heteroscedasticity—non-constant error variance—can invalidate the assumption of consistent error distribution. In a homoscedastic environment, the errors have a constant variance across all levels of predictor variables. However, in situations exhibiting heteroscedasticity, the variance changes, often increasing with the magnitude of predictors. For example, in modeling the fuel consumption of a vehicle, the variability in the fuel consumption measurements might increase significantly at higher speeds. This non-constant error variance violates the OLS assumption. The model, because it weighs the errors equally, will be disproportionately influenced by the low-variance data points and will not be able to properly capture the relationships in regions with higher variance. This produces a biased regression fit, and its confidence intervals are unreliable.

Finally, correlated errors, often seen in time-series data, can significantly disrupt the assumptions of independent errors. When data are correlated (e.g., data points close to each other in time have similar errors), the standard errors of the regression coefficients become inaccurate. Autocorrelation, a particular type of correlation where errors are related to previous errors, leads to inflated R-squared values, giving a false sense of accuracy and statistical significance. My attempts to predict network traffic patterns, for example, often fell victim to this issue. Daily traffic volumes are generally correlated, exhibiting a pattern where today’s volume is heavily influenced by the volumes observed on recent days. Directly applying OLS on such data, without explicit consideration of this autocorrelation, will lead to highly unreliable forecasts and coefficient estimates.

To illustrate these problems, consider the following Python code snippets:

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Scenario 1: Non-linear relationship
X = np.linspace(0, 10, 100)
y = 2 * X**2 + 3*X + np.random.normal(0, 20, 100) # Quadratic relationship
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)
y_pred = model.predict(X.reshape(-1, 1))
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.title("OLS Failing with Non-Linear Data")
plt.show()

```

*Commentary:* In this example, the underlying data-generating process is quadratic. The linear model cannot capture this curvature and produces a poor fit. The scatter plot of the original data shows a parabolic trend while the superimposed linear fit demonstrates the model's poor ability to represent the data. The residuals (not displayed) would be large in this case and not normally distributed.

```python
# Scenario 2: Outliers
X = np.linspace(0, 10, 100)
y = 2 * X + np.random.normal(0, 1, 100) # Linear relationship
y[np.random.randint(0, 100)] += 40    # Introduce an outlier
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)
y_pred = model.predict(X.reshape(-1, 1))
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.title("OLS Affected by Outliers")
plt.show()
```

*Commentary:* Here, the data has an underlying linear trend, but an outlier is introduced. The least squares model shifts to accommodate the outlier, distorting the fit on the rest of the data. The linear fit is skewed by the outlier. Instead of accurately representing the trend of the data, the line has noticeably changed its angle and is further away from the data.

```python
# Scenario 3: Heteroscedasticity
X = np.linspace(0, 10, 100)
errors = np.random.normal(0, X/2, 100) # Error variance increases with X
y = 2 * X + errors
model = LinearRegression()
model.fit(X.reshape(-1, 1), y)
y_pred = model.predict(X.reshape(-1, 1))
plt.scatter(X, y)
plt.plot(X, y_pred, color='red')
plt.title("OLS Failing with Heteroscedasticity")
plt.show()
```

*Commentary:* This code demonstrates heteroscedasticity by making the error variance directly proportional to the magnitude of X. The fitted regression line is not accurately representing the relationship because the model treats all errors equally, even though the spread in Y values becomes progressively larger as the value of X increases. The error bars around the mean trend would be larger for higher values of x.

To address these failures, several alternative methods and preprocessing steps can be employed. For non-linear relationships, non-linear regression models, such as polynomial regression, can be explored. Alternatively, transformations of the variables might make the relationship more linear. For handling outliers, robust regression methods like RANSAC or techniques such as median regression, can mitigate the influence of outliers. When heteroscedasticity is detected, weighted least squares (WLS) or transformations on the dependent variable can be utilized. For handling correlated errors, techniques from time series analysis, such as ARIMA models, might be necessary. Thorough evaluation using diagnostics, such as residual plots, is crucial to identify the specific type of failure and to determine the appropriate corrective action.

For deeper understanding, I recommend examining texts on statistical modeling such as “An Introduction to Statistical Learning” by Gareth James et al. This resource provides a broad overview of different regression methods and their assumptions. Additionally, more focused textbooks on time series analysis, such as “Time Series Analysis and Its Applications” by Robert H. Shumway and David S. Stoffer, will help address the issue of correlated errors and related techniques. Books focusing on robust statistics can provide insights into dealing with data impacted by outliers. These resources emphasize the importance of critically evaluating the assumptions underlying statistical methods and the need for adapting those methods or using alternative techniques when those assumptions are violated. The specific steps should be informed by a thorough examination of the data.
