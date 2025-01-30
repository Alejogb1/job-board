---
title: "Why does my model have low MAE and low R2 simultaneously?"
date: "2025-01-30"
id: "why-does-my-model-have-low-mae-and"
---
Low mean absolute error (MAE) and low R-squared (R²) simultaneously indicate a fundamental mismatch between your model's predictions and the true underlying data generating process, despite seemingly good point-wise prediction accuracy.  My experience in developing predictive models for financial time series, specifically in volatility forecasting, has shown this scenario arising consistently from specific data characteristics and modeling choices.  The key issue isn't simply a bad model, but rather a model that captures a specific, limited aspect of the data while failing to represent the overall relationship.

The low MAE suggests your model's predictions are numerically close to the actual values on average.  However, a low R² signifies a poor model fit, implying a significant portion of the variance in the target variable remains unexplained. This discrepancy arises when the model is accurate in its magnitude estimations, but completely fails to capture the overall trend or variability within the data.  This often occurs in datasets where the target variable has a strong non-linear component or exhibits heteroskedasticity (non-constant variance).  Furthermore, the presence of outliers, even a few, can significantly skew the R² metric, especially if your model is relatively insensitive to them, due to its local-optimizing nature.  Finally, a simple model applied to complex data will inherently generate low R².


Let's illustrate this with some examples.  Consider a dataset where the true relationship between the independent and dependent variable is highly complex, possibly non-linear or involving multiple interactions.


**Example 1:  Linear Model on Non-linear Data**

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Generate non-linear data
np.random.seed(42)
X = np.linspace(-5, 5, 100).reshape(-1, 1)
y = 2 * np.sin(X) + np.random.normal(0, 0.5, 100)

# Fit a linear model
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# Calculate MAE and R²
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")
```

This code generates data following a sinusoidal pattern and then fits a simple linear regression model.  The linear model will produce a relatively low MAE because it might capture the general magnitude of the data, predicting values relatively close to the true values on average.  However, it dramatically underperforms in capturing the curvature inherent in the data, leading to a very low R².  The low R² highlights the linear model's inability to capture the true underlying relationship.  This illustrates that low MAE and low R² can coexist when a simplistic model is applied to complex data.


**Example 2:  Outliers Impact on Model Fit**

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Generate data with an outlier
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + np.random.normal(0, 1, 100)
y[-1] = 100  # Outlier

X = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")
```

Here, we introduce a single outlier to an otherwise linearly related dataset. A linear regression will still achieve a relatively low MAE because it minimizes the sum of absolute errors. However, the outlier significantly impacts the R², pulling the regression line towards it and reducing the overall explained variance.  The model, while making reasonably accurate predictions for the bulk of the data, does not capture the overall trend due to the influence of outliers.  Robust regression methods could alleviate this problem.


**Example 3:  Heteroskedasticity and Model Assumptions**

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Generate data with heteroskedasticity
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 2 * X + X * np.random.normal(0, 1, 100)

X = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.2f}")
```

This example illustrates heteroskedasticity, where the error variance is not constant across the range of X.  A linear regression assumes homoskedasticity; violating this assumption leads to inefficient and potentially biased estimates.  While the MAE might still be reasonably low, the R² will be substantially lower because the model fails to capture the increasing variance in the data.  Transformations on the dependent variable or using weighted least squares can improve the model fit in such scenarios.


In conclusion, low MAE and low R² are not contradictory but rather indicative of a mismatch between your model and the data.  The model might predict values close to the actuals on average, but fails to capture the broader trends, variance, or non-linear relationships present in the data.  Diagnosing the exact reason requires a careful examination of the dataset for non-linearity, heteroskedasticity, outliers, and a thoughtful assessment of the model's suitability for the data.


**Resource Recommendations:**

*  Introductory texts on statistical modeling and regression analysis.  A strong understanding of the assumptions underlying different regression models is paramount.
*  Advanced texts covering non-linear modeling techniques, such as generalized additive models (GAMs) or neural networks.  These techniques offer greater flexibility for modeling complex data relationships.
*  Resources on robust regression methods to handle outliers effectively and reduce the impact of extreme values on model estimation.


Thorough data exploration and visualization are crucial initial steps.  Consider various diagnostic plots (residual plots, Q-Q plots) to detect deviations from model assumptions.   Remember to consider the limitations of the metrics used; low R² doesn't automatically mean a poor model, particularly when dealing with inherently noisy data or data with limitations in explanatory variables.  Choose models suitable for your specific problem and carefully assess the context of the low R².  It's a valuable indicator of model inadequacy, but it needs further investigation to understand precisely *why* the model is failing to capture the complete data relationship.
