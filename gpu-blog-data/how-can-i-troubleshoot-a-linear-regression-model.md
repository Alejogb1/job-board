---
title: "How can I troubleshoot a linear regression model trained on train data?"
date: "2025-01-30"
id: "how-can-i-troubleshoot-a-linear-regression-model"
---
Linear regression models, while conceptually straightforward, often present challenges when translating from theory to practical application. Based on my experience debugging numerous machine learning pipelines, the root cause of poor performance in a linear regression model trained on training data can typically be traced back to issues in data quality, model assumptions, or overfitting, and a systematic approach to analysis is essential for effective troubleshooting.

**Understanding the Core Issues**

A linear regression assumes a linear relationship between the independent (predictor) variables and the dependent (target) variable. The model learns coefficients that define this linear relationship by minimizing a cost function, typically the mean squared error (MSE). If the underlying relationship is non-linear, or the data violate core assumptions such as homoscedasticity, the model will inevitably perform poorly, even on the training data. Overfitting is another crucial consideration. A model that fits the training data too closely might not generalize to unseen data because it has essentially memorized the training set's noise.

**A Systematic Debugging Process**

My typical troubleshooting process begins with data quality inspection followed by a meticulous analysis of model performance and diagnostics. The following points cover key elements I've found helpful.

*   **Data Inspection and Preparation:** Before even considering the model, I meticulously examine the training data. This involves:
    *   **Outlier Detection:** I identify and analyze outliers in both the predictor and target variables. These extreme values can unduly influence the regression coefficients, skewing the model's fit.
    *   **Data Distributions:** I assess the distributions of each variable. Highly skewed distributions can cause problems, and transformations, such as logarithmic scaling, might be necessary.
    *   **Missing Values:** I explicitly identify any missing values and decide on an appropriate strategy to deal with them – either imputation using statistical measures like mean or median, or removal.
    *   **Data Scaling:** Feature scaling or standardization is crucial for numerical stability of the algorithms and prevent the algorithm to learn relationships based on the scale of one feature over the other.
    *   **Feature Engineering:** The set of available data points may not fully represent the problem. Hence, feature engineering with expert domain knowledge can provide the model with better explanatory and predictive capabilities.

*   **Model Evaluation Metrics:** Evaluate the model using a comprehensive set of metrics beyond just MSE.
    *   **R-squared:** This value shows what percentage of variance in target is explained by the model. It helps to understand the goodness of fit. Low R-squared value may indicate poor performance.
    *   **Adjusted R-squared:** It is similar to R-squared but penalizes model complexity, especially if new features are being considered.
    *   **Mean Absolute Error (MAE):** MAE is more robust to outliers than MSE and helps us understand the magnitude of errors in the same units as the target variable.
    *   **Root Mean Squared Error (RMSE):** RMSE has a higher weight on large errors. It provides insights in the magnitude of errors.
    *   **Residual Analysis:** Visualizing the model's residuals – the differences between predicted and actual values – is very important. If residuals exhibit patterns, such as heteroscedasticity (non-constant variance) or non-linearity, it shows the assumptions of linear regression model are not met.
    *   **Residuals vs Predicted Plot:** This plot helps us understand if residuals are homoscedastic or heteroscedastic.
    *   **QQ-Plot:** This plot helps us understand if the residuals are normally distributed or not.

*   **Model Complexity and Overfitting:** Overfitting is a critical issue.
    *   **Regularization:** Regularization techniques, such as L1 (Lasso) and L2 (Ridge) regularization, add a penalty term to the cost function, discouraging excessively large coefficient values, and reducing model variance.
    *   **Cross-Validation:** Employ k-fold cross-validation to assess how well a model generalizes to unseen data. A significant gap between training performance and validation performance is a telltale sign of overfitting.

**Code Examples with Commentary**

To illustrate how these concepts translate to practice, consider these code examples using Python with common data science libraries.

**Example 1: Investigating Residuals**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create some example data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
X_train_sm = sm.add_constant(X_train) # Add a constant to the design matrix
model = sm.OLS(y_train, X_train_sm).fit()

# Calculate predictions and residuals on the training set
predictions = model.predict(X_train_sm)
residuals = y_train - predictions

# Plot residuals
plt.figure(figsize=(8, 6))
plt.scatter(predictions, residuals)
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.axhline(y=0, color='r', linestyle='--')
plt.show()
```

*Commentary:* This code snippet creates a synthetic dataset, fits a linear regression model using `statsmodels`, calculates training residuals, and then plots them. The red horizontal line indicates 0. Ideally, the residuals should be randomly scattered around zero. Patterns in the residual plot, such as a funnel shape or curvature, indicate that the linear regression assumptions might be violated.

**Example 2: Using Regularization**

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error

# Creating a dataset with a non-linear relationship to test the power of regularization
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + 2 * X**2 + np.random.randn(100, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature engineering to make model complex
poly = PolynomialFeatures(degree=5)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_poly)
X_test_scaled = scaler.transform(X_test_poly)

# Fit a Ridge model
ridge = Ridge(alpha=1) # Set regularization strength
ridge.fit(X_train_scaled, y_train)

# Evaluate the model
y_train_pred = ridge.predict(X_train_scaled)
y_test_pred = ridge.predict(X_test_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

print("Training RMSE:", train_rmse)
print("Test RMSE:", test_rmse)
```

*Commentary:* This example generates data with a non-linear underlying relationship and applies polynomial feature engineering to demonstrate how regularization is needed in cases of overfitting. It demonstrates the application of polynomial features and the impact of Ridge regularization, a form of L2 regularization. By comparing training and test RMSE, you would see how this reduces the test error and generalizes better than the standard regression on non-linear dataset.

**Example 3: Examining R-squared and Adjusted R-squared**

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Create some example data
np.random.seed(42)
X = 2 * np.random.rand(100, 3)  # Three predictor variables
y = 4 + 3 * X[:, 0] + 2 * X[:, 1] + 1* X[:,2] + np.random.randn(100, 1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit linear regression model
X_train_sm = sm.add_constant(X_train)
model = sm.OLS(y_train, X_train_sm).fit()

# Calculate R-squared and Adjusted R-squared on the training set
predictions = model.predict(X_train_sm)
r_squared = r2_score(y_train, predictions)
adjusted_r_squared = model.rsquared_adj

print(f"R-squared on the training data: {r_squared}")
print(f"Adjusted R-squared on the training data: {adjusted_r_squared}")
```

*Commentary:* Here, we create a multivariate dataset and fit a linear model using `statsmodels`. After the model fit, we examine the R-squared and Adjusted R-squared. If we add a non-informative feature to the model, the R-squared might go up, but Adjusted R-squared might go down showing that the added feature isn't improving the model.

**Resource Recommendations**

For a deeper dive, I recommend exploring resources dedicated to statistical modeling. Books on regression analysis, such as those by Kutner et al., or by Gelman et al., offer comprehensive coverage of the theoretical underpinnings and practical aspects of linear regression. Additionally, online resources dedicated to statistical learning with Python, using libraries like scikit-learn and statsmodels, will prove very valuable. Researching on regularized regression models and cross-validation is recommended for more advanced techniques. These resources can aid in developing a more sophisticated and nuanced understanding of model behavior and troubleshooting.
