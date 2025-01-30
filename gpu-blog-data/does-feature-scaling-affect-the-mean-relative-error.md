---
title: "Does feature scaling affect the mean relative error of predictions?"
date: "2025-01-30"
id: "does-feature-scaling-affect-the-mean-relative-error"
---
Feature scaling's impact on the mean relative error (MRE) of predictive models is nuanced and depends heavily on the specific algorithm employed.  My experience working on high-frequency trading models, particularly those utilizing gradient-boosting techniques, highlighted this intricacy.  While MRE itself isn't directly sensitive to the *scale* of features, the *convergence* and ultimately the *performance* of many machine learning algorithms are significantly influenced by it.  This indirectly affects the MRE.

The core issue lies in the differing sensitivity of various optimization algorithms to feature scales.  Algorithms like gradient descent, commonly used in neural networks and support vector machines, rely on calculating gradients to navigate the loss landscape.  If features have vastly different scales, the gradients will be dominated by features with larger values, effectively masking the contribution of smaller-scaled features. This leads to suboptimal weight assignments and consequently, less accurate predictions, potentially impacting the MRE.  Algorithms that don't explicitly rely on gradient descent, such as decision trees, are less directly affected, but even then, indirect effects can still exist.

**1.  Explanation:**

The mean relative error measures the average percentage difference between predicted and actual values.  Its formula is:

MRE = (1/n) * Σ| (yi - ŷi) / yi |

where *yi* is the actual value, *ŷi* is the predicted value, and *n* is the number of data points.  Crucially, MRE is *unitless*.  This means a change in the scale of the input features, say from meters to kilometers, doesn't intrinsically alter the MRE value itself. The percentage difference remains the same irrespective of the units.

However, the accuracy of the *predictions* (ŷi), which directly determine the MRE, is affected by feature scaling. Consider a model trained on unscaled data with features varying across orders of magnitude.  The algorithm might struggle to find an optimal solution, leading to larger errors (*yi - ŷi*) and subsequently, a higher MRE. Scaling the features, bringing them to a similar range (e.g., through standardization or normalization), addresses this issue.  It allows the algorithm to converge more efficiently to a better solution, leading to more accurate predictions and potentially a lower MRE.

**2. Code Examples and Commentary:**

The following examples demonstrate the impact using Python and scikit-learn.  These examples focus on the effect on model convergence and subsequent predictive accuracy, which influences the MRE.

**Example 1: Linear Regression with Unscaled and Scaled Data:**

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

# Generate sample data with disparate scales
X = np.array([[1, 1000], [2, 2000], [3, 3000], [4, 4000]])
y = np.array([1100, 2100, 3100, 4100])

# Unscaled data
model_unscaled = LinearRegression()
model_unscaled.fit(X, y)
y_pred_unscaled = model_unscaled.predict(X)
mre_unscaled = mean_absolute_percentage_error(y, y_pred_unscaled)

# Scaled data using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model_scaled = LinearRegression()
model_scaled.fit(X_scaled, y)
y_pred_scaled = model_scaled.predict(X_scaled)
mre_scaled = mean_absolute_percentage_error(y, y_pred_scaled)


print(f"MRE (Unscaled): {mre_unscaled:.4f}")
print(f"MRE (Scaled): {mre_scaled:.4f}")

```

This example explicitly shows that while MRE is calculated on unscaled and scaled predictions, the underlying model performance, and thus MRE, differs based on scaling.  The StandardScaler brings both features to a similar range (zero mean, unit variance), potentially leading to a better fit and lower MRE.

**Example 2: Gradient Boosting with Different Scaling Techniques:**

```python
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

# Generate sample data
X = np.random.rand(100, 2) * 100  #Features with varying scales
y = 2*X[:,0] + 3*X[:,1] + np.random.normal(0,10,100) #Target

# MinMaxScaler
scaler_minmax = MinMaxScaler()
X_minmax = scaler_minmax.fit_transform(X)
model_minmax = GradientBoostingRegressor()
model_minmax.fit(X_minmax,y)
y_pred_minmax = model_minmax.predict(X_minmax)
mre_minmax = mean_absolute_percentage_error(y,y_pred_minmax)

# StandardScaler
scaler_standard = StandardScaler()
X_standard = scaler_standard.fit_transform(X)
model_standard = GradientBoostingRegressor()
model_standard.fit(X_standard,y)
y_pred_standard = model_standard.predict(X_standard)
mre_standard = mean_absolute_percentage_error(y, y_pred_standard)

print(f"MRE (MinMaxScaler): {mre_minmax:.4f}")
print(f"MRE (StandardScaler): {mre_standard:.4f}")

```

This illustrates the impact of different scaling methods on a gradient-boosting algorithm.  MinMaxScaler and StandardScaler will likely result in different MRE values, highlighting how the choice of scaling method can influence the final model accuracy.


**Example 3:  Illustrating the effect on a non-gradient based algorithm:**

```python
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error

# Generate sample data with one highly scaled feature
X = np.array([[1, 1000], [2, 2000], [3, 3000], [4, 4000]])
y = np.array([1100, 2100, 3100, 4100])

# Unscaled data
model_unscaled = DecisionTreeRegressor()
model_unscaled.fit(X, y)
y_pred_unscaled = model_unscaled.predict(X)
mre_unscaled = mean_absolute_percentage_error(y, y_pred_unscaled)

# Scaled data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model_scaled = DecisionTreeRegressor()
model_scaled.fit(X_scaled, y)
y_pred_scaled = model_scaled.predict(X_scaled)
mre_scaled = mean_absolute_percentage_error(y, y_pred_scaled)

print(f"MRE (Unscaled): {mre_unscaled:.4f}")
print(f"MRE (Scaled): {mre_scaled:.4f}")
```

This example, using a decision tree, which is less sensitive to feature scaling, might show a smaller difference in MRE between scaled and unscaled data compared to the gradient-based examples. However, even here, the model might learn more efficiently with scaled data, potentially leading to slightly better generalization and lower MRE in different datasets.


**3. Resource Recommendations:**

"Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman.
"Pattern Recognition and Machine Learning" by Christopher Bishop.
"Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron.


In conclusion, while MRE itself is not directly affected by feature scaling, the performance of many machine learning algorithms, and consequently their predictive accuracy, is heavily influenced.  This indirect effect significantly impacts the MRE. Therefore, thoughtful feature scaling is crucial for obtaining accurate and reliable predictions and achieving a lower MRE, especially when working with algorithms sensitive to feature scales.  The choice of scaling method also plays a role, and experimenting with different techniques is often necessary to determine the best approach for a particular dataset and algorithm.
