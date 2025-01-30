---
title: "Do weighted fits lead to irrational predictions?"
date: "2025-01-30"
id: "do-weighted-fits-lead-to-irrational-predictions"
---
Weighted least squares, when improperly applied, can indeed lead to predictions that appear irrational, particularly if the underlying assumptions about data variances are incorrect. My experience building a predictive model for sensor calibration, where the inherent noise characteristics of each sensor varied significantly, provides a concrete example of this. The core principle lies in how weights impact the cost function and thus the resulting model parameters. In standard least squares regression, we minimize the sum of the squared errors between predicted and observed values. Weighted least squares modifies this by introducing weights associated with each data point. Specifically, the cost function becomes the sum of the squared errors multiplied by these weights. The aim is to give greater influence to data points with lower variance (higher precision) and less influence to those with higher variance.

The rationale behind this approach is statistically sound. When errors are heteroscedastic, meaning their variances are not constant across all data points, ordinary least squares produces parameter estimates that are unbiased, but less efficient than weighted least squares. Efficiency in this context refers to the variance of parameter estimates; higher variance implies less confidence in the accuracy of the parameters, and consequently, less confidence in the predictions. Weighted least squares seeks to mitigate this by down-weighting data points that are less reliable. However, this approach introduces a potential problem: the choice of weights significantly impacts the model. Ill-defined or incorrectly specified weights can cause the model to be overly sensitive to data points that are given undue importance.

Let’s examine the mechanics with a few examples using Python and `numpy`. Suppose we are trying to model a linear relationship where the error variance is not uniform across the dataset. The correct weighting approach would involve assigning weights that are inversely proportional to the variances of the error terms.

**Example 1: Correct Weighting**

Here, let’s assume we have data with a true linear relationship `y = 2x + 1` but with heteroscedastic noise. The variance increases with increasing values of `x`. We generate some noisy data and apply weights based on the simulated variances.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.linspace(1, 10, 20)
y_true = 2*x + 1
variances = 0.2 * x  # Increase variance with x
y = y_true + np.random.normal(0, np.sqrt(variances))

weights = 1 / variances

# Add a column of ones to x for the intercept
X = np.vstack((np.ones(len(x)), x)).T
# calculate wLS
w = np.diag(weights)
wX = np.matmul(w,X)
wy = np.matmul(w,y)
params = np.linalg.solve(np.matmul(X.T,wX),np.matmul(X.T,wy))


y_pred = np.matmul(X,params)

print("Parameters:", params)

plt.scatter(x, y, label='Data', color = 'blue')
plt.plot(x, y_pred, label='Weighted fit', color = 'red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Weighted Linear Regression')
plt.show()
```
In this example, the weighted least squares fitting correctly approximates the underlying linear relationship because the weights reflect the actual variances in the data.

**Example 2: Incorrect Weighting**
Now, consider a case where we incorrectly assume that the variance decreases with increasing x. We invert the weighting approach, applying larger weights to the data points that have larger variances, effectively creating an imbalance.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.linspace(1, 10, 20)
y_true = 2*x + 1
variances = 0.2 * x # Increase variance with x
y = y_true + np.random.normal(0, np.sqrt(variances))

# INCORRECT WEIGHTS
weights_incorrect = variances # Larger weights for larger variances

X = np.vstack((np.ones(len(x)), x)).T
# calculate wLS with incorrect weights
w = np.diag(weights_incorrect)
wX = np.matmul(w,X)
wy = np.matmul(w,y)
params_incorrect = np.linalg.solve(np.matmul(X.T,wX),np.matmul(X.T,wy))
y_pred_incorrect = np.matmul(X,params_incorrect)


print("Incorrect Parameters:", params_incorrect)

plt.scatter(x, y, label='Data', color = 'blue')
plt.plot(x, y_pred_incorrect, label='Incorrectly Weighted Fit', color = 'red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression with Incorrect Weights')
plt.show()
```

The result of incorrect weighting is a model that deviates significantly from the actual relationship. The regression line is pulled towards the points with high variance because those have been given undue influence in the fit, which results in an inaccurate prediction model, especially where the weights are highest.

**Example 3: Extreme Weighting (Outlier Dominance)**

Suppose that a single data point has a considerably lower expected variance and is consequently given an extraordinarily high weight. This scenario could arise in real datasets if a sensor experienced less noise for a brief period.

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
x = np.linspace(1, 10, 20)
y_true = 2*x + 1
variances = 0.2 * x
y = y_true + np.random.normal(0, np.sqrt(variances))

weights_extreme = 1 / variances # Correct weights
weights_extreme[10] = weights_extreme[10] * 100 # Give a data point extremely high weight

X = np.vstack((np.ones(len(x)), x)).T
# calculate wLS with extreme weight
w = np.diag(weights_extreme)
wX = np.matmul(w,X)
wy = np.matmul(w,y)
params_extreme = np.linalg.solve(np.matmul(X.T,wX),np.matmul(X.T,wy))
y_pred_extreme = np.matmul(X,params_extreme)

print("Extreme Weight Parameters:", params_extreme)


plt.scatter(x, y, label='Data', color = 'blue')
plt.plot(x, y_pred_extreme, label='Extreme Weighted Fit', color = 'red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression with Extreme Weighting')
plt.show()
```
Here, even a single outlier with an extremely high weight will disproportionately skew the model. The model almost pivots around this single, highly-weighted point. In real-world scenarios, such extreme weights could be the result of faulty variance estimates or poorly chosen prior assumptions about data reliability. The resulting predictions are very sensitive to the individual point, and are therefore likely unreliable.

The examples demonstrate that the use of weights must be justified. It's not simply a case of applying weights to any regression problem.  The weights must be informed by a deep understanding of the data and the error distribution. Without this foundation, weighted least squares can easily produce models that, in retrospect, will make little sense, yielding what could be considered ‘irrational predictions.’ The underlying problem is not with the method itself but with improper use. I have seen in several projects that an uncritical application of a technique, particularly one like weighted least squares that can amplify existing errors, can lead to very poor model performance.

For practitioners to avoid such issues, the following are important:
1. **Carefully evaluate error variance**: Before applying weighted least squares, rigorously examine if heteroscedasticity is present in the data. This examination may require plotting residuals, using statistical tests, or employing domain knowledge to inform your understanding of the error structure.
2. **Proper weight calculation**: Choose appropriate weights that are inversely proportional to the estimated variances, or some other appropriate metric. Avoid assigning arbitrary weights or those that are not grounded in the underlying statistical framework.
3. **Sensitivity analysis**: After creating a weighted model, perform sensitivity analysis to see how model output changes with varying weights. This analysis can highlight if a model has become overly influenced by a particular data point or subset of data.
4. **Outlier detection**: Inherent in the proper use of weighted least squares is the assessment of whether the weights assigned are reasonable and meaningful, which implicitly makes the approach robust to outliers. Outlier removal should be considered if weighting is not sufficient.

I recommend further study of generalized least squares and the implications of heteroscedasticity.  Consulting texts focusing on statistical modeling and linear regression will deepen your understanding of how to effectively implement weighted least squares techniques, while remaining aware of its pitfalls. Reviewing materials on regression diagnostics is also vital in helping you determine if your model assumptions are valid and your chosen approach has been correct.
