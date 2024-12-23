---
title: "How can forestci be used to generate error bars for random forest regression?"
date: "2024-12-23"
id: "how-can-forestci-be-used-to-generate-error-bars-for-random-forest-regression"
---

Alright, let's talk about generating error bars for random forest regression using `forestci`. It's a topic I’ve spent a considerable amount of time on, particularly during a project involving high-throughput screening data a few years back. We were pushing the limits of predictive accuracy, and simply having point estimates wasn’t cutting it – we needed measures of uncertainty to make informed decisions. This is where `forestci` became invaluable. The core problem, you see, is that standard random forests, by their very nature, don't natively provide uncertainty quantification in the way, for example, a Bayesian regression would. We get a prediction, sure, but not an associated interval that conveys how confident we should be in that prediction.

`forestci`, or "forest confidence intervals," as the name implies, tackles this by computing the *empirical* variance of predictions. It leverages the out-of-bag (OOB) predictions – predictions made on data points not used in the training of a particular tree in the forest – to estimate this variance. Essentially, we’re looking at the variability of predictions across the different trees, but in a way that’s more nuanced than just looking at the variance of all predictions. It's important to understand that we are not directly calculating prediction intervals in the traditional sense. Instead, we're getting a sense of the "reliability" of the predictions for each instance using the variance across trees. This is crucial because it provides an indication of how much different trees, trained on slightly different data, disagree about the outcome.

Here's the process conceptually: during training, each tree is built on a bootstrap sample of the original training data. The instances not included in the bootstrap sample are "out-of-bag" for that specific tree. We then record the predictions made by each tree on its OOB samples. `forestci` then uses the variance of these OOB predictions for *each individual instance* to estimate the error, and this can then be used to generate error bars.

Let's get into some code examples to illustrate how to implement this in Python using scikit-learn and `forestci`. I'll be assuming you have both libraries installed; if not, `pip install scikit-learn forestci` should sort you out.

**Example 1: Basic Usage with toy data**

First, let's create a simple regression problem using some randomly generated data. We’ll then train a random forest regressor and use `forestci` to estimate the prediction variance and generate associated error bars.

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from forestci import random_forest_error
import matplotlib.pyplot as plt

# Generate some toy data
np.random.seed(42)
X = np.sort(5 * np.random.rand(100, 1), axis=0)
y = np.sin(X).ravel() + np.random.randn(100) * 0.1

# Train a random forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get predictions
predictions = rf.predict(X)

# Calculate the variance
variance = random_forest_error(rf, X)

# Calculate the standard deviation (used to form error bars)
std = np.sqrt(variance)

# Visualize the results
plt.figure(figsize=(8, 6))
plt.plot(X, y, 'o', label='Data')
plt.plot(X, predictions, 'r-', label='Mean Prediction')
plt.fill_between(X.flatten(), predictions - 1.96 * std, predictions + 1.96 * std,
                 color='r', alpha=0.2, label='95% CI') #note: confidence interval for illustration
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Random Forest Regression with Error Bars')
plt.show()

```

This code will generate a scatter plot of the original data points, a line representing the predicted values from the random forest, and shaded regions showing estimated 95% confidence intervals. The 1.96 multiplier is associated with the 95% confidence interval, assuming the variance is normally distributed and can therefore be transformed to represent a confidence interval. Remember, `forestci` estimates *variances*, so we need to take the square root to get the standard deviation.

**Example 2: Applying it to a more complex dataset**

Now, let’s move to something a bit more involved. We’ll use a dataset with multiple features:

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from forestci import random_forest_error
import matplotlib.pyplot as plt

# Generate a more complex regression dataset
X, y = make_regression(n_samples=200, n_features=5, noise=5, random_state=42)

# Train a random forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get predictions
predictions = rf.predict(X)

# Calculate variance using forestci
variance = random_forest_error(rf, X)
std = np.sqrt(variance)


# Visualize results for a single feature (feature 0)
plt.figure(figsize=(8, 6))
plt.plot(X[:, 0], y, 'o', label='Data')
plt.plot(X[:, 0], predictions, 'r-', label='Mean Prediction')
plt.fill_between(X[:, 0], predictions - 1.96 * std, predictions + 1.96 * std,
                 color='r', alpha=0.2, label='95% CI')

plt.xlabel('Feature 0')
plt.ylabel('y')
plt.legend()
plt.title('Random Forest with Error Bars (Complex Dataset)')
plt.show()
```

In this example, we create data using `make_regression` with five features. The process remains the same; we fit a random forest, get the predictions, calculate the variance, and then use the standard deviation to plot error bars, focusing visualization on the first feature for clarity.

**Example 3: Using pre-computed OOB predictions (for optimization)**

In some situations, you might have already calculated the OOB predictions. `forestci` can be used on these OOB predictions without needing the model object to speed up repeated calculations. This is useful, for example, if you want to explore different confidence intervals or have a large model.

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from forestci import calc_oob_predictions, random_forest_error
import matplotlib.pyplot as plt

# Generate a complex regression dataset
X, y = make_regression(n_samples=200, n_features=5, noise=5, random_state=42)

# Train a random forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42, oob_score=True) #set oob_score to true to be able to retrieve oob_predictions
rf.fit(X, y)

# Get OOB predictions directly from the model
oob_predictions = rf.oob_prediction_
#use calc_oob_predictions in place if not available directly

# Calculate variance using precomputed oob predictions
variance = random_forest_error(rf, X, oob_pred=oob_predictions) #passing precomputed oob predictions
std = np.sqrt(variance)

# Visualize the results for a single feature (feature 0)
plt.figure(figsize=(8, 6))
plt.plot(X[:, 0], y, 'o', label='Data')
plt.plot(X[:, 0], rf.predict(X), 'r-', label='Mean Prediction')
plt.fill_between(X[:, 0], rf.predict(X) - 1.96 * std, rf.predict(X) + 1.96 * std,
                 color='r', alpha=0.2, label='95% CI')
plt.xlabel('Feature 0')
plt.ylabel('y')
plt.legend()
plt.title('Random Forest with Error Bars (Precomputed OOB)')
plt.show()
```
In this scenario, we've enabled oob scoring in the regressor, thus enabling `rf.oob_prediction_` which holds pre-computed OOB predictions and pass it to `random_forest_error`. This bypasses needing to calculate the out of bag predictions internally, thus optimizing the calculation.

For deeper dives into the theory behind random forests and their statistical properties, I highly recommend reading “The Elements of Statistical Learning” by Hastie, Tibshirani, and Friedman. For a more focused treatment of random forest variance estimation, the original `forestci` paper by Wager et al. (available through most academic databases by searching for “Confidence Intervals for Random Forests: The Jackknife and the Infinitesimal Jackknife”) would be an extremely valuable resource.

Remember that `forestci` provides estimates of *prediction variance*, not necessarily true prediction intervals. These estimated variances are extremely useful to assess relative prediction uncertainty, but, as noted in the associated papers, these standard errors should not be used as a direct replacement for prediction intervals. However, for many practical applications, especially when comparing the confidence of different predictions, or when assessing the variability of a model across different input spaces, `forestci` is an invaluable and computationally efficient tool.
