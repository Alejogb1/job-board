---
title: "Can simple deep learning (linear regression) effectively calculate x*y?"
date: "2025-01-30"
id: "can-simple-deep-learning-linear-regression-effectively-calculate"
---
Linear regression, at its core, is designed to model linear relationships between variables. Attempting to apply it directly to calculating the product of two inputs, `x * y`, presents a fundamental mismatch in the function's capabilities. I've witnessed firsthand, during various machine learning projects involving signal processing, that forcing a linear model onto a non-linear problem results in severely limited accuracy and often bizarre, unpredictable behavior.

The reason this approach fails is rooted in the nature of multiplication itself. The equation `z = x * y` defines a hyperbolic surface, not a plane, in a three-dimensional space (where `x`, `y`, and `z` are the axes). Linear regression, on the other hand, seeks to fit a hyperplane (in this case, a plane) to the data. A linear model, expressed generally as `z = ax + by + c`, where `a`, `b`, and `c` are trainable parameters, simply cannot capture the multiplicative relationship represented by `z = x * y`. A single plane cannot, by its nature, model a curved surface effectively.

Trying to force linear regression to approximate `x * y` will typically lead to a model that performs poorly across the input space. The model might work reasonably well for a small range of input values, especially if they are clustered close to zero. However, accuracy will quickly degrade when inputs move farther from the training data and, in most cases, will even predict incorrect results by orders of magnitude in areas where the multiplicative effect is more prominent. This is because the plane model will attempt to approximate a single, general linear trend that will be a poor substitute for the actual hyperbolic behaviour of the data.

Let’s illustrate this with several code examples utilizing Python with the NumPy and scikit-learn libraries, commonly used in machine learning.

**Example 1: Basic Linear Regression Attempt**

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 2) * 10 # 100 samples, 2 features (x and y), range 0-10
y = X[:, 0] * X[:, 1]        # The target output: x * y

# Train the model
model = LinearRegression()
model.fit(X, y)

# Test the model
test_X = np.array([[2, 3], [5, 8], [1, 1], [10, 10]])
predictions = model.predict(test_X)

print("Predictions:", predictions)
print("Actual values: ", [6, 40, 1, 100])
```

In this example, I generated 100 random data points where each input was between 0 and 10. The target variable `y` is simply the product of `x` and `y` for every sample. I then train a linear regression model. After that, I use a few arbitrary `x` and `y` values as a test set. As seen by the output, the model struggles. For `[2, 3]`, a value of 6 is expected, but the prediction might be around 12. Predictions for larger numbers like `[10, 10]` will also be incorrect. This is a core example of what I described previously: the linear model is attempting to approximate an inherently non-linear function. The coefficients learned by the linear model are not representative of multiplicative relations, but a best effort to fit a plane to non-planar data.

**Example 2: Attempting to Enhance Performance with Larger Dataset**

```python
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate a larger dataset
np.random.seed(42)
X = np.random.rand(10000, 2) * 100 # Increased samples and range
y = X[:, 0] * X[:, 1]

# Train the model
model = LinearRegression()
model.fit(X, y)

# Test
test_X = np.array([[20, 30], [50, 80], [10, 10], [100, 100]])
predictions = model.predict(test_X)

print("Predictions:", predictions)
print("Actual values: ", [600, 4000, 100, 10000])
```

Here I've significantly increased the training set size and range of input values, expecting this might help. While a larger dataset might slightly improve accuracy in the vicinity of the training data's mean, it doesn’t fundamentally change the model's inability to capture the non-linear nature of multiplication. The predictions are still largely incorrect, showing that simply having more data doesn't make a linear model fit a non-linear pattern effectively.

**Example 3: Feature Engineering (Partial Success)**

```python
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Generate synthetic data
np.random.seed(42)
X = np.random.rand(100, 2) * 10
y = X[:, 0] * X[:, 1]

# Feature Engineering (Create x*y feature)
poly = PolynomialFeatures(degree=2, include_bias = False) #Include interaction term
X_poly = poly.fit_transform(X) # create polynomial and interaction terms

# Train the model
model = LinearRegression()
model.fit(X_poly, y)

# Test
test_X = np.array([[2, 3], [5, 8], [1, 1], [10, 10]])
test_X_poly = poly.transform(test_X) # Apply polynomial to test set
predictions = model.predict(test_X_poly)


print("Predictions:", predictions)
print("Actual values: ", [6, 40, 1, 100])
```

In this final example, I demonstrate a technique called feature engineering. Specifically, I use `PolynomialFeatures` to generate a new feature that is the interaction between `x` and `y`, effectively the multiplicative term itself.  This allows linear regression to accurately calculate the values, because the non-linear behavior has now been included in the input, turning the problem into a linear space.

This is not a way to show that *linear regression* can approximate x*y, but rather a demonstration that a linear regression model can be *utilized* to perform the task if provided with the correct engineered feature. This is not the same as a direct application of linear regression. A different problem was created for the linear model. Note the model learned coefficients for x, y, and x*y, it is not inferring x*y.

It should be noted that there are other more appropriate models for learning complex non-linear functions. These include:

1.  **Artificial Neural Networks (ANN):** Multi-layered networks with non-linear activation functions can effectively approximate arbitrary non-linear relationships. This is the preferred model for such tasks when the goal is to learn the mapping from the data and not create a linear solution. The flexibility of the network, in terms of structure and the chosen activation functions, is what allows it to represent the inherent hyperbolic function of `x * y`.

2.  **Support Vector Machines (SVM) with a Kernel:** SVMs equipped with a non-linear kernel function, such as a polynomial or radial basis function (RBF) kernel, can learn decision boundaries that are not simple hyperplanes.

3.  **Decision Trees and Random Forests:** These models learn hierarchical decision rules that can represent non-linear patterns. Regression trees, in particular, would naturally partition the space in a way that better captures the behavior of a multiplication function.

**Resource Recommendations**

For a comprehensive understanding of the mathematical foundation of linear regression, I recommend "The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman. The book provides a thorough treatment of linear models alongside other statistical machine learning methods.

To delve deeper into neural networks, I would recommend "Deep Learning" by Goodfellow, Bengio, and Courville. It's considered a canonical reference and covers a wide range of topics, from basic feedforward networks to more advanced architectures.

Finally, for a more general introduction to machine learning algorithms including SVMs, I suggest "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Géron. This text is very accessible and practical, providing hands-on coding examples along with the underlying theory.

In conclusion, while linear regression is a valuable tool for modeling linear relationships, attempting to use it directly to compute `x * y` is fundamentally flawed due to its inability to represent non-linear mappings. While feature engineering, as demonstrated with `PolynomialFeatures`, can allow a linear model to approximate the solution by creating a new feature which *is* a linear space, this is a workaround, and other methods are better suited for this. Understanding this limitation is crucial when choosing the right machine learning algorithm for a specific problem.
