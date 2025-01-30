---
title: "How can linear regression be minimized using a quadratic cost function?"
date: "2025-01-30"
id: "how-can-linear-regression-be-minimized-using-a"
---
Minimizing a linear regression model’s error using a quadratic cost function, specifically Mean Squared Error (MSE), is a cornerstone of many statistical learning techniques. Having spent considerable time developing predictive models for various industrial applications, I've consistently observed that the choice of a quadratic cost function, despite its simplicity, allows for robust and efficient optimization. The rationale stems from the properties of the squared error, particularly its differentiability and convexity, which are critical for employing gradient-based optimization algorithms.

Linear regression aims to find a linear relationship between a set of input variables (features) and an output variable (target). This relationship is modeled by:

*y* = *X* *β* + *ε*

where:
*   *y* is a vector of observed target values,
*   *X* is a matrix of input feature values,
*   *β* is a vector of unknown coefficients we aim to estimate, and
*   *ε* is a vector of random error terms.

The core problem is to estimate the *β* that best fits the observed data, and the choice of the cost function determines what we mean by ‘best fit’. The quadratic cost function, or MSE, is calculated as:

MSE = (1/n) * Σ(*y<sub>i</sub>* - *ŷ<sub>i</sub>*)<sup>2</sup>

where:
*   *n* is the number of data points,
*   *y<sub>i</sub>* is the observed target value for the *i*-th data point,
*   *ŷ<sub>i</sub>* is the predicted target value for the *i*-th data point (calculated as *X<sub>i</sub>* *β*).

The key advantage of MSE lies in its mathematical properties. It’s a convex function, meaning any local minimum is also a global minimum. This is vital because it guarantees that optimization algorithms will converge to an optimal solution. Moreover, MSE is differentiable with respect to *β*, enabling the use of gradient descent and its variants, which iteratively adjust the coefficients to minimize the cost.  Calculating the gradient involves taking the derivative of the MSE with respect to each element in *β*. This yields:

∇MSE = (-2/n) * *X*<sup>T</sup>(*y* - *X* *β*)

The iterative process of gradient descent updates the coefficient vector *β* in the direction opposite the gradient, moving toward lower values of the cost function. The update rule is given by:

*β*<sub>t+1</sub> = *β*<sub>t</sub> - α * ∇MSE

where:
*   *β*<sub>t</sub> is the coefficient vector at the *t*-th iteration,
*   α is the learning rate, a parameter that controls the step size.

This process is repeated until the cost function converges or a predefined number of iterations is reached.  Furthermore, analytical solutions to the minimization problem can also be found. Setting the gradient of the MSE to zero and solving for *β* provides the ordinary least squares solution:

*β* = (*X*<sup>T</sup>*X*)<sup>-1</sup>*X*<sup>T</sup>*y*

This direct solution avoids iterative optimization but has the computational expense of matrix inversion. The suitability of each method often depends on the scale of the dataset. Iterative methods are preferable for large datasets.

Here are three examples of how linear regression is minimized using the quadratic cost function in practice, using Python with NumPy for demonstration:

**Example 1: Gradient Descent with a Single Feature**

```python
import numpy as np

def gradient_descent(X, y, learning_rate=0.01, num_iterations=1000):
    n = len(y)
    # Initialize the coefficient beta as a scalar, for a single feature
    beta = 0

    for _ in range(num_iterations):
        y_predicted = X * beta
        # Calculate the gradient for the single beta
        gradient = (-2/n) * np.sum(X * (y - y_predicted))
        # Update beta
        beta = beta - learning_rate * gradient

    return beta

# Example Usage
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

optimal_beta = gradient_descent(X, y)
print("Optimal beta using gradient descent:", optimal_beta)
```

In this first example, gradient descent optimizes a simplified linear regression model with a single feature, using a predefined learning rate and number of iterations. The code explicitly implements the MSE gradient and updates the single coefficient.  The initial coefficient is set to zero, and through each iteration, it’s modified until convergence. The output showcases a scalar beta value representing the estimated slope.

**Example 2:  Gradient Descent with Multiple Features (Vectorized)**

```python
import numpy as np

def gradient_descent_multifeature(X, y, learning_rate=0.01, num_iterations=1000):
    n = len(y)
    m = X.shape[1] # Number of features
    #Initialize beta as a vector with the appropriate number of elements
    beta = np.zeros(m)

    for _ in range(num_iterations):
      y_predicted = np.dot(X, beta)
      #Vectorized gradient computation with matrix operations
      gradient = (-2/n) * np.dot(X.T, (y - y_predicted))
      # Vectorized update of beta
      beta = beta - learning_rate * gradient

    return beta

# Example Usage
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])  #Adding a column of ones as the intercept
y = np.array([2, 4, 5, 4, 5])

optimal_beta = gradient_descent_multifeature(X, y)
print("Optimal beta using multi-feature gradient descent:", optimal_beta)
```

This second example builds on the first by extending it to multiple features using matrix operations. This vectorized implementation significantly improves efficiency, particularly for high-dimensional datasets.  A column of ones is added to the input matrix *X* to implicitly represent the intercept term in the regression model.  The output is a vector representing the regression coefficients including the intercept.

**Example 3: Closed-Form Solution (Ordinary Least Squares)**

```python
import numpy as np

def ordinary_least_squares(X, y):
    # Closed form solution to compute coefficients
    beta = np.linalg.inv(X.T @ X) @ X.T @ y
    return beta

# Example Usage
X = np.array([[1, 1], [1, 2], [1, 3], [1, 4], [1, 5]])
y = np.array([2, 4, 5, 4, 5])

optimal_beta = ordinary_least_squares(X, y)
print("Optimal beta using closed-form solution:", optimal_beta)
```

This third example shows the direct analytical solution using the ordinary least squares method. This is implemented using matrix inversion. While computationally more intensive, it directly computes the coefficients minimizing MSE. The result is a vector of coefficients equivalent to those found through gradient descent when that method converges. The choice between gradient descent and ordinary least squares largely depends on the computational resources available and the size of the dataset being considered.

For resources to further understand and implement linear regression and cost function minimization, I would suggest the following (excluding direct website links):

1.  **"The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman.** This book provides a comprehensive and rigorous treatment of statistical learning concepts, including detailed explanations of linear regression and its optimization using quadratic cost functions. It is an excellent resource for anyone interested in the theoretical foundations and practical implementations of statistical learning.
2.  **"Pattern Recognition and Machine Learning" by Christopher Bishop.** This work focuses on probabilistic machine learning but also gives an in-depth treatment of linear regression with thorough mathematical details. It’s an excellent resource for readers with a strong mathematical background.
3.  **"Introduction to Machine Learning with Python" by Andreas Mueller and Sarah Guido.** This book offers a more practical approach, providing many examples and code snippets for implementing machine learning algorithms, including linear regression, with Python. It’s more application-oriented and suitable for readers who want to quickly start developing models.

Each of these resources presents a different perspective and level of mathematical detail, but collectively they provide a strong understanding of linear regression and its minimization using quadratic cost functions. In practical applications, the proper understanding of underlying theory, numerical computations, and the computational limits of each method can be critically important for robust and efficient model development.
