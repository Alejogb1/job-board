---
title: "How can gradient descent be used to approximate functions in Python?"
date: "2025-01-30"
id: "how-can-gradient-descent-be-used-to-approximate"
---
Gradient descent, while fundamentally an optimization algorithm, can effectively approximate functions by learning the parameters of a parameterized model that represents the target function. This works by treating the function approximation problem as a minimization problem: specifically, minimizing the difference between the model's output and the target function's output. My experience has shown this to be a surprisingly versatile technique, extending beyond just curve-fitting to complex, multi-dimensional relationships.

The core principle relies on defining a loss function that quantifies the error between the model’s prediction and the true value. A common choice for regression problems is the mean squared error (MSE). For a given training set comprising inputs (x) and target outputs (y), and a model with parameters (θ), the loss function L(θ) can be represented as:

L(θ) = (1/N) * Σ (model(x_i, θ) - y_i)^2

where N is the number of training samples, model(x_i, θ) represents the model's prediction for input x_i using parameters θ, and y_i is the corresponding target value.

Gradient descent iteratively adjusts the model parameters θ in the direction that reduces the loss. This adjustment is guided by the gradient of the loss function with respect to the parameters. The gradient, a vector of partial derivatives, indicates the direction of steepest ascent of the loss function. Consequently, we move in the opposite direction, the negative gradient, to minimize the loss. This update rule is expressed as:

θ_new = θ_old - α * ∇L(θ_old)

where α is the learning rate, a hyperparameter that controls the step size during parameter updates, and ∇L(θ_old) denotes the gradient of the loss function evaluated at the current parameters.

The iterative process continues until a predefined stopping criterion is met, such as a maximum number of iterations, or until the change in loss falls below a specified threshold. The final, learned parameters will represent an approximation of the target function embedded in the chosen parameterized model.

Here are three examples demonstrating this in Python using `numpy` for numerical computation and `matplotlib` for visualization:

**Example 1: Linear Regression**

This example approximates a linear function using a simple linear model.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Linear model
def model(X, theta):
    return theta[0] + theta[1] * X

# Loss function (MSE)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Gradient of the loss function
def gradient(X, y_true, y_pred):
    n = len(y_true)
    d_theta0 = (1/n) * np.sum(y_pred - y_true)
    d_theta1 = (1/n) * np.sum((y_pred - y_true) * X)
    return np.array([d_theta0, d_theta1])

# Gradient descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    costs = []
    for i in range(iterations):
        y_pred = model(X, theta)
        grad = gradient(X, y, y_pred)
        theta = theta - learning_rate * grad
        cost = mse(y, y_pred)
        costs.append(cost)
    return theta, costs

# Initialize parameters and train the model
theta = np.array([np.random.randn(), np.random.randn()]) # random initialization
learning_rate = 0.01
iterations = 1000
theta_final, costs = gradient_descent(X, y, theta, learning_rate, iterations)

# Make predictions
X_test = np.linspace(0, 2, 100).reshape(-1, 1)
y_pred = model(X_test, theta_final)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, label='Data')
plt.plot(X_test, y_pred, color='red', label='Model')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()
```
In this example, a linear model `y = theta[0] + theta[1] * x` is used to approximate a linear relationship between `X` and `y`. The gradient is derived explicitly, and then the gradient descent updates iteratively learn the optimal values for `theta[0]` and `theta[1]`. The cost is tracked at each iteration to demonstrate the convergence behavior of the algorithm, visually represented in the second subplot. Random initialization is crucial as it avoids bias in the learning process.

**Example 2: Polynomial Regression**

Approximating a non-linear function using a polynomial model.

```python
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 2 + 5*X + 3 * X**2 + np.random.randn(100, 1)

# Polynomial model (degree 2)
def model(X, theta):
    return theta[0] + theta[1] * X + theta[2] * X**2

# Loss function (MSE)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Gradient of the loss function
def gradient(X, y_true, y_pred):
    n = len(y_true)
    d_theta0 = (1/n) * np.sum(y_pred - y_true)
    d_theta1 = (1/n) * np.sum((y_pred - y_true) * X)
    d_theta2 = (1/n) * np.sum((y_pred - y_true) * X**2)
    return np.array([d_theta0, d_theta1, d_theta2])

# Gradient descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    costs = []
    for i in range(iterations):
        y_pred = model(X, theta)
        grad = gradient(X, y, y_pred)
        theta = theta - learning_rate * grad
        cost = mse(y, y_pred)
        costs.append(cost)
    return theta, costs

# Initialize parameters and train the model
theta = np.array([np.random.randn(), np.random.randn(), np.random.randn()])
learning_rate = 0.005
iterations = 1000
theta_final, costs = gradient_descent(X, y, theta, learning_rate, iterations)

# Make predictions
X_test = np.linspace(0, 2, 100).reshape(-1, 1)
y_pred = model(X_test, theta_final)

# Visualization
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(X, y, label='Data')
plt.plot(X_test, y_pred, color='red', label='Model')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(costs)
plt.xlabel('Iteration')
plt.ylabel('Loss')

plt.tight_layout()
plt.show()
```
This illustrates that gradient descent can be used with non-linear models. Here, a quadratic polynomial function `y = theta[0] + theta[1]*x + theta[2]*x^2` is employed to approximate a curved relationship between `X` and `y`. The model function and gradient update are adapted accordingly, and the visualization showcases the effectiveness of this approach in learning a more complex functional relationship. The adjustment of the learning rate becomes important, and often requires fine-tuning based on the specific task.

**Example 3: Multi-Dimensional Input**

Extending gradient descent to approximate functions with multiple input variables.

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Generate synthetic data (2D input, 1D output)
np.random.seed(42)
X = 2 * np.random.rand(100, 2)
y = 3 + 2 * X[:, 0] - 1.5 * X[:, 1] + np.random.randn(100)

# Linear model with 2 inputs
def model(X, theta):
  return theta[0] + theta[1] * X[:, 0] + theta[2] * X[:, 1]

# Loss function (MSE)
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)

# Gradient of the loss function
def gradient(X, y_true, y_pred):
    n = len(y_true)
    d_theta0 = (1/n) * np.sum(y_pred - y_true)
    d_theta1 = (1/n) * np.sum((y_pred - y_true) * X[:, 0])
    d_theta2 = (1/n) * np.sum((y_pred - y_true) * X[:, 1])
    return np.array([d_theta0, d_theta1, d_theta2])


# Gradient descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    costs = []
    for i in range(iterations):
        y_pred = model(X, theta)
        grad = gradient(X, y, y_pred)
        theta = theta - learning_rate * grad
        cost = mse(y, y_pred)
        costs.append(cost)
    return theta, costs

# Initialize parameters and train the model
theta = np.array([np.random.randn(), np.random.randn(), np.random.randn()])
learning_rate = 0.01
iterations = 1000
theta_final, costs = gradient_descent(X, y, theta, learning_rate, iterations)


# Create grid for predictions
x_range = np.linspace(0, 2, 50)
y_range = np.linspace(0, 2, 50)
xx, yy = np.meshgrid(x_range, y_range)
X_grid = np.c_[xx.ravel(), yy.ravel()]
Z = model(X_grid, theta_final).reshape(xx.shape)

# Visualization
fig = plt.figure(figsize=(10, 5))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], y, label='Data')
ax1.plot_surface(xx, yy, Z, color='red', alpha=0.5)
ax1.set_xlabel('X1')
ax1.set_ylabel('X2')
ax1.set_zlabel('y')
ax1.legend()

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(costs)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Loss')


plt.tight_layout()
plt.show()
```

This code illustrates gradient descent for a function `y =  theta[0] + theta[1] * x_1 + theta[2] * x_2`, taking two input variables (x1, x2) and producing a single output `y`. The visualization shifts from a 2D scatter plot to a 3D scatter plot, with the predicted values represented as a surface. The approach and implementation are consistent, simply extending the model and gradient calculations to accommodate multiple input dimensions.

When learning complex functions, remember that the model architecture becomes critical. Increasing the complexity of the model (e.g., using a deep neural network with multiple layers) can enable the approximation of more complex functions, at the cost of increased computation and the necessity of fine-tuning parameters like the learning rate. Furthermore, regularization techniques are commonly employed to avoid overfitting the training data. For further study on gradient descent and function approximation, resources on numerical optimization methods, machine learning, and deep learning can be valuable. Specific topics would include stochastic gradient descent variants, regularization techniques, and different model architectures, such as neural networks, and how they are applied to function approximation. Textbooks covering these fields often include practical, hands-on examples.
