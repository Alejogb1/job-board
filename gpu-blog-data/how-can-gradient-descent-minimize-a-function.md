---
title: "How can gradient descent minimize a function?"
date: "2025-01-30"
id: "how-can-gradient-descent-minimize-a-function"
---
Gradient descent's efficacy in function minimization stems from the principle of iteratively adjusting parameters in the direction of the steepest decrease of the function's output. It’s not a magical process; it's an optimization algorithm relying on calculus and linear algebra. My experience, honed over several projects involving neural network training and curve fitting, underscores its practical significance and underlying mechanism. Essentially, gradient descent uses the function's derivative (the gradient, in a multi-dimensional context) to determine how to modify the input values to move closer to the function's minimum, be it a local or global minimum.

The process unfolds as follows. Given a function, commonly denoted as *J(θ)*, where *θ* represents the input parameters (weights, biases, etc.), we seek the *θ* that results in the lowest possible *J(θ)*. The gradient, ∇*J(θ)*, points in the direction of the function's greatest increase at a particular point in the parameter space. Consequently, to minimize the function, we take steps in the *opposite* direction of the gradient. This principle is encapsulated in the update rule:

*θ*<sub>new</sub> = *θ*<sub>old</sub> - *α* ∇*J(θ*<sub>old</sub>)*

Here, *α* is the learning rate, a hyperparameter that controls the step size during each iteration. A large learning rate risks overshooting the minimum, while a small one can lead to slow convergence.

Initially, we begin with a random or heuristically chosen set of parameters, *θ*<sub>0</sub>. The gradient, ∇*J(θ*<sub>0</sub>), is computed using the derivative of the function evaluated at *θ*<sub>0</sub>. If *J(θ)* is a complex function (for example, the error surface of a neural network), calculating the derivative analytically may be impossible. Therefore, we often resort to numerical methods, such as backpropagation, in neural networks, to calculate these gradients. The gradient vector's components indicate how much each parameter contributes to the function's output value and in what direction we should adjust these parameters to reduce the output value. The parameters are then updated using the aforementioned update rule. This process is repeated iteratively until a convergence criterion is met, such as a sufficiently small change in the function value or the number of iterations reaches a maximum limit. This repeated adjustment allows the algorithm to "descend" toward a minimum in the function's landscape.

However, gradient descent’s convergence is not guaranteed to find the global minimum, particularly when the function landscape contains many local minima. The algorithm may get stuck in one of these local minima, depending on the initial starting point and the optimization method used. Several variations of gradient descent, including stochastic gradient descent (SGD), mini-batch gradient descent, and adaptive learning rate methods (e.g., Adam, RMSprop), have been developed to mitigate these issues and improve convergence speed and reliability. These variations primarily adjust the way the gradients are calculated or how the learning rate is adapted, offering different performance profiles and trade-offs.

Let's consider several scenarios with illustrative code examples.

**Example 1: Linear Regression**

Suppose we want to minimize the mean squared error (MSE) for a simple linear regression problem, where *y* = *mx* + *b*, and we aim to find the optimal values for *m* (slope) and *b* (intercept).

```python
import numpy as np

def mse(y_true, y_pred):
  return np.mean((y_true - y_pred)**2)

def gradient_mse(x, y_true, m, b):
  n = len(y_true)
  d_m = (-2/n) * np.sum(x * (y_true - (m*x + b)))
  d_b = (-2/n) * np.sum(y_true - (m*x + b))
  return d_m, d_b

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Initial parameters
m = 0
b = 0
learning_rate = 0.01
iterations = 1000

for i in range(iterations):
  y_pred = m * x + b
  grad_m, grad_b = gradient_mse(x, y, m, b)
  m = m - learning_rate * grad_m
  b = b - learning_rate * grad_b

  if i % 100 == 0:
    print(f"Iteration {i}, MSE: {mse(y, y_pred):.4f}")


print(f"Optimal m: {m:.4f}, Optimal b: {b:.4f}")

```

Here, we implement the gradient computation for the MSE loss function and use gradient descent to update the parameters *m* and *b*. The `gradient_mse` function calculates the partial derivatives of the MSE loss with respect to *m* and *b*, allowing us to determine the direction of steepest descent. The loop updates *m* and *b* iteratively, displaying the MSE value periodically to monitor the optimization progress. In a real-world scenario, a larger dataset and more refined stopping criteria would be applied.

**Example 2: Minimizing a Simple Quadratic Function**

Let's consider minimizing the function *J(θ)* = *θ*<sup>2</sup> - 4*θ* + 5. This is a simple convex function and useful for understanding how gradient descent descends to the minimum.

```python
import numpy as np

def quadratic(theta):
  return theta**2 - 4*theta + 5

def gradient_quadratic(theta):
  return 2*theta - 4

theta = 0 # Initial theta
learning_rate = 0.1
iterations = 100

for i in range(iterations):
  grad = gradient_quadratic(theta)
  theta = theta - learning_rate * grad
  if i % 10 == 0:
     print(f"Iteration {i}, Theta: {theta:.4f}, J(theta): {quadratic(theta):.4f}")

print(f"Optimal theta: {theta:.4f}")

```

The code directly computes the derivative of the quadratic function. Gradient descent iteratively updates `theta`, converging towards the value that minimizes the quadratic function. The periodic printing shows the progress towards the minimum. This example clearly demonstrates the iterative nature of the process.

**Example 3: One Step of Gradient Descent on a complex non-convex Function**
Consider a simple non-convex function, f(x) = x^3 - 3x, which can have more than one local minimum. Let’s perform a single step of gradient descent.

```python
import numpy as np

def non_convex(x):
    return x**3 - 3*x

def gradient_non_convex(x):
    return 3*x**2 - 3


x_current = 0.5  # Starting value
learning_rate = 0.1
gradient_value = gradient_non_convex(x_current)
x_new = x_current - learning_rate * gradient_value

print(f"Initial x: {x_current:.4f}, f(x): {non_convex(x_current):.4f}")
print(f"Gradient at x: {gradient_value:.4f}")
print(f"New x after one step: {x_new:.4f}, f(x_new): {non_convex(x_new):.4f}")

```

In this code, a specific initial value is chosen and we perform a single update step based on the calculated gradient. This shows the actual change in position given the gradient's direction at that point. This single step elucidates the algorithm’s behaviour at a non-trivial location, illustrating the movement towards a (possibly local) minimum.

For further study, several resources provide in-depth explanations. Texts on numerical optimization frequently cover gradient descent and its variations. Books dedicated to machine learning will extensively feature gradient descent, especially concerning neural networks. Furthermore, academic papers on optimization methods often present new research developments and algorithmic improvements related to gradient descent. I would encourage the exploration of these resources to deepen your understanding.
