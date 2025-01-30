---
title: "Where do gradients not exist for a variable?"
date: "2025-01-30"
id: "where-do-gradients-not-exist-for-a-variable"
---
Gradients, fundamental to the optimization process in machine learning and numerical computation, fail to exist at points where the function defining them is not differentiable. I've encountered this issue repeatedly during my work in developing custom neural network architectures, particularly when dealing with activation functions and loss landscapes. The lack of a gradient at a specific location implies that the function's slope is either undefined, discontinuous, or exhibits a sharp change, making standard gradient-based optimization algorithms unable to ascertain a direction for descent or ascent. This fundamentally limits the efficacy of optimization procedures that rely on derivative calculations.

A function’s differentiability at a point necessitates both continuity at that point and the existence of a unique tangent line. The formal definition of the derivative involves a limit that must approach the same value from both sides of the point in question. Where these conditions are not met, the limit defining the derivative either does not exist or diverges to infinity, resulting in an undefined gradient. This can manifest in several distinct scenarios.

Firstly, consider *points of discontinuity*. When a function jumps from one value to another abruptly, the limit of the function as it approaches the point from different directions is not equal, implying that the derivative cannot be defined at that point. For example, a step function exhibits a discontinuity at the point of the step; the function's value makes an immediate jump, leaving no defined slope. Secondly, *points of non-smoothness* are problematic. This occurs when the function’s tangent line does not exist, which can result from a sharp bend or corner in the function. One classic instance of this is the absolute value function. Although continuous everywhere, the derivative is undefined at its vertex because the slope changes instantaneously. Another scenario where differentiability breaks down is at *vertical tangents*. When a function possesses an infinitely steep tangent line at a certain point, the gradient approaches infinity, and thus is not a defined value.

The practical consequences of these issues in machine learning are substantial. Optimization algorithms like gradient descent become unreliable. While these algorithms often attempt to take a small step in the direction of the negative gradient of the loss function, if the gradient is undefined, there is no appropriate direction to take. In fact, they may misbehave, and learning fails. One way to visualize this is to imagine a landscape with sharp cliffs; if you stand at the edge of the cliff, you don’t know whether to go left or right because your “gradient” is undefined.

Let's examine some illustrative code examples, demonstrating how gradient calculations fail. I'll use Python with NumPy for numerical operations since it's a common environment for these tasks.

**Example 1: Absolute Value Function**

```python
import numpy as np
import matplotlib.pyplot as plt

def absolute_value(x):
    return np.abs(x)

def numerical_gradient(func, x, h=1e-8):
    return (func(x + h) - func(x)) / h

x_values = np.linspace(-2, 2, 400)
y_values = absolute_value(x_values)

#Approximate numerical gradient, not a true gradient calculation
grad_values = [numerical_gradient(absolute_value, x) for x in x_values]

#Visualize
plt.figure(figsize=(10, 5))
plt.plot(x_values, y_values, label='f(x) = |x|')
plt.plot(x_values, grad_values, label='Numerical Gradient')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Absolute Value Function and Numerical Gradient')
plt.legend()
plt.grid(True)
plt.show()

x_at_zero = 0
grad_approx_at_zero = numerical_gradient(absolute_value, x_at_zero)

print(f"Approximate numerical gradient at x=0: {grad_approx_at_zero}")
```
In this example, I define the absolute value function using NumPy and approximate the gradient using a finite difference method since standard library gradient implementations cannot handle points of non-differentiability. Notice that in the visualization, around x = 0, the gradient changes abruptly; also, the numerical gradient at 0 is not truly defined, although it has a value. This demonstrates a failure to get meaningful information at that critical point. When using automatic differentiation, you will also find similar behavior; the automatic differentiation will be unable to provide a proper derivative.

**Example 2: Heaviside Step Function**

```python
def heaviside(x):
    return np.where(x >= 0, 1, 0)

x_values = np.linspace(-2, 2, 400)
y_values = heaviside(x_values)

grad_values = [numerical_gradient(heaviside, x) for x in x_values]

plt.figure(figsize=(10, 5))
plt.plot(x_values, y_values, label='Heaviside(x)')
plt.plot(x_values, grad_values, label='Numerical Gradient')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Heaviside Function and Numerical Gradient')
plt.legend()
plt.grid(True)
plt.show()

x_at_zero = 0
grad_approx_at_zero = numerical_gradient(heaviside, x_at_zero)

print(f"Approximate numerical gradient at x=0: {grad_approx_at_zero}")

```
Here, the step function is implemented using NumPy’s `where` function to handle the discontinuity. The numerical gradient approximation shows that away from x = 0, the gradient is zero; however, near zero, the numerical gradient spikes, illustrating that around the discontinuity, the derivative is undefined. Automatic differentiation will typically not be able to provide a sensible derivative at 0.

**Example 3: Function with Vertical Tangent**

```python
def function_with_vertical_tangent(x):
    return np.sqrt(np.abs(x))

x_values = np.linspace(-1, 1, 400)
y_values = function_with_vertical_tangent(x_values)

grad_values = [numerical_gradient(function_with_vertical_tangent, x) for x in x_values]

plt.figure(figsize=(10, 5))
plt.plot(x_values, y_values, label='f(x) = sqrt(|x|)')
plt.plot(x_values, grad_values, label='Numerical Gradient')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Function with Vertical Tangent and Numerical Gradient')
plt.legend()
plt.grid(True)
plt.show()

x_at_zero = 0
grad_approx_at_zero = numerical_gradient(function_with_vertical_tangent, x_at_zero)

print(f"Approximate numerical gradient at x=0: {grad_approx_at_zero}")
```

This function, defined as the square root of the absolute value of x, is continuous at x=0 but possesses a vertical tangent at that point. Using the numerical gradient, we observe it tends towards infinity at x=0, demonstrating the non-differentiability where a function's slope is infinitely steep. Again, automatic differentiation will be unable to handle this specific situation.

These examples highlight situations where gradients do not exist due to discontinuities, non-smoothness, or vertical tangents. In the context of machine learning, these situations can hinder the optimization process. It is often necessary to choose alternative, differentiable functions, or to apply smoothing techniques to enable stable training. For example, leaky ReLU activations are used as an alternative to ReLU to prevent zero gradients.

For further exploration, I recommend reviewing advanced calculus texts that rigorously cover differentiability, particularly in single and multivariate contexts, specifically topics such as partial derivatives and the Jacobian matrix. Additionally, consulting resources on numerical analysis can help deepen the understanding of how computers approximate derivatives and their limitations. Texts dealing with optimization algorithms often discuss the assumptions of gradient-based methods and how to handle non-differentiable scenarios in practical implementations. The mathematical underpinnings of automatic differentiation can provide additional insight into how derivatives are calculated algorithmically and its limitations when dealing with non-differentiable functions.
