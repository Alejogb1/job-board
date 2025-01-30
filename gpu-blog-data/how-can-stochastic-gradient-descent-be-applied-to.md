---
title: "How can stochastic gradient descent be applied to a piecewise non-differentiable function?"
date: "2025-01-30"
id: "how-can-stochastic-gradient-descent-be-applied-to"
---
The core challenge in applying stochastic gradient descent (SGD) to a piecewise non-differentiable function lies in the inability to directly compute the gradient at points of non-differentiability.  My experience optimizing complex reinforcement learning environments, specifically those involving discontinuous reward functions, highlighted this precisely.  Instead of relying on the gradient, we must approximate it using subgradients or other techniques that account for the function's discontinuities.

**1.  Understanding the Problem and Subgradients**

A piecewise non-differentiable function, by definition, possesses points where the derivative is undefined.  Traditional SGD relies on the gradient to guide the parameter updates.  Without a defined gradient, the algorithm lacks direction at these critical points.  However, the concept of a subgradient provides a solution.  A subgradient at a point x is a vector that satisfies the following inequality for all y in the function's domain:

f(y) ≥ f(x) + g<sup>T</sup>(y - x)

where f(x) is the function's value at x, and g is the subgradient at x.  Essentially, the subgradient provides a lower bound on the function's value, providing a direction of descent even in the absence of a true gradient.  Several approaches exist for computing subgradients, depending on the specific structure of the piecewise non-differentiable function.  For example, if the function is composed of differentiable pieces, the subgradient can be computed as the gradient of the active piece.


**2.  Code Examples and Commentary**

Let’s illustrate this with three examples demonstrating different approaches to handling piecewise non-differentiable functions within an SGD framework.


**Example 1:  Using Subgradients for a Simple Piecewise Linear Function**

Consider the function:

f(x) = max(0, x)

This function is non-differentiable at x = 0.  The subgradient is:

g(x) = 1, if x > 0
g(x) = 0, if x < 0
g(x) ∈ [0, 1], if x = 0


The following Python code demonstrates SGD using a subgradient:

```python
import numpy as np

def subgradient(x):
    if x > 0:
        return 1
    elif x < 0:
        return 0
    else:
        return np.random.rand() # Random subgradient at x=0

def sgd(learning_rate, iterations):
    x = np.random.rand()
    for i in range(iterations):
        g = subgradient(x)
        x -= learning_rate * g
    return x

learning_rate = 0.1
iterations = 1000
result = sgd(learning_rate, iterations)
print(f"SGD result: {result}")

```

This example highlights the arbitrary nature of the subgradient at the non-differentiable point.  The choice within the [0,1] range impacts the convergence but doesn’t prevent it.  This is characteristic of subgradient methods; the solution might not always converge to the exact minimum, but rather to a neighbourhood of the minimum.


**Example 2:  Handling a Function with Multiple Non-Differentiable Points**

Now, let’s extend this to a function with multiple non-differentiable points:

f(x) = |x| + |x - 1|

This function is non-differentiable at x = 0 and x = 1. The subgradient can be defined piecewise:

```python
import numpy as np

def subgradient(x):
    if x > 1:
        return 2
    elif x < 0:
        return -2
    elif x == 0:
        return np.random.choice([-1,1])
    else:
        return np.random.choice([-1,1])

# ... (rest of the SGD implementation remains the same as Example 1)
```

This code adapts the subgradient calculation to address multiple non-differentiable points.  The selection of a subgradient at the non-differentiable points is again crucial, and the stochastic nature introduces variability in the convergence process.


**Example 3:  Approximating with a Smooth Function**

For functions where computing subgradients proves challenging, an alternative involves approximating the piecewise non-differentiable function with a smooth function.  This can be accomplished through techniques like convolution with a smoothing kernel.  For example, the absolute value function |x| can be approximated by √(x² + ε), where ε is a small positive constant.  This introduces a controllable level of smoothness, allowing for gradient-based optimization.  The choice of ε influences the accuracy of the approximation and the convergence behaviour.

```python
import numpy as np

def smooth_approx(x, epsilon):
  return np.sqrt(x**2 + epsilon)

def gradient(x, epsilon):
  return x / np.sqrt(x**2 + epsilon)

def sgd_smooth(learning_rate, iterations, epsilon):
  x = np.random.rand()
  for i in range(iterations):
    g = gradient(x, epsilon)
    x -= learning_rate * g
  return x

learning_rate = 0.1
iterations = 1000
epsilon = 0.01
result = sgd_smooth(learning_rate, iterations, epsilon)
print(f"SGD result (smoothed): {result}")
```

This approach trades off accuracy for differentiability, providing a practical solution when dealing with complex piecewise functions.  The choice of the smoothing parameter (ε) requires careful consideration, balancing the smoothness of the approximation with the accuracy of the solution.


**3. Resource Recommendations**

For a deeper understanding, I strongly recommend consulting texts on convex optimization and subgradient methods.  Specialized literature on non-smooth optimization and stochastic approximation will provide more advanced techniques.  Exploring literature related to proximal gradient methods and their application in machine learning would also prove beneficial.  Finally,  reviewing papers on reinforcement learning that deal with discontinuous reward functions will provide practical insights into applying these concepts to real-world problems.
