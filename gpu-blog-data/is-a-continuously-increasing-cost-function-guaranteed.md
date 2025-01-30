---
title: "Is a continuously increasing cost function guaranteed?"
date: "2025-01-30"
id: "is-a-continuously-increasing-cost-function-guaranteed"
---
The monotonic increase of a cost function during optimization is not guaranteed, contrary to a common misconception.  My experience working on large-scale model training, particularly within the context of reinforcement learning and non-convex optimization problems, has repeatedly demonstrated this. While many algorithms *aim* for a monotonically decreasing cost function (as the goal is minimization),  the actual behavior is contingent on various factors including the algorithm's specifics, the data's properties, and the presence of local minima or saddle points in the cost landscape.

Let's clarify this with a structured explanation.  A cost function, or loss function, quantifies the error between a model's prediction and the ground truth. The optimization process aims to find the model parameters that minimize this function.  Many popular optimization algorithms, like gradient descent and its variants (Adam, RMSprop), iteratively update parameters based on the negative gradient of the cost function.  The negative gradient points towards the direction of steepest descent, theoretically leading to a reduction in cost. However, this guarantee falls apart in several scenarios.

1. **Non-Convex Optimization Landscapes:**  Many real-world problems, including those encountered in deep learning and complex system modeling, have non-convex cost functions. These functions exhibit multiple local minima and saddle points, which are points where the gradient is zero but aren't necessarily global minima.  An optimization algorithm might get stuck in a local minimum, leading to oscillations or even an increase in the cost function during subsequent iterations. The algorithm might escape the local minimum eventually, but the cost function won't be monotonically decreasing.  This is particularly problematic in high-dimensional spaces where exhaustive search is infeasible.

2. **Learning Rate Selection:** The learning rate dictates the step size taken during each parameter update.  An inappropriately large learning rate can cause the algorithm to overshoot the minimum, leading to oscillations and potential increases in the cost function.  Conversely, a learning rate that is too small can result in slow convergence and an excessively long optimization process, where transient increases are possible.  Optimal learning rate scheduling, often involving techniques like learning rate decay or cyclical learning rates, aims to mitigate this.

3. **Stochastic Gradient Descent (SGD) and its Variants:**  When dealing with large datasets, SGD and its variations are typically employed.  These methods calculate the gradient using only a subset of the data at each iteration (mini-batch).  The stochastic nature of the gradient estimation introduces noise, leading to fluctuations in the cost function.  Even if the average gradient points towards the minimum, the noisy gradient updates might temporarily increase the cost.  The overall trend, however, should still be downward over many iterations.


Let’s illustrate with three code examples using Python and a fictitious cost function for simplicity.  These examples showcase different scenarios impacting monotonicity.

**Example 1: Non-convex cost function leading to oscillations:**

```python
import numpy as np
import matplotlib.pyplot as plt

def cost_function(x):
  return x**4 - 4*x**2 + 5 # Example non-convex function

x_values = np.arange(-2, 2, 0.1)
y_values = cost_function(x_values)
plt.plot(x_values, y_values)
plt.xlabel("x")
plt.ylabel("Cost")
plt.title("Non-Convex Cost Function")
plt.show()

x = 1.5  # Initial point
learning_rate = 0.1
costs = []
for i in range(100):
  gradient = 4*x**3 - 8*x
  x = x - learning_rate * gradient
  costs.append(cost_function(x))

plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function Value Over Iterations")
plt.show()

```

This example uses a simple non-convex function. The plot of the cost function shows multiple local minima.  The gradient descent algorithm, even with a carefully chosen learning rate, might exhibit oscillations and temporarily increase the cost before converging to a local minimum.


**Example 2: Impact of learning rate:**

```python
import numpy as np
import matplotlib.pyplot as plt

def cost_function(x):
  return x**2 # Simple convex function

x = 5
learning_rate_high = 2
learning_rate_low = 0.01
costs_high = []
costs_low = []

for i in range(10):
  gradient = 2*x
  x_high = x - learning_rate_high * gradient
  x_low = x - learning_rate_low * gradient
  costs_high.append(cost_function(x_high))
  costs_low.append(cost_function(x_low))
  x = x_high

plt.plot(costs_high, label="High Learning Rate")
plt.plot(costs_low, label="Low Learning Rate")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.legend()
plt.title("Effect of Learning Rate on Cost Function")
plt.show()


```

This illustrates how a high learning rate (overshooting) can cause oscillations and temporary cost increases, while a low learning rate leads to a smoother but potentially slower convergence.


**Example 3:  Stochasticity in SGD:**

```python
import numpy as np
import matplotlib.pyplot as plt

def cost_function(x):
  return x**2

x = 5
learning_rate = 0.1
costs = []

for i in range(100):
  # Simulate stochastic gradient: add noise
  noisy_gradient = 2*x + np.random.normal(0, 0.5) #Gaussian Noise
  x = x - learning_rate * noisy_gradient
  costs.append(cost_function(x))

plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost Function with Stochastic Gradient Descent")
plt.show()
```

Here, noise is introduced to mimic the stochastic nature of SGD.  The cost function's fluctuations clearly demonstrate how a temporary increase is possible even with a well-behaved convex function.


In summary, while the goal of optimization is to minimize the cost function, a monotonically decreasing cost during training is not a guaranteed outcome.  Understanding the underlying reasons, primarily the characteristics of the cost function and the optimization algorithm employed, is crucial.


**Resources:**

I would recommend consulting standard textbooks on optimization algorithms and machine learning, covering topics like convex optimization, gradient descent methods, stochastic optimization, and the practical challenges encountered in high-dimensional problems.  A thorough understanding of numerical optimization techniques is also beneficial.  Focusing on these areas will provide a strong foundation to address this and related complexities in the field.
