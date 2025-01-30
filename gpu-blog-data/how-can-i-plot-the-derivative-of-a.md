---
title: "How can I plot the derivative of a function using PyTorch?"
date: "2025-01-30"
id: "how-can-i-plot-the-derivative-of-a"
---
The inherent challenge in plotting the derivative of a function using PyTorch stems from the fact that PyTorch primarily deals with computational graphs and automatic differentiation, rather than directly providing symbolic derivatives.  My experience working on high-dimensional optimization problems within the context of generative adversarial networks (GANs) has frequently necessitated precise derivative visualization for debugging and analysis.  The approach, therefore, isn't about directly obtaining a symbolic derivative as you might with SymPy, but leveraging PyTorch's autograd functionality to numerically approximate the derivative.

**1.  Clear Explanation:**

The core concept involves using PyTorch's `autograd` engine to calculate the gradient of the function at various points. We then use these gradient values to plot the derivative.  This process leverages the power of automatic differentiation, which efficiently computes derivatives without explicit derivation, especially beneficial for complex functions. The accuracy of the plot is directly dependent on the spacing of the points at which we evaluate the gradient – finer spacing provides a smoother, more accurate representation, but with increased computational cost.  It's crucial to understand that what we're plotting is a numerical approximation, not an analytical derivative.


**2. Code Examples with Commentary:**

**Example 1:  Simple Polynomial**

This example demonstrates the process for a simple polynomial function, illustrating the fundamental approach.

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# Define the function
def f(x):
  return x**3 - 2*x**2 + x - 1

# Create input tensor and require gradient calculation
x = torch.linspace(-2, 2, 100, requires_grad=True)

# Calculate function value
y = f(x)

# Calculate gradient
y.backward(torch.ones_like(x))  #Gradient w.r.t. x

# Extract gradient values
derivative = x.grad.detach().numpy()

# Plot the function and its derivative
plt.figure(figsize=(10, 6))
plt.plot(x.detach().numpy(), y.detach().numpy(), label='f(x)')
plt.plot(x.detach().numpy(), derivative, label='f\'(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Function and its Derivative')
plt.grid(True)
plt.show()
```

This code first defines the function `f(x)`.  `torch.linspace` generates evenly spaced points for x.  Crucially, `requires_grad=True` enables gradient tracking. `y.backward()` computes the gradient of `y` with respect to `x`.  The `torch.ones_like(x)` argument specifies that the gradient should be computed for each element of x, essentially calculating the derivative at each point. `.detach().numpy()` converts the PyTorch tensor to a NumPy array for plotting with `matplotlib`.

**Example 2:  Trigonometric Function with Central Difference Approximation**

For functions exhibiting rapid oscillations or discontinuities, a simple backward pass may not be sufficient for an accurate derivative representation. Central difference approximation can enhance accuracy.

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

#Define the function
def g(x):
  return torch.sin(2*torch.pi*x)

# Create input tensor
x = torch.linspace(0, 1, 100)

#Compute Central difference
h = 1e-4
g_prime = (g(x + h) - g(x - h)) / (2 * h)

#Plotting
plt.figure(figsize=(10, 6))
plt.plot(x.detach().numpy(), g(x).detach().numpy(), label='g(x)')
plt.plot(x.detach().numpy(), g_prime.detach().numpy(), label='g\'(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Function and its Derivative (Central Difference)')
plt.grid(True)
plt.show()
```

Here, the central difference method approximates the derivative using values slightly before and after each point, improving accuracy, especially near local extrema.  Note that we don't require `requires_grad=True` since we're manually computing the derivative.

**Example 3:  Handling Multiple Variables with `torch.autograd.grad`**

In scenarios with multiple variables, a more sophisticated approach using `torch.autograd.grad` becomes necessary.

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

#Define a multivariable function
def h(x, y):
    return x**2 + y**2

#Create input tensors
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(1.0, requires_grad=True)

#Compute function value
z = h(x, y)

#Compute partial derivatives using autograd.grad
dz_dx = torch.autograd.grad(z, x)[0]
dz_dy = torch.autograd.grad(z, y)[0]

print(f"Partial derivative w.r.t x: {dz_dx}")
print(f"Partial derivative w.r.t y: {dz_dy}")


#For plotting purposes (demonstrating partial derivative w.r.t x)
x_vals = torch.linspace(-5, 5, 100)
y_val = 1.0
z_vals = h(x_vals, torch.tensor(y_val))

partial_x = torch.autograd.grad(z_vals, x_vals, grad_outputs=torch.ones_like(z_vals))[0]


plt.figure(figsize=(10, 6))
plt.plot(x_vals.detach().numpy(), z_vals.detach().numpy(), label='h(x,y)')
plt.plot(x_vals.detach().numpy(), partial_x.detach().numpy(), label='∂h/∂x')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Function and Partial Derivative w.r.t x')
plt.grid(True)
plt.show()


```

This example showcases computing partial derivatives using `torch.autograd.grad`.  This function explicitly calculates the gradient of `z` with respect to `x` and `y`.  The example focuses on plotting the partial derivative with respect to x. For more complex scenarios involving many variables, visualization might require techniques like contour plots or other multivariate visualization methods.


**3. Resource Recommendations:**

The official PyTorch documentation, focusing on the `autograd` section.  A comprehensive textbook on numerical methods and calculus, to understand approximation techniques.  A book covering scientific visualization techniques for better graphical representation of multivariate functions and their derivatives.
