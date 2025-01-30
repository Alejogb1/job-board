---
title: "How do I find the Hessian matrix of this function?"
date: "2025-01-30"
id: "how-do-i-find-the-hessian-matrix-of"
---
The second partial derivatives of a function, arranged as a matrix, form the Hessian. This matrix is crucial in optimization and analysis of multi-variable functions, providing insights into the function's local curvature and stationary points. Having spent several years developing numerical optimization algorithms for robotics simulations, I've repeatedly encountered the need to compute Hessians, and the nuances involved. I will outline the process for obtaining the Hessian matrix, incorporating both theoretical understanding and practical implementation details.

The Hessian matrix, denoted as *H(f)* or ∇²*f*, for a scalar-valued function *f(x₁, x₂, ..., xₙ)* of *n* variables, is an *n x n* matrix. Each entry *Hᵢⱼ* is the second partial derivative of *f* with respect to the *i*-th and *j*-th variables: *Hᵢⱼ* = ∂²*f*/∂*xᵢ*∂*xⱼ*. For functions with continuous second derivatives, the Hessian matrix is symmetric, meaning *Hᵢⱼ* = *Hⱼᵢ*. This property often simplifies calculations.

To find the Hessian matrix, one must compute all necessary second partial derivatives. This process involves two steps: first, calculating the first partial derivatives of *f* with respect to each variable, and then, differentiating each of those first partial derivatives with respect to all variables again.

Let's consider a specific function to illustrate the process. Assume our function is: *f(x, y) = x²y + 3y² - x*. This function has two variables, *x* and *y*, thus its Hessian will be a 2x2 matrix.

First, we calculate the first partial derivatives:
∂*f*/∂*x* = 2*xy* - 1
∂*f*/∂*y* = *x²* + 6*y*

Now, we compute the second partial derivatives:

∂²*f*/∂*x*² = ∂(2*xy* - 1)/∂*x* = 2*y*
∂²*f*/∂*y*∂*x* = ∂(2*xy* - 1)/∂*y* = 2*x*
∂²*f*/∂*x*∂*y* = ∂(*x²* + 6*y*)/∂*x* = 2*x*
∂²*f*/∂*y*² = ∂(*x²* + 6*y*)/∂*y* = 6

The Hessian matrix for *f(x, y)* is thus:

H(f) =  [  2*y*   2*x* ]
         [  2*x*    6  ]

The symmetry of the matrix is evident; ∂²*f*/∂*y*∂*x* = ∂²*f*/∂*x*∂*y*.

The process extends analogously to functions of more variables, although the manual differentiation can become cumbersome. For high-dimensional cases, numerical methods or automated differentiation techniques are preferable. However, knowing how the analytic calculation unfolds gives critical intuition.

Now, consider code examples illustrating how to obtain a Hessian in different contexts using numerical methods, as pure symbolic derivation can be tedious or infeasible for many practical cases.

**Example 1: Numerical Approximation using Finite Differences (Python)**

```python
import numpy as np

def function_example(x):
  """A simple 2D function."""
  return x[0]**2 * x[1] + 3 * x[1]**2 - x[0]

def numerical_hessian(func, x, h=1e-5):
    """Computes the Hessian matrix numerically using central differences.
    Args:
      func: The function for which to compute the Hessian.
      x: The point at which to compute the Hessian.
      h: The step size for finite differences.
    Returns:
      A numpy array representing the Hessian matrix.
    """
    n = len(x)
    hessian = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            x_plus_i_plus_j = x.copy()
            x_plus_i_plus_j[i] += h
            x_plus_i_plus_j[j] += h

            x_plus_i_minus_j = x.copy()
            x_plus_i_minus_j[i] += h
            x_plus_i_minus_j[j] -= h

            x_minus_i_plus_j = x.copy()
            x_minus_i_plus_j[i] -= h
            x_minus_i_plus_j[j] += h

            x_minus_i_minus_j = x.copy()
            x_minus_i_minus_j[i] -= h
            x_minus_i_minus_j[j] -= h

            hessian[i, j] = (func(x_plus_i_plus_j) - func(x_plus_i_minus_j) - func(x_minus_i_plus_j) + func(x_minus_i_minus_j)) / (4 * h * h)
    return hessian

# Example usage
point = np.array([2.0, 3.0])
hessian_approx = numerical_hessian(function_example, point)
print(hessian_approx)
```
This Python code implements a numerical approximation of the Hessian using central finite differences. It perturbates the input vector `x` along each dimension *i* and *j* by a small step *h* and applies the central difference formula. Note, that while conceptually simple, finite difference approximation can be sensitive to the step size *h* and may suffer from numerical cancellation for very small *h*.

**Example 2: Symbolic Differentiation with SymPy (Python)**

```python
import sympy

x, y = sympy.symbols('x y')

f = x**2 * y + 3 * y**2 - x

# First partial derivatives
fx = sympy.diff(f, x)
fy = sympy.diff(f, y)

# Second partial derivatives
fxx = sympy.diff(fx, x)
fxy = sympy.diff(fx, y)
fyx = sympy.diff(fy, x)
fyy = sympy.diff(fy, y)

# Construct the Hessian matrix
hessian_matrix = sympy.Matrix([[fxx, fxy], [fyx, fyy]])

print(hessian_matrix)

# Example of evaluation (specific point)
x_value = 2
y_value = 3

hessian_evaluated = hessian_matrix.subs({x: x_value, y: y_value})
print(hessian_evaluated)
```

This example employs SymPy, a Python library for symbolic computation. It defines symbolic variables *x* and *y*, represents the function *f*, and then performs symbolic differentiation. The result is a symbolic Hessian matrix. This approach produces exact derivatives without numerical approximation. The `.subs()` method allows evaluation of the matrix at specific values. Symbolic differentiation provides an accurate representation of the Hessian, but can become computationally expensive for complex functions.

**Example 3: Automatic Differentiation (AD) with JAX (Python)**

```python
import jax
import jax.numpy as jnp

def jax_function_example(x):
  """A function in JAX. Input must be a JAX array."""
  return x[0]**2 * x[1] + 3 * x[1]**2 - x[0]

# Create the Hessian function using jax.hessian()
hessian_jax = jax.hessian(jax_function_example)

# Evaluate at a point
point_jax = jnp.array([2.0, 3.0])
hessian_jax_evaluated = hessian_jax(point_jax)

print(hessian_jax_evaluated)
```

This code leverages JAX, a library well-suited for high-performance numerical computation and automatic differentiation. The `jax.hessian()` function takes a function and automatically generates a function for its Hessian. JAX efficiently calculates derivatives by tracking the computational graph of the provided function, enabling efficient automatic differentiation. This is a robust approach when exact symbolic forms aren't necessary, yet fast and accurate numerical results are crucial.

For further exploration, textbooks on numerical optimization offer comprehensive coverage of the theory and applications of the Hessian matrix. Additionally, many advanced numerical methods texts cover various techniques for computing and utilizing the Hessian, including topics like quasi-Newton methods which approximate the Hessian without directly calculating second derivatives. Books focusing on scientific computing with Python can provide practical guidance on using libraries like SciPy, SymPy, and JAX. Lastly, open-source documentation for numerical and symbolic computation libraries often provides example code and explanations.
