---
title: "How can local minima of an unknown convex function be found?"
date: "2025-01-30"
id: "how-can-local-minima-of-an-unknown-convex"
---
Convex functions, while possessing the desirable property of a single global minimum, present a challenge when their analytical form is unknown.  Direct methods relying on derivatives or gradient information are inapplicable in such scenarios.  My experience with optimization problems in high-dimensional spaces, particularly those arising from inverse problems in geophysical modelling, has led me to appreciate the efficacy of derivative-free optimization methods for addressing this very issue.  These methods cleverly navigate the function landscape without relying on explicit knowledge of its gradient or Hessian.

The core strategy for finding local minima (and, by extension, the global minimum in a convex function) in this context involves iterative exploration of the function's domain.  Each iteration involves sampling the function at various points, using the collected information to refine the search direction and step size, ultimately converging towards a minimum.  The effectiveness of these algorithms depends on several factors, including the sampling strategy, the search algorithm employed, and the inherent "roughness" of the function.

The most robust strategies in my experience fall under the umbrella of model-based optimization. These methods build a surrogate model of the unknown function based on the observed function values at sampled points.  This surrogate model, often a polynomial or radial basis function interpolation, is then optimized to suggest the next point for evaluation.  This iterative process continues until a convergence criterion is met, typically based on the change in function value or the distance between successive iterates.  Let's examine three prominent examples:

**1. Nelder-Mead Simplex Method:**  This method, while not strictly model-based, is remarkably effective and easy to implement.  It maintains a simplex, a geometric figure (a triangle in two dimensions, a tetrahedron in three, and so on) whose vertices represent points at which the function has been evaluated.  The algorithm iteratively modifies the simplex based on the function values at its vertices.  Points with higher function values are reflected, expanded, or contracted, while the simplex is shrunk if the algorithm fails to make sufficient progress.  Its simplicity makes it computationally inexpensive, but its convergence rate can be slow compared to more sophisticated methods.

```python
import numpy as np
from scipy.optimize import minimize

def unknown_convex_function(x):
    # Replace this with your actual unknown function
    return np.sum(x**2) + np.sum(np.sin(x)) #Example of a convex function

# Initial guess
x0 = np.array([1.0, 2.0, 3.0])

# Nelder-Mead optimization
result = minimize(unknown_convex_function, x0, method='Nelder-Mead')

print(result) #Prints the optimized parameters and other relevant information
```

The `scipy.optimize` library provides a readily available implementation of the Nelder-Mead algorithm.  Note that the `unknown_convex_function` placeholder should be replaced with the actual function under consideration.  The algorithm's simplicity is evident in its minimal input requirements—only the function and an initial guess are needed.  The output contains the optimized parameter values, function value at the minimum, and other convergence information.  This makes it exceptionally practical for initial explorations.


**2. Pattern Search Methods:**  These methods are based on systematic exploration of the function's domain using a grid-based or directional search.  They iteratively refine the search region, gradually reducing the step size, until a minimum is located.  Pattern search methods are particularly appealing for their robustness and ease of implementation, even in high-dimensional spaces.  They are less susceptible to getting stuck in local minima, particularly relevant when dealing with noisy function evaluations.  However, their convergence rate can be relatively slow.

```python
from scipy.optimize import basinhopping

# Define the unknown convex function (same as above)
#...

# Initial guess
x0 = np.array([1.0, 2.0, 3.0])

# Basin hopping optimization
minimizer_kwargs = {"method": "Nelder-Mead"} #Can use other methods for local search
result = basinhopping(unknown_convex_function, x0, minimizer_kwargs=minimizer_kwargs, niter=100)

print(result)
```

This example utilizes `basinhopping`, a global optimization algorithm in `scipy.optimize` that incorporates local optimization (here, Nelder-Mead) within a broader exploration scheme. The `niter` parameter controls the number of iterations.  The combination enhances robustness against getting trapped in local minima, a critical consideration for non-smooth functions. The `minimizer_kwargs` dictionary allows for tuning the underlying local optimization method.


**3. Surrogate Model-Based Optimization with Radial Basis Functions (RBF):** This approach constructs a surrogate model using radial basis functions to approximate the unknown function. The surrogate model allows for inexpensive evaluations, leading to quicker convergence.  After constructing the RBF model, the next evaluation point is determined by optimizing the surrogate model, often using gradient-based methods. This process iteratively refines the surrogate model and the approximation of the minimum. This offers superior convergence speed compared to simplex or pattern search, but requires careful selection of hyperparameters such as the RBF kernel and regularization parameters.

```python
from scipy.interpolate import Rbf
from scipy.optimize import minimize

# Sample points and function values (obtained from evaluations of the unknown function)
x = np.array([[1, 2], [3, 4], [5, 6]])  # Example sample points
y = np.array([unknown_convex_function(xi) for xi in x])  # Corresponding function values

# Create RBF model
rbf = Rbf(*x.T, y)

# Define objective function for surrogate model optimization
def objective_function(p):
    return rbf(*p)

# Optimize the surrogate model
result = minimize(objective_function, np.array([2,3]), method='BFGS') #BFGS is a gradient-based method

print(result) #Optimized point from the surrogate model
```


This code snippet shows a basic framework.  A more robust implementation would involve iterative model refinement and appropriate stopping criteria.  The chosen method, BFGS, requires gradient information (which is readily obtained from the surrogate model).  The use of an interpolation model accelerates the convergence compared to the previous methods.


**Resource Recommendations:**

* Numerical Optimization textbooks by Nocedal and Wright, and Bertsekas.  These provide a rigorous treatment of various optimization algorithms and their convergence properties.
* Books on approximation theory, focusing on radial basis functions and spline interpolation.  Understanding the underlying surrogate model is essential for effective implementation of model-based methods.
* Research papers on derivative-free optimization, focusing on specific algorithms and their application domains.


In conclusion, selecting an appropriate method for locating the minimum of an unknown convex function requires careful consideration of the problem's specific characteristics and computational resources.  While the Nelder-Mead method offers simplicity, pattern search excels in robustness, and RBF-based methods provide potentially faster convergence; a tailored approach often yields the best results.  The choice often involves a trade-off between computational cost and the desired accuracy of the solution.
