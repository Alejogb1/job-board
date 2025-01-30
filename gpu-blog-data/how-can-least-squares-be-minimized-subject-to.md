---
title: "How can least squares be minimized subject to a constraint?"
date: "2025-01-30"
id: "how-can-least-squares-be-minimized-subject-to"
---
The core challenge in minimizing least squares subject to a constraint lies in the inherent conflict between finding the point of minimum error and adhering to the imposed restriction.  Direct application of standard least-squares techniques will generally fail to satisfy the constraint.  My experience working on robust sensor fusion algorithms for autonomous vehicle navigation highlighted this repeatedly.  Effectively, we needed to find the optimal solution within a defined feasible region, a necessity driven by physical limitations and safety considerations.  This often required the application of constrained optimization techniques.

The most straightforward approach, and one I frequently employed, involves the method of Lagrange multipliers. This technique elegantly incorporates the constraint into the objective function, allowing for the simultaneous minimization of the error and satisfaction of the constraint.  Let's consider the standard least-squares problem:

Minimize:  `f(x) = ||Ax - b||²`

where `A` is an `m x n` matrix, `x` is an `n x 1` vector of unknowns, and `b` is an `m x 1` vector of observations.  This is the unconstrained problem.

Now, introduce a constraint of the form:  `g(x) = 0`. This could represent a variety of limitations, such as a fixed sum of parameters, a limited range of values for certain parameters, or adherence to a specific model.

The Lagrangian function is constructed as:

`L(x, λ) = f(x) + λg(x) = ||Ax - b||² + λg(x)`

where λ is the Lagrange multiplier, a scalar value representing the penalty associated with violating the constraint.  The optimal solution is found by solving the system of equations formed by setting the gradient of the Lagrangian to zero:

∇ₓL(x, λ) = 0
g(x) = 0

Solving this system yields the values of `x` and `λ` that minimize the least-squares objective while simultaneously satisfying the constraint.


**Code Example 1: Equality Constraint using `scipy.optimize.minimize`**

Consider a simple example where we want to minimize the sum of squared errors between a linear model and data points, subject to the constraint that the sum of the model parameters must equal one.  This is frequently encountered when dealing with probability distributions.

```python
import numpy as np
from scipy.optimize import minimize

# Sample data
A = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([7, 8, 9])

# Constraint: sum(x) - 1 = 0
def constraint(x):
    return np.sum(x) - 1

# Objective function
def objective(x):
    return np.sum((np.dot(A, x) - b)**2)

# Initial guess
x0 = np.array([0.5, 0.5])

# Optimization using SLSQP (Sequential Least Squares Programming)
result = minimize(objective, x0, constraints={'type': 'eq', 'fun': constraint}, method='SLSQP')

print(result.x)  # Optimal parameters
print(result.fun) # Minimum objective function value
```

This code utilizes the `scipy.optimize.minimize` function with the SLSQP method, which is particularly well-suited for handling equality constraints. The `constraint` function defines the equality constraint, and the `objective` function defines the least-squares objective to be minimized.  SLSQP's strength lies in its ability to efficiently solve these problems, even with many parameters.


**Code Example 2: Inequality Constraint using `cvxpy`**

Inequality constraints are equally important. Imagine needing to estimate model parameters within a specific range.  Convex optimization solvers offer a robust approach.

```python
import cvxpy as cp
import numpy as np

# Sample data (same as before)
A = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([7, 8, 9])

# Define the variable
x = cp.Variable(2)

# Define the objective function
objective = cp.Minimize(cp.sum_squares(A @ x - b))

# Define the constraints (x1 >= 0, x2 >= 0)
constraints = [x >= 0]

# Formulate the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve()

print("Optimal value:", problem.value)
print("Optimal parameters:", x.value)
```

This example employs `cvxpy`, a powerful Python library for convex optimization problems. This allows for the specification of inequality constraints directly, resulting in a more concise and readable code. The `problem.solve()` function automatically selects a suitable solver based on the problem's structure.  I found `cvxpy` particularly useful when dealing with complex systems exhibiting convexity.


**Code Example 3:  Quadratic Programming Formulation**

Many constrained least squares problems can be reformulated as quadratic programming (QP) problems.  This offers computational advantages, particularly for large-scale problems.

```python
import numpy as np
from scipy.optimize import quadprog

# Sample data (same as before)
A = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([7, 8, 9])

# Define the matrices for the QP problem
P = 2 * np.dot(A.T, A) # Hessian of the objective function
q = -2 * np.dot(A.T, b) # Gradient of the objective function
G = -np.eye(2) # Inequality constraint matrix (x >= 0)
h = np.zeros(2) # Inequality constraint vector

# Solve the QP problem
result = quadprog.solve_qp(P, q, G, h)

print("Optimal parameters:", result[0])
print("Minimum objective function value:", np.sum((np.dot(A, result[0]) - b)**2))

```

This code leverages `scipy.optimize.quadprog` to directly solve the QP formulation.  The `P` and `q` matrices represent the quadratic and linear terms of the objective function, while `G` and `h` define the inequality constraints. This approach is computationally efficient and well-suited for problems with numerous parameters and constraints.  My preference for this method stemmed from its direct application to the QP structure, avoiding the overhead of general-purpose solvers.


**Resource Recommendations:**

*  *Numerical Optimization* by Jorge Nocedal and Stephen J. Wright
*  *Convex Optimization* by Stephen Boyd and Lieven Vandenberghe
*  A textbook on linear algebra covering matrix decompositions and vector spaces.
*  Documentation for `scipy.optimize` and `cvxpy` Python libraries.


These resources provide a comprehensive understanding of the underlying mathematical principles and practical techniques for solving constrained least-squares problems. Choosing the optimal method depends on the specific problem structure, the nature of the constraints, and the available computational resources.  Understanding the strengths and weaknesses of each approach, as developed through practical experience, is crucial for effective problem-solving.
