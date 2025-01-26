---
title: "How can SLSQP be used with nonlinear constraints?"
date: "2025-01-26"
id: "how-can-slsqp-be-used-with-nonlinear-constraints"
---

Sequential Least Squares Programming (SLSQP), despite its name, is not fundamentally a least-squares method when dealing with nonlinear constraints. Its core mechanism lies in iterative quadratic programming; each iteration refines a model of the objective function and constraints using Taylor expansions, then solves a quadratic subproblem. This contrasts with methods focusing directly on minimizing squared errors, and this distinction is critical for proper application. My experience across various optimization problems, including those in process control and financial modeling, has highlighted the nuanced process of effectively using SLSQP with nonlinear constraints.

SLSQP tackles problems of the following form: minimize f(x) subject to g(x) <= 0, h(x) = 0, where f(x) is the objective function, g(x) represents inequality constraints, and h(x) represents equality constraints, all potentially nonlinear. Unlike methods relying on exact second derivatives (Hessian matrices), SLSQP utilizes a quasi-Newton update scheme, typically BFGS, to approximate the Hessian of the Lagrangian. This avoids computationally expensive full second-order calculations. The crucial element regarding nonlinear constraints is their incorporation within the quadratic programming subproblem solved at each iteration. The original problem is thus approximated by a quadratic objective and linearized versions of the constraints at each step. The solution to this subproblem determines the search direction for the next iterate.

Here’s a breakdown of the process in practical implementation, referencing `scipy.optimize.minimize`, a common implementation of SLSQP:

**1. Problem Formulation:** Define the objective function, inequality constraints, and equality constraints as Python functions. Each constraint function must return the value of the constraint at the given point `x`. For inequality constraints, the function returns a value that should be less than or equal to zero. For equality constraints, it returns a value that should be exactly zero.

**2. Constraint Representation:** `scipy.optimize.minimize` requires constraints to be provided as a list of dictionaries. Each dictionary specifies a single constraint. The `type` key indicates whether it's an inequality (`'ineq'`) or equality (`'eq'`) constraint. The `fun` key stores the function implementing the constraint logic. Additional options, such as a gradient (using `jac`) might improve convergence.

**3. Initial Guess:** Providing a good initial guess, `x0`, is paramount for the success of SLSQP. This is not unique to SLSQP, but its sensitivity to the initial guess can be noticeable, especially with complex, non-convex problems. A poor starting point might lead to a local minimum, divergence, or slow convergence.

**4. Optimization:** Finally, call `scipy.optimize.minimize`, passing the objective function, initial guess, constraint list, and specifying `method='SLSQP'`.

Here are three code examples demonstrating various aspects of nonlinear constraint application:

**Example 1: Simple Constrained Optimization**

```python
import numpy as np
from scipy.optimize import minimize

# Objective function: minimize x^2 + y^2
def objective(x):
    return x[0]**2 + x[1]**2

# Inequality constraint: x + y - 1 >= 0 (equivalent to x + y - 1 <= 0 by negating)
def constraint1(x):
    return -(x[0] + x[1] - 1) # Note the negative sign for <= 0

# Equality constraint: x - y = 0
def constraint2(x):
    return x[0] - x[1]

# Define constraints as a list of dictionaries
cons = ({'type': 'ineq', 'fun': constraint1},
        {'type': 'eq', 'fun': constraint2})

# Initial guess
x0 = np.array([2, 0])

# Perform optimization
result = minimize(objective, x0, method='SLSQP', constraints=cons)

print(result)
```

This first example illustrates a simple case involving two variables and two constraints: one inequality constraint and one equality constraint. Notice the negation within `constraint1` to convert the "greater than or equal to" to a "less than or equal to" form expected by `minimize`. The output of the `result` object contains the optimized variables in `x`, the objective function's minimized value in `fun`, and several other details such as `success` which indicates if the optimization converged correctly. This simple setup highlights how one can set up constraints within the specified structure.

**Example 2: Nonlinear Inequality Constraint**

```python
import numpy as np
from scipy.optimize import minimize

# Objective function: minimize x^2 + y^2
def objective(x):
    return x[0]**2 + x[1]**2

# Nonlinear inequality constraint: x^2 + y^2 - 1 <= 0
def constraint1(x):
  return x[0]**2 + x[1]**2 - 1

# Initial guess
x0 = np.array([2, 2])

# Define constraints as a list of dictionaries
cons = ({'type': 'ineq', 'fun': constraint1})

# Perform optimization
result = minimize(objective, x0, method='SLSQP', constraints=cons)

print(result)
```
In this second example, we replace the linear constraint with a nonlinear inequality, namely a circular constraint of radius one. The objective function is kept identical to illustrate the independent nature of constraints. Despite using `SLSQP` on a circular constraint, which by itself represents a significant challenge for other methods based on the first-order approximation, this example will often converge to the minimum with an initial guess reasonably close to the constraint boundary. The key is how SLSQP forms a quadratic approximation to the objective and uses linearizations of the *constraint* function. The final optimized result typically lands on the boundary of the feasible region (`x[0]**2 + x[1]**2` is approximately 1), as expected from this type of problem.

**Example 3: Multiple Constraints, Including a More Complex One**

```python
import numpy as np
from scipy.optimize import minimize

# Objective function: minimize (x - 2)^2 + (y - 3)^2
def objective(x):
    return (x[0] - 2)**2 + (x[1] - 3)**2

# Inequality constraint: x + y - 4 <= 0
def constraint1(x):
    return x[0] + x[1] - 4

# Nonlinear equality constraint: x^2 + y^2 - 5 = 0
def constraint2(x):
    return x[0]**2 + x[1]**2 - 5

# Initial guess
x0 = np.array([0, 0])

# Define constraints as a list of dictionaries
cons = ({'type': 'ineq', 'fun': constraint1},
         {'type': 'eq', 'fun': constraint2})

# Perform optimization
result = minimize(objective, x0, method='SLSQP', constraints=cons)

print(result)
```

This third example demonstrates the combined usage of nonlinear equality constraints and linear inequality constraints. The nonlinear equality constraint in this example (a circle with radius `sqrt(5)`) coupled with a simple linear inequality constraint and an objective function that seeks the minimum distance to (2, 3) forces the optimization process to find the closest point on the circle satisfying the inequality. This showcases the power of SLSQP in complex scenarios with multiple constraints where standard methods might struggle. This will often converge to a point along the circle such that x+y <= 4.

Through these examples, it is clear that `SLSQP` effectively works with nonlinear constraints by iteratively approximating the feasible region. Each iteration solves a quadratic program that considers linearized approximations of the constraints; this allows the algorithm to move towards the optimum with each step, adapting to the complexity of nonlinear constraints effectively. One might use more complex functions than just polynomial ones, so long as these are computationally feasible.

For further study, consult Numerical Optimization by Nocedal and Wright, a standard text covering optimization algorithms in depth. Also explore the documentation for scipy.optimize, which provides details on the various solvers and their parameters, and provides a guide to best practices. Books on calculus and linear algebra should also complement the understanding of the foundational mathematical principles behind these methods. Finally, understanding the limitations of optimization algorithms in terms of convergence and robustness is essential to make well-informed decisions.
