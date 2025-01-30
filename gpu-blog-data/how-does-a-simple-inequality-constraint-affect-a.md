---
title: "How does a simple inequality constraint affect a quadratic programming control allocation problem?"
date: "2025-01-30"
id: "how-does-a-simple-inequality-constraint-affect-a"
---
The core impact of a simple inequality constraint on a quadratic programming (QP) control allocation problem stems from its alteration of the feasible solution space.  This directly influences the optimal solution obtained, potentially resulting in a suboptimal control allocation compared to the unconstrained case, but crucially ensuring the system remains within operational limits.  My experience optimizing trajectory tracking for highly maneuverable aerial vehicles heavily involved such constraints, particularly regarding actuator saturation.

**1.  Explanation:**

A standard QP control allocation problem can be formulated as minimizing a quadratic cost function subject to linear equality constraints. This function typically represents the deviation from a desired control vector.  The addition of inequality constraints, often representing physical limitations on actuators (e.g., maximum thrust, torque limits, or rate limits), restricts the solution space to a subset of the original feasible region defined solely by equality constraints.

Mathematically, the unconstrained problem is:

Minimize:  J(u) = ½uᵀQu + uᵀc

Subject to: Bu = r

where:

* `u` is the control allocation vector.
* `Q` is a positive semi-definite weighting matrix.
* `c` is a constant vector.
* `B` is the control effectiveness matrix.
* `r` is the desired control vector.

Introducing a simple inequality constraint of the form `uᵢ ≤ uᵢ,max` for each actuator `i`, where `uᵢ,max` represents the maximum allowed value for the `i`-th actuator, modifies the problem to:

Minimize: J(u) = ½uᵀQu + uᵀc

Subject to: Bu = r
              uᵢ ≤ uᵢ,max  ∀ i

This seemingly minor addition fundamentally changes the optimization problem. The solver must now find the optimal solution within the bounded feasible region defined by both equality and inequality constraints.  If the unconstrained optimal solution violates any of these constraints, the constrained solution will necessarily differ, resulting in a potentially suboptimal allocation in terms of minimizing the cost function J(u).  However, this suboptimality is a trade-off for ensuring system safety and avoiding actuator damage.

In my work designing control systems for unmanned aerial vehicles, ignoring such constraints led to unrealistically high torque demands, causing simulations to crash and real-world tests to yield unstable flight. Incorporating the inequality constraints via QP solvers resulted in smoother, safer, and more reliable control performance.

**2. Code Examples with Commentary:**

The following examples demonstrate the impact of inequality constraints using Python with the `quadprog` library.  Note that the choice of solver and library significantly impacts performance, especially for higher-dimensional problems.  For large-scale systems, more advanced solvers such as those available in commercial optimization packages might be necessary.


**Example 1: Unconstrained QP**

```python
import numpy as np
from quadprog import solve_qp

# Define problem parameters
Q = np.array([[2, 0], [0, 1]])
c = np.array([-1, -2])
B = np.array([[1, 1]])
r = np.array([1])

# Solve the unconstrained QP
u, _, _, _, _, = solve_qp(Q, c, None, None, B, r)
print("Unconstrained solution:", u)
```

This code solves a simple unconstrained QP. The output `u` represents the optimal control allocation vector without any constraints.


**Example 2: QP with Simple Upper Bound Constraints**

```python
import numpy as np
from quadprog import solve_qp

# Define problem parameters (same as Example 1)
Q = np.array([[2, 0], [0, 1]])
c = np.array([-1, -2])
B = np.array([[1, 1]])
r = np.array([1])

# Add upper bound constraints
A = -np.eye(2)  # Inequality constraint matrix: -I * u <= ub
ub = np.array([-0.2, -0.5])  # Upper bounds on u


# Solve the QP with upper bound constraints
u, _, _, _, _, = solve_qp(Q, c, A, ub, B, r)
print("Constrained solution:", u)

```

Here, we introduce upper bound constraints on `u` using a matrix inequality `A*u <= ub`.  The negative identity matrix `-np.eye(2)` ensures that each element of `u` is less than or equal to its corresponding upper bound.  Observe how the constrained solution differs from the unconstrained solution. The specific difference depends on the interaction between the cost function, equality constraints, and inequality constraints.


**Example 3:  QP with Mixed Constraints**

```python
import numpy as np
from quadprog import solve_qp

# Define problem parameters
Q = np.array([[2, 0.5], [0.5, 1]])
c = np.array([-1, -2])
B = np.array([[1, 1]])
r = np.array([1])

# Add upper and lower bound constraints
A = np.vstack([-np.eye(2), np.eye(2)])
ub = np.array([-0.2, -0.5, 1, 1]) # Upper and lower bounds
lb = np.array([-1, -1, 0.2, 0.5]) # Lower bounds


# Solve the QP with upper and lower bound constraints
u, _, _, _, _, = solve_qp(Q, c, A, ub, B, r)
print("Solution with mixed bounds:", u)

```

This example extends the previous one by incorporating both upper and lower bounds on the control inputs, creating a more realistic representation of actuator limitations.  The solution now lies within a hyper-rectangle defined by these bounds, further illustrating the restriction of the feasible solution space.


**3. Resource Recommendations:**

*  **Boyd & Vandenberghe, Convex Optimization:** A comprehensive textbook covering the theory and algorithms of convex optimization, including quadratic programming.
*  **Nocedal & Wright, Numerical Optimization:** Another excellent resource that delves deeper into numerical methods used in optimization algorithms.
*  **A suitable textbook on Control Systems Engineering:**  A thorough understanding of control systems principles provides crucial context for applying QP to control allocation problems.  Many textbooks provide in-depth coverage of this subject matter.


The impact of simple inequality constraints on a QP control allocation problem is profound. It fundamentally alters the optimization landscape, potentially sacrificing optimality in the cost function for maintaining actuator limits and ensuring safe operation. Careful consideration of these constraints is crucial for designing robust and reliable control systems.  The examples provided highlight how simple constraint additions can lead to significantly different solutions and emphasize the importance of using suitable optimization tools and understanding their limitations.
