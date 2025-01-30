---
title: "Why do Gurobi and OSQP yield different QP optimization results?"
date: "2025-01-30"
id: "why-do-gurobi-and-osqp-yield-different-qp"
---
Quadratic programming (QP) solvers, despite aiming for the same mathematical objective, can produce differing results due to inherent algorithmic differences and numerical tolerances.  My experience optimizing large-scale portfolio problems frequently highlighted this discrepancy between Gurobi and OSQP.  The core issue stems from the distinct approaches these solvers employ: Gurobi uses a primal-dual interior-point method, while OSQP leverages an operator splitting method based on ADMM (Alternating Direction Method of Multipliers). These fundamental differences in methodology lead to variations in solution accuracy and convergence behavior, particularly when dealing with ill-conditioned or numerically challenging problems.


**1.  Algorithmic Divergence and Numerical Precision:**

Interior-point methods, like the one implemented in Gurobi, directly tackle the Karush-Kuhn-Tucker (KKT) conditions of the QP problem.  They iteratively refine primal and dual variables until they satisfy these conditions within a specified tolerance.  This tolerance, however, represents a critical point of divergence.  Gurobi offers fine-grained control over these tolerances, allowing users to prioritize solution accuracy or computational speed.

OSQP, on the other hand, employs an iterative ADMM approach. ADMM decomposes the problem into smaller, more manageable subproblems, solved sequentially. This decomposition inherently introduces approximations and iterative refinements. While generally robust,  ADMM's convergence rate can be sensitive to problem structure and parameter tuning.  The default tolerances within OSQP might be less stringent than those typically used in Gurobi, leading to a less precise solution, although potentially with a faster solution time.  Furthermore, the underlying linear algebra operations employed by each solver introduce subtle numerical errors that accumulate differently across iterations.  These cumulative errors, particularly significant in large or ill-conditioned problems, can further contribute to observed discrepancies in the final solutions.


**2. Code Examples and Commentary:**

Let's illustrate this with three Python examples, focusing on the impact of problem structure and solver parameters.

**Example 1:  A Simple Well-Conditioned QP:**

```python
import numpy as np
from osqp import OSQP
import gurobipy as gp

# Define a simple QP problem
P = np.array([[2, 0], [0, 1]])
q = np.array([-1, -2])
A = np.array([[1, 1]])
l = np.array([1])
u = np.array([1])

# Solve with Gurobi
m = gp.Model("qp")
x = m.addVars(2, lb=-gp.GRB.INFINITY)
m.setObjective(0.5 * x[0] * x[0] + x[0] + 0.5 * x[1] * x[1] + 2 * x[1], gp.GRB.MINIMIZE)
m.addConstr(x[0] + x[1] == 1)
m.optimize()
gurobi_sol = np.array([x[i].x for i in range(2)])

# Solve with OSQP
prob = OSQP()
prob.setup(P=P, q=q, A=A, l=l, u=u)
osqp_sol = prob.solve().x

print("Gurobi Solution:", gurobi_sol)
print("OSQP Solution:", osqp_sol)

```

In this simple, well-conditioned problem, both solvers are likely to yield very similar results. The differences, if any, are likely attributable to rounding errors and minor differences in the termination criteria.


**Example 2:  An Ill-Conditioned QP:**

```python
import numpy as np
from osqp import OSQP
import gurobipy as gp

# Define an ill-conditioned QP problem (nearly singular P)
P = np.array([[1e-6, 0], [0, 1]])
q = np.array([-1, -2])
A = np.array([[1, 1]])
l = np.array([1])
u = np.array([1])

# Solve using Gurobi and OSQP (as in Example 1)

# ... (same Gurobi and OSQP solving code as Example 1) ...
```

Here, the near singularity of matrix P makes the problem numerically challenging. The discrepancies between Gurobi and OSQP solutions are more pronounced due to the different ways the solvers handle numerical instability.  Gurobi, with its more sophisticated handling of numerical issues, may produce a more accurate solution compared to OSQP.


**Example 3:  Impact of Solver Parameters:**

```python
import numpy as np
from osqp import OSQP
import gurobipy as gp

# ... (Define a QP problem –  choose any from examples above) ...


# Gurobi with tighter tolerances
m = gp.Model("qp")
m.Params.OptimalityTol = 1e-10 # Tighter tolerance
# ... (rest of Gurobi setup) ...


# OSQP with altered settings
prob = OSQP()
prob.setup(P=P, q=q, A=A, l=l, u=u, eps_abs=1e-10, eps_rel=1e-10) # Adjust tolerances

# ... (Solve using Gurobi and OSQP) ...
```

This example illustrates the impact of altering solver parameters. By adjusting tolerances (`OptimalityTol` in Gurobi, `eps_abs` and `eps_rel` in OSQP), we can influence the solution accuracy and computational cost.  Tighter tolerances lead to more accurate but potentially slower solutions. This experiment is crucial for understanding the trade-offs between precision and speed.


**3. Resource Recommendations:**

For a deeper understanding, I recommend studying the theoretical underpinnings of interior-point methods and ADMM.  Consult textbooks on optimization theory and numerical linear algebra.  Furthermore, the documentation for both Gurobi and OSQP provides comprehensive explanations of their algorithms and parameters.  Examining the parameter settings for each solver will clarify their impact on solution quality.  Finally, exploring research articles comparing the performance of different QP solvers on various problem classes would be highly beneficial.
