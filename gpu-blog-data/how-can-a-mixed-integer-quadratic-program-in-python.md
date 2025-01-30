---
title: "How can a mixed-integer quadratic program in Python, using CVXPY, optimize a portfolio with weight constraints?"
date: "2025-01-30"
id: "how-can-a-mixed-integer-quadratic-program-in-python"
---
The core challenge in optimizing a portfolio with weight constraints using a mixed-integer quadratic program (MIQP) in CVXPY lies in effectively representing the integrality constraints while maintaining numerical stability and computational tractability.  My experience optimizing large-scale financial models has shown that naive formulations often lead to solver failures or excessively long computation times.  Careful problem structuring is paramount.

**1.  Clear Explanation:**

A portfolio optimization problem often aims to maximize expected return subject to risk constraints and weight restrictions.  The quadratic nature stems from incorporating a risk measure, typically variance or covariance, into the objective function.  Mixed-integer programming is necessary when imposing integrality constraints on the portfolio weights – for instance, requiring that a particular asset is either fully included or entirely excluded (binary constraint) or restricting asset holdings to discrete quantities (integer constraint).

Formally, the problem can be stated as follows:

Maximize:  `μᵀx - λxᵀΣx`

Subject to:

* `∑ᵢ xᵢ = 1`  (Full investment)
* `lᵢ ≤ xᵢ ≤ uᵢ` (Weight bounds for each asset i)
* `xᵢ ∈ {0, 1}` or `xᵢ ∈ Z⁺` (Integrality constraints: binary or integer, respectively)

Where:

* `x` is the vector of asset weights.
* `μ` is the vector of expected returns.
* `Σ` is the covariance matrix of asset returns.
* `λ` is the risk aversion parameter (a scalar).
* `lᵢ` and `uᵢ` represent the lower and upper bounds for the weight of asset `i`.
* `Z⁺` denotes the set of non-negative integers.

The choice between binary and integer constraints depends on the specific application.  Binary constraints enforce strict selection or exclusion, while integer constraints allow for multiple units of each asset.  The efficient handling of these constraints within the CVXPY framework requires careful consideration of solver selection and problem structuring.


**2. Code Examples with Commentary:**

**Example 1: Binary Constraints – Portfolio Selection with Asset Exclusion**

```python
import cvxpy as cp
import numpy as np

# Sample data (replace with your actual data)
n = 5  # Number of assets
mu = np.array([0.1, 0.15, 0.08, 0.12, 0.05])  # Expected returns
Sigma = np.array([[0.04, 0.01, 0.005, 0.015, 0.002],
                  [0.01, 0.09, 0.01, 0.02, 0.008],
                  [0.005, 0.01, 0.06, 0.008, 0.005],
                  [0.015, 0.02, 0.008, 0.12, 0.01],
                  [0.002, 0.008, 0.005, 0.01, 0.03]]) # Covariance matrix
lambda_risk = 1 # Risk aversion parameter

# Variables
x = cp.Variable(n, boolean=True) # Binary variables for asset selection

# Objective function
objective = cp.Maximize(mu @ x - lambda_risk * cp.quad_form(x, Sigma))

# Constraints
constraints = [cp.sum(x) == 1] # Full investment

# Problem definition and solving
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.ECOS) # or cp.SCS, or other suitable solver

# Solution
print("Optimal portfolio weights:", x.value)
print("Optimal objective value:", problem.value)
```

This example utilizes boolean variables (`boolean=True`) directly within the CVXPY variable definition to enforce binary constraints.  The ECOS solver is a good choice for smaller problems, while SCS is a robust alternative for larger-scale instances.  The choice of solver depends heavily on the problem size and structure.

**Example 2: Integer Constraints – Portfolio with Discrete Asset Holdings**

```python
import cvxpy as cp
import numpy as np

# Sample data (same as Example 1)

#Variables
x = cp.Variable(n, integer=True) # Integer variables

# Objective function (same as Example 1)

# Constraints
constraints = [cp.sum(x) == 10, x >= 0] # Total holdings of 10 units

#Problem definition and solving (same as Example 1, but might require a different solver like CPLEX or Gurobi)
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.GUROBI)

# Solution (same as Example 1)
```

This demonstrates the use of integer variables (`integer=True`).  Note that this problem now requires a solver capable of handling integer programming, such as Gurobi or CPLEX.  These solvers are generally commercial, offering superior performance on complex MIQP problems compared to open-source alternatives like ECOS or SCS.  The `constraints` are adjusted to reflect the total number of units being invested.

**Example 3: Combining Binary and Integer Constraints**

```python
import cvxpy as cp
import numpy as np

# Sample data (same as Example 1)

# Variables
x_binary = cp.Variable(3, boolean=True) # Binary selection of three asset groups
x_integer = cp.Variable(n, integer=True) # Integer holdings within each group

# Asset grouping (example: Assets 0-1 in group 0, 2-3 in group 1, 4 in group 2)
group_mapping = [[0,1],[2,3],[4]]

# Objective function
objective = cp.Maximize(mu @ x_integer - lambda_risk * cp.quad_form(x_integer, Sigma))

# Constraints
constraints = [cp.sum(x_binary) == 2, # Select two groups
               x_integer[group_mapping[0]] <= 5*x_binary[0], # Limit holdings based on group selection
               x_integer[group_mapping[1]] <= 3*x_binary[1],
               x_integer[group_mapping[2]] <= 2*x_binary[2],
               x_integer >= 0]

# Problem definition and solving (requires a solver like Gurobi or CPLEX)
problem = cp.Problem(objective, constraints)
problem.solve(solver=cp.GUROBI)

# Solution (display weights for each asset)
print("Optimal portfolio weights:", x_integer.value)
print("Optimal objective value:", problem.value)

```

This example showcases a more complex scenario combining binary and integer variables.  Here, we select groups of assets using binary variables, then define integer constraints within each selected group.  The constraints enforce the restriction of investing only in selected groups and limiting the quantity of investment within each chosen group.  This requires a significantly more powerful solver like Gurobi or CPLEX due to the increased complexity.


**3. Resource Recommendations:**

The CVXPY documentation itself is an invaluable resource.  Furthermore, a solid understanding of linear and integer programming theory is crucial.  Textbooks on optimization, particularly those focusing on mixed-integer programming, are highly beneficial.  Finally, the documentation for commercial solvers such as Gurobi and CPLEX should be consulted for advanced techniques and performance tuning.  Familiarity with numerical linear algebra will aid in interpreting and troubleshooting potential issues related to the covariance matrix.
