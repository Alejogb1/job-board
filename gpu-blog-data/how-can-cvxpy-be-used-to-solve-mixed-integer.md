---
title: "How can cvxpy be used to solve mixed-integer second-order cone programs?"
date: "2025-01-30"
id: "how-can-cvxpy-be-used-to-solve-mixed-integer"
---
Mixed-integer second-order cone programs (MISOCPs) represent a challenging class of optimization problems due to the combination of continuous variables within second-order cone constraints and the added complexity of integer constraints on some variables. I've encountered these frequently in supply chain optimization and resource allocation problems where discrete choices interact with continuous resource levels. `cvxpy`, while primarily designed for convex problems, does provide the framework to tackle MISOCPs by leveraging external solvers capable of handling mixed-integer programming.

The core challenge isn't that `cvxpy` inherently solves MISOCPs itself, but rather that it acts as a modeling language that can formulate the problem in a way suitable for external solvers like `ECOS_BB`, `SCIP`, or `GUROBI`. `cvxpy` translates the problem description into a form that these solvers understand, thus allowing us to abstract away the low-level solver-specific details while still harnessing their power for non-convex problems.  It is crucial to understand that when encountering integer constraints, the underlying optimization landscape moves from convex to generally non-convex. This fundamentally changes the solution methodologies and computational complexity.

The basic principle of leveraging `cvxpy` for an MISOCP involves two primary steps: problem formulation and solver selection. First, one must precisely define all variables, including identifying which are integer and which are continuous. Next, the objective function and all constraint functions need to be formulated. Crucially, any second-order cone constraints must adhere to the format `norm(Ax + b) <= c^T x + d` where `A`, `b`, `c`, and `d` are constant matrices and vectors, and `x` is the variable vector. Finally, when solving the problem via `prob.solve()`, specifying an appropriate mixed-integer capable solver is required through the `solver` keyword argument. Without providing a valid solver, `cvxpy` will attempt to utilize a convex-only solver, which will fail to handle integer constraints.

Let’s examine a concrete example. Suppose we have a simplified resource allocation problem. We want to decide whether or not to build two different factory types, with capacities to produce goods given we choose to build. The decision to build is a binary integer variable;  the actual production quantity will be a continuous variable, but it is subject to a second-order cone constraint depending on the building choice. This ensures that we don't attempt to produce more than our capacity.

```python
import cvxpy as cp
import numpy as np

# Number of factory types
n = 2

# Variables
x = cp.Variable(n, integer=True)  # Binary integer variable (0 or 1) for factory choice
p = cp.Variable(n) # Production quantity (continuous) for each factory

# Parameters
capacity = np.array([10, 15]) # Production capacity of each factory (when built)
cost_per_unit = np.array([5, 7]) # Cost per unit produced in each factory
fixed_cost = np.array([50, 75]) # Fixed cost of building each factory
demand = 20  # Target demand we want to meet

# Objective function: minimize cost
objective = cp.Minimize(cp.sum(cost_per_unit @ p) + fixed_cost @ x)

# Constraints
constraints = [
    cp.sum(p) >= demand, # Satisfy the target demand
    0 <= p,
    p <= capacity * x, # Production depends on factory selection (if a factory is not selected x=0, and thus p=0)
    # Second-order cone constraint: production <= capacity_coefficient * sqrt(capacity * x)
    # This constraint is used in more complex cases with diminishing return and the example is for demonstration purposes
    cp.norm(p) <= np.sqrt(np.sum(capacity*x)) # Illustrative example constraint
]

# Formulate problem
prob = cp.Problem(objective, constraints)

# Solve problem using SCIP as a solver (requires SCIP installation, example using SCIP)
# The solver must be explicitly stated and must support MISOCP
prob.solve(solver=cp.SCIP, verbose=True)

# Print solution
if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
    print("Optimal solution found.")
    print("Optimal factory selections (0 means no, 1 means yes):", x.value)
    print("Production quantities:", p.value)
    print("Optimal cost:", prob.value)
else:
    print("Solution not optimal or failed to solve.")
    print("Solver status:", prob.status)

```

In this example, the `x` variable represents the choice of whether to build a factory. The `p` variable represents the quantity produced by each factory. The key addition here is that `x` is declared as `integer=True`. The second-order cone constraint `cp.norm(p) <= np.sqrt(np.sum(capacity*x))` ties production capacity with factory selection.  When `x[i]` is 0 (the factory is not built), the production limit is 0.  Crucially, without `solver=cp.SCIP`, the solve attempt would not succeed because the default solvers do not handle integer constraints.  The verbose output is included to help with debugging and understanding the performance of the solve. This illustrative second-order constraint is not strictly necessary for simple allocation, but it highlights the structure that would be used for more complex modeling.

Next, consider an example involving portfolio optimization with a budget constraint and a minimum quantity requirement for buying specific types of assets, with each asset type having varying risk profiles, formalized via a second order cone constraint.  Here, integer constraints on assets purchased, combined with a risk threshold modeled as a second-order cone constraint, produce a viable MISOCP.

```python
import cvxpy as cp
import numpy as np

# Number of asset types
n = 3

# Variables
x = cp.Variable(n, integer=True)  # Integer number of assets to buy
w = cp.Variable(n) # Fractional amounts invested in each asset
risk_factor = np.array([2, 1.5, 3]) # Risk associated with each asset
price = np.array([10, 20, 15]) # Price of each asset

budget = 100
min_assets = 1 # Minimum number of each asset purchased, if purchased.
target_return = 0.1 # Target return of the portfolio, relative to investment
expected_return = np.array([0.1, 0.05, 0.15]) # Expected return of each asset

# Objective function: maximize return
objective = cp.Maximize(expected_return @ w)

constraints = [
  w >= 0, # Fractional investment must be non-negative
  cp.sum(price @ x) <= budget, # total spending must be under budget
  # Second-order cone constraint, limiting the portfolio risk, based on a risk factor
  cp.norm(risk_factor @ w) <= 0.4, # total risk (in terms of fraction of total portfolio) is limited by 0.4
  cp.sum(w) == 1, # Sum of all the assets must be 1 (i.e. fractions add up to 1)
  w <= x, # If an asset is not purchased (x = 0), then no investment can be made in that asset (w=0)
  x * min_assets <= w
  ]
# Formulate problem
prob = cp.Problem(objective, constraints)

# Solve problem using GUROBI
prob.solve(solver=cp.GUROBI, verbose = True)

# Print solution
if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
    print("Optimal solution found.")
    print("Optimal asset purchases (number):", x.value)
    print("Fractional investments:", w.value)
    print("Optimal total return:", prob.value)
else:
    print("Solution not optimal or failed to solve.")
    print("Solver status:", prob.status)

```

In this scenario, the `x` variables represent the quantity of each asset bought, and they must be integers. The continuous variables `w` determine the fraction of capital allocated to the assets. A second-order cone constraint limits the portfolio risk based on asset allocation. The objective is to maximize the expected return given budget limitations and risk limits. This code requires GUROBI to be installed, highlighting the need for different solvers in MISOCP problems. Note the additional constraint `x * min_assets <= w`, which in combination with the constraint `w <= x` ensures that the minimum number of each asset is purchased when a positive fractional investment is allocated.

Finally, let’s examine a location selection problem, common in network design or facility placement, involving a combination of discrete choices to build a facility and continuous flow decisions. This highlights the interplay of different decision types.

```python
import cvxpy as cp
import numpy as np

# Number of possible facility locations
n = 3

# Number of customers
m = 2

# Variables
x = cp.Variable(n, integer=True)  # Binary variable: build facility (1) or not (0)
f = cp.Variable((m, n))  # Flow between customer and facility

# Parameters
demand = np.array([10, 15]) # Demand from each customer
distance = np.array([[2, 4, 3], [5, 1, 2]]) # Distance between customer and facility
capacity = np.array([20, 25, 18]) # Capacity of each facility
fixed_cost = np.array([50, 60, 45]) # Fixed cost of building a facility
transport_cost = 1

# Objective function: minimize total cost
objective = cp.Minimize(cp.sum(fixed_cost @ x) + cp.sum(transport_cost * distance @ f))

# Constraints
constraints = [
    cp.sum(f, axis=1) == demand,  # Satisfy each customer demand
    f >= 0, # Flow must be non-negative
    cp.sum(f, axis=0) <= capacity * x, # Facility capacity is limited by the facility location choice
     # Second-order cone constraint limiting the maximum flow to a fraction of the capacity if a facility is built
    cp.norm(cp.sum(f, axis=0)) <= 0.8 * capacity*x # Illustrative example constraint: maximum flow <= 80% capacity
 ]

# Formulate problem
prob = cp.Problem(objective, constraints)

# Solve problem using ECOS_BB
prob.solve(solver=cp.ECOS_BB, verbose=True)

# Print solution
if prob.status == cp.OPTIMAL or prob.status == cp.OPTIMAL_INACCURATE:
    print("Optimal solution found.")
    print("Optimal facility locations (0 means no, 1 means yes):", x.value)
    print("Optimal flow allocation:", f.value)
    print("Optimal cost:", prob.value)
else:
    print("Solution not optimal or failed to solve.")
    print("Solver status:", prob.status)
```

Here, the `x` variables represent binary decision of building facilities, and `f` represents the continuous flow from the customer to each facility. The second-order cone constraint further constrains the total flow allowed into each facility relative to its capacity. The solver ECOS_BB is used, a solver specifically created to be interfaced with cvxpy. This demonstrates the practical challenge of choosing the correct solver for the given type of problem.

For further exploration of MISOCP concepts and solvers, I recommend consulting the documentation for `cvxpy`, which provides detailed information about how to utilize the library for various problem types.  Additionally, reviewing the specific documentation for the solvers, such as `SCIP`, `GUROBI`, and `ECOS_BB`, is essential to fully understand the capabilities and nuances of each solver.  Finally, academic literature on optimization techniques provides theoretical background for understanding the complexities of both convex and non-convex optimization and how integer programming alters solutions.
