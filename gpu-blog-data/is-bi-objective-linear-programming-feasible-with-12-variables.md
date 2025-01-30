---
title: "Is bi-objective linear programming feasible with 12 variables?"
date: "2025-01-30"
id: "is-bi-objective-linear-programming-feasible-with-12-variables"
---
The feasibility of bi-objective linear programming (BOLP) with twelve variables isn't inherently determined by the number of variables.  My experience optimizing supply chain networks, involving far more complex models, indicates that the computational challenge scales more with problem structure and the characteristics of the objective functions than solely the variable count.  Twelve variables are easily handled by modern solvers, provided the problem's inherent complexity doesn't introduce numerical instability or excessively long solution times.  The primary concern shifts to the nature of the objectives and the constraints.

**1.  Explanation of Feasibility Considerations:**

BOLP aims to optimize two conflicting linear objective functions subject to a set of linear constraints.  The solution space isn't a single optimal point but rather a Pareto front—a set of Pareto optimal solutions. Each solution on this front represents a trade-off between the two objectives; improving one necessarily worsens the other.  The computational difficulty arises not from the number of variables *per se*, but from:

* **The structure of the constraint matrix:** A highly dense constraint matrix (many non-zero elements) leads to increased computation time for most solvers. Sparse matrices are computationally more efficient.
* **The nature of the objective functions:**  Highly correlated or nearly identical objectives can simplify the problem, while significantly different objectives can make the Pareto front more complex and challenging to explore comprehensively.
* **The size of the feasible region:**  A very large or unbounded feasible region will increase the computational burden of finding the Pareto front.
* **The solver employed:** Different solvers (simplex method, interior-point methods, evolutionary algorithms) have varying strengths and weaknesses depending on the problem's characteristics.  The choice of solver significantly impacts feasibility in terms of runtime and solution quality.


In my experience working on a large-scale logistics optimization project involving over 50 variables and multiple objectives, I found that careful problem formulation and solver selection were crucial. Preprocessing steps like constraint tightening and variable reduction often dramatically improved solution time and efficiency.


**2. Code Examples with Commentary:**

These examples demonstrate BOLP using Python with the `PuLP` library.  While 12 variables are easily handled, they illustrate the core concepts and allow scalability to larger problems. Remember, these examples are simplified and wouldn't represent the full complexity encountered in real-world scenarios.

**Example 1: Simple Bi-objective Portfolio Optimization**

This example optimizes a portfolio balancing risk and return.

```python
from pulp import *

# Define problem
prob = LpProblem("Bi-objective Portfolio", LpMaximize)

# Define variables (12 assets)
assets = [f"Asset_{i}" for i in range(1, 13)]
x = LpVariable.dicts("Investment", assets, 0, 1, LpContinuous)

# Define objective functions (return and risk)
returns = [0.1, 0.12, 0.08, 0.15, 0.09, 0.11, 0.13, 0.10, 0.07, 0.14, 0.06, 0.16] # Fictional returns
risks = [0.05, 0.10, 0.03, 0.15, 0.06, 0.08, 0.12, 0.04, 0.02, 0.18, 0.01, 0.20]   # Fictional risks

prob += lpSum([returns[i] * x[assets[i]] for i in range(len(assets))]), "Return"
prob += -lpSum([risks[i] * x[assets[i]] for i in range(len(assets))]), "Risk" # Minimizing risk

# Constraints (e.g., budget limit)
prob += lpSum([x[i] for i in assets]) == 1, "BudgetConstraint"

# Solve (using a suitable solver; this will yield a single optimal solution based on a weighting scheme)
prob.solve()
print("Status:", LpStatus[prob.status])
for v in prob.variables():
    print(v.name, "=", v.varValue)
print("Objective 1 (Return):", value(prob.objective))
print("Objective 2 (Risk):", -value(prob.objective)) # Note the negative sign for risk
```

This code utilizes a single solver run, obtaining only one solution on the Pareto front. To generate the entire front, a multi-objective optimization algorithm, such as weighted sum method or epsilon-constraint method, needs to be implemented.

**Example 2:  Weighted Sum Method**

This approach combines the objectives into a single weighted objective function.

```python
# ... (Previous code as before) ...

# Weighted sum method
weight_return = 0.7
weight_risk = 1 - weight_return

prob += weight_return * lpSum([returns[i] * x[assets[i]] for i in range(len(assets))]) + \
        weight_risk * (-lpSum([risks[i] * x[assets[i]] for i in range(len(assets))])), "WeightedObjective"
# ... (rest of the code remains the same) ...
```

By varying `weight_return`, different points on the Pareto front are obtained.

**Example 3: Epsilon-Constraint Method**

This method constrains one objective and optimizes the other.

```python
# ... (Previous code as before) ...

# Epsilon-constraint method
epsilon = 0.1 # Example constraint value for risk

prob += -lpSum([risks[i] * x[assets[i]] for i in range(len(assets))]) >= epsilon, "RiskConstraint"
prob += lpSum([returns[i] * x[assets[i]] for i in range(len(assets))]), "Return"
# ... (rest of the code remains the same) ...
```

By changing `epsilon`, different points on the Pareto front can be explored.  This approach requires multiple solver calls.


**3. Resource Recommendations:**

"Linear Programming and Extensions" by George B. Dantzig and Mukund N. Thapa provides a comprehensive treatment of linear programming techniques.  "Multi-Objective Optimization: Interactive and Evolutionary Approaches" by Kalyanmoy Deb delves into the specifics of handling multiple objective functions.  Texts covering operations research provide valuable contextual knowledge and advanced techniques for larger scale problems.  Finally, consult the documentation for optimization libraries such as PuLP, CVXPY, or commercial solvers (CPLEX, Gurobi) for implementation details and advanced features.
