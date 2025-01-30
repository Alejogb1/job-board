---
title: "How can linear programming problems be formulated using SciPy?"
date: "2025-01-30"
id: "how-can-linear-programming-problems-be-formulated-using"
---
Linear programming (LP) problems, characterized by a linear objective function subject to linear equality and inequality constraints, can be effectively solved using SciPy's `optimize.linprog` function. This function provides a robust and efficient method for finding optimal solutions within the defined feasible region. My professional experience often involves optimizing resource allocation within complex systems, and I've frequently found `linprog` to be the appropriate tool.

The core idea behind `linprog` is to minimize (or maximize, with a minor adjustment) a linear objective function, expressed as a vector dot product `c^T * x`, where `c` represents the cost or benefit coefficients and `x` are the decision variables. The feasible region for these variables is constrained by a set of linear inequalities `A_ub * x <= b_ub` and equalities `A_eq * x == b_eq`. Lower and upper bounds for the variables, if present, are also incorporated. `linprog` employs an iterative algorithm, often a variant of the simplex method, to navigate the feasible region and converge upon an optimal solution.

I find the key to effective utilization of `linprog` is proper formulation of the problem into the required matrix and vector representations. This involves clearly identifying the decision variables, objective function, constraints, and bounds. In practical applications, this process is often the most challenging, as the problem description may not directly translate into mathematical form. A meticulous approach here is essential.

For demonstration, consider a classic resource allocation problem. Assume a factory produces two products, A and B, using two resources, X and Y. Product A requires 2 units of X and 1 unit of Y, while product B requires 1 unit of X and 3 units of Y. We have 10 units of X and 15 units of Y available. The profit per unit of A is $5, and per unit of B is $8. Our aim is to determine the number of units of A and B to produce in order to maximize total profit.

Here's the code example illustrating this:

```python
from scipy.optimize import linprog
import numpy as np

# Define the objective function coefficients
# We are maximizing profit so multiply coefficients by -1 (minimization problem by linprog)
c = np.array([-5, -8])

# Define the inequality constraint matrix (A_ub) and vector (b_ub)
# 2A + 1B <= 10 (X resource constraint)
# 1A + 3B <= 15 (Y resource constraint)
A_ub = np.array([[2, 1], [1, 3]])
b_ub = np.array([10, 15])

# Define the bounds for x (A and B production cannot be negative)
bounds = [(0, None), (0, None)]

# Solve the linear programming problem
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

# Output results
print("Optimal solution (A, B):", result.x)
print("Maximum profit:", -result.fun)
```

In this example, the objective function is formulated as minimizing `-5A - 8B`, which is equivalent to maximizing `5A + 8B`. The constraints are converted into the form `A_ub * x <= b_ub`. The `bounds` parameter specifies that the production quantities for both products are non-negative. The result provides the optimal production quantities `result.x` and the optimal objective function value, which we negate back to retrieve profit through `-result.fun`.

Another scenario I frequently encounter involves a diet planning problem. Imagine needing to minimize the cost of a meal while meeting specific nutritional requirements. Assume there are three food items (1, 2, and 3) with varying costs and nutritional content. Let's say food item 1 costs $2 per unit and provides 2 units of nutrient A, 1 unit of nutrient B, and 0 units of nutrient C. Food item 2 costs $3 per unit providing 1, 3, and 2 units of A, B, and C, respectively. Finally food item 3 costs $5 per unit and provides 3, 0, and 1 units of A, B, and C. We require at least 8 units of nutrient A, 6 units of nutrient B, and 4 units of nutrient C.

Here's the code to formulate this:

```python
from scipy.optimize import linprog
import numpy as np

# Objective function is to minimize cost: 2x1 + 3x2 + 5x3
c = np.array([2, 3, 5])

# Inequality constraints (A_ub, b_ub):
# 2x1 + 1x2 + 3x3 >= 8 (nutrient A)
# 1x1 + 3x2 + 0x3 >= 6 (nutrient B)
# 0x1 + 2x2 + 1x3 >= 4 (nutrient C)

# The constraints need to be in <= form, so multiply the inequalities by -1:
A_ub = np.array([[-2, -1, -3], [-1, -3, 0], [0, -2, -1]])
b_ub = np.array([-8, -6, -4])

# Bounds are zero since we cannot have a negative food amount
bounds = [(0, None), (0, None), (0, None)]

# Solve
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)

print("Optimal amounts (food1, food2, food3):", result.x)
print("Minimum cost:", result.fun)
```

In this example, the objective function is to minimize `2x1 + 3x2 + 5x3`. The constraints for the required nutrients are initially written with `greater or equal` operators. I then multiplied each constraint by -1, converting them to `less or equal` form to conform to the `linprog` requirement. The bounds specify that the quantities of each food item cannot be negative.

Lastly, consider a transportation problem. A company has three warehouses (W1, W2, W3) with capacities of 100, 150, and 120 units, respectively. They need to ship these units to two retailers (R1, R2) who require 200 and 170 units, respectively. The cost of shipping from each warehouse to each retailer is given below:
- W1 to R1: $5
- W1 to R2: $8
- W2 to R1: $4
- W2 to R2: $6
- W3 to R1: $7
- W3 to R2: $3

The objective is to minimize the total shipping cost.

```python
from scipy.optimize import linprog
import numpy as np

# Define the cost matrix, converted to a vector
c = np.array([5, 8, 4, 6, 7, 3])

# Define the equality constraint matrix (A_eq) and vector (b_eq)
# The matrix contains two types of equality constraints:
# 1) sum of shipments from warehouse = warehouse capacity
# 2) sum of shipments to retailer = retailer demand
A_eq = np.array([[1, 1, 0, 0, 0, 0],  # W1 capacity constraint
                 [0, 0, 1, 1, 0, 0],  # W2 capacity constraint
                 [0, 0, 0, 0, 1, 1],  # W3 capacity constraint
                 [1, 0, 1, 0, 1, 0], # R1 demand constraint
                 [0, 1, 0, 1, 0, 1]]) # R2 demand constraint

b_eq = np.array([100, 150, 120, 200, 170])

# Bounds: shipments cannot be negative
bounds = [(0, None)] * 6

# Solve
result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds)


print("Optimal shipping amounts (W1->R1, W1->R2, W2->R1, W2->R2, W3->R1, W3->R2):", result.x)
print("Minimum total cost:", result.fun)
```

In this scenario, the decision variables are the quantities shipped between each warehouse-retailer pair. The constraints are equalities ensuring that the total shipments from each warehouse match its capacity and the total shipments to each retailer meet its demand.  The solution reveals the optimal shipping quantities `result.x` and the minimal shipping cost `result.fun`.

For further exploration and a deeper understanding of linear programming and its application using SciPy, I recommend referring to books on numerical optimization and the official SciPy documentation.  Additionally, articles on Operations Research can provide a broader context for the types of problems well-suited for linear programming approaches. Consulting resources focusing on mathematical programming would also be beneficial for a thorough grasp of the underlying theory and algorithm details.
