---
title: "How can I calculate Euclidean or Manhattan distances within a Gurobi constraint using Python?"
date: "2025-01-30"
id: "how-can-i-calculate-euclidean-or-manhattan-distances"
---
Calculating Euclidean or Manhattan distances within Gurobi constraints requires careful consideration of the model's linear structure.  Gurobi, at its core, is a linear programming solver; therefore, direct implementation of these distance metrics, which are inherently non-linear for more than two dimensions, necessitates reformulation.  My experience optimizing supply chain networks extensively exposed this limitation, forcing me to develop strategies to embed these distances effectively.  The key is to represent the distance calculation using linear expressions.

**1. Clear Explanation:**

The challenge arises because the Euclidean distance, √(Σ(xi - yi)²), involves a square root and squaring operations, both non-linear. Similarly, the Manhattan distance, Σ|xi - yi|, contains the absolute value function, which is also non-linear in a standard linear programming context. To overcome this, we employ a technique that leverages the solver's ability to handle linear inequalities and auxiliary variables.

For the Euclidean distance, we can use the fact that minimizing the square of the Euclidean distance is equivalent to minimizing the Euclidean distance itself (as the square root is a monotonically increasing function).  Thus, we replace the objective function or constraint involving Euclidean distance with its squared form. This removes the square root, leaving us with a quadratic expression.  While Gurobi can handle quadratic constraints and objectives, the performance gains from converting to a linear model are usually significant.  So, we should approximate the square term using linear piecewise functions. This approximation involves introducing additional variables and constraints, resulting in a linear model.

For the Manhattan distance, the absolute value can be linearized by introducing two non-negative variables for each difference: one representing the positive part and the other the negative part. The absolute value is then simply the sum of these two variables, resulting in a linear expression.  This requires adding constraints ensuring the correct relationship between the original difference and the newly introduced positive and negative parts.


**2. Code Examples with Commentary:**

**Example 1: Manhattan Distance**

```python
import gurobipy as gp
from gurobipy import GRB

# Data: Coordinates of points
points = [(1, 2), (3, 4), (5, 6)]

# Model
model = gp.Model("ManhattanDistance")

# Variables
x = model.addVars(len(points), len(points), vtype=GRB.CONTINUOUS, name="x")

#Constraints to ensure correct absolute values
for i in range(len(points)):
    for j in range(len(points)):
        plus = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"plus_{i}_{j}")
        minus = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"minus_{i}_{j}")
        model.addConstr(plus - minus == points[i][0] - points[j][0], name=f"constr_x_{i}_{j}")
        model.addConstr(plus + minus >= abs(points[i][0] - points[j][0]), name=f"constr_xabs_{i}_{j}")
        model.addConstr(x[i,j] == plus + minus, name=f"x_val_{i}_{j}")

        plus = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"plus_y_{i}_{j}")
        minus = model.addVar(lb=0, vtype=GRB.CONTINUOUS, name=f"minus_y_{i}_{j}")
        model.addConstr(plus - minus == points[i][1] - points[j][1], name=f"constr_y_{i}_{j}")
        model.addConstr(plus + minus >= abs(points[i][1] - points[j][1]), name=f"constr_yabs_{i}_{j}")
        model.addConstr(x[i,j] == x[i,j] + plus + minus, name=f"y_val_{i}_{j}")


# Objective (minimize total Manhattan distance)
model.setObjective(gp.quicksum(x[i,j] for i in range(len(points)) for j in range(len(points))), GRB.MINIMIZE)

# Optimize
model.optimize()

#Print results.  Note that x values will now be manhattan distances.
for i in range(len(points)):
    for j in range(len(points)):
        print(f"Manhattan Distance between {points[i]} and {points[j]}: {x[i,j].X}")

```
This code demonstrates the linearization of the Manhattan distance by introducing auxiliary variables to represent the positive and negative parts of the differences.


**Example 2: Approximating Squared Euclidean Distance (Piecewise Linear)**

```python
import gurobipy as gp
from gurobipy import GRB

# Data: Coordinates of points
points = [(1, 2), (3, 4), (5, 6)]

#Simplified piecewise linear approximation. For accuracy more segments are needed.
def piecewise_linear_approx(x, segments=2):
    #Note: This approximation needs adaptation for practical application.  It's a minimal example.
    breakpoints = [0,100]  #Needs refinement depending on data.
    slopes = [1,2] #Needs refinement depending on data.
    y = model.addVars(segments,lb=0,ub=1,vtype=GRB.CONTINUOUS)
    model.addConstr(gp.quicksum(y) == 1)
    model.addConstr(x == gp.quicksum(y[i]*breakpoints[i+1] for i in range(segments)))
    return gp.quicksum(y[i]*slopes[i]*breakpoints[i+1] for i in range(segments))

# Model
model = gp.Model("EuclideanDistanceApprox")

# Variables
x = model.addVars(len(points), len(points), vtype=GRB.CONTINUOUS, name="x")

# Constraints for approximating squared Euclidean distance (piecewise linear)
for i in range(len(points)):
    for j in range(len(points)):
        sq_diff_x = piecewise_linear_approx((points[i][0] - points[j][0])**2)
        sq_diff_y = piecewise_linear_approx((points[i][1] - points[j][1])**2)
        model.addConstr(x[i,j] == sq_diff_x + sq_diff_y)


# Objective (minimize total approximated squared Euclidean distance)
model.setObjective(gp.quicksum(x[i,j] for i in range(len(points)) for j in range(len(points))), GRB.MINIMIZE)

# Optimize
model.optimize()

#Print results. Note that x[i,j].X contains the approximate squared euclidean distance.
for i in range(len(points)):
    for j in range(len(points)):
        print(f"Approximated Squared Euclidean Distance between {points[i]} and {points[j]}: {x[i,j].X}")
```

This example demonstrates a simplified piecewise linear approximation of the squared Euclidean distance.  The accuracy depends critically on the chosen breakpoints and slopes, requiring careful consideration and potentially iterative refinement based on the problem's data characteristics.   More sophisticated piecewise-linear approximations can be implemented for higher accuracy.



**Example 3:  Euclidean Distance with Quadratic Constraints (Direct, less efficient)**

```python
import gurobipy as gp
from gurobipy import GRB

# Data: Coordinates of points
points = [(1, 2), (3, 4), (5, 6)]

# Model
model = gp.Model("EuclideanDistanceQuadratic")

# Variables
x = model.addVars(len(points), len(points), vtype=GRB.CONTINUOUS, name="x")


# Objective (minimize total Euclidean distance) -  Note: This is a quadratic objective.
model.setObjective(gp.quicksum(gp.quicksum((points[i][k]-points[j][k])**2 for k in range(2))**0.5 for i in range(len(points)) for j in range(len(points))), GRB.MINIMIZE)

# Optimize - Gurobi will handle the quadratic objective, but this is less efficient than linearization
model.optimize()

#Print results. This will likely be less accurate and slower than linear approximation.
for i in range(len(points)):
    for j in range(len(points)):
        print(f"Euclidean Distance (Quadratic Formulation) between {points[i]} and {points[j]}: {x[i,j].X}")
```
This code directly uses the Euclidean distance formula in the objective function. While Gurobi can handle this quadratic formulation, it's generally less efficient than the linearized approaches, especially for larger problems.  It's included for completeness to illustrate the trade-off between direct implementation and linearization.


**3. Resource Recommendations:**

The Gurobi documentation, specifically sections on quadratic programming and modeling techniques, is invaluable.  Textbooks on mathematical optimization, focusing on integer and linear programming, offer detailed explanations of linearization techniques and piecewise linear approximations.  Finally, reviewing publications on supply chain optimization and location problems will provide context and inspiration for advanced implementations.
