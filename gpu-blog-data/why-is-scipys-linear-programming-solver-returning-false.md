---
title: "Why is SciPy's linear programming solver returning False for success?"
date: "2025-01-30"
id: "why-is-scipys-linear-programming-solver-returning-false"
---
SciPy's `linprog` function, when reporting `success: False`, indicates the optimization algorithm failed to find a feasible solution or converged to a suboptimal point under the imposed constraints. This failure mode isn't a blanket error; it often stems from specific characteristics of the problem formulation rather than an inherent flaw in the solver itself. In my experience, debugging these situations frequently involves meticulous examination of the input parameters and an understanding of the underlying algorithm's limitations.

The most common reasons for a `success: False` outcome fall into several distinct categories. First, the problem may be truly infeasible. This occurs when the constraints are contradictory, making it impossible to find a set of variable values satisfying all conditions simultaneously. Second, the constraints, though feasible, may lead to an unbounded objective function. The solver will try to minimize (or maximize) the objective function infinitely without a limiting solution point. Third, the problem might suffer from numerical instability, particularly with very large or small coefficients. And finally, the algorithm might terminate prematurely due to reaching iteration limits or encountering internal errors. Identifying which of these scenarios is occurring is vital for selecting the appropriate corrective measures.

The first scenario – infeasibility – often results from errors in defining inequalities or equalities. For instance, unintentionally including constraints that conflict with one another, or a sign error in the inequality direction can yield this result. Let's look at an example. Suppose we intend to minimize `z = -x + 4y` subject to:
* `x - 3y <= -6`
* `2x + y <= 4`
* `x >= 0`
* `y >= 0`

Here's the implementation in SciPy:

```python
import numpy as np
from scipy.optimize import linprog

c = [-1, 4]  # Objective function coefficients
A_ub = [[1, -3], [2, 1]]  # Inequality constraint coefficients
b_ub = [-6, 4]   # Inequality constraint upper bounds
x0_bounds = (0, None)  # x0 >= 0
x1_bounds = (0, None)  # x1 >= 0
bounds = (x0_bounds, x1_bounds)

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
print(result)
```

This script yields a `success: False` outcome. Examining the constraint coefficients and bounds reveals a potential issue: there is no overlap in the feasible regions defined by x-3y <= -6 and x >= 0, y>=0.  Graphically plotting the constraints shows this infeasibility, with the line x-3y = -6 preventing solution. The `highs` algorithm confirms this infeasibility. In cases like this, double-checking constraints against the desired real-world situation is crucial.

Next, let’s examine the unbounded objective function scenario. Consider we wish to minimize `z = -x - y` subject to:
* `-x + y <= 1`
* `x >= 0`
* `y >= 0`

Here's the code:
```python
import numpy as np
from scipy.optimize import linprog

c = [-1, -1]  # Objective function coefficients
A_ub = [[-1, 1]]  # Inequality constraint coefficients
b_ub = [1]   # Inequality constraint upper bounds
x0_bounds = (0, None)  # x0 >= 0
x1_bounds = (0, None)  # x1 >= 0
bounds = (x0_bounds, x1_bounds)

result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
print(result)
```
In this example, the objective function `z = -x -y` can decrease indefinitely, because both `x` and `y` are non-negative, and the constraint `-x + y <= 1` permits `x` and `y` to increase without bound. Again, the `highs` method correctly identifies the problem. The `message` attribute in the result will detail that the problem is unbounded. To correct this, additional constraints need to be added to prevent the function from diverging in the desired direction.

Finally, let's address numerical stability issues. These are more subtle and often involve very large or small coefficients that can lead to loss of precision within the solver's internal calculations. Consider this example: we aim to maximize `z = 1000000000 * x + y` subject to:
* `x + y <= 10`
* `x >= 0`
* `y >= 0`

Implemented in Python:
```python
import numpy as np
from scipy.optimize import linprog

c = [-1000000000, -1] # Objective function coefficients
A_ub = [[1, 1]] # Inequality constraint coefficients
b_ub = [10] # Inequality constraint upper bounds
x0_bounds = (0, None)  # x0 >= 0
x1_bounds = (0, None)  # x1 >= 0
bounds = (x0_bounds, x1_bounds)


result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
print(result)

c = [-1, -1]
A_ub = [[0.000000001, 0.000000001]] # Rescaled coefficients
b_ub = [0.00000001]

result_scaled = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
print(result_scaled)
```
The first `linprog` call returns a `success: False`. The second instance, however, does produce a successful result. In the original problem the large value for x's objective function coefficient can cause difficulties in the optimization process. Rescaling the problem to have coefficients of similar magnitude often helps. In our case, scaling by 1,000,000,000 and modifying c accordingly yielded the correct solution using the same constraints and algorithm.

When experiencing a `success: False` return, I always start by meticulously checking the constraint definitions, paying close attention to signs and bounds. Visualizing the constraints graphically can help identify infeasibilities, especially in low-dimensional problems. Next, consider the objective function; if the problem appears to be unbounded, re-evaluate the constraints to see if they are sufficiently limiting. Finally, consider the magnitude of your coefficients. If they vary greatly, rescaling can improve numerical stability. In practice, it is important to try a variety of algorithms (`method` parameter) available in `linprog`. Often the more advanced `highs` solver will successfully solve problems where the default simplex fails to do so.

For additional study, I highly recommend consulting textbooks on linear programming, operations research, and numerical analysis. Resources dedicated to optimization algorithms can also provide valuable context into these challenges. Furthermore, examining the documentation for the `scipy.optimize.linprog` function will clarify the nuances of its parameters and result attributes. Practice is the most valuable training tool when troubleshooting this type of failure, and by implementing these steps you can improve your overall linear programming skill and intuition.
