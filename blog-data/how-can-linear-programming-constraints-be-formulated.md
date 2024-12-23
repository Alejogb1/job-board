---
title: "How can linear programming constraints be formulated?"
date: "2024-12-23"
id: "how-can-linear-programming-constraints-be-formulated"
---

, let's tackle the formulation of linear programming constraints. This isn't just academic; I've seen firsthand how critical a clear and accurate formulation is to actually solving real-world optimization problems. It’s often the bottleneck, believe me. It's one thing to understand the theory; it's another to translate business or engineering requirements into a set of mathematically tractable constraints. I remember a particularly grueling project where we were optimizing a supply chain, and a single incorrectly formulated constraint led to utterly nonsensical production schedules. So, yeah, details matter here.

Linear programming (lp) fundamentally revolves around optimizing a linear objective function subject to a set of linear constraints. These constraints define the feasible region—the space of solutions that satisfy all conditions. Formulating them precisely is paramount. At their core, constraints are mathematical inequalities or equalities that limit the possible values of the decision variables in your model. These decision variables, often represented by x, y, z, etc., are the things you're actually trying to find the optimal values for.

The general forms of constraints are:

1.  **Less than or equal to (<=):** This represents a limit or maximum capacity. For example, the total number of hours spent on projects must be less than or equal to the available workforce hours.

2.  **Greater than or equal to (>=):** This represents a minimum requirement. Perhaps a certain amount of materials must be procured, or a certain level of production must be met.

3.  **Equal to (=):** This enforces a precise equality. It might represent a balance requirement, such as total output equal to total demand.

Let's dive into the nitty-gritty with some practical examples and code snippets. I’ll use python with `scipy.optimize.linprog`, as it's a common tool for this, although the concepts apply irrespective of the specific solver.

**Example 1: Resource Allocation**

Imagine you have two types of products, A and B, to manufacture. Each requires different amounts of raw materials and labor, and you have limited resources.

Let:
* `x` be the number of units of product A produced.
* `y` be the number of units of product B produced.
* Product A requires 2 units of material and 3 hours of labor per unit.
* Product B requires 1 unit of material and 4 hours of labor per unit.
* You have 10 units of material and 20 hours of labor available.

The constraints can be formulated as:

*   Material constraint: 2x + 1y <= 10
*   Labor constraint: 3x + 4y <= 20
*   Non-negativity constraints: x >= 0, y >= 0 (you can’t produce negative products)

Here’s how you might express this using `scipy`:

```python
import numpy as np
from scipy.optimize import linprog

# Objective: Let's assume you want to maximize 3*x + 5*y (profit)
c = [-3, -5] #coefficients of the objective function (negative for maximization)

# inequality constraints: Ax <= b (we are defining as the transpose to match the expected form)
A = np.array([[2, 1], [3, 4]])
b = np.array([10, 20])

# Bounds for x and y (non-negativity)
x0_bounds = (0, None)
x1_bounds = (0, None)

# Solve the linear programming problem
result = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds], method='highs')

print("Optimal x:", result.x[0])
print("Optimal y:", result.x[1])
print("Optimal objective value:", -result.fun) #negate back to get the actual maximized value
```

**Example 2: Blending Problem**

Consider a blending problem. A food company wants to produce a feed mix using two ingredients, Grain X and Grain Y. Each grain contains different amounts of nutrients A, B, and C. The company wants to meet specific minimum levels of nutrients while minimizing cost.

Let:
*  `x` be the amount (in kg) of Grain X in the mix.
*  `y` be the amount (in kg) of Grain Y in the mix.
* Grain X contains 0.3 kg of A, 0.2 kg of B, and 0.1 kg of C per kg.
* Grain Y contains 0.1 kg of A, 0.3 kg of B, and 0.2 kg of C per kg.
* The minimum required nutrients in the mix are 2 kg of A, 2.5 kg of B, and 1 kg of C.
* The cost of Grain X is $2 per kg and Grain Y is $3 per kg.

The constraints are:
*   Nutrient A constraint: 0.3x + 0.1y >= 2
*   Nutrient B constraint: 0.2x + 0.3y >= 2.5
*   Nutrient C constraint: 0.1x + 0.2y >= 1
*  Non-negativity constraints: x >= 0, y >= 0

Here’s the python implementation:

```python
import numpy as np
from scipy.optimize import linprog

# Objective: minimize 2*x + 3*y (cost)
c = [2, 3]

# inequality constraints: Ax >= b
A = np.array([[0.3, 0.1], [0.2, 0.3], [0.1, 0.2]])
b = np.array([2, 2.5, 1])

# Bounds for x and y
x0_bounds = (0, None)
x1_bounds = (0, None)

# Solve the linear programming problem
result = linprog(c, A_ub=-A, b_ub=-b, bounds=[x0_bounds, x1_bounds], method='highs')
print("Optimal x:", result.x[0])
print("Optimal y:", result.x[1])
print("Minimum cost:", result.fun)
```

Note that because `linprog` by default uses A\*x <= b format we multiply our matrix A and vector b by -1

**Example 3: Workforce Scheduling**

Let's look at something slightly different. A store needs a certain number of employees during each shift throughout the day. The goal is to minimize the total cost while meeting staffing requirements.

Let:
*   `x1` be the number of employees starting at shift 1.
*   `x2` be the number of employees starting at shift 2.
*   `x3` be the number of employees starting at shift 3.
* Each shift lasts 8 hours.
* Shift 1 covers 6 am - 2 pm.
* Shift 2 covers 10 am - 6 pm.
* Shift 3 covers 2 pm - 10 pm.
* Requirements:
    * 6 am - 10 am: minimum 10 employees needed.
    * 10 am - 2 pm: minimum 15 employees needed.
    * 2 pm - 6 pm: minimum 12 employees needed.
    * 6 pm - 10 pm: minimum 8 employees needed.
*  Assume each employee costs $150 for the 8-hour shift.

The constraints are:

*   6 am - 10 am: x1 >= 10
*   10 am - 2 pm: x1 + x2 >= 15
*   2 pm - 6 pm: x2 + x3 >= 12
*   6 pm - 10 pm: x3 >= 8
*   Non-negativity constraints: x1 >= 0, x2 >= 0, x3 >= 0

Here's the Python code for this scenario:

```python
import numpy as np
from scipy.optimize import linprog

# Objective: minimize 150*x1 + 150*x2 + 150*x3 (total cost)
c = [150, 150, 150]

# inequality constraints Ax >= b
A = np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1], [0, 0, 1]])
b = np.array([10, 15, 12, 8])

# Bounds for x1, x2, and x3
x0_bounds = (0, None)
x1_bounds = (0, None)
x2_bounds = (0, None)

# Solve the linear programming problem
result = linprog(c, A_ub=-A, b_ub=-b, bounds=[x0_bounds, x1_bounds, x2_bounds], method='highs')
print("Optimal x1:", result.x[0])
print("Optimal x2:", result.x[1])
print("Optimal x3:", result.x[2])
print("Minimum cost:", result.fun)
```

Again note that because `linprog` by default uses A\*x <= b format we multiply our matrix A and vector b by -1

These examples highlight a few key things. Firstly, the constraints need to accurately reflect the limitations of the problem you're trying to solve. A misstep here and the model won't be useful, which I've encountered far too often. Secondly, pay very close attention to the direction of your inequalities (<= vs. >=) and their corresponding interpretation. Thirdly, the non-negativity constraints, although simple, are often essential.

For deeper study, I highly recommend "Linear Programming and Network Flows" by Bazaraa, Jarvis, and Sherali. It is a comprehensive textbook that covers the subject in depth. Additionally, “Introduction to Operations Research” by Hillier and Lieberman is an excellent general reference that goes beyond linear programming, but covers it very well, along with practical application. If you want to dive into algorithms and the computational side, "Numerical Optimization" by Nocedal and Wright is a fantastic choice.

In conclusion, effective formulation of linear programming constraints is more than just plugging numbers into an equation. It involves a careful translation of a problem's requirements into a mathematical form that captures its essence while remaining solvable. I’ve spent years refining my understanding of this, and it's been invaluable in my work. Hopefully, these examples and references will be helpful to you.
