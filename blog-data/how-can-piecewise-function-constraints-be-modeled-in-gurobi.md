---
title: "How can piecewise function constraints be modeled in Gurobi?"
date: "2024-12-23"
id: "how-can-piecewise-function-constraints-be-modeled-in-gurobi"
---

, let's tackle piecewise functions in Gurobi. This isn't always straightforward, and I've certainly had my share of headaches dealing with them, particularly when performance is critical. It’s a situation that often comes up when you're modeling non-linear phenomena within a linear or mixed-integer programming framework. I recall once, back in my quant days, when we were trying to model the price impact of trades in a highly fragmented market – the non-linearities were a significant challenge.

The key to handling piecewise functions in Gurobi, or indeed any similar optimization solver, lies in transforming the discontinuous or non-linear function into a set of linear constraints. This often involves introducing auxiliary variables and constraints to properly represent the different segments or "pieces" of your function. There are a few common approaches depending on the function's characteristics, but they all revolve around this linearization technique. We'll go through a few of the more prevalent methods and look at some concrete examples.

First, let's consider a simple continuous piecewise linear function. Imagine you have a function *f(x)* that behaves differently over two intervals. For example:

*   *f(x) = 2x* for *x ≤ 5*
*   *f(x) = x + 5* for *x > 5*

To model this in Gurobi, we introduce a new variable, let’s call it *y*, that will represent *f(x)*, and additional binary variables that indicate which interval we are in. Here's the approach:

1.  **Introduce binary variables:** We'll use a binary variable *z* to determine which segment is active. *z = 0* indicates the first interval (*x ≤ 5*), and *z = 1* indicates the second interval (*x > 5*).

2.  **Constraint construction:** We need to link *x*, *y*, and *z* through a set of linear constraints that force the right behavior. We will introduce *lower_bound*, *upper_bound*, *lower_y* and *upper_y* that are based on the breakpoints, and are used to make the constraints tighter.
    *   *x ≤ 5 + M(1 - z)* . M is some large number (Big-M). This ensures x is limited by the bound when *z=0*. When *z=1* this constraint is effectively inactive.
    *   *x ≥ lower_bound - Mz* . This forces x to be greater or equal to a lower bound. When *z=0* this is an active constraint and when *z=1* this constraint is inactive.
    *   *y ≥ lower_y + 2x - 2 * lower_bound - M(1 - z)* This ensures that when *z=0*, then y is equal to the first piece (2x). When *z=1* then this constraint is inactive.
    *   *y ≥ lower_y + (x - lower_bound) + 2 * lower_bound  - Mz*  When *z=1* we ensure that y is forced to be at least equal to *x+5*.
    *   *y ≤ upper_y + 2x - 2 * lower_bound + M(1 - z)* When *z=0* this forces y to be at most *2x*. When *z=1* it is inactive.
     *   *y ≤ upper_y + (x- lower_bound) + 2 * lower_bound + Mz*  When *z=1* it forces y to be at most *x+5*.

Here is Python code using `gurobipy` that demonstrates this approach:

```python
import gurobipy as gp

def model_piecewise_function_continuous():
    model = gp.Model("Piecewise_Continuous")
    x = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="x")
    y = model.addVar(vtype=gp.GRB.CONTINUOUS, name="y")
    z = model.addVar(vtype=gp.GRB.BINARY, name="z")

    lower_bound = 0
    upper_bound = 10
    lower_y = 0
    upper_y = 15
    M = 1000  # Big-M value

    # Constraint for selecting the segment: x <= 5 when z=0, otherwise not constrained
    model.addConstr(x <= 5 + M * (1 - z), name="segment_select_upper")
    model.addConstr(x >= lower_bound - M * z, name="segment_select_lower")

     # Constraint for y when x <= 5 (z = 0)
    model.addConstr(y >= lower_y + 2 * x - 2 * lower_bound - M * (1 - z), name="y_segment1_lower")

    # Constraint for y when x > 5 (z = 1)
    model.addConstr(y >= lower_y + (x - lower_bound) + 2 * lower_bound - M * z, name="y_segment2_lower")

    model.addConstr(y <= upper_y + 2 * x - 2 * lower_bound + M*(1-z), name = "y_segment1_upper")
    model.addConstr(y <= upper_y + (x-lower_bound) + 2* lower_bound + M*z, name = "y_segment2_upper")

    # Objective (example)
    model.setObjective(y, gp.GRB.MAXIMIZE)

    # Solve
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
        print(f"Optimal x: {x.x}, Optimal y: {y.x}, z value: {z.x}")
    else:
        print("No optimal solution found")

    return model

if __name__ == '__main__':
     model = model_piecewise_function_continuous()
```
This approach works well for continuous piecewise linear functions where we can transition smoothly between segments at the breakpoints.

Now, let's consider a discontinuous case. Suppose we have:

*   *f(x) = 10*  for *x < 3*
*   *f(x) = 20* for *x ≥ 3*

Here, we're not just dealing with a change in slope, but a jump in the function's value. This calls for a slight modification in how we use our binary variable. Again we will introduce *y* as the function representation and *z* as the binary variable that determines whether we are above or below *x=3*. Here's the approach:

1. **Binary Variables**: Same as before, *z=0* for *x<3* and *z=1* for *x>=3*.

2. **Constraints**: We must force y to take one of the two values:
    *   *x < 3 + Mz*  This constraint makes sure that x is below 3 if z is 0.
    *   *x >= 3 - M(1-z)* This constraint makes sure x is above 3 if z is 1.
    *   *y >= 10 - Mz*  This forces y to be at least 10 if z=0.
    *   *y <= 10 + Mz* This forces y to be at most 10 if z=0.
    *   *y >= 20 - M(1-z)* This forces y to be at least 20 if z=1.
    *   *y <= 20 + M(1-z)* This forces y to be at most 20 if z=1.

Here's the Gurobi code:

```python
import gurobipy as gp

def model_piecewise_function_discontinuous():
    model = gp.Model("Piecewise_Discontinuous")
    x = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="x")
    y = model.addVar(vtype=gp.GRB.CONTINUOUS, name="y")
    z = model.addVar(vtype=gp.GRB.BINARY, name="z")

    M = 1000  # Big-M value

    # Constraint for selecting the segment: x < 3 when z = 0, x>=3 when z=1
    model.addConstr(x < 3 + M * z, name="segment_select_upper")
    model.addConstr(x >= 3 - M * (1-z), name="segment_select_lower")

    # Constraint for y when x < 3 (z = 0)
    model.addConstr(y >= 10 - M * z, name="y_segment1_lower")
    model.addConstr(y <= 10 + M * z, name="y_segment1_upper")

    # Constraint for y when x >= 3 (z = 1)
    model.addConstr(y >= 20 - M * (1-z), name="y_segment2_lower")
    model.addConstr(y <= 20 + M * (1-z), name="y_segment2_upper")


    # Objective (example)
    model.setObjective(y, gp.GRB.MAXIMIZE)

    # Solve
    model.optimize()

    if model.status == gp.GRB.OPTIMAL:
       print(f"Optimal x: {x.x}, Optimal y: {y.x}, z value: {z.x}")
    else:
        print("No optimal solution found")
    return model

if __name__ == '__main__':
    model = model_piecewise_function_discontinuous()
```

Finally, for more complex situations, such as arbitrary piecewise linear functions with more breakpoints, a more general approach is needed. For a piecewise function with n breakpoints, we use n+1 binary variables. Consider this piecewise function as an example.
*  f(x) = x + 1 when x is between 0 and 1
*  f(x) = 2x when x is between 1 and 2
*  f(x) = 3 when x is between 2 and 3

We will introduce *y* as usual as the function representation. We will use binary variables *z_1*, *z_2*, and *z_3*, to show which segment is active. If the variable *x* is in segment i, then *z_i=1*, otherwise it is *0*. We also need a constraint to force only one *z_i* to be 1 at all times. Then the constraints will be:

1.  **Binary variables:** *z_1*, *z_2*, *z_3*, to indicate the active segment.
2.  **Constraints**
   *   Sum of all *z_i* is 1: we can only be in one segment.
   *   *0 <= x - 0*
   *   *x - 0 <= 1 + M(1 - z_1)*
   *   *1 <= x - 0 + M(1 - z_2)*
   *   *x-0 <= 2 + M(1 - z_2)*
   *   *2 <= x -0 + M(1-z_3)*
   *   *x - 0 <= 3 + M(1-z_3)*
    *  *y >= (x - 0) + 1 - M(1-z_1)*
    *  *y <= (x - 0) + 1 + M(1-z_1)*
    *  *y >= 2*(x-0) - M(1-z_2)*
    *  *y <= 2*(x-0) + M(1-z_2)*
    *  *y >= 3 - M(1-z_3)*
    *   *y <= 3 + M(1-z_3)*

Here is the corresponding Gurobi code:

```python
import gurobipy as gp

def model_piecewise_function_general():
    model = gp.Model("Piecewise_General")
    x = model.addVar(lb=0, vtype=gp.GRB.CONTINUOUS, name="x")
    y = model.addVar(vtype=gp.GRB.CONTINUOUS, name="y")
    z = model.addVars(3, vtype=gp.GRB.BINARY, name="z")

    M = 1000 #Big M value
    breakpoints = [0,1,2,3]
    # Only one segment can be active
    model.addConstr(sum(z[i] for i in range(3)) == 1, name="one_segment")
    # Force the x to follow the segment values
    model.addConstr(x - breakpoints[0] >= 0, name="x_lower0")
    model.addConstr(x - breakpoints[0] <= breakpoints[1] - breakpoints[0] + M*(1-z[0]), name="x_upper1")
    model.addConstr(x - breakpoints[0] >= breakpoints[1] - breakpoints[0] - M*(1-z[1]), name="x_lower1")
    model.addConstr(x - breakpoints[0] <= breakpoints[2] - breakpoints[0] + M*(1-z[1]), name="x_upper2")
    model.addConstr(x - breakpoints[0] >= breakpoints[2] - breakpoints[0] - M*(1-z[2]), name="x_lower2")
    model.addConstr(x - breakpoints[0] <= breakpoints[3] - breakpoints[0] + M*(1-z[2]), name="x_upper3")


    # Force the y to follow the piecewise function values
    model.addConstr(y >= (x - breakpoints[0]) + 1 - M*(1-z[0]), name ="y_lower1")
    model.addConstr(y <= (x - breakpoints[0]) + 1 + M*(1-z[0]), name="y_upper1")
    model.addConstr(y >= 2*(x-breakpoints[0]) - M*(1-z[1]), name="y_lower2")
    model.addConstr(y <= 2*(x-breakpoints[0]) + M*(1-z[1]), name="y_upper2")
    model.addConstr(y >= 3 - M * (1-z[2]), name="y_lower3")
    model.addConstr(y <= 3 + M*(1-z[2]), name="y_upper3")

    # Objective (example)
    model.setObjective(y, gp.GRB.MAXIMIZE)

    # Solve
    model.optimize()
    if model.status == gp.GRB.OPTIMAL:
        print(f"Optimal x: {x.x}, Optimal y: {y.x}, Active Z: {[z[i].x for i in range(3)]}")
    else:
        print("No optimal solution found")

    return model

if __name__ == '__main__':
    model = model_piecewise_function_general()
```

These techniques are foundational. When dealing with more complex functions, or in scenarios where performance is paramount, you might need to delve into more advanced techniques like the SOS2 (Special Ordered Sets of Type 2) constraints which are available in Gurobi, or more advanced decomposition algorithms when large sets are involved.

For further study, I'd recommend "Integer Programming" by Laurence A. Wolsey as a great resource for the mathematical underpinnings of these methods, and "Applied Mathematical Programming" by Bradley, Hax, and Magnanti which covers a wide array of modeling techniques, including those relevant to piecewise functions. "Optimization Modeling with Spreadsheets" by Baker, which is a simpler introduction to optimization, might be helpful for newcomers. For a more direct approach, and a wealth of information on the details of using Gurobi, the Gurobi documentation itself is invaluable. Lastly, exploring papers on piecewise linearization techniques in optimization literature, particularly those focusing on specific applications, will provide further insight into these topics. I've found that understanding the underlying mathematical structure helps greatly when things get complex, or when the standard approaches just don’t quite cut it. Remember, as with anything in optimization, it's often an iterative process, refining your model and constraints until you achieve the desired accuracy and performance.
