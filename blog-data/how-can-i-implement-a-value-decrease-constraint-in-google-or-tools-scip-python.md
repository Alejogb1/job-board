---
title: "How can I implement a value decrease constraint in Google OR-Tools SCIP (Python)?"
date: "2024-12-23"
id: "how-can-i-implement-a-value-decrease-constraint-in-google-or-tools-scip-python"
---

, let's talk about value decrease constraints in Google or-tools scip, something I've definitely bumped into more than once during various optimization projects. It’s a common need: you often find yourself needing to ensure certain decision variables only ever decrease, and not increase, across different stages or conditions in your model. Let's get into how to do that practically in Python, along with some examples.

The core issue is that standard linear programming solvers, like the underlying engine in scip, don't inherently understand a "decreasing value" concept; they look at constraints purely in terms of equations and inequalities. We need to explicitly encode that behavior through carefully chosen constraints. The general pattern I use is to essentially force subsequent variables to be less than or equal to their predecessors. This approach works well for integer, continuous, and binary variables.

To understand this better, imagine you're modeling inventory levels over time. At each time step, the inventory should either remain the same or decrease, representing consumption or use. We definitely don't want an arbitrary solver increasing the inventory level just because it finds some mathematically permissible solution. I faced precisely this in a large-scale logistics optimization model a few years ago, when I failed to properly constrain inventory values across periods. It led to some seriously nonsensical outputs until I debugged it. The lesson there was definitely to make decrease constraints explicit from the beginning, and to double-check my reasoning (even with my years of experience!).

Let's solidify this with a few code snippets. We’ll assume you've already set up your basic ortools model, meaning you have your `model = cp_model.CpModel()` instance going, and know how to add variables.

**Example 1: Simple Linear Decrease with Integer Variables**

Here, I'll demonstrate the most basic case: sequentially decreasing integer variables. Imagine representing a decreasing quota over multiple periods.

```python
from ortools.sat.python import cp_model

def linear_decrease_example(num_periods):
    model = cp_model.CpModel()

    quotas = [model.NewIntVar(0, 100, f'quota_{i}') for i in range(num_periods)]

    # Constraint to enforce the decreasing order
    for i in range(num_periods - 1):
        model.Add(quotas[i+1] <= quotas[i])

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        for i in range(num_periods):
            print(f"Period {i}: Quota = {solver.Value(quotas[i])}")
    else:
        print("No solution found.")

linear_decrease_example(5)
```

This is probably the simplest way to enforce a decreasing sequence. The core of it lies in the loop where we iterate through the `quotas` list and add a constraint for each consecutive pair, making sure the next is less than or equal to the previous one. It’s important to explicitly add each of these `less than or equal to` constraints; scip doesn't automatically infer the decreasing requirement.

**Example 2: Decreasing Continuous Variables with a Lower Bound**

Here, I'll demonstrate a decrease constraint applied to continuous variables, with a non-zero lower bound. Suppose these represent fluid levels in a tank that might have some remaining after each usage, but should never refill.

```python
from ortools.sat.python import cp_model

def continuous_decrease_example(num_periods):
    model = cp_model.CpModel()

    tank_levels = [model.NewFloatVar(5.0, 50.0, f'tank_level_{i}') for i in range(num_periods)]

    # Constraint to enforce the decreasing order
    for i in range(num_periods - 1):
        model.Add(tank_levels[i+1] <= tank_levels[i])


    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
         for i in range(num_periods):
             print(f"Period {i}: Tank Level = {solver.Value(tank_levels[i])}")
    else:
         print("No solution found.")


continuous_decrease_example(5)
```

The principle remains the same: we add constraints that force a `less than or equal to` relationship between successive variables. The key difference here is the type of variables and the bounds we set. This illustrates that this pattern can be used for different types of decision variables. The lower bound of 5.0 prevents them from going below a certain level, but allows for decreases from initial values within the upper bound of 50.0.

**Example 3: Conditional Decrease Based on Binary Variables**

Now, let's get a bit more sophisticated. Imagine a scenario where you only want a resource level to decrease if a certain process is active. We can control this using a binary variable for process activation.

```python
from ortools.sat.python import cp_model

def conditional_decrease_example(num_periods):
    model = cp_model.CpModel()

    resource_levels = [model.NewIntVar(0, 100, f'resource_{i}') for i in range(num_periods)]
    process_active = [model.NewBoolVar(f'process_active_{i}') for i in range(num_periods - 1)]

    # Conditional constraint: resource decrease only if process is active
    for i in range(num_periods - 1):
        model.AddImplication(process_active[i], resource_levels[i+1] <= resource_levels[i])
        # If process is not active, resource level can be the same as the previous
        model.AddImplication(process_active[i].Not(), resource_levels[i+1] == resource_levels[i])

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
         for i in range(num_periods - 1):
             print(f"Period {i}: Resource = {solver.Value(resource_levels[i])}, Process Active = {solver.Value(process_active[i])}")
         print(f"Period {num_periods - 1}: Resource = {solver.Value(resource_levels[num_periods-1])}")
    else:
         print("No solution found.")

conditional_decrease_example(5)
```

Here, we use the `AddImplication` method. It allows us to specify, "if the `process_active` variable is true, then the next `resource_level` must be less than or equal to the current level." The alternative constraint ensures that if `process_active` is false, the `resource_level` remains the same. This kind of conditional logic often shows up in more complex optimization problems and, when using the right approach, becomes very manageable.

**General Recommendations and Further Study**

These examples should provide a solid practical foundation. But if you want a deeper dive, I would suggest looking into the core texts related to mathematical programming. *Integer Programming* by Wolsey is a classic, and it provides the theoretical underpinning of the modeling techniques I’ve discussed. You might also find *Model Building in Mathematical Programming* by Paul Williams useful as a more applied text focusing on how to translate real-world problems to mathematical models. Finally, for practical implementations and algorithm explanations, read the original scip publications. They provide the most direct understanding into how the solvers operate and why these particular constraints work.

The crucial takeaway from this discussion is this: implementing value decrease constraints in ortools scip involves explicitly using inequalities to make sure each variable in a sequence is less than or equal to its predecessor, or using conditional logic to achieve the same behavior within more complex conditions. Don't assume the solver will 'understand' or 'infer' this; it needs to be explicitly told. Doing so methodically will save you some debugging hours down the road, trust me on this one!
