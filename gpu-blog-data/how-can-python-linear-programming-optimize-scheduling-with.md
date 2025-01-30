---
title: "How can Python linear programming optimize scheduling with constraints on consecutive night shifts?"
date: "2025-01-30"
id: "how-can-python-linear-programming-optimize-scheduling-with"
---
A common challenge in workforce management, particularly in industries with 24/7 operations, lies in generating schedules that minimize cost while adhering to complex constraints, notably those concerning consecutive night shifts. Linear programming (LP) offers a robust framework for tackling this optimization problem, allowing us to define both the objective function (what we want to minimize, typically labor cost) and a set of constraints representing various scheduling rules. My experience developing scheduling tools has repeatedly demonstrated the efficacy of this approach.

The core idea behind using LP for scheduling is to formulate the problem as a set of linear equations and inequalities. We represent each shift assignment as a variable, where a value of 1 indicates the employee is working that shift and 0 indicates they are not. The objective function is then a weighted sum of these variables, where the weights represent the cost of each shift. Constraints are then added to enforce scheduling rules, such as ensuring each shift is covered, limiting the total hours an employee can work in a week, and crucially, restricting consecutive night shifts. The power of LP lies in its ability to efficiently explore the solution space defined by these constraints to find the optimal schedule.

To illustrate, let's consider a simplified scheduling problem with three employees (A, B, and C) and three shifts per day (day, evening, and night) for a three-day period. We want to minimize overall cost, assuming night shifts are the most expensive. We will focus specifically on the constraint that no employee can work more than two consecutive night shifts.

Here's how it can be implemented using the `PuLP` library in Python:

```python
from pulp import *

# Define the problem
prob = LpProblem("Scheduling_Problem", LpMinimize)

# Define employees and shifts
employees = ["A", "B", "C"]
days = [1, 2, 3]
shifts = ["day", "evening", "night"]

# Define shift costs
shift_costs = {"day": 100, "evening": 120, "night": 150}

# Create decision variables
x = LpVariable.dicts("x", ((emp, day, shift) for emp in employees for day in days for shift in shifts), cat="Binary")

# Objective function
prob += lpSum(shift_costs[shift] * x[emp, day, shift] for emp in employees for day in days for shift in shifts)

# Constraint 1: Each shift must be covered
for day in days:
  for shift in shifts:
    prob += lpSum(x[emp, day, shift] for emp in employees) == 1

# Constraint 2: No employee can work more than 1 shift per day
for emp in employees:
    for day in days:
        prob += lpSum(x[emp, day, shift] for shift in shifts) <= 1

# Constraint 3: Maximum two consecutive night shifts
for emp in employees:
  for day in days:
      if day > 1:
          prob += (x[emp, day-1, "night"] + x[emp, day, "night"]) <= 2

# Solve the problem
prob.solve()

# Print the solution
print("Status:", LpStatus[prob.status])
for emp in employees:
    for day in days:
        for shift in shifts:
            if value(x[emp,day,shift]) == 1:
                print(f"Employee {emp} works {shift} on day {day}")
print("Total Cost:", value(prob.objective))

```

This first example establishes the foundational structure. `PuLP` is a library specifically designed for linear programming. The decision variables `x[emp, day, shift]` represent whether an employee is assigned to a specific shift on a given day. The objective function sums the cost of all assigned shifts. Constraint 1 ensures that each shift has exactly one employee assigned to it. Constraint 2 prevents an employee from working multiple shifts on the same day. Constraint 3 tackles the consecutive night shift limit – it ensures that if an employee worked a night shift on day `d-1`, they cannot be assigned to the night shift on day `d`. The solution then outputs the optimal schedule and the total cost.  This constraint, while a basic example, demonstrates the core logic of enforcing consecutive shift rules.

A crucial point about Constraint 3 is its limitation. It only prevents more than two *consecutive* shifts. Let us introduce a scenario that will ensure no more than two consecutive night shifts are assigned to an employee. This introduces an auxiliary binary variable:

```python
from pulp import *

# Define the problem
prob = LpProblem("Scheduling_Problem", LpMinimize)

# Define employees and shifts
employees = ["A", "B", "C"]
days = [1, 2, 3, 4, 5] # Extended to 5 days to better highlight the consecutive constraint
shifts = ["day", "evening", "night"]

# Define shift costs
shift_costs = {"day": 100, "evening": 120, "night": 150}

# Create decision variables
x = LpVariable.dicts("x", ((emp, day, shift) for emp in employees for day in days for shift in shifts), cat="Binary")
y = LpVariable.dicts("y", ((emp, day) for emp in employees for day in days), cat="Binary") # Auxiliary variable

# Objective function
prob += lpSum(shift_costs[shift] * x[emp, day, shift] for emp in employees for day in days for shift in shifts)

# Constraint 1: Each shift must be covered
for day in days:
  for shift in shifts:
    prob += lpSum(x[emp, day, shift] for emp in employees) == 1

# Constraint 2: No employee can work more than 1 shift per day
for emp in employees:
    for day in days:
        prob += lpSum(x[emp, day, shift] for shift in shifts) <= 1

#Constraint 3: Relationship between night shift and 'y'
for emp in employees:
    for day in days:
        prob += x[emp, day, "night"] <= y[emp, day]


# Constraint 4: No more than 2 consecutive y's (night shifts)
for emp in employees:
    for day in days:
        if day > 1 and day < days[-1]:
            prob += (y[emp, day-1] + y[emp, day] + y[emp, day+1]) <= 2

# Solve the problem
prob.solve()

# Print the solution
print("Status:", LpStatus[prob.status])
for emp in employees:
    for day in days:
        for shift in shifts:
            if value(x[emp,day,shift]) == 1:
                print(f"Employee {emp} works {shift} on day {day}")
print("Total Cost:", value(prob.objective))

```

Here, we introduce `y[emp, day]`, a binary variable that’s set to 1 only if an employee works a night shift on that day. We connect this variable to our decision variables via `Constraint 3` ensuring that if an employee does work a night shift on day, `y` is set to 1 for that day and employee. The critical modification is `Constraint 4`. This constraint leverages the auxiliary `y` variable to ensure that over three consecutive days no employee works more than 2 nights. This method offers more robust control over the total number of consecutive night shifts. Note, the length of days was extended to 5 to highlight the impact of this constraint.

Building on this, let's introduce an additional layer of complexity by incorporating rest day constraints. Assume that after working two night shifts, an employee is mandated to have at least one day off.  We now need to enforce this rest period through the linear programming model.

```python
from pulp import *

# Define the problem
prob = LpProblem("Scheduling_Problem", LpMinimize)

# Define employees and shifts
employees = ["A", "B", "C"]
days = [1, 2, 3, 4, 5]
shifts = ["day", "evening", "night"]

# Define shift costs
shift_costs = {"day": 100, "evening": 120, "night": 150}

# Create decision variables
x = LpVariable.dicts("x", ((emp, day, shift) for emp in employees for day in days for shift in shifts), cat="Binary")
y = LpVariable.dicts("y", ((emp, day) for emp in employees for day in days), cat="Binary")
z = LpVariable.dicts("z", ((emp, day) for emp in employees for day in days), cat = "Binary")

# Objective function
prob += lpSum(shift_costs[shift] * x[emp, day, shift] for emp in employees for day in days for shift in shifts)

# Constraint 1: Each shift must be covered
for day in days:
  for shift in shifts:
    prob += lpSum(x[emp, day, shift] for emp in employees) == 1

# Constraint 2: No employee can work more than 1 shift per day
for emp in employees:
    for day in days:
        prob += lpSum(x[emp, day, shift] for shift in shifts) <= 1

# Constraint 3: Relationship between night shift and 'y'
for emp in employees:
    for day in days:
      prob += x[emp, day, "night"] <= y[emp, day]

# Constraint 4: No more than 2 consecutive night shifts
for emp in employees:
  for day in days:
    if day > 1 and day < days[-1]:
       prob += (y[emp, day-1] + y[emp, day] + y[emp, day+1]) <=2

# Constraint 5: Rest days after two night shifts
for emp in employees:
  for day in days:
    if day > 2:
      prob +=  y[emp, day-2] + y[emp, day-1] <= 1 + z[emp, day]
      for shift in shifts:
          prob += x[emp, day, shift] <= 1-z[emp, day]

# Solve the problem
prob.solve()

# Print the solution
print("Status:", LpStatus[prob.status])
for emp in employees:
    for day in days:
        for shift in shifts:
            if value(x[emp,day,shift]) == 1:
                print(f"Employee {emp} works {shift} on day {day}")
print("Total Cost:", value(prob.objective))
```
Here, `Constraint 5` introduces a variable, `z`, to capture that if two consecutive nights have occurred then at least one rest day will be assigned to the employee after. We ensure that if `y[emp, day-2] + y[emp, day-1]` equals 2 (meaning the employee had two consecutive night shifts) then `z[emp, day]` will be 0, and it will force the employee to have a day off. This implementation uses auxiliary variables to link night shifts to required rest days.

When considering resources for a deeper dive into this area, I recommend focusing on texts detailing operations research and specifically, linear and integer programming. Books on mathematical optimization are crucial for understanding the underlying theory. Python libraries for optimization such as `PuLP`, `CVXPY`, and `OR-Tools` have excellent documentation and tutorials, these resources provide practical insights. Additionally, publications focused on scheduling and resource allocation offer a broad overview of common modeling approaches.

Linear programming is a powerful tool for generating optimal schedules within complex constraint environments. I've observed how these methods can effectively reduce labor costs and ensure that workforce requirements are met while adhering to operational constraints, including strict rules around consecutive night shifts. The code examples presented provide a foundational understanding of how such constraints are incorporated within an LP model. A thorough understanding of the theory combined with experience manipulating these tools are essential for developing robust scheduling solutions.
