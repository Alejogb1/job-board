---
title: "How can precedence constraints be modeled in a Pulp LP problem when time is an index for variables?"
date: "2025-01-30"
id: "how-can-precedence-constraints-be-modeled-in-a"
---
Precedence constraints, when incorporated into Linear Programming (LP) models indexed by time, require a careful formulation that ensures activities are sequenced correctly, respecting their temporal dependencies. I’ve encountered this challenge numerous times while optimizing production schedules for a multi-stage manufacturing process, where the completion of one operation directly dictates the start time of another. The key lies in expressing these relationships as linear inequalities that relate variables defined across different time points.

Typically, in LP models, decision variables represent the level of an activity at a specific time step. Let's assume `x[i,t]` represents the amount of activity `i` performed at time `t`. A precedence constraint between activity `i` and `j`, requiring `i` to be completed before `j` can begin, must then tie the time-indexed variables `x[i,t]` and `x[j,t]`. It's important to recognize that the completion of activity `i` is not defined by a single `x[i,t]` but is rather the accumulation of all units of activity `i` across the time steps.

Therefore, we need to define a separate variable or derived value that indicates when a specific activity is considered "finished," and then use this information to constrain when the succeeding activity can start.  One common approach involves introducing a binary completion variable, which, for practical purposes in a real production environment, I've found to be computationally demanding. More frequently, I employ a "start time" variable, which represents the earliest time activity can start, rather than an explicit completion indicator. This approach leads to a more straightforward formulation. Specifically, we enforce that an activity `j` can start only after the completion of all predecessors, as defined by a minimum start time, derived from the variables `x[i,t]` for those predecessors.

To illustrate, consider two activities, A and B, where B cannot start before A is finished. Let `t` be the time index, and `d_A` be the processing duration for activity A. Let's define `start_A` as the start time of activity A and `start_B` as the start time for activity B. We don't have the variable `x[i,t]` explicitly here; instead, we're assuming the problem can derive `start_A`, which is the usual output of a LP that is designed to schedule. Thus our constraint would be:

```python
from pulp import *

# Problem data
duration_A = 5  # Duration of activity A
duration_B = 3  # Duration of activity B

# Model
model = LpProblem("Precedence_Example", LpMinimize)

# Decision Variables
start_A = LpVariable("start_A", lowBound=0, cat='Integer')
start_B = LpVariable("start_B", lowBound=0, cat='Integer')

# Objective function (placeholder, as scheduling often optimizes on makespan)
model += start_B + duration_B # minimizing completion time of B

# Precedence Constraint
model += start_B >= start_A + duration_A

# Solving
model.solve()

# Output
print("Start time of A:", value(start_A))
print("Start time of B:", value(start_B))
```

This simple example defines `start_A` and `start_B` directly as variables. The core precedence constraint, `start_B >= start_A + duration_A`, dictates that activity B cannot start until activity A has been completed. This approach is effective when activities are treated as atomic units with definite starting times.

Let's consider a more practical case. Suppose we have multiple activities and multiple units of work (e.g., jobs). We can represent the activity being processed by another variable `y[i, t]`, which we assume can be derived from the variable `x[i,t]` in a larger model. Then the start time for any activity `j` must be determined dynamically within the optimization, considering the times that the preceding activity `i` has been performed. Let's assume that the duration of any activity `i` is a known quantity `d_i`. We also have a time horizon `T`.

```python
from pulp import *

# Problem data
activities = ["A", "B", "C"]
precedence = {"B": ["A"], "C": ["A"]}
durations = {"A": 2, "B": 3, "C": 4}
T = 10 # Time horizon

# Model
model = LpProblem("Precedence_MultipleActivities", LpMinimize)

# Decision Variables
start_times = LpVariable.dicts("start", activities, lowBound=0, cat='Integer')
completion_times = LpVariable.dicts("completion", activities, lowBound=0, cat='Integer')
activity_at_time = LpVariable.dicts("Activity_at_Time", [(i, t) for i in activities for t in range(T)], 0, 1, cat='Binary')
# This is the x[i,t], but we need to use y[i,t]

# Objective function
model += lpSum(completion_times.values())

# Constraints
# Define completion time based on the start time and duration
for activity in activities:
    model += completion_times[activity] == start_times[activity] + durations[activity]

# Precedence Constraints
for successor, predecessors in precedence.items():
    for predecessor in predecessors:
        model += start_times[successor] >= completion_times[predecessor]

# Activity start time constraints based on which activity is in process at which time
for activity in activities:
  model += lpSum(t * activity_at_time[activity, t] for t in range(T)) == start_times[activity]

# Ensure an activity can only have work completed in its assigned time window
for activity in activities:
    model += lpSum(activity_at_time[activity, t] for t in range(T)) == 1


# Each time can only have at most 1 activity in progress at a given time
for t in range(T):
    model += lpSum(activity_at_time[activity, t] for activity in activities) <= 1



# Solving
model.solve()

# Output
for activity in activities:
    print(f"Activity {activity} Start Time: {value(start_times[activity])}")
    print(f"Activity {activity} Completion Time: {value(completion_times[activity])}")
```

This expanded example introduces a dictionary `precedence` that defines which activities must finish before another begins. It iterates through these precedence relationships, applying the constraint that the start time of the successor is greater than or equal to the completion time of the predecessor. The `activity_at_time` variable has to exist because we did not specify directly in the earlier example how to formulate the start time variable in terms of time. This example shows one way to derive that `start_time` variable. This is quite common when a more general formulation is needed.

Let’s consider one more specific scenario, in the context of scheduling batches in a chemical process, where multiple batches of the same material might exist. Now we have to consider not only precedence relations, but also the fact that some activities have to occur continuously across the time horizon. The following formulation focuses on a simplified batch process. It uses indicator variables to denote the starting time of a series of activities. Here, we are assuming we have defined the `batches` variable which defines the number of batches of activity that exists

```python
from pulp import *

# Problem Data
activities = ["A", "B"]
durations = {"A": 3, "B": 2}
batches = 2 # Number of batches for each activity
T = 10 # Time Horizon


# Model
model = LpProblem("Precedence_Batches", LpMinimize)

# Decision variables

start_times = LpVariable.dicts("start", [(a, b) for a in activities for b in range(batches)], lowBound=0, cat='Integer')
completion_times = LpVariable.dicts("completion", [(a,b) for a in activities for b in range(batches)], lowBound=0, cat='Integer')
activity_start_time_indicator = LpVariable.dicts("activity_start_time_indicator", [(a,b,t) for a in activities for b in range(batches) for t in range(T)], 0, 1, cat='Binary')

# Objective function
model += lpSum(completion_times.values())

# Constraints

# Define completion time
for activity in activities:
    for batch in range(batches):
      model += completion_times[(activity, batch)] == start_times[(activity, batch)] + durations[activity]

# batch start time indicator constraint
for activity in activities:
  for batch in range(batches):
    model += lpSum(t * activity_start_time_indicator[(activity,batch,t)] for t in range(T)) == start_times[(activity, batch)]

# make sure batch is only scheduled once
for activity in activities:
  for batch in range(batches):
      model += lpSum(activity_start_time_indicator[(activity,batch,t)] for t in range(T)) == 1


# Precedence -  All batches of A complete before any of B starts
for batch_b in range(batches):
  for batch_a in range(batches):
    model += start_times[("B",batch_b)] >= completion_times[("A", batch_a)]

# Solving
model.solve()

# Output
for activity in activities:
    for batch in range(batches):
        print(f"Activity {activity} Batch {batch} Start Time: {value(start_times[(activity,batch)])}")
        print(f"Activity {activity} Batch {batch} Completion Time: {value(completion_times[(activity,batch)])}")

```
This example incorporates both the batch concept, and a way to formulate the start time of that batch. This is typically more complicated than the previous examples, but the main point is that one has to derive these "start time" variables from the other time indexed variables, `x[i,t]` as we discussed earlier.

For resources on this topic, I recommend exploring textbooks focused on linear programming and operations research, which usually have sections on scheduling and resource allocation that delve into time-indexed models. I also suggest referring to documentation on advanced optimization packages, including practical examples. Academic papers on production scheduling, particularly those involving mathematical programming, frequently demonstrate these techniques. Finally, case studies on supply chain optimization often showcase real-world applications of precedence constraints in LP.
