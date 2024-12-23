---
title: "How can ortools minimize L1 distances?"
date: "2024-12-23"
id: "how-can-ortools-minimize-l1-distances"
---

Alright, let's tackle minimizing L1 distances using ortools. It's a problem I’ve frequently encountered, particularly when dealing with resource allocation and route optimization tasks in the past. I remember one particularly thorny situation where we had to optimize delivery routes while considering a penalty proportional to the *city block* distance, not the euclidean distance. That's when L1 optimization became crucial, and ortools proved invaluable.

Minimizing L1 distances, also known as Manhattan or taxicab distances, differs fundamentally from minimizing squared euclidean distances. Unlike squared euclidean distance which, due to its differentiability, works well with gradient-based optimization techniques, L1 distances are not differentiable at zero. This characteristic poses a challenge for many standard optimization algorithms. However, ortools, with its diverse suite of solvers, provides several avenues to effectively handle L1 minimization problems.

Let's delve into the specifics. The L1 norm, defined as the sum of the absolute values of the differences, essentially means we're calculating the sum of distances along each axis rather than the straight-line distance. This can be particularly useful when dealing with constraints that are grid-like or movement that is restricted along cardinal directions.

Ortools doesn't directly have a specific solver labeled "L1 Minimization," but it provides tools and techniques that can be combined to achieve this. The most common method involves transforming the L1 problem into a mixed-integer linear programming (MILP) problem, which ortools’ CP-SAT solver handles quite capably.

The core idea is to introduce auxiliary variables. For each absolute value term, say `|x - y|`, we introduce two non-negative variables, `d_plus` and `d_minus`, such that:

1. `x - y = d_plus - d_minus`
2. `d_plus >= 0`
3. `d_minus >= 0`
4. `objective += d_plus + d_minus`

The constraint that `d_plus` and `d_minus` are non-negative enforces the absolute value, because in a minimization problem, the solver will always choose the smallest values of `d_plus` and `d_minus` possible, ensuring that their sum equals `|x - y|`.

To illustrate, let's consider three example scenarios, each showing a slightly different context:

**Example 1: Simple 1D L1 Minimization**

Suppose we want to minimize the sum of L1 distances from a single point `x` to several given points `a[i]` along a single axis. Let's assume points `a` are `[1, 5, 8]`. Here's how to implement that:

```python
from ortools.sat.python import cp_model

def minimize_l1_1d(a):
    model = cp_model.CpModel()
    x = model.NewIntVar(-10, 20, 'x') # Setting a reasonable range for x. Adjust as needed.

    objective = 0
    for val in a:
        d_plus = model.NewIntVar(0, 100, f'd_plus_{val}')
        d_minus = model.NewIntVar(0, 100, f'd_minus_{val}')

        model.Add(x - val == d_plus - d_minus)
        objective += d_plus + d_minus

    model.Minimize(objective)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Optimal value for x: {solver.Value(x)}")
        print(f"Minimized L1 distance: {solver.ObjectiveValue()}")
    else:
        print("No solution found")

minimize_l1_1d([1, 5, 8])

```

This code sets up the model using the introduced auxiliary variables and constraints. The `objective` is set to the sum of the `d_plus` and `d_minus` variables for each `a[i]`, which enforces the L1 minimization. The output will give you the best value for `x` that minimizes the sum of the L1 distances.

**Example 2: 2D L1 Minimization**

Let’s elevate the complexity to a 2D scenario, where we aim to minimize the L1 distance between point (x, y) and a set of points (ax[i], ay[i]). Consider the given points `[(1, 2), (4, 6), (8, 3)]`.

```python
from ortools.sat.python import cp_model

def minimize_l1_2d(points):
  model = cp_model.CpModel()
  x = model.NewIntVar(-10, 20, 'x')
  y = model.NewIntVar(-10, 20, 'y')

  objective = 0
  for ax, ay in points:
      dx_plus = model.NewIntVar(0, 100, f'dx_plus_{ax}_{ay}')
      dx_minus = model.NewIntVar(0, 100, f'dx_minus_{ax}_{ay}')
      dy_plus = model.NewIntVar(0, 100, f'dy_plus_{ax}_{ay}')
      dy_minus = model.NewIntVar(0, 100, f'dy_minus_{ax}_{ay}')

      model.Add(x - ax == dx_plus - dx_minus)
      model.Add(y - ay == dy_plus - dy_minus)
      objective += dx_plus + dx_minus + dy_plus + dy_minus

  model.Minimize(objective)
  solver = cp_model.CpSolver()
  status = solver.Solve(model)

  if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
      print(f"Optimal value for x: {solver.Value(x)}")
      print(f"Optimal value for y: {solver.Value(y)}")
      print(f"Minimized L1 distance: {solver.ObjectiveValue()}")
  else:
      print("No solution found.")

minimize_l1_2d([(1, 2), (4, 6), (8, 3)])
```

Here, we introduce separate auxiliary variables for the x and y dimensions and then sum them in the objective function. The CP-SAT solver then optimizes, finds the appropriate coordinates for (x,y), and outputs the minimum L1 distance sum.

**Example 3: L1 minimization within a larger optimization problem**

Often, minimizing L1 distances is just part of a bigger picture. Here, I'll show a very simplified example of such a situation where, in addition to choosing a point (x,y), we also have a binary decision variable that affects the cost:

```python
from ortools.sat.python import cp_model

def minimize_l1_with_binary(points):
  model = cp_model.CpModel()

  x = model.NewIntVar(-10, 20, 'x')
  y = model.NewIntVar(-10, 20, 'y')
  use_binary = model.NewBoolVar('use_binary')
  binary_cost = 10

  objective = 0
  for ax, ay in points:
      dx_plus = model.NewIntVar(0, 100, f'dx_plus_{ax}_{ay}')
      dx_minus = model.NewIntVar(0, 100, f'dx_minus_{ax}_{ay}')
      dy_plus = model.NewIntVar(0, 100, f'dy_plus_{ax}_{ay}')
      dy_minus = model.NewIntVar(0, 100, f'dy_minus_{ax}_{ay}')

      model.Add(x - ax == dx_plus - dx_minus)
      model.Add(y - ay == dy_plus - dy_minus)
      objective += dx_plus + dx_minus + dy_plus + dy_minus

  objective += use_binary * binary_cost # adding the binary decision cost to the objective function
  model.Minimize(objective)

  solver = cp_model.CpSolver()
  status = solver.Solve(model)

  if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
      print(f"Optimal value for x: {solver.Value(x)}")
      print(f"Optimal value for y: {solver.Value(y)}")
      print(f"Use binary decision: {solver.Value(use_binary)}")
      print(f"Minimized total cost: {solver.ObjectiveValue()}")
  else:
      print("No solution found.")

minimize_l1_with_binary([(1, 2), (4, 6), (8, 3)])
```

This expanded example introduces a binary variable that incurs a cost when it is set to 1. The solver will now optimize both the L1 distance and the binary decision, thus providing a comprehensive optimization result.

In all of these cases, setting appropriate bounds on variables can be crucial for performance. Ortoll’s documentation is, frankly, indispensable for understanding the finer details and options available in the CP-SAT solver. For a more theoretical background on optimization techniques, specifically linear programming and mixed-integer programming, I recommend reading "Linear Programming" by Vasek Chvatal. Additionally, "Convex Optimization" by Stephen Boyd and Lieven Vandenberghe offers a deep dive into optimization techniques which, though not directly addressing L1 minimization, is fundamental knowledge for any serious practitioner.

In practice, it is important to remember that transforming an L1 problem into a MILP form can introduce a significant number of variables and constraints, which can affect solving time. Thus, careful consideration of your model structure and efficient constraint encoding are very worthwhile pursuits. I've seen cases where preprocessing steps that try to remove the number of terms can significantly improve runtimes. However, with some effort, and a good grasp of how ortools can handle linear programs, minimizing L1 distances can be very effective for various real-world applications.
