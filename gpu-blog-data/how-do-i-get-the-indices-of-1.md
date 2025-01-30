---
title: "How do I get the indices of '1' values from a list of OR-Tools variables?"
date: "2025-01-30"
id: "how-do-i-get-the-indices-of-1"
---
Working extensively with constraint programming, specifically OR-Tools, reveals a common necessity: extracting the specific indices where binary variables evaluate to '1' within a list. This need arises frequently when interpreting solution vectors, translating solver outputs into tangible actions, or performing post-processing calculations. Neglecting the efficient retrieval of these indices can lead to cumbersome, and potentially less performant, code.

The core challenge stems from the fact that OR-Tools variables aren't standard Python lists; they're specialized objects representing decision variables within the optimization model. Directly iterating and inspecting their values before a solution is found results in accessing the underlying expression, not the solution assignment. Therefore, one must access the solution values after the solver finds an optimal solution by interacting with the model's solution. This involves iterating through the variable list and using methods specific to the solver's solution object.

The strategy is straightforward: after the solver successfully finds a solution, its solution object provides the necessary mechanism to query the concrete values assigned to variables. We access these solution values, typically using the `Value()` method specific to OR-Tools variables, and then build a new list of the indices that correspond to a value of '1'. The process generally involves iterating through the list of variables and appending the index to a result list when the variable's value evaluates to '1'.

Let's consider a scenario where I'm managing resource allocation across different tasks. A list of Boolean variables, `task_assignments`, represents whether a task is assigned to a particular resource. Each index within the `task_assignments` corresponds to a specific task-resource pair. After solving, I need to determine exactly which task-resource pairings were selected (i.e., where the binary variables have a value of '1').

Here’s a basic implementation:

```python
from ortools.sat.python import cp_model

def get_indices_of_ones_basic(variables, solver):
  """
  Retrieves indices where variables in a list have a value of 1 after solving.

  Args:
      variables: A list of OR-Tools BoolVar variables.
      solver: The OR-Tools CP-SAT solver object.

  Returns:
      A list of integers representing indices where variables are 1.
  """
  indices = []
  for i, var in enumerate(variables):
    if solver.Value(var) == 1:
      indices.append(i)
  return indices

# Example Usage
model = cp_model.CpModel()
num_vars = 5
vars = [model.NewBoolVar(f'x_{i}') for i in range(num_vars)]

# Simple constraint example: at least one variable must be true
model.Add(sum(vars) >= 1)

solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    indices_of_ones = get_indices_of_ones_basic(vars, solver)
    print(f"Indices with value 1: {indices_of_ones}")
```

This function `get_indices_of_ones_basic` takes a list of OR-Tools boolean variables and the solver object as input. It iterates through the list, and using the `solver.Value()` function, retrieves the assigned value from the solved instance. If that value is 1, the index `i` is added to the result list. This initial example is direct but could benefit from a more concise approach.

My second project involved handling a more complex routing problem where various segments of a route were represented by boolean decision variables. In this instance, processing the returned indices as a NumPy array proved significantly faster for subsequent analysis and manipulation of the routes, including distance calculations and clustering operations. A NumPy-based version is often more advantageous with large datasets due to its optimized numerical operations.

Here is the NumPy-optimized approach:

```python
import numpy as np
from ortools.sat.python import cp_model

def get_indices_of_ones_numpy(variables, solver):
    """
    Retrieves indices using numpy where variables in a list have a value of 1 after solving.

    Args:
        variables: A list of OR-Tools BoolVar variables.
        solver: The OR-Tools CP-SAT solver object.

    Returns:
        A NumPy array of integers representing indices where variables are 1.
    """
    values = np.array([solver.Value(var) for var in variables])
    indices = np.where(values == 1)[0]
    return indices

# Example Usage (same model as before)
model = cp_model.CpModel()
num_vars = 5
vars = [model.NewBoolVar(f'x_{i}') for i in range(num_vars)]
model.Add(sum(vars) >= 1) #Simple constraint
solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    indices_of_ones_np = get_indices_of_ones_numpy(vars, solver)
    print(f"Indices with value 1 (NumPy): {indices_of_ones_np}")

```

This function `get_indices_of_ones_numpy` first uses a list comprehension to create a Python list of the values obtained using `solver.Value()`, which is then converted to a NumPy array. Then, `np.where` identifies all indices where the array values are equal to '1'. The returned result is directly a NumPy array, which allows for further vectorized processing. The code demonstrates similar functionality, but with the performance benefit of NumPy operations for large variable sets. It avoids the explicit looping used in the basic version.

Finally, there are scenarios, especially in larger models or instances with frequent solution queries, where the most computationally efficient approach might involve minimizing the calls to `solver.Value()`. In cases like these, it's more efficient to store all solution values once and process that storage.

Here is the efficient values-caching implementation:

```python
from ortools.sat.python import cp_model

def get_indices_of_ones_cached(variables, solver, solution_cache):
    """
    Retrieves indices where variables in a list have a value of 1 after solving, using a pre-computed cache.

    Args:
        variables: A list of OR-Tools BoolVar variables.
        solver: The OR-Tools CP-SAT solver object.
        solution_cache: a dictionary containing the solution values for all variables

    Returns:
        A list of integers representing indices where variables are 1.
    """

    indices = []
    for i, var in enumerate(variables):
      if solution_cache[var] == 1:
          indices.append(i)
    return indices

# Example Usage (same model as before)
model = cp_model.CpModel()
num_vars = 5
vars = [model.NewBoolVar(f'x_{i}') for i in range(num_vars)]
model.Add(sum(vars) >= 1)

solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    solution_cache = {}
    for var in vars:
        solution_cache[var] = solver.Value(var) # Store solution values once
    indices_of_ones_cached = get_indices_of_ones_cached(vars, solver, solution_cache)
    print(f"Indices with value 1 (Cached): {indices_of_ones_cached}")
```

In `get_indices_of_ones_cached`, a dictionary `solution_cache` is populated with the solution values of all variables using a single loop after the solver has found a solution.  This approach is useful for situations with many queries, where recalculating solution values each time is inefficient.

For further exploration, the official OR-Tools documentation is invaluable for understanding the nuances of different solver types (CP-SAT, MIP, etc.) and their respective interfaces. The examples section within the documentation provides many different problem formulations. Constraint Programming textbooks detail theoretical foundations, offering insights into problem modeling best practices. Texts on optimization will broaden one's perspective on algorithms and techniques. Finally, Python’s own standard documentation for list comprehension and NumPy documentation is useful to understand efficiency improvements when processing data. These are good starting points to advance your skill using the OR-Tools.
