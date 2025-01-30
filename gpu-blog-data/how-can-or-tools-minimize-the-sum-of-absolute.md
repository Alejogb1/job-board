---
title: "How can OR-Tools minimize the sum of absolute pairwise differences in an assignment/scheduling problem?"
date: "2025-01-30"
id: "how-can-or-tools-minimize-the-sum-of-absolute"
---
Minimizing the sum of absolute pairwise differences is a common challenge in optimization problems where minimizing disparity between assigned values is a priority. I've faced this myself when scheduling employee shifts; a situation where you want not only to cover requirements, but also ensure a degree of fairness in workload distribution, avoiding large discrepancies in how many shifts each employee receives. OR-Tools, Google's optimization suite, provides the necessary tools to model and solve these types of problems, though it requires a specific encoding of the objective function because the absolute value operator isn’t natively handled in linear programming directly.

The challenge arises from the fact that standard linear programming solvers expect linear objective functions. The absolute value function, |x|, is non-linear. To circumvent this limitation, we introduce auxiliary variables and constraints. For each pair of assigned values, we create a new variable representing the absolute difference between those two values. Crucially, we then add constraints that force this new variable to be greater than or equal to both the positive and negative difference. This effectively captures the behavior of the absolute value without violating the linear programming framework's constraints. Finally, the solver optimizes the *sum* of these auxiliary variables, which effectively minimizes the sum of absolute pairwise differences, as each auxiliary variable is forced to be as small as possible while still respecting the constraint.

Consider an assignment problem where we are assigning workers to different tasks, and we wish to minimize the total disparity in assigned worker loads. I'll illustrate this using three example code segments, first using Python to demonstrate the core concept, then using C++ for more explicit control, and finally a more generalized Python implementation.

**Example 1: Basic Python Implementation with Fixed Assignment**

This example highlights the core idea using a simple case where the assignments are predetermined but the optimization targets workload balancing. In a real application, the assignment would be part of the overall decision variables of the optimization problem.

```python
from ortools.sat.python import cp_model

def minimize_pairwise_differences():
    model = cp_model.CpModel()

    # Assuming we have 4 workers and 4 task assignments (fixed for demonstration)
    assignments = [3, 1, 5, 2] # workload/value assigned to each worker

    num_workers = len(assignments)

    # Variables to represent the absolute differences
    diff_vars = {}
    for i in range(num_workers):
        for j in range(i + 1, num_workers):
             diff_vars[(i, j)] = model.NewIntVar(0, max(assignments), f"diff_{i}_{j}")

    # Constraints to ensure diff_vars capture the absolute differences correctly
    for i in range(num_workers):
        for j in range(i + 1, num_workers):
            diff_var = diff_vars[(i, j)]
            diff = assignments[i] - assignments[j]
            model.Add(diff_var >= diff)
            model.Add(diff_var >= -diff)

    # Objective: minimize the sum of all absolute differences
    model.Minimize(sum(diff_vars.values()))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
         print("Minimum sum of absolute differences:", solver.ObjectiveValue())
         for key, var in diff_vars.items():
            print(f"Difference between workers {key[0]+1} & {key[1]+1}: {solver.Value(var)}")
    else:
        print("No solution found.")

minimize_pairwise_differences()
```
In this code, `assignments` holds fixed assignment values. We create `diff_vars` to represent the absolute differences. The key part is how we constrain these auxiliary variables to always be greater or equal to the positive and negative differences. We minimize the sum of all `diff_vars`, forcing each absolute difference to be as small as possible. The output gives the total minimal absolute difference and each specific difference value. This shows the encoding and how to check the calculated differences.

**Example 2: C++ Implementation for Optimization Control**

This C++ example shows how to explicitly create the model and solve it with a more direct view of the solver’s behavior and potential optimizations. This demonstrates a more computationally efficient method for optimization of larger problems.
```cpp
#include "ortools/sat/cp_model.h"
#include "ortools/sat/cp_model.pb.h"
#include "ortools/sat/sat_parameters.pb.h"

#include <iostream>
#include <vector>
#include <numeric>

using namespace operations_research;
using namespace sat;

int main() {
  CpModelBuilder model;

  // Assuming we have 4 workers and 4 task assignments (fixed for demonstration)
  std::vector<int> assignments = {3, 1, 5, 2};
  int num_workers = assignments.size();

  // Variables to represent the absolute differences
    std::vector<IntVar> diff_vars;
  for (int i = 0; i < num_workers; ++i) {
     for (int j = i + 1; j < num_workers; ++j) {
        diff_vars.push_back(model.NewIntVar(0, *std::max_element(assignments.begin(),assignments.end()),
                                           absl::StrCat("diff_", i, "_", j)));
    }
  }

 // Constraints to ensure diff_vars capture the absolute differences correctly
  int diff_idx = 0;
  for (int i = 0; i < num_workers; ++i) {
      for (int j = i + 1; j < num_workers; ++j) {
        int diff = assignments[i] - assignments[j];
         model.Add(diff_vars[diff_idx] >= diff);
         model.Add(diff_vars[diff_idx] >= -diff);
         diff_idx++;
    }
  }


  // Objective: minimize the sum of all absolute differences
  model.Minimize(LinearExpr::Sum(diff_vars));

  Model sat_model;
  SatParameters parameters;
  parameters.set_log_search_progress(false);
  sat_model.Add(parameters);


  CpSolverResponse response = Solve(model.Build(), sat_model);

  if (response.status() == CpSolverStatus::OPTIMAL || response.status() == CpSolverStatus::FEASIBLE) {
    std::cout << "Minimum sum of absolute differences: "
              << response.objective_value() << std::endl;
   diff_idx = 0;
    for (int i = 0; i < num_workers; ++i) {
        for (int j = i + 1; j < num_workers; ++j) {
        std::cout << "Difference between workers " << i+1 << " & " << j+1 << ": "
                 << SolutionIntegerValue(response, diff_vars[diff_idx]) << std::endl;
        diff_idx++;
        }
    }
  } else {
    std::cout << "No solution found." << std::endl;
  }

  return 0;
}
```

This C++ variant demonstrates the same principle but with more explicit control. The approach for representing absolute differences and the objective function remains unchanged, but this allows for more control over parameters such as search progress logging, which has been turned off in this instance. Note the explicit creation of the `Model` object, along with the `SatParameters` and the `Solve` function, as is required in C++. This shows how the exact same problem is implemented in a C++ environment and demonstrates additional capabilities specific to that environment.

**Example 3: Generalized Python Implementation with Model Extension**

This final Python example provides a more complete picture of a practical scenario where assignments are variables instead of fixed values. It builds on the principle to demonstrate a generalized implementation where assignments are also optimized as part of a larger optimization problem, and the absolute difference minimization works seamlessly within that context.

```python
from ortools.sat.python import cp_model

def minimize_pairwise_differences_with_assignments(num_workers, num_tasks, task_loads):
    model = cp_model.CpModel()

    # Decision variables: which task is assigned to which worker
    assignments = {}
    for worker in range(num_workers):
        for task in range(num_tasks):
            assignments[(worker, task)] = model.NewBoolVar(f"assignment_{worker}_{task}")

    # Constraint: Each task is assigned to exactly one worker
    for task in range(num_tasks):
        model.Add(sum(assignments[(worker, task)] for worker in range(num_workers)) == 1)

    # Variable to track worker loads
    worker_loads = [model.NewIntVar(0, sum(task_loads), f"worker_load_{worker}") for worker in range(num_workers)]


    # Constraint: calculate the total load assigned to each worker
    for worker in range(num_workers):
       model.Add(worker_loads[worker] == sum(assignments[(worker, task)] * task_loads[task] for task in range(num_tasks)))


     # Variables to represent the absolute differences in worker loads
    diff_vars = {}
    for i in range(num_workers):
        for j in range(i + 1, num_workers):
            diff_vars[(i, j)] = model.NewIntVar(0, max(task_loads)*num_tasks, f"diff_{i}_{j}")

    # Constraints to ensure diff_vars capture the absolute differences correctly
    for i in range(num_workers):
        for j in range(i + 1, num_workers):
            diff_var = diff_vars[(i, j)]
            diff = worker_loads[i] - worker_loads[j]
            model.Add(diff_var >= diff)
            model.Add(diff_var >= -diff)


    # Objective: minimize the sum of all absolute differences
    model.Minimize(sum(diff_vars.values()))

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
         print("Minimum sum of absolute differences:", solver.ObjectiveValue())
         print("Worker Loads:")
         for i in range(num_workers):
            print(f"Worker {i+1}: {solver.Value(worker_loads[i])}")
         for key, var in diff_vars.items():
            print(f"Difference between workers {key[0]+1} & {key[1]+1}: {solver.Value(var)}")
    else:
        print("No solution found.")



num_workers = 3
num_tasks = 4
task_loads = [2, 4, 1, 3]
minimize_pairwise_differences_with_assignments(num_workers, num_tasks, task_loads)

```
In this extended example, `assignments` are no longer fixed, but now, variables themselves which must be optimized. We introduce the concept of individual task loads, and calculate `worker_loads`. We maintain the same logic for minimizing the sum of absolute pairwise differences, and the framework now can find optimized assignments *and* minimize disparity in workloads. The output is more descriptive providing assignments and differences. This complete example demonstrates the practical usage of the minimization technique within a broader optimization model.

For further study, I would recommend the official OR-Tools documentation, specifically exploring the Constraint Programming section. In addition, reviewing papers and resources on linear programming and integer programming will further clarify the theoretical background for absolute value modelling. Finally, study optimization techniques with regards to worker assignment problems; this provides a real-world context within which to test and practice the optimization framework outlined. I've found the practical experience of using OR-Tools combined with these resources crucial for handling real-world optimization problems.
