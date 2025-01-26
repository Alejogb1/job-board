---
title: "What are the roles of limits and symmetry in OR-Tools?"
date: "2025-01-26"
id: "what-are-the-roles-of-limits-and-symmetry-in-or-tools"
---

In my experience optimizing complex logistics networks, I've consistently found that effectively leveraging both limits and symmetry is crucial for achieving tractable solutions with Google's OR-Tools. These concepts, while seemingly distinct, often interact in intricate ways to define the search space and computational efficiency of constraint programming solvers. Ignoring either can lead to excessively long solve times or suboptimal results.

Limits, primarily, act as constraints on the exploration of the solution space. They are implemented to prevent unbounded search, which would be impractical even for moderately sized problems. OR-Tools uses a variety of limit types. One of the most common is the **time limit**, which restricts the solver's execution to a specific duration. This is critical in scenarios where a perfect solution is not required, but a reasonable solution within a time constraint is paramount, such as real-time decision-making systems. Consider a vehicle routing problem, where the solver could theoretically search indefinitely if no time limit was imposed. Another type of limit is the **solution limit**, which terminates the search once a specified number of solutions have been discovered. This can be useful for scenarios where the goal is to explore multiple alternatives, not necessarily find the absolute best. Furthermore, the **number of conflicts limit**, applicable when employing a conflict-directed search algorithm, dictates the maximum number of times the solver will backtrack due to constraint violations, providing another control mechanism over the search process. Finally, **branching limits** can restrict the number of decisions the solver makes. These limits prevent the solver from getting lost in unproductive branches and can drastically reduce run time.

Symmetry, conversely, is about the inherent structure of the problem itself. When a problem exhibits symmetry, multiple solutions that differ only in the arrangement of indistinguishable elements exist. For instance, in a job shop scheduling problem, if two machines are identical, interchanging the tasks assigned to these machines generates a solution of equivalent cost. In other words, these different solutions are actually isomorphic. This redundancy is problematic for search algorithms because they must exhaustively explore the solution space regardless of the symmetry. The search algorithms waste resources considering solutions that are fundamentally the same.

Breaking symmetry can dramatically reduce the search space and significantly enhance solver performance. OR-Tools supports several methods for symmetry breaking, mostly implemented via adding additional constraints.  One common technique involves **value ordering**, forcing certain variables to take specific values first and thereby ruling out symmetric permutations. Another, more sophisticated, method involves adding **lexicographical ordering** constraints, effectively imposing an artificial ordering among symmetrical groups of variables. This type of approach prevents the solver from exploring equivalent permutations. In many cases the most effective method of handling symmetry is to carefully design the constraint model to eliminate the source of symmetry. This may involve adding redundant variables or more complex constraints.

Let's illustrate these concepts with code examples, utilizing the Python interface of OR-Tools.

**Example 1: Time Limit**

Here, a simplified assignment problem is set up, followed by the application of a time limit. Without the time limit, the solver will run until all possible solutions have been explored, which can be quite extended depending on problem size.

```python
from ortools.sat.python import cp_model

def assignment_with_time_limit(num_workers, num_tasks, costs):
    model = cp_model.CpModel()
    x = {}
    for worker in range(num_workers):
        for task in range(num_tasks):
            x[worker, task] = model.NewBoolVar(f'x_{worker}_{task}')

    for worker in range(num_workers):
        model.Add(sum(x[worker, task] for task in range(num_tasks)) <= 1)

    for task in range(num_tasks):
        model.Add(sum(x[worker, task] for worker in range(num_workers)) == 1)


    objective = sum(costs[worker][task] * x[worker, task] for worker in range(num_workers) for task in range(num_tasks))
    model.Minimize(objective)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    status = solver.Solve(model)


    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Total Cost: {solver.ObjectiveValue()}")
        for worker in range(num_workers):
          for task in range(num_tasks):
            if solver.Value(x[worker, task]):
              print(f"Worker {worker} assigned to task {task}")
    elif status == cp_model.INFEASIBLE:
      print("No solution found")
    else:
      print("Solution not found or proven optimal within time limit")

num_workers = 3
num_tasks = 3
costs = [[2, 5, 7], [4, 2, 9], [1, 6, 8]]
assignment_with_time_limit(num_workers, num_tasks, costs)
```

In this code, I explicitly set the `max_time_in_seconds` parameter to 5 seconds. The solver will attempt to find the optimal solution within that time frame and will output what was achieved within the time limit. If the solver doesn't find the solution, it will tell the user via the status checks. Without setting this time limit, the program would be guaranteed to eventually find the solution, but this could take much longer for larger problems.

**Example 2: Value Ordering for Symmetry Breaking**

Consider a scheduling problem involving identical machines. Without breaking symmetry, the solver will explore different arrangements of tasks on these identical machines, even though the cost is the same. Here's an example illustrating this. Note that this example will not completely remove all symmetry, which can be a challenging and complex process, but it will provide an example of basic symmetry breaking.

```python
from ortools.sat.python import cp_model

def job_shop_with_symmetry_breaking(num_machines, num_jobs, processing_times):
    model = cp_model.CpModel()
    horizon = sum(max(times) for times in processing_times)
    
    starts = {}
    ends = {}
    intervals = {}
    
    for job in range(num_jobs):
      for machine in range(num_machines):
        starts[(job, machine)] = model.NewIntVar(0, horizon, f'start_j{job}_m{machine}')
        ends[(job, machine)] = model.NewIntVar(0, horizon, f'end_j{job}_m{machine}')
        intervals[(job, machine)] = model.NewIntervalVar(starts[(job, machine)], processing_times[job][machine], ends[(job, machine)], f'interval_j{job}_m{machine}')
        
    for job in range(num_jobs):
        model.Add(starts[(job, 0)] >= 0)
        for machine in range(num_machines-1):
          model.Add(starts[(job, machine+1)] >= ends[(job, machine)])

    for machine in range(num_machines):
        model.AddNoOverlap([intervals[(job,machine)] for job in range(num_jobs)])


    
    model.Add(starts[(0,0)] == 0) #value ordering symmetry breaking, first job must start at 0

    objective = model.NewIntVar(0, horizon, 'objective')
    for job in range(num_jobs):
        model.Add(objective >= ends[(job, num_machines-1)])
    model.Minimize(objective)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"Makespan: {solver.ObjectiveValue()}")
        for job in range(num_jobs):
            for machine in range(num_machines):
                print(f"Job {job} on machine {machine}: Starts at {solver.Value(starts[(job, machine)])}, Ends at {solver.Value(ends[(job, machine)])}")
    else:
        print("No solution found.")


num_machines = 2
num_jobs = 3
processing_times = [[3,2],[2,4],[1,3]]
job_shop_with_symmetry_breaking(num_machines, num_jobs, processing_times)
```

Here, I added `model.Add(starts[(0,0)] == 0)` to the problem. This explicitly forces the first job on the first machine to start at 0. While simplistic, this kind of value ordering prevents the solver from exploring solutions that only differ in how jobs are shuffled among the machines. Without this added constraint, the solver might spend time examining equivalent possibilities.

**Example 3: Conflict Limit**

In this example, I will set a limit on the number of conflicts the solver will allow. Conflicts are a way to measure how much the solver had to backtrack. A high conflict count is a sign of a problem that is highly constrained and that the solver is having difficulties with.

```python
from ortools.sat.python import cp_model
import random

def conflict_limit(num_vars, num_constraints):
    model = cp_model.CpModel()
    vars = {}
    for i in range(num_vars):
      vars[i] = model.NewBoolVar(f'x_{i}')

    for i in range(num_constraints):
        var1 = random.randint(0,num_vars-1)
        var2 = random.randint(0,num_vars-1)
        model.Add(vars[var1] + vars[var2] <= 1)


    solver = cp_model.CpSolver()
    solver.parameters.max_number_of_conflicts = 1000
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print("Solution Found")
    elif status == cp_model.INFEASIBLE:
      print("No solution found")
    else:
      print("Solution not proven optimal before conflict limit reached")
    print(f"Conflicts found: {solver.NumConflicts()}")
    print(f"Branches found: {solver.NumBranches()}")

num_vars = 5
num_constraints = 15
conflict_limit(num_vars, num_constraints)
```

In this example, a number of boolean variables are created, and then constraints are added to them such that for each constraint, no more than one of a pair of random variables can be set to true. I then set the `max_number_of_conflicts` parameter in the solver. If the limit is reached before the solution is found the solver will return. This example also outputs both the conflicts and the branches in the solution process. By limiting conflicts it is possible to allow the program to run quickly and fail fast.

To deepen your understanding of these concepts within OR-Tools, I recommend exploring the official documentation, focusing on the sections related to solver parameters and symmetry detection. There are also excellent textbooks and scholarly articles on constraint programming that can further elucidate the theoretical basis of these ideas. Additionally, practical application experience, particularly with diverse problem types, is vital for mastering the nuances of limits and symmetry in solving complex optimization challenges. I've personally learned the most through hands on work.
