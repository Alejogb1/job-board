---
title: "How can I add a time dimension to each node in Google OR-Tools?"
date: "2025-01-30"
id: "how-can-i-add-a-time-dimension-to"
---
The core challenge when modeling time in Google OR-Tools, especially when dealing with network flows or vehicle routing, stems from the fact that nodes themselves don’t inherently possess a time dimension. OR-Tools primarily focuses on optimizing paths or schedules based on costs and constraints associated with *edges* or *arcs* between nodes. Therefore, introducing a time-based attribute to nodes requires a careful re-interpretation of how you represent your problem and leverage OR-Tools' capabilities. My experiences implementing complex logistics optimization systems have highlighted that direct node-level time assignment isn't readily available; it's more about constraining *when* a transition to or from a node can occur.

To effectively integrate time into the problem formulation, you must transform the concept of a "node" into a time-aware entity. I've found two prevalent approaches, often used in combination: introducing time *windows* associated with visiting or departing from a node and modeling node visits as *intervals* with corresponding start and end times. These approaches rely heavily on the concept of variables that encode time information rather than attaching time directly to node identifiers.

The first approach, using time windows, is effective for cases where there are specific time periods during which a node can be active or accessible. We model the start and end of a time window, typically expressed as integers representing discrete time steps, for each node. When formulating the problem in OR-Tools, we then utilize constraints to ensure that any variable representing an activity at the node is bound to occur within the node's defined time window. This approach, however, does not explicitly capture how long a vehicle or entity remains at a node. It simply restricts the time *when* the node can be entered.

The second approach, more comprehensive in my experience, employs *intervals*. Here, each node visit or activity is modeled as a decision variable that dictates *when* an action at the node begins and ends, along with its duration. This allows for capturing not only the time window within which an event can happen, but also the time it spends at the node. I find this method more adaptable to nuanced problems, especially those involving loading and unloading or other tasks performed at specific locations, and where the amount of time spent at each node matters as a resource constraint.

Let me illustrate with three practical code examples. These snippets are in Python, leveraging the OR-Tools Python library, and represent fragments of larger models focusing on time integration.

**Example 1: Time Windows using a Routing Model**

This first example demonstrates adding a time window to locations in a Vehicle Routing Problem (VRP). We define a simple VRP where each location must be visited within a specific time period.

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['time_windows'] = [
        (0, 5), (2, 7), (3, 9), (5, 10), (6, 12) # (start, end) time windows for each location.
    ]
    data['time_matrix'] = [
        [0, 1, 2, 3, 4],
        [1, 0, 1, 2, 3],
        [2, 1, 0, 1, 2],
        [3, 2, 1, 0, 1],
        [4, 3, 2, 1, 0] # travel time matrix
    ]
    data['num_vehicles'] = 1
    data['depot'] = 0
    return data

def main():
    data = create_data_model()
    manager = pywrapcp.RoutingIndexManager(len(data['time_matrix']),
                                              data['num_vehicles'], data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def travel_time_callback(from_index, to_index):
      """Returns the travel time between the two locations."""
      from_node = manager.IndexToNode(from_index)
      to_node = manager.IndexToNode(to_index)
      return data['time_matrix'][from_node][to_node]

    travel_time_callback_index = routing.RegisterTransitCallback(travel_time_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(travel_time_callback_index)

    time_dimension_name = 'Time'
    routing.AddDimension(
      travel_time_callback_index,
      60,  # allow time windows and slacks of up to this value.
      60,  # max time per vehicle
      False,
      time_dimension_name)
    time_dimension = routing.GetDimensionOrDie(time_dimension_name)

    for location_idx, time_window in enumerate(data['time_windows']):
        index = manager.NodeToIndex(location_idx)
        time_dimension.CumulVar(index).SetRange(time_window[0], time_window[1])

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
      routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
      print('Objective: {}'.format(solution.ObjectiveValue()))
      time_var = time_dimension.CumulVar(manager.NodeToIndex(1))
      print("Visit time for node 1: start {}, end {}".format(solution.Value(time_var), solution.Value(time_var) + 1 ))


if __name__ == '__main__':
    main()
```

In this example, I used `routing.AddDimension` to create a time dimension. Then, I iterated over the `data['time_windows']` list, setting upper and lower bounds to the time dimension's cumul variables for each node. This constrains node visits within their specified time windows using `SetRange()`. The `CumulVar(index)` variable stores the cumulative time up to the associated node, ensuring that visits occur within the defined windows, while respecting the provided `travel_time_callback` function.

**Example 2: Modeling Node Activities as Intervals**

Here, I model node activities as intervals within a scheduling model. This allows not only time windows, but also a duration for tasks. The example has three tasks. Each task is an interval and has a starting and ending time, which I constrain.

```python
from ortools.sat.python import cp_model

def main():
  model = cp_model.CpModel()
  horizon = 20  # Maximum end time for all tasks

  # Creates the interval variables.
  tasks = {}
  tasks[1] =  model.NewIntervalVar(2,5,7, 'task 1')   #start, duration, end, name
  tasks[2] = model.NewIntervalVar(4, 3, 7, 'task 2')
  tasks[3] = model.NewIntervalVar(8, 2, 10, 'task 3')


  # Adds constraint for temporal ordering: the tasks must be consecutive.
  model.Add(cp_model.StartVar(tasks[2]) >= cp_model.EndVar(tasks[1]))
  model.Add(cp_model.StartVar(tasks[3]) >= cp_model.EndVar(tasks[2]))

  # Creates the solver and solve.
  solver = cp_model.CpSolver()
  status = solver.Solve(model)

  if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
      for i in tasks:
          print(f"Task {i}: start={solver.Value(cp_model.StartVar(tasks[i]))}, end={solver.Value(cp_model.EndVar(tasks[i]))}")
  else:
      print("No solution found.")

if __name__ == '__main__':
  main()
```

In this example, I utilized `model.NewIntervalVar` to represent node activities as interval variables. Each interval has a start time, a duration, and an end time. I use these variables to apply sequencing constraints, ensuring that tasks are completed sequentially and each task is performed within its respective defined interval bounds. This allows us to manage both *when* an activity begins and *how long* it takes.

**Example 3: Combining Time Windows and Intervals**

This example combines the techniques of time windows and intervals in a resource scheduling scenario. I have two tasks, each with a minimum duration. There is a time window for each task and they have to be done in consecutive order.

```python
from ortools.sat.python import cp_model

def main():
  model = cp_model.CpModel()
  horizon = 20

  tasks = {}
  tasks[1] =  model.NewIntervalVar(0, 2, horizon, 'task 1')
  tasks[2] = model.NewIntervalVar(0, 3, horizon, 'task 2')

  # Time windows.
  model.Add(cp_model.StartVar(tasks[1]) >= 2) # start greater or equal to 2.
  model.Add(cp_model.EndVar(tasks[1]) <= 8) # end less or equal to 8.
  model.Add(cp_model.StartVar(tasks[2]) >= 5) # start greater or equal to 5.
  model.Add(cp_model.EndVar(tasks[2]) <= 10) # end less or equal to 10.

  # Constraint for temporal ordering: the tasks must be consecutive.
  model.Add(cp_model.StartVar(tasks[2]) >= cp_model.EndVar(tasks[1]))


  solver = cp_model.CpSolver()
  status = solver.Solve(model)

  if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
      for i in tasks:
          print(f"Task {i}: start={solver.Value(cp_model.StartVar(tasks[i]))}, end={solver.Value(cp_model.EndVar(tasks[i]))}")
  else:
    print("No solution found.")

if __name__ == '__main__':
  main()
```

This example combines both time window and duration. The `model.NewIntervalVar` defines the intervals for the tasks. The task's start time and end time are bound using `model.Add()` and using their respective start and end interval variables through `cp_model.StartVar()` and `cp_model.EndVar()`. The two tasks are also consecutive. This illustrates how to blend different techniques for richer time-based constraints.

To expand your knowledge in this area, I would recommend several resources beyond the official Google OR-Tools documentation. I have found the book "Constraint Programming in Python" to be invaluable. It provides in-depth coverage of constraint programming techniques that are fundamental to working effectively with the OR-Tools library. Other useful materials are found on academic websites, especially those from universities with strong operations research departments, like those of MIT or Stanford. Also, consider exploring resources that cover constraint satisfaction problems more generally, which will help provide a foundational understanding of the principles behind this problem-solving method, allowing for more creativity when facing new challenges.
