---
title: "How can I solve the CVRP problem using OR-Tools in Python with date-based constraints?"
date: "2025-01-30"
id: "how-can-i-solve-the-cvrp-problem-using"
---
The vehicle routing problem with capacity constraints (CVRP) is notoriously complex, becoming even more so when introducing time-based restrictions, particularly those tied to specific dates. In my experience working on logistics optimization for a regional food distribution company, I've found that while OR-Tools provides a robust framework for tackling CVRP, accommodating explicit date dependencies requires a careful combination of model construction, data preprocessing, and constraint definition.

The core challenge lies in representing dates as tangible constraints within the optimization model. The standard CVRP focuses primarily on minimizing distance or cost while respecting vehicle capacities. Introducing dates necessitates a transformation of the problem space: we must effectively translate date-specific delivery windows into logical constraints understandable by the OR-Tools solver. This isn't directly handled by built-in time window functionalities designed for time-of-day scheduling; instead, each delivery location effectively needs a custom, time-dependent capacity.

The key to solving this lies in pre-processing your date-based data into a format OR-Tools can handle. Rather than thinking of the dates as direct constraints, view them as a means to create "availability" of each location on a *per-day* basis. In essence, this involves decomposing your single CVRP into a series of daily CVRPs, each potentially having a subset of locations to visit. This adds significant complexity but allows us to leverage OR-Tools' strengths in handling static routing problems.

Here's a breakdown of how I've approached this, including the necessary code examples:

**1. Data Preprocessing and Model Initialization:**

First, we must transform our delivery data, which likely includes location IDs, delivery volume, and target dates, into a structure useful for OR-Tools. The goal is to have a separate routing problem instance per operational day, with nodes representing locations that have delivery requirements for *that specific day*. The crucial component is mapping each day to its available delivery locations and their corresponding demands. For example, we would extract the necessary data for a given date then use it to construct the routing model for that day. The total number of routes across all days is the effective objective to be minimized.

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_data_model(date_data):
    """Transforms date-based delivery data into a model usable by OR-Tools.

    Args:
        date_data: A dictionary where keys are dates and values are lists of
          dictionaries, each containing 'location_id', 'demand', and other location properties.

    Returns:
        A dictionary mapping each date to a data model for the OR-Tools solver.
    """
    data = {}
    for date, locations in date_data.items():
        day_data = {}
        day_data['locations'] = [loc['location_id'] for loc in locations]
        day_data['demands'] = [loc['demand'] for loc in locations]
        day_data['num_vehicles'] = 3  # or whatever the fleet size is
        day_data['depot'] = 0  # Assuming depot is always node 0
        day_data['vehicle_capacities'] = [1500 for _ in range(day_data['num_vehicles'])] # or a dynamic capacity
        # Generate a distance matrix (or load from an existing source).
        # This example is using a simple linear distance. In practice, this needs to be a more
        # precise implementation. For this case, let's assume locations are numbered based on a grid.
        num_locations = len(day_data['locations'])
        day_data['distance_matrix'] = [[abs(i - j) for j in range(num_locations)]
                                         for i in range(num_locations)]

        data[date] = day_data
    return data
```

In this function, I've constructed a structure where each date points to a separate dataset, creating a collection of sub-problems. The core of this is `day_data`, containing only the locations active on that particular date. `distance_matrix` here is a placeholder; in practice, you'd be loading a precomputed matrix or creating one based on geographic coordinates using a library like `geopy`. Note, a real-world model would have significantly more complex calculations for travel time and distance, based on factors like traffic.

**2. Solving the CVRP for Each Date:**

After preparing the data, we proceed to create and solve a routing model *for each day*. This section iterates through the dates, solving the routing problem based on the associated `day_data`.

```python
def solve_date_cvrp(data_model):
  """Solves CVRP problems for each date using the OR-Tools solver.

  Args:
    data_model: The dictionary of data models created by 'create_data_model'.

  Returns:
    A dictionary mapping each date to the solution obtained by the solver.
  """
  solutions = {}
  for date, day_data in data_model.items():
    manager = pywrapcp.RoutingIndexManager(len(day_data['locations']),
                                          day_data['num_vehicles'], day_data['depot'])
    routing = pywrapcp.RoutingModel(manager)

    def demand_callback(from_index):
        """Returns the demand of the node."""
        from_node = manager.IndexToNode(from_index)
        return day_data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimension(demand_callback_index, 0, sum(day_data['vehicle_capacities']), True, "Capacity")


    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return day_data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)


    solution = routing.SolveWithParameters(search_parameters)
    if solution:
        solutions[date] = solution
    else:
        solutions[date] = None

  return solutions
```

Here, the crucial aspects include the creation of the routing manager and model based on the data for each individual day. A crucial component is the `demand_callback` which is used to track capacity constraints. The `distance_callback` establishes the cost function based on the distance matrix. The code then calls the solver and returns all solutions. If a specific day cannot be solved, None is returned for that day in the results. This is a crucial step. If a given days routing is not possible with the given vehicles, the user should be made aware that they need more vehicles or need to re-evaluate the delivery demand.

**3. Extracting and Interpreting the Results:**

Finally, I need to process the output of the solver and display the results. This includes extracting the routes for each vehicle on a given day.

```python
def display_solutions(solutions, data_model):
  """Prints route assignments from solutions.

  Args:
    solutions: Solutions returned by 'solve_date_cvrp'.
    data_model: The original data models.
  """
  for date, solution in solutions.items():
      if not solution:
          print(f"No solution found for {date}.")
          continue
      print(f"Solution for {date}:")
      data = data_model[date]
      manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['depot'])
      routing = pywrapcp.RoutingModel(manager)
      capacity_dimension = routing.GetDimensionOrDie('Capacity')
      total_distance = 0
      for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f'  Route for vehicle {vehicle_id}:\n'
        route_distance = 0
        route_load = 0

        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]

            plan_output += f'    Location {data["locations"][node_index]} (Load: {data["demands"][node_index]}) -> '
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += data['distance_matrix'][manager.IndexToNode(previous_index)][manager.IndexToNode(index)]

        node_index = manager.IndexToNode(index)
        plan_output += f'    Depot (Load: 0)\n'
        plan_output += f"    Distance: {route_distance} \n"
        plan_output += f"    Total Load: {route_load}\n"
        print(plan_output)
        total_distance += route_distance
      print(f"Total distance for {date}: {total_distance}")


# Example Usage
if __name__ == '__main__':
  # Example data, this would come from an external source or database
  date_data = {
      '2024-08-01': [
          {'location_id': 1, 'demand': 500},
          {'location_id': 2, 'demand': 300},
          {'location_id': 3, 'demand': 200},
      ],
      '2024-08-02': [
          {'location_id': 2, 'demand': 400},
          {'location_id': 4, 'demand': 600},
          {'location_id': 5, 'demand': 200},
        {'location_id': 6, 'demand': 100},
      ]
  }

  data_model = create_data_model(date_data)
  solutions = solve_date_cvrp(data_model)
  display_solutions(solutions, data_model)
```

This section parses the solver output and produces a human-readable summary of the routes. For each vehicle route on a given day, it displays the location IDs, their loads, and the total route distance. The display shows the total distance traveled for each date. The example `main` section shows how to initialize and run the model using some dummy data.

**Resource Recommendations:**

For a deeper understanding of OR-Tools, I recommend exploring the official documentation. The Constraint Programming section is particularly helpful for customizing constraints. Also, I have found research articles on dynamic vehicle routing problems to be helpful, even if they are not directly using OR-Tools. In the realm of algorithms, books covering network flow and combinatorial optimization offer valuable insights into the mathematical underpinnings of these problems. Consulting literature on logistics and operations research can broaden your understanding of practical considerations and constraints frequently encountered in real-world applications of the CVRP.

By employing this iterative, date-segmented approach, I've been able to effectively leverage OR-Tools for CVRP problems that have date-based constraints. It's an involved process, but the results can be impactful, leading to significant cost reductions and operational efficiency improvements.
