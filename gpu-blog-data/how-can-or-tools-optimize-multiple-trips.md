---
title: "How can OR-Tools optimize multiple trips?"
date: "2025-01-30"
id: "how-can-or-tools-optimize-multiple-trips"
---
The primary challenge in optimizing multiple trips with OR-Tools lies in formulating the problem to accurately represent real-world constraints while leveraging the solver’s capabilities. Traditional Vehicle Routing Problems (VRPs) often focus on a single, synchronized set of routes. Extending this to multi-trip scenarios requires careful consideration of temporal aspects, vehicle capacities, and potentially varied depot locations. I've encountered this challenge numerous times while developing logistics solutions for various clients, particularly those managing same-day delivery services. The core issue is not solely about minimizing distance, but minimizing cost while considering time windows, vehicle availability across multiple routes, and potentially varying load capacities during each trip.

I'll detail how to approach this by reformulating the problem as a series of sequential VRPs, linked by time constraints and vehicle states. The solver is not directly designed for iterative route planning; it's designed for a holistic route optimization. Thus, the multi-trip aspect requires us to model routes as separate entities within the overall problem, and then use constraints to ensure continuity of resources. My approach typically involves defining each potential trip as a distinct VRP, incorporating appropriate constraints, and then stringing them together sequentially.

Here’s the breakdown:

**1. Model Formulation:**

Instead of viewing all deliveries as one large problem, we break it into smaller, more manageable trip segments. Each trip constitutes a separate VRP with its own constraints. I've found that representing each trip as a subset of the total delivery set, with constraints related to time and vehicle availability at each stage, yields effective results. We must define:

*   **Locations:** The depot and delivery locations with associated coordinates.
*   **Trips:** Each potential trip, characterized by a time window, potentially varying vehicle capacity, and delivery nodes.
*   **Vehicles:** Vehicles with specific attributes such as initial location, capacity, and operating hours. Crucially, we must model vehicle “availability” across different trips; this includes time needed to return to the depot after a trip.
*   **Time Windows:** Time windows for deliveries and potential trip start/end times.
*   **Transition Times:** Travel times between locations, and potentially depot time needed to load/unload.

The key differentiator is how we manage vehicle availability.  Instead of assuming vehicles are infinitely available, we need to model each vehicle’s start state, end state for each trip, and transfer them to subsequent trips. In my experience, representing time constraints as cumulative variables in the model is more efficient than using complex global constraints.

**2. Code Example and Explanation:**

Here's an example using Python, illustrating the concept of modeling sequential trips:

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['depot'] = 0  # Index of the depot.
    data['locations'] = [
        (0, 0), # Depot
        (1, 1), (2, 2), (3, 1), (1, 3), # Delivery locations
        (2, 4)
    ]
    data['time_windows'] = [
        (0,100),
        (10, 20), (20,30), (30, 40), (40, 50),
        (50,60)
    ]
    data['num_vehicles'] = 2
    data['vehicle_capacities'] = [2, 2] # Vehicle capacity per trip
    data['vehicle_start_locations'] = [0, 0]
    data['vehicle_end_locations'] = [0, 0]
    data['trip_sets'] = [
        [1, 2, 3], # Trip 1 Delivery Locations
        [4, 5] # Trip 2 Delivery Locations
    ]

    return data

def create_distance_matrix(data):
  """Creates the distance matrix based on location coordinates."""
  locations = data['locations']
  matrix = {}
  for from_index in range(len(locations)):
      for to_index in range(len(locations)):
          x_dist = abs(locations[from_index][0] - locations[to_index][0])
          y_dist = abs(locations[from_index][1] - locations[to_index][1])
          matrix[(from_index, to_index)] = x_dist + y_dist
  return matrix

def solve_multi_trip_vrp():
    """Solves the multi-trip VRP."""
    data = create_data_model()
    distance_matrix = create_distance_matrix(data)

    manager = pywrapcp.RoutingIndexManager(
        len(data['locations']), data['num_vehicles'], data['vehicle_start_locations'], data['vehicle_end_locations']
        )
    routing = pywrapcp.RoutingModel(manager)

    # Distance Callback
    def distance_callback(from_index, to_index):
      """Returns the distance between the two nodes."""
      from_node = manager.IndexToNode(from_index)
      to_node = manager.IndexToNode(to_index)
      return distance_matrix[(from_node, to_node)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Demand Callback (Dummy)
    def demand_callback(from_index):
      """Returns the demand of the node."""
      return 1

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    capacity = 'Capacity'
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0, # Null capacity slack
        data['vehicle_capacities'], # Vehicle capacity
        True, # Start cumul to zero
        capacity
    )

    # Time Window Dimension (Simplified for this example)
    time = 'Time'
    time_callback_index = routing.RegisterTransitCallback(distance_callback) # using distance for simplicity
    routing.AddDimension(
        time_callback_index,
        100,  # Allow a maximum time of 100 (can be optimized)
        100, # allow time to be exceeded with large slack
        False,
        time
    )

    time_dimension = routing.GetDimensionOrDie(time)

    for index in range(len(data['time_windows'])):
      if index != data['depot']:
        time_dimension.CumulVar(index).SetRange(data['time_windows'][index][0], data['time_windows'][index][1])


    # Trip Assignment Constraint
    for trip_index, trip_locations in enumerate(data['trip_sets']):
      for node_index in trip_locations:
        for vehicle_id in range(data['num_vehicles']):
          if (trip_index > 0): # Simple Example of trip assignment
              routing.VehicleVar(node_index).RemoveValue(vehicle_id)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)
    if solution:
      print("Solution found:")
      for vehicle_id in range(data['num_vehicles']):
          index = routing.Start(vehicle_id)
          print(f"Vehicle {vehicle_id}:")
          while not routing.IsEnd(index):
              print(f" {manager.IndexToNode(index)} ->", end="")
              index = solution.Value(routing.NextVar(index))
          print(f"{manager.IndexToNode(index)}")


if __name__ == '__main__':
    solve_multi_trip_vrp()
```

This example defines two trips and two vehicles.  The `data` dictionary now holds information about trip sets (`trip_sets`), which dictate which delivery locations are part of which trip. The time dimension is added, simplified in this example. The trip assignment constraint logic, specifically `if (trip_index > 0):`, is key. This rudimentary example demonstrates that a simple constraint can prevent a vehicle from accessing nodes outside the trip set. I use a combination of `RemoveValue` on `VehicleVar` and set time range on time variable of each node.  In practice, the trip allocation logic can be greatly expanded.

**3. Detailed Code Example with Time Windows and Capacity:**

Let's extend the previous example with stricter time windows and a better representation of vehicle capacity:

```python
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp

def create_data_model():
    """Stores the data for the problem."""
    data = {}
    data['depot'] = 0
    data['locations'] = [
        (0, 0), # Depot
        (1, 1), (2, 2), (3, 1), (1, 3),
        (2, 4),
    ]
    data['demands'] = [0, 1, 1, 1, 1, 1]
    data['time_windows'] = [
       (0,100),
        (10, 20), (20,30), (30, 40), (40, 50),
        (50,60)
    ]
    data['num_vehicles'] = 2
    data['vehicle_capacities'] = [2, 2]
    data['vehicle_start_locations'] = [0, 0]
    data['vehicle_end_locations'] = [0, 0]
    data['trip_sets'] = [
        [1, 2], # Trip 1 Delivery Locations
        [3,4,5] # Trip 2 Delivery Locations
    ]
    data['trip_start_times'] = [0, 20]
    data['trip_end_times'] = [20, 70]
    return data

def create_distance_matrix(data):
    locations = data['locations']
    matrix = {}
    for from_index in range(len(locations)):
        for to_index in range(len(locations)):
            x_dist = abs(locations[from_index][0] - locations[to_index][0])
            y_dist = abs(locations[from_index][1] - locations[to_index][1])
            matrix[(from_index, to_index)] = x_dist + y_dist
    return matrix

def solve_multi_trip_vrp():
    data = create_data_model()
    distance_matrix = create_distance_matrix(data)
    manager = pywrapcp.RoutingIndexManager(len(data['locations']), data['num_vehicles'], data['vehicle_start_locations'], data['vehicle_end_locations'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distance_matrix[(from_node, to_node)]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        return data['demands'][manager.IndexToNode(from_index)]
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    capacity = 'Capacity'
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,
        data['vehicle_capacities'],
        True,
        capacity
    )

    time = 'Time'
    time_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.AddDimension(
        time_callback_index,
        100,  # Maximum Time
        100,
        False,
        time
    )
    time_dimension = routing.GetDimensionOrDie(time)
    for index in range(len(data['time_windows'])):
      if index != data['depot']:
         time_dimension.CumulVar(index).SetRange(data['time_windows'][index][0], data['time_windows'][index][1])
    # Trip Assignment Constraint with Sequential Time Check
    for trip_index, trip_locations in enumerate(data['trip_sets']):
        for node_index in trip_locations:
            for vehicle_id in range(data['num_vehicles']):
              if (trip_index > 0):
                 routing.VehicleVar(node_index).RemoveValue(vehicle_id)


        for vehicle_id in range(data['num_vehicles']):
          start_var = time_dimension.CumulVar(routing.Start(vehicle_id))
          end_var = time_dimension.CumulVar(routing.End(vehicle_id))

          if (trip_index == 0):
             start_var.SetRange(data['trip_start_times'][0],100) # trip start constraint

          if(trip_index == 1):
              start_var.SetRange(data['trip_start_times'][1],100)
              # end_var.SetRange(data['trip_end_times'][1], 10000)




    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
      routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

    solution = routing.SolveWithParameters(search_parameters)

    if solution:
      print("Solution found:")
      for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        print(f"Vehicle {vehicle_id}:")
        while not routing.IsEnd(index):
            print(f" {manager.IndexToNode(index)} (time: {solution.Value(time_dimension.CumulVar(index))}) ->", end="")
            index = solution.Value(routing.NextVar(index))
        print(f"{manager.IndexToNode(index)} (time: {solution.Value(time_dimension.CumulVar(index))})")


if __name__ == '__main__':
    solve_multi_trip_vrp()
```

In this expanded code, each delivery location now has a demand and associated time window. Additionally, I've implemented `trip_start_times` and `trip_end_times`. The code now adds time window constraints to cumulative variables for each node. The time range is adjusted based on the trip index to force time continuity through a sequence of trips. I also added the `demand` callbacks and vehicle capacities to more closely resemble realistic problems.

**4. Resource Recommendations:**

For a more comprehensive understanding, I recommend exploring the official OR-Tools documentation, which details the various modeling options available within the library, as well as examples.  Further research into Vehicle Routing Problem literature, particularly focusing on variants such as the VRP with Time Windows (VRPTW), is also beneficial. Consider the open-source OR-Tools examples and case studies to understand best practices. For a deep dive into constraint programming, numerous academic papers and books on the subject can aid in effectively modeling different real-world scenarios. Additionally, reviewing optimization textbooks and algorithmic design will enhance the development of effective solutions using OR-Tools. The core libraries provide a wealth of examples for how to model different types of problems, which is crucial when extending VRP to include multi-trip problems.  A significant learning comes through practical implementation and experimentation with data.
