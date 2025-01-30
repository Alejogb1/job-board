---
title: "How can OR-Tools be used for facility location?"
date: "2025-01-30"
id: "how-can-or-tools-be-used-for-facility-location"
---
Facility location problems, a core component of operations research, often involve complex optimization challenges. My experience working on logistics optimization for a large-scale e-commerce fulfillment network highlighted the significant advantages of using Google OR-Tools' capabilities in solving such problems.  Specifically, OR-Tools' flexibility in handling various constraints and its robust solver implementations proved invaluable in achieving near-optimal solutions within acceptable computational timeframes, even with datasets encompassing hundreds of potential facility locations and thousands of customer demands.

The core of applying OR-Tools to facility location lies in formulating the problem as a mathematical model, typically a mixed-integer programming (MIP) problem.  This involves defining decision variables representing the location of facilities, the assignment of customers to facilities, and potentially other relevant factors like facility capacities and transportation costs. The objective function, which OR-Tools aims to optimize, usually minimizes the total cost, encompassing the cost of establishing facilities and the cost of transportation between facilities and customers.

This model can take several forms depending on the specific problem characteristics.  Common variations include the p-median problem (locating *p* facilities to minimize the total distance to customers), the uncapacitated facility location problem (UFLP), and capacitated facility location problems (CFLP), each with variations based on distance metrics, fixed costs, and capacity constraints.  The choice of model significantly impacts the complexity and solution time.  My experience has shown that carefully defining the problem and selecting an appropriate model is crucial for effective OR-Tools implementation.  Overly simplistic models might not accurately capture the real-world nuances, while overly complex models can lead to intractable computational times.

Let's illustrate this with three code examples using Python and OR-Tools' CP-SAT solver, suitable for smaller to medium-sized instances of the problem.  Larger instances often benefit from the MIP solver, which I’ve extensively used for the aforementioned e-commerce application.

**Example 1: Uncapacitated Facility Location Problem (UFLP)**

This example demonstrates a basic UFLP where the objective is to minimize the total cost of opening facilities and assigning customers to them.

```python
from ortools.sat.python import cp_model

# Data
num_facilities = 3
num_customers = 5
fixed_costs = [10, 15, 20]  # Fixed cost for each facility
distances = [
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15]
]

# Model
model = cp_model.CpModel()

# Decision variables
x = {}  # x[i, j] = 1 if customer i is assigned to facility j, 0 otherwise
y = {}  # y[j] = 1 if facility j is open, 0 otherwise
for i in range(num_customers):
    for j in range(num_facilities):
        x[i, j] = model.NewBoolVar(f'x_{i}_{j}')
for j in range(num_facilities):
    y[j] = model.NewBoolVar(f'y_{j}')

# Constraints
# Each customer must be assigned to exactly one facility
for i in range(num_customers):
    model.AddExactlyOne(x[i, j] for j in range(num_facilities))

# If a customer is assigned to a facility, the facility must be open
for i in range(num_customers):
    for j in range(num_facilities):
        model.AddImplication(x[i, j], y[j])

# Objective function
objective = sum(fixed_costs[j] * y[j] for j in range(num_facilities)) + \
            sum(distances[j][i] * x[i, j] for i in range(num_customers) for j in range(num_facilities))
model.Minimize(objective)

# Solve
solver = cp_model.CpSolver()
status = solver.Solve(model)

# Print solution
if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Total cost:", solver.ObjectiveValue())
    for j in range(num_facilities):
        if solver.Value(y[j]) == 1:
            print(f"Facility {j + 1} is open")
            for i in range(num_customers):
                if solver.Value(x[i, j]) == 1:
                    print(f"  Customer {i + 1} is assigned to facility {j + 1}")
else:
    print("No solution found.")

```

This code defines the UFLP, creates the model, adds constraints, sets the objective function, and solves it using the CP-SAT solver.  The output shows the optimal facility locations and customer assignments.  Note that for larger instances, the CP-SAT solver might struggle; the MIP solver would be a more appropriate choice.


**Example 2: Incorporating Capacity Constraints (CFLP)**

This expands the previous example by adding capacity constraints to each facility.

```python
# ... (previous code) ...

# Add capacity constraints
capacities = [5, 7, 10] # Capacity of each facility
demands = [2, 3, 1, 4, 2] # Demand of each customer

for j in range(num_facilities):
    model.AddLessOrEqual(sum(demands[i] * x[i, j] for i in range(num_customers)), capacities[j])

# ... (rest of the code) ...
```

Here, we introduce `capacities` and `demands` lists to represent the facility capacities and customer demands.  The added constraint ensures that the total demand assigned to each facility does not exceed its capacity.

**Example 3:  Using a Different Distance Metric**

This example demonstrates using a different distance metric – Euclidean distance – instead of the predefined distance matrix.

```python
import math
# ... (previous code, excluding the distances matrix) ...


coordinates_facilities = [(1,2), (4,5), (7,8)]
coordinates_customers = [(2,3), (5,6), (1,1), (8,7), (3,4)]

# Calculate Euclidean distances
distances = [[math.sqrt((coordinates_facilities[j][0] - coordinates_customers[i][0])**2 + (coordinates_facilities[j][1] - coordinates_customers[i][1])**2) for j in range(num_facilities)] for i in range(num_customers)]

# ... (rest of the code remains the same) ...
```

This example computes Euclidean distances between facilities and customers dynamically, offering more flexibility in modeling real-world scenarios.

These examples highlight the versatility of OR-Tools in tackling facility location problems.  Remember,  problem formulation, including the choice of model and solver, significantly impacts performance.  For larger-scale problems, consider using the MIP solver, exploring different solution strategies like heuristics and metaheuristics, and potentially leveraging parallel processing.


**Resource Recommendations:**

Consult the official OR-Tools documentation for detailed explanations of the available solvers and their functionalities. Explore textbooks on operations research and optimization to gain a solid theoretical foundation in formulating and solving facility location problems.  Consider studying advanced techniques like Benders decomposition for handling particularly challenging instances.  Familiarize yourself with different MIP solver parameters to fine-tune performance.  Finally, investigate papers on advanced facility location problems and solution methodologies to deepen your understanding of the field.  These resources, combined with practical experience, will empower you to effectively utilize OR-Tools for a wide range of facility location challenges.
