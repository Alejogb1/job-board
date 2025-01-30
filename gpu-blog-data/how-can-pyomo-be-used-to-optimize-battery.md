---
title: "How can Pyomo be used to optimize battery charging/discharging schedules?"
date: "2025-01-30"
id: "how-can-pyomo-be-used-to-optimize-battery"
---
Pyomo's strength lies in its ability to formulate and solve optimization problems using a declarative modeling approach.  My experience with large-scale energy management systems has shown that this declarative style is particularly beneficial when tackling the complexities inherent in optimizing battery charging and discharging schedules.  The key is leveraging Pyomo's support for mixed-integer linear programming (MILP) to effectively model the discrete nature of charging states and the continuous nature of power flows.

**1.  A Clear Explanation of the Optimization Process**

Optimizing battery charging/discharging schedules fundamentally involves determining the optimal power flow into and out of the battery at each time step within a given planning horizon. This optimization considers various constraints, including:

* **Battery State of Charge (SOC):**  The SOC must remain within predefined minimum and maximum limits at all times.  This prevents deep discharges which can damage the battery and ensures sufficient charge is available when needed.
* **Charging/Discharging Rates:**  Batteries have limitations on how quickly they can charge and discharge.  These limits must be respected to avoid exceeding the battery's thermal and electrical capacity.
* **Power Availability:** The amount of power available for charging may be limited by the source (e.g., solar panel output, grid connection). Similarly, the amount of power that can be discharged might be limited by downstream demand or grid regulations.
* **Cycle Life:**  Frequent and deep charge/discharge cycles can degrade the battery's lifespan.  The optimization can incorporate penalties for excessive cycling to extend the battery's useful life.
* **Cost:**  Electricity prices often fluctuate throughout the day. The optimization aims to minimize charging costs by favoring periods with lower electricity prices.  Similarly, it could maximize revenue by discharging during peak demand periods if the battery is connected to a grid-tied system.


The optimization problem is formulated as an MILP problem.  We define decision variables representing the power flow into and out of the battery at each time step. The objective function minimizes the overall cost (or maximizes profit) while satisfying the constraints outlined above.  Pyomo's modeling capabilities allow for a structured and readable representation of this problem, significantly improving the maintainability and scalability of the model.  My past work involved handling hundreds of batteries across diverse locations using this very approach.



**2. Code Examples with Commentary**

These examples demonstrate core aspects of Pyomo's application to this problem.  They are simplified for illustrative purposes but highlight the key elements.


**Example 1:  Basic Charging/Discharging Optimization**

```python
from pyomo.environ import *

model = ConcreteModel()

# Time horizon (in hours)
model.T = RangeSet(24)

# Battery capacity (kWh)
battery_capacity = 10

# Maximum charging/discharging rate (kW)
max_rate = 5

# Decision variables: power flow into (+) and out of (-) battery (kW)
model.P = Var(model.T, domain=Reals, bounds=(-max_rate, max_rate))

# Battery state of charge (kWh)
model.SOC = Var(model.T, domain=NonNegativeReals, bounds=(0, battery_capacity))

# Objective: minimize total energy cost (assuming a constant price for simplicity)
model.obj = Objective(expr=sum(model.P[t] for t in model.T), sense=minimize)

# Constraints
model.SOC_init = Constraint(expr=model.SOC[1] == 5) # Initial SOC
model.SOC_balance = Constraint(model.T, rule=lambda model, t: model.SOC[t+1] == model.SOC[t] + model.P[t] if t < 24 else Constraint.Skip)
model.rate_limit = Constraint(model.T, rule=lambda model, t: -max_rate <= model.P[t] <= max_rate)
model.SOC_limit = Constraint(model.T, rule=lambda model, t: 0 <= model.SOC[t] <= battery_capacity)

# Solve the model
solver = SolverFactory('glpk')
results = solver.solve(model)

# Print results (example)
print(results.solver.termination_condition)
for t in model.T:
    print(f"Time {t}: SOC = {model.SOC[t].value:.2f} kWh, P = {model.P[t].value:.2f} kW")
```

This example uses a simplified cost function and assumes a constant electricity price.  The `SOC_balance` constraint tracks the SOC over time, while `rate_limit` and `SOC_limit` enforce the charging/discharging rate and SOC bounds respectively.  `glpk` is used as a solver; other solvers like CBC or CPLEX can also be used depending on license availability and problem size.


**Example 2: Incorporating Time-Varying Electricity Prices**

```python
from pyomo.environ import *

# ... (previous code as before) ...

# Electricity price at each time step ($/kWh)
electricity_prices = [0.10, 0.12, 0.15, 0.18, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10, 0.08, 0.07, 0.06, 0.07, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.18, 0.15, 0.12] #Example prices


# Objective: minimize total energy cost
model.obj = Objective(expr=sum(model.P[t] * electricity_prices[t-1] for t in model.T), sense=minimize)

# ... (rest of the code remains similar) ...

```

This expands on the previous example by introducing time-varying electricity prices, making the objective function more realistic.  The cost is now calculated by multiplying the power flow at each time step with the corresponding electricity price.


**Example 3:  Handling Renewable Energy Source Integration**

```python
from pyomo.environ import *

# ... (previous code as before) ...

# Renewable energy generation at each time step (kW)
renewable_generation = [0, 2, 5, 8, 10, 12, 10, 8, 5, 3, 1, 0, 0, 0, 0, 0, 1, 3, 5, 7, 9, 10, 8, 5] #Example generation


# Constraint: charging power cannot exceed renewable generation + grid power
model.grid_power = Var(model.T, domain=NonNegativeReals) # Added grid power variable
model.renewable_constraint = Constraint(model.T, rule=lambda model, t: model.P[t] <= model.grid_power[t] + renewable_generation[t-1] if model.P[t] >= 0 else Constraint.Skip)

# ... (rest of the code remains similar) ...
```

This example demonstrates incorporating renewable energy sources into the optimization.  The `renewable_constraint` ensures that charging power does not exceed the sum of available renewable energy and grid power.  The model now also includes a decision variable `grid_power` representing the amount of power drawn from the grid for charging.


**3. Resource Recommendations**

For a deeper understanding of Pyomo, I recommend consulting the official Pyomo documentation and tutorials.  Furthermore, textbooks on optimization modeling and linear programming will provide valuable background knowledge.  Finally, exploring case studies on energy system optimization, readily available in academic journals, will offer practical insights into real-world applications.  Familiarizing yourself with different solvers’ capabilities and limitations is also crucial for efficient model solving.
