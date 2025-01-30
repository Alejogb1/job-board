---
title: "How can Pyomo be used to optimize a thermal storage system coupled with a boiler and heat pump?"
date: "2025-01-30"
id: "how-can-pyomo-be-used-to-optimize-a"
---
My experience in industrial process optimization has frequently involved integrating disparate energy systems. A common challenge is managing thermal loads using a mix of energy sources, which requires careful optimization to minimize costs and maintain system performance. In this case, controlling a thermal storage system coupled with a boiler and heat pump presents a compelling optimization problem readily addressed by Pyomo.

Pyomo, a Python-based optimization modeling language, allows us to define the system components and their interactions mathematically. Specifically, we can formulate a mixed-integer linear program (MILP) or a nonlinear program (NLP), depending on the complexity and relationships we wish to represent. The process primarily entails three stages: 1) defining model variables that represent operational parameters, 2) formulating the objective function that quantifies our target (e.g., minimizing total cost), and 3) describing system constraints through mathematical relationships.

The core components of this system include:

*   **Thermal Storage:** A tank or similar device that stores thermal energy. Its state is represented by the stored energy level (e.g., in kWh) or its temperature, and charging/discharging rates are defined by power values (e.g., kW).
*   **Boiler:** A device that generates thermal energy using fuel. The model considers its power output (kW) and, importantly, its fuel consumption rate, which is often linearly related to the power output with an efficiency factor.
*   **Heat Pump:** An electrically driven device that transfers heat from a low-temperature source to a higher one. The model uses its heating or cooling capacity (kW) and its Coefficient of Performance (COP), which quantifies its energy efficiency.

To illustrate, let's construct a simplified model using Pyomo. The system will operate over several discrete time steps, and we seek to minimize the overall cost. This initial formulation will use linear constraints, which is often an initial good approach. Note that a key challenge with these systems is often the non-linear dependence of heat pump efficiency on the temperatures; for the sake of simplicity, we'll use a constant COP to keep our model linear. I will present the model piece-by-piece.

First, we define the Pyomo model and its sets and parameters:

```python
from pyomo.environ import *

# Define model
model = ConcreteModel()

# Set of time steps
model.T = RangeSet(0, 23)  # Assuming 24 hourly time steps

# Parameters
model.load_demand = Param(model.T, initialize={
    0: 100, 1: 90, 2: 80, 3: 70, 4: 60, 5: 50, 6: 40, 7: 40, 8: 50, 9: 60, 10: 70, 11: 80,
    12: 90, 13: 100, 14: 110, 15: 120, 16: 130, 17: 140, 18: 150, 19: 140, 20: 130, 21: 120, 22: 110, 23: 100
}) # Hourly heat demand in kW

model.boiler_efficiency = Param(initialize=0.85)  # Boiler efficiency
model.heat_pump_cop = Param(initialize=3.0)  # Heat pump COP
model.fuel_cost = Param(initialize=0.05) # cost per kWh of fuel
model.electricity_cost = Param(initialize=0.10) # cost per kWh of electricity
model.storage_capacity = Param(initialize=500)  # Maximum storage capacity (kWh)
model.storage_initial_level = Param(initialize=250) # Initial storage level (kWh)

model.storage_discharge_max_rate = Param(initialize=100) # Maximum discharge rate of the storage in kW
model.storage_charge_max_rate = Param(initialize=100) # Maximum charge rate of the storage in kW
```
Here, we initialize the model, the time step set, demand profile, efficiencies, cost parameters, and constraints like storage capacity. This is a relatively generic instantiation of key system parameters which will drive the optimization.

Next, I define the decision variables, which are the model’s choices. These variables include the boiler power output, the heat pump power, storage charge/discharge, and storage level:

```python
# Decision Variables
model.boiler_power = Var(model.T, within=NonNegativeReals)    # Boiler output (kW)
model.heat_pump_power = Var(model.T, within=NonNegativeReals) # Heat pump output (kW)
model.storage_charge = Var(model.T, within=NonNegativeReals)   # Storage charge rate (kW)
model.storage_discharge = Var(model.T, within=NonNegativeReals) # Storage discharge rate (kW)
model.storage_level = Var(model.T, within=NonNegativeReals)    # Storage level (kWh)

```

These variables allow the model to decide how to operate the system at each hour. The use of `NonNegativeReals` ensures that they are physically meaningful.

Next, we define the objective function (in this case, to minimize total cost) and model constraints. The constraints will define the system's operational boundaries and make sure the demand is satisfied:

```python
# Objective function: minimize the total operational cost
def obj_rule(model):
    return sum(
        model.boiler_power[t] / model.boiler_efficiency * model.fuel_cost
        + model.heat_pump_power[t] / model.heat_pump_cop * model.electricity_cost
    for t in model.T)

model.obj = Objective(rule=obj_rule, sense=minimize)

# Constraints

# Energy balance constraint: demand = boiler + heat_pump + storage discharge - storage charge
def energy_balance_rule(model, t):
  if t == 0:
    return Constraint.Skip
  return model.load_demand[t] == model.boiler_power[t] + model.heat_pump_power[t] + model.storage_discharge[t] - model.storage_charge[t]
model.energy_balance = Constraint(model.T, rule=energy_balance_rule)

# Storage level constraint
def storage_level_rule(model, t):
  if t == 0:
      return model.storage_level[t] == model.storage_initial_level
  return model.storage_level[t] == model.storage_level[t-1] + model.storage_charge[t] - model.storage_discharge[t]
model.storage_level_balance = Constraint(model.T, rule=storage_level_rule)

# Storage level max/min constraint
def storage_level_minmax_rule(model, t):
  return (0, model.storage_level[t], model.storage_capacity)
model.storage_level_minmax = Constraint(model.T, rule=storage_level_minmax_rule)

# Storage charge/discharge rate constraint
def storage_charge_max_rate_rule(model, t):
    return (0, model.storage_charge[t], model.storage_charge_max_rate)
model.storage_charge_max_rate_cons = Constraint(model.T, rule = storage_charge_max_rate_rule)

def storage_discharge_max_rate_rule(model, t):
    return (0, model.storage_discharge[t], model.storage_discharge_max_rate)
model.storage_discharge_max_rate_cons = Constraint(model.T, rule = storage_discharge_max_rate_rule)
```
The objective function calculates the total cost of operating both the boiler and the heat pump. The energy balance constraint guarantees that the total heat production from the boiler, heat pump, and storage, minus the charge, satisfies demand. The storage level constraint ensures that the storage level follows charging/discharging patterns and always stays within the capacity.

Finally, to solve the problem and print the results:

```python
# Solve the model
solver = SolverFactory('ipopt')  # Choose a solver, e.g., IPOPT
results = solver.solve(model)

# Print results
if results.solver.termination_condition == TerminationCondition.optimal:
    print("Solution found:")
    print("Total cost:", model.obj())
    for t in model.T:
      if t > 0:
          print(f'Time: {t}, Demand: {model.load_demand[t]}, Boiler: {model.boiler_power[t].value:.2f}, Heat pump: {model.heat_pump_power[t].value:.2f}, Charge: {model.storage_charge[t].value:.2f}, Discharge: {model.storage_discharge[t].value:.2f}, Storage Level: {model.storage_level[t].value:.2f}')

else:
    print("No optimal solution found.")
    print(results.solver.termination_condition)
```
This code solves the formulated optimization problem. The results are printed if an optimal solution was found, including the objective value and the individual values for each time step.

This example uses a basic linear model.  More complex behaviors can be added, including:

1.  **Nonlinear Heat Pump Efficiency:** The COP of a heat pump often depends on source and sink temperature. We would implement this by expressing the COP as a function of the storage and ambient temperature, introducing a non-linear constraint, moving from MILP to NLP.
2.  **Variable Electricity Costs:** The cost of electricity often varies with time. We could create a time-dependent parameter `electricity_cost` to reflect this, influencing the optimal dispatch pattern.
3.  **Storage Heat Losses:**  Storage systems always lose some heat to their surroundings. We could introduce a term proportional to the storage level or temperature to model these losses.

For those looking to expand on the basics, I recommend further exploring documentation of the Pyomo project; focusing on topics such as advanced constraint formulation and solver selection is often beneficial.  Furthermore, literature on energy system optimization, specifically on thermal storage management and the integration of different energy sources, provides crucial background and modeling insights.
