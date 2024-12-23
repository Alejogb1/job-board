---
title: "How can constraints be added in DOCPLEX with Python?"
date: "2024-12-23"
id: "how-can-constraints-be-added-in-docplex-with-python"
---

 I recall a particularly thorny project back in '18, developing a supply chain optimization model where we had to intricately weave in various constraints to accurately reflect the operational realities. Doplex, for all its power, sometimes requires a nuanced approach to constraint implementation. It's not always as straightforward as it seems initially. Let’s unpack how to effectively add constraints using docplex with python, focusing on the mechanics and best practices I’ve accumulated over time.

Constraints, in essence, are the limitations or rules that govern the solution space of our optimization problem. In docplex, we express these using expressions that relate variables to constants, or each other. The core concept revolves around the `Model` object, where we define our variables, objective, and constraints. When I say "constraint," i'm referring to restrictions like "the total cost must not exceed a certain value," or "the number of items produced cannot be negative."

We primarily leverage three constraint types: equality constraints (`==`), less than or equal to (`<=`), and greater than or equal to (`>=`). Docplex handles these quite effectively when they are composed of linear expressions – expressions where variables are multiplied by constants and added together. This forms the backbone of most linear programming problems. In the instance mentioned earlier, we were modeling inventory levels and had to ensure no negative stock on hand, using a series of `>=` constraints.

Now, let’s get into the practical side with some code examples.

**Example 1: Simple Capacity Constraint**

Suppose we’re modeling production at a factory with a limited machine capacity. We have two products, 'A' and 'B', and we want to ensure the total machine time used doesn’t exceed a weekly limit.

```python
from docplex.mp.model import Model

# Create a new model
mdl = Model(name='factory_capacity')

# Define decision variables: the quantity of each product to produce
prod_A = mdl.integer_var(name='production_A')
prod_B = mdl.integer_var(name='production_B')

# Define constants
time_per_A = 2 # hours per unit of product A
time_per_B = 3 # hours per unit of product B
total_time_limit = 100 # hours available

# Add the capacity constraint
mdl.add_constraint(time_per_A * prod_A + time_per_B * prod_B <= total_time_limit, ctname='capacity_limit')

# Define objective to maximize total production (for demonstration)
mdl.maximize(prod_A + prod_B)

# Solve the model
solution = mdl.solve()

# Print the results
if solution:
    print(f"Optimal production of A: {solution.get_value(prod_A)}")
    print(f"Optimal production of B: {solution.get_value(prod_B)}")
    print(f"Total machine time used: {time_per_A * solution.get_value(prod_A) + time_per_B * solution.get_value(prod_B)}")
else:
    print("No solution found.")

```

In this snippet, we define our decision variables (`prod_A`, `prod_B`), specify the constraint expressing that total time utilized cannot exceed the `total_time_limit`, and then we define a trivial objective and solve the model. The core part here is how we construct the constraint via the `add_constraint` method, taking a linear expression and a name for clarity. The name 'capacity_limit' can be useful in complex models when retrieving specific constraints, something I found crucial when debugging issues in larger-scale projects.

**Example 2: Demand Constraints with Conditional Logic**

Now let’s add a layer of complexity. Assume our product B can be sold in two distinct markets. Market 1 requires a minimum of 10 units and market 2 requires minimum of 15 units for product B to be produced. We’ll model this using indicator variables. In practice, implementing these correctly often took me multiple attempts. You must get the indexing and the logical operation correct or you might find your solution space is completely off.

```python
from docplex.mp.model import Model

# Create model
mdl = Model(name='demand_constraint')

# Decision variables
prod_A = mdl.integer_var(name='prod_A', lb=0)  # Lower bound
prod_B = mdl.integer_var(name='prod_B', lb=0) # Lower bound

# Binary indicator variables for markets
market1_active = mdl.binary_var(name='market1_active')
market2_active = mdl.binary_var(name='market2_active')

# Demand for product B at each market
min_demand_market1 = 10
min_demand_market2 = 15

# Constraints
mdl.add(prod_B >= min_demand_market1 * market1_active, ctname="market1_min_demand")
mdl.add(prod_B >= min_demand_market2 * market2_active, ctname="market2_min_demand")
mdl.add(prod_B <= 1000*(market1_active + market2_active), ctname="market_sum_limit")

# Example objective (for demonstration)
mdl.maximize(prod_A + prod_B)

# Solve the model
solution = mdl.solve()

# Print the solution
if solution:
    print(f"Optimal production of A: {solution.get_value(prod_A)}")
    print(f"Optimal production of B: {solution.get_value(prod_B)}")
    print(f"Market 1 active: {solution.get_value(market1_active)}")
    print(f"Market 2 active: {solution.get_value(market2_active)}")
else:
    print("No solution found.")

```

Here we introduce binary decision variables (`market1_active` and `market2_active`). When one of the markets is active, it ensures a minimum demand is met. If neither market is active the total production of `prod_B` must be zero, effectively modeling a conditional constraint using binary variables. The third constraint ensures that if both markets are *not* active then no production of `prod_B` is allowed. In my experience, I've noticed that indicator variable usage can often become tricky very quickly, so keeping a clean and concise approach is crucial.

**Example 3: Time-Indexed Constraints**

Let’s consider a scenario with a temporal dimension: our production must satisfy time-dependent demand over 4 weeks. Each week has a separate demand requirement, and we must meet all of these demands. This one gets pretty close to the kind of complexity I faced when working on inventory management systems, and it highlights the power of using loops in docplex.

```python
from docplex.mp.model import Model

# Create a new model
mdl = Model(name='time_dependent_demand')

# Define time horizon
num_weeks = 4

# Define demand for each week
weekly_demand = [50, 60, 70, 80]

# Decision variables for production each week
prod_weekly = [mdl.integer_var(lb=0, name=f'prod_week_{i+1}') for i in range(num_weeks)]

# Inventory for each week
inventory_weekly = [mdl.integer_var(lb=0, name=f'inventory_week_{i+1}') for i in range(num_weeks+1)]
inventory_weekly[0] = 0

# Capacity for production
capacity_per_week = 100

# Constraints
for week in range(num_weeks):
    mdl.add_constraint(prod_weekly[week] <= capacity_per_week, ctname=f'capacity_week_{week+1}')
    mdl.add_constraint(inventory_weekly[week+1] == inventory_weekly[week] + prod_weekly[week] - weekly_demand[week], ctname=f'inventory_balance_{week+1}')
    mdl.add_constraint(inventory_weekly[week+1] >=0, ctname=f'inventory_non_negative_{week+1}')

# Example Objective function (for demonstration)
mdl.minimize(sum(prod_weekly))


# Solve the model
solution = mdl.solve()

# Print the results
if solution:
    for week in range(num_weeks):
        print(f"Production in week {week+1}: {solution.get_value(prod_weekly[week])}")
    for week in range(num_weeks+1):
        print(f"Inventory at week {week}: {solution.get_value(inventory_weekly[week])}")
else:
    print("No solution found.")

```

Here we introduce a list comprehension to generate variables for each week. We use a `for` loop to generate time-indexed constraints such as production capacity, inventory balance, and non-negative inventory. This shows how we can elegantly handle time-series or other data dimensions within docplex. It's a pattern I often used during my project to build more complex models from simpler components.

**Recommendations**

For a deeper dive into the mathematical underpinnings of these concepts, I would highly recommend “Linear Programming: Foundations and Extensions” by Robert J. Vanderbei. It provides a thorough mathematical framework of optimization. For more on the specifics of modeling optimization problems using python, the documentation for `docplex` itself is your best resource alongside the examples in the `cplex` installation folder. I would also suggest checking "Optimization Methods for a Complex World" by James E. Smith, as it provides a good framework to apply these techniques to problems in the real world.

These examples should give a solid foundation for adding and handling constraints in docplex. Remember, when dealing with complex problems, start with a simple model, incrementally add complexities, and rigorously test each step. That’s what I learned over a few years working with these tools. Keep your expressions clear, your constraints well-defined, and your model will become much more manageable. Good luck!
