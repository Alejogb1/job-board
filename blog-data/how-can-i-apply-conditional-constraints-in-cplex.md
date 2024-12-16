---
title: "How can I apply conditional constraints in Cplex?"
date: "2024-12-16"
id: "how-can-i-apply-conditional-constraints-in-cplex"
---

Alright, let’s tackle conditional constraints in cplex. It's a topic that's tripped up many, and frankly, I've spent a good number of late nights debugging models because of it. There are multiple ways to skin this cat, and your specific approach depends largely on the nature of your conditional and the structure of your model. I’m drawing here from experiences building optimization models for supply chain logistics and resource allocation; places where seemingly straightforward rules often turn into a labyrinth of logical conditions.

Essentially, conditional constraints boil down to this: you want a constraint to only become active under certain conditions, usually depending on the values of other variables. The core issue is that cplex directly handles linear constraints. Logical if-then statements as they appear in programming languages aren’t directly supported, so we need to translate our conditional logic into equivalent linear forms. This transformation is often referred to as “big-m” modeling or using indicator variables. The key is to use binary variables and a very large constant (the 'big-m') to effectively switch constraints on and off.

Let’s dive into specific methods.

**Method 1: Indicator Constraints (the cleaner approach, where available)**

The most straightforward method, and my preferred route when possible, is using indicator constraints. This cplex feature directly expresses the dependency between a binary variable and a constraint. Basically, you’re telling cplex, "if this variable is 1, then this constraint holds." This avoids some of the clumsiness inherent in the big-m method. However, indicator constraints have limitations in which type of constraints can be used. They work better with linear constraints rather than more complex constraints.

Here's a conceptual example where a production run only starts if a specific setup is in place. Let's assume 'x[t]' is the amount produced at time 't', 'y[t]' is a binary setup variable (1 if a setup is active, 0 otherwise), and 'min_production' is a minimum production amount if there’s a setup.

```python
import cplex
from cplex.exceptions import CplexError

try:
  my_prob = cplex.Cplex()

  # --- Variables ---
  my_prob.variables.add(names = [f"x_{t}" for t in range(10)],
                        lb = [0]*10) #production variable
  my_prob.variables.add(names = [f"y_{t}" for t in range(10)],
                        types = [my_prob.variables.type.binary]*10) #setup indicator


  # --- Parameters ---
  min_production = 100

  #--- Constraints ---
  for t in range(10):
     # Indicator constraint: if y[t]=1, then x[t] >= min_production
     my_prob.indicator_constraints.add(
      indvar = f"y_{t}",
      complemented=0, #meaning if y[t]==1
      lin_expr = cplex.SparsePair(ind = [f"x_{t}"], val = [1]),
      sense='G', #greater or equal
      rhs=min_production)

     #another constraint just for the example
     my_prob.linear_constraints.add(
       lin_expr = cplex.SparsePair(ind = [f"x_{t}"], val=[1]),
       sense = "L", # less than or equal
       rhs = 5000
    )


  # Objective (just an example)
  my_prob.objective.set_sense(my_prob.objective.sense.minimize)
  my_prob.objective.set_linear([(f"x_{t}",1) for t in range(10)])
  my_prob.solve()


  print("Solution status = ", my_prob.solution.get_status_string())
  print("Solution value  = ", my_prob.solution.get_objective_value())

  for t in range(10):
    print(f"x_{t} = {my_prob.solution.get_values(f'x_{t}')}, y_{t} = {my_prob.solution.get_values(f'y_{t}')}")


except CplexError as exc:
  print(f"Cplex encountered an error: {exc}")
```
In this example, the `indicator_constraints.add()` method creates an indicator constraint. If `y[t]` is 1, then `x[t]` must be greater than or equal to `min_production`. If `y[t]` is 0, then that constraint is essentially inactive. Note, the `complemented = 0` setting specifies that the indicator constraint is activated when `y[t] = 1`. If it was `1`, it would be activated when `y[t] = 0`. This is a much more readable and less error-prone way to handle these sorts of constraints than the big-m approach.

**Method 2: The Big-M Method (the classic workhorse)**

The 'big-m' approach is the more general method, even if it’s less elegant than using indicator constraints when you can. It relies on introducing a large constant (M) and manipulating inequalities to achieve the desired conditional behavior. It’s especially useful when your conditions involve more complex logic than simply a straightforward activation based on one variable. This is because it's quite flexible and applies to nearly any constraint.

Let's assume you have a constraint that `a*x + b*y <= c` which is activated if a binary variable 'z' is 1. Otherwise, that constraint is essentially ignored.

Here's how that looks with a big-m:

```python
import cplex
from cplex.exceptions import CplexError

try:
  my_prob = cplex.Cplex()

  # --- Variables ---
  my_prob.variables.add(names = ["x", "y"],
                        lb = [0,0])

  my_prob.variables.add(names = ["z"],
                        types = [my_prob.variables.type.binary]) # binary variable


  # --- Parameters ---
  a = 2
  b = 3
  c = 10
  big_M = 1000 #A sufficiently large number

  #--- Constraints ---
  # If z==1, then a*x + b*y <= c
  # If z == 0, we relax the constraint with Big M

  my_prob.linear_constraints.add(
       lin_expr = cplex.SparsePair(ind = ["x","y"], val=[a,b]),
       sense = "L", # less than or equal
       rhs = c + big_M*(1-my_prob.solution.get_values("z") if my_prob.solution.get_status() == 101 else 1))


  my_prob.objective.set_sense(my_prob.objective.sense.minimize)
  my_prob.objective.set_linear([("x",1)])
  my_prob.solve()

  print("Solution status = ", my_prob.solution.get_status_string())
  print("Solution value  = ", my_prob.solution.get_objective_value())
  print(f"x = {my_prob.solution.get_values('x')}, y = {my_prob.solution.get_values('y')}, z = {my_prob.solution.get_values('z')}")

except CplexError as exc:
  print(f"Cplex encountered an error: {exc}")
```
The trick here is `rhs = c + big_M*(1-z)`. If `z` is 1, the right-hand side becomes `c`, and the constraint `a*x + b*y <= c` is active. If `z` is 0, the right-hand side becomes `c + big_M`, a very large number, effectively relaxing the constraint so it's no longer limiting the solution. The critical part is to pick a `big_M` that's large enough to dominate the problem context but not so large that it causes numerical instability.

**Method 3: Using SOS Sets (Specific use cases)**

Another technique, particularly for mutually exclusive choices, is using Special Ordered Sets (SOS). SOS1 sets enforce that at most one variable in a set can be non-zero, and SOS2 sets impose ordering constraints among the variables. While they don't directly enforce if-then rules like big-m, they're very useful for modeling when a constraint is active based on a choice between multiple, mutually exclusive outcomes.

Let’s illustrate with a simple example. Suppose we can only choose one production line, each with its own production constraint. 'x_i' represents production in line i. Each x_i is only active in its corresponding production line is chosen, so it has a mutually exclusive relationship.

```python
import cplex
from cplex.exceptions import CplexError

try:
    my_prob = cplex.Cplex()
    num_lines = 3


    # --- Variables ---
    my_prob.variables.add(names = [f"x_{i}" for i in range(num_lines)],
                          lb = [0]*num_lines)

    my_prob.variables.add(names = [f"z_{i}" for i in range(num_lines)],
                          types = [my_prob.variables.type.binary] * num_lines) #binary variable if production line i is selected

    # --- Parameters ---
    max_production = [200,300,400]


    #--- Constraints ---
    for i in range(num_lines):
      #production limit for each production line
      my_prob.linear_constraints.add(
         lin_expr = cplex.SparsePair(ind=[f"x_{i}"], val = [1]),
         sense = "L",
         rhs = max_production[i]
      )

    #SOS type 1 sets: enforces only one production line to be active
    my_prob.SOS.add(type = "1", names = ["sos1"],
      vars = [f"z_{i}" for i in range(num_lines)],
      weights = [1]*num_lines) #weights don't matter in SOS1


    #linking indicator variable to production level for each production line
    big_M = 1000

    for i in range(num_lines):
      my_prob.linear_constraints.add(
      lin_expr=cplex.SparsePair(ind = [f"x_{i}"], val=[1]),
      sense = "L",
      rhs = big_M * my_prob.solution.get_values(f"z_{i}") if my_prob.solution.get_status() == 101 else big_M)




    # Objective
    my_prob.objective.set_sense(my_prob.objective.sense.maximize)
    my_prob.objective.set_linear([(f"x_{i}", 1) for i in range(num_lines)])

    my_prob.solve()

    print("Solution status = ", my_prob.solution.get_status_string())
    print("Solution value  = ", my_prob.solution.get_objective_value())

    for i in range(num_lines):
      print(f"x_{i} = {my_prob.solution.get_values(f'x_{i}')}, z_{i} = {my_prob.solution.get_values(f'z_{i}')}")


except CplexError as exc:
    print(f"Cplex encountered an error: {exc}")

```
In this example, `my_prob.SOS.add(type = "1" ...)` creates an SOS1 set, ensuring that at most one `z_i` is equal to one, effectively choosing one production line. I often see this sort of structure in production planning problems where each production line has specific constraints.

**Concluding Thoughts**

The key takeaway is that there’s no single perfect approach, each has its strengths. Start with indicator constraints when they're directly applicable, especially if the condition is simply "if binary variable is 1, then constraint must hold." Move to the Big-M method for more complex, customized logical dependencies. Finally, utilize SOS sets when dealing with mutually exclusive options or ordered sets of variables.

For further reading on these techniques, I'd recommend exploring the *Handbook of Optimization in Logistics and Supply Chain Management* by Carlos F. Daganzo, particularly the sections on mixed-integer programming. You might also find insights in *Integer Programming* by Laurence A. Wolsey, which gets into the details of formulations with binary variables. Finally, the cplex documentation itself is the most authoritative resource regarding usage of their specific features, such as indicator constraints, which often get updated in newer releases.

Remember, formulation is often as much an art as it is a science. Start simple, and incrementally increase complexity, keeping an eye on both model accuracy and computational performance. I've spent considerable time wrestling with these very same topics, so don't worry if there's a learning curve. Just iterate, test, and keep refining.
