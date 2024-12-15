---
title: "How to use Gurobipy: How to add a constraint that limits the number of times a certain tupledict value can appear in the optimal solution?"
date: "2024-12-15"
id: "how-to-use-gurobipy-how-to-add-a-constraint-that-limits-the-number-of-times-a-certain-tupledict-value-can-appear-in-the-optimal-solution"
---

alright so, you’re facing a classic combinatorial optimization problem here, and gurobipy is definitely the tool for the job. limiting the number of times a certain tupledict value appears in the optimal solution… yeah, i’ve been there. it’s surprisingly common and can get tricky depending on how your model is structured. i've spent countless hours staring at my screen debugging similar stuff, trust me.

so, let’s break this down. i’m assuming you have a tupledict, maybe something like `x[(i,j)]`, which represents a decision variable indicating if, say, a link between node `i` and node `j` is active. and you need to restrict how many times a specific link, say `x[(3,7)]`, can be activated in the optimal solution.

the core idea is to use gurobi’s addconstr method to add a restriction to your model which translates to: the sum of those variables that can be of interest is less than or equal to the maximum number allowed times of activation.

here's how i typically approach this, starting with the most basic scenario, and then adding a bit more complexity with examples.

**basic example:**

imagine you're working with a transportation network. you've got some locations, let's say labelled 1 to 10, and you want to know the optimal way to transport items among them. the decision variables would be `x[(i, j)]` which equals 1 if you ship something from location `i` to location `j`. and 0 otherwise. let’s suppose you have a constraint that prevents you from using the route from 3 to 7 more than 2 times in the optimal solution.

```python
import gurobipy as gp
from gurobipy import GRB

# sample data - this would come from your real data
locations = range(1, 11)
edges = [(i, j) for i in locations for j in locations if i != j]
cost = {(i, j): (i*j) % 11  for i, j in edges} # just some random cost

# Create the model
model = gp.Model("transportation_model")

# Decision variables
x = model.addVars(edges, vtype=GRB.BINARY, name="x")

# Objective function
model.setObjective(gp.quicksum(cost[i,j] * x[i, j] for i, j in edges), GRB.MINIMIZE)

# constraint on how many times route (3,7) can be used: max 2.
model.addConstr(x[(3, 7)] <= 2, "max_use_of_3_7")


# other constraints would go here.
# lets add just an dummy one just to have a feasible solution
model.addConstr(gp.quicksum(x[1,j] for j in range(2,11))>=1, "dummy_feasibility")

# Solve the model
model.optimize()

# Print the results
if model.status == GRB.OPTIMAL:
    print("Optimal objective value:", model.objVal)
    for i,j in edges:
       if x[i,j].x > 0.5: #check only if it is part of the solution.
           print(f"Ship from {i} to {j}")

else:
    print("No optimal solution found")
```
in this case `model.addConstr(x[(3, 7)] <= 2, "max_use_of_3_7")` is exactly what we need to restrict the number of times that link is activated. in essence you are limiting the upper bound of the binary variable. pretty direct.
but, what if you have more restrictions?.

**more complex constraint:**

now, let’s say instead of a specific edge, you want to limit the total number of edges involving a particular node, like the total number of links involving location 3,  that can be active. you’d have to sum all of them and add that as a constraint. it's very similar to the previous approach but instead of adding one variable, you would need to add all the variables that are linked to a specific node.

```python
import gurobipy as gp
from gurobipy import GRB

# Sample data
locations = range(1, 11)
edges = [(i, j) for i in locations for j in locations if i != j]
cost = {(i, j): (i*j) % 11  for i, j in edges}

# Create the model
model = gp.Model("transportation_model")

# Decision variables
x = model.addVars(edges, vtype=GRB.BINARY, name="x")

# Objective function
model.setObjective(gp.quicksum(cost[i,j] * x[i, j] for i, j in edges), GRB.MINIMIZE)

# constraint on how many times edges with 3 as origin or destination can be used: max 4
model.addConstr(gp.quicksum(x[3,j] for j in locations if j!=3) + gp.quicksum(x[i,3] for i in locations if i!=3) <= 4 , "max_use_of_node_3")

# other constraints would go here.
model.addConstr(gp.quicksum(x[1,j] for j in range(2,11))>=1, "dummy_feasibility")

# Solve the model
model.optimize()

# Print the results
if model.status == GRB.OPTIMAL:
    print("Optimal objective value:", model.objVal)
    for i,j in edges:
       if x[i,j].x > 0.5: #check only if it is part of the solution.
           print(f"Ship from {i} to {j}")

else:
    print("No optimal solution found")

```
here `model.addConstr(gp.quicksum(x[3,j] for j in locations if j!=3) + gp.quicksum(x[i,3] for i in locations if i!=3) <= 4 , "max_use_of_node_3")` adds a constraint that sums all the incoming and outcoming links of location 3, and limits that sum to 4, in this case.

**using tuplelists for even more flexibility:**

now, let’s go even more abstract. what if the items you want to limit have not a specific pattern, but are stored in a python list. and you have a list of edges `[(1,2),(3,4),(5,6),(7,8)]` you want to limit the total sum of these in the optimal solution to, lets say, 3.

```python
import gurobipy as gp
from gurobipy import GRB

# Sample data
locations = range(1, 11)
edges = [(i, j) for i in locations for j in locations if i != j]
cost = {(i, j): (i*j) % 11  for i, j in edges}

# items to limit: these can come from the data of your problem
items_to_limit = [(1,2),(3,4),(5,6),(7,8)]

# Create the model
model = gp.Model("transportation_model")

# Decision variables
x = model.addVars(edges, vtype=GRB.BINARY, name="x")

# Objective function
model.setObjective(gp.quicksum(cost[i,j] * x[i, j] for i, j in edges), GRB.MINIMIZE)


# constraint on how many times the items in "items_to_limit" can be used: max 3
model.addConstr(gp.quicksum(x[i,j] for i, j in items_to_limit)<=3 , "max_use_of_items_to_limit")

# other constraints would go here.
model.addConstr(gp.quicksum(x[1,j] for j in range(2,11))>=1, "dummy_feasibility")

# Solve the model
model.optimize()

# Print the results
if model.status == GRB.OPTIMAL:
    print("Optimal objective value:", model.objVal)
    for i,j in edges:
       if x[i,j].x > 0.5: #check only if it is part of the solution.
           print(f"Ship from {i} to {j}")

else:
    print("No optimal solution found")

```
in this case, we leverage a python list called `items_to_limit` to store the relevant edges and we create a constraint that limits the sum of these variables that belong to `items_to_limit` to be at most 3.

**some things to keep in mind:**

*   **binary vs. integer variables:** the examples i’ve given use binary variables (`vtype=GRB.BINARY`). if your decision variables can take integer values greater than one, you’d want to adjust the right-hand side of the `addConstr` accordingly. the logic behind of adding the constraint remains the same.
*   **performance**: if you find your model is becoming slow, investigate the structure of your constraints. sometimes, a subtle change in how you express the constraint can dramatically impact performance. also, using `gp.quicksum()` is better than a python `sum` when summing up over many terms of gurobi variables, as it will be done within gurobi’s core.
*   **debugging:** always print the model status and the variables values to diagnose if the model is solving properly. i tend to write small test cases first, to test only one constraint. that's a good practice.

**additional resources:**

for a deeper dive, you could check out "modeling languages in mathematical programming" by h. paul williams. it provides a theoretical foundation for what you are doing here. there’s also plenty of documentation on the gurobi website, particularly the reference manual which is quite handy. and there are some case studies at sites like or-tools that often show how to express constraints in similar modeling languages. some of these cases may have different tools (not gurobi) but give ideas of how to formulate the models, i found that quite useful when i was starting. and do not forget stackoverflow where it may be other useful questions that resemble yours.

i remember this one time, i had a model with thousands of binary variables and a constraint similar to this one, and i ended up using a lambda function to create a dictionary that indexed all the variables by their associated index on a multi dimensional tuple, just to speed up things. but i think that falls outside the scope of this question, just a funny story, i guess. anyway… if you have further questions feel free to ask, i've been down that road more than once. happy modeling.
