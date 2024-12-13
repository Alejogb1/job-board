---
title: "pulp optimization linear programming usage?"
date: "2024-12-13"
id: "pulp-optimization-linear-programming-usage"
---

Okay so you want to talk about linear programming for pulp optimization yeah I've been down that road a few times let me tell you it's like trying to find the perfect balance in a really complex recipe except instead of ingredients we're dealing with constraints and an objective function and instead of eating it at the end you get some numbers that hopefully make sense

I first bumped into this whole linear optimization thing back in my early days trying to optimize resource allocation for a small cloud based service we were launching we had a ton of VMs different sizes different costs different performance profiles and the users were hitting it sporadically it was a chaos show honestly initially I tried to do it all by hand spreadsheets were involved it was a disaster let me tell you I was manually balancing everything and it was like trying to herd cats on roller skates I quickly realized I needed something way more systematic and efficient and that's when I really dove into linear programming 

Pulp it’s a killer tool for this sort of thing really user friendly lets you define your problems in code makes it all so much easier if you've used any kind of python library before its a total breeze 

Here's the thing linear programming is at its core about solving a problem where you have a linear objective function like profit you're trying to maximize or cost you're trying to minimize and then a set of linear constraints these constraints are limitations or requirements that must be satisfied its all about finding the feasible solution within that space and then the best of those feasible solutions based on the objective so for me my constraint could be like how many vCPUs available how much RAM we had how much we could actually spend on cloud computing and objective was minimum cost for maximum performance

So lets give you a simple example imagine we’re running a factory we make two products let's call them A and B each needs different amounts of two resources lets call them material X and material Y lets say product A needs 2 units of X and 1 unit of Y while product B needs 1 unit of X and 3 units of Y we have 10 units of X and 15 units of Y available and we make a profit of 10 for every unit of A we make and 15 for every unit of B we make so the objective is to maximize profit based on available resources

```python
from pulp import *

# Create the LP problem
prob = LpProblem("Factory_Production", LpMaximize)

# Define decision variables
A = LpVariable("A", lowBound=0, cat='Integer') # Number of units of A
B = LpVariable("B", lowBound=0, cat='Integer') # Number of units of B

# Objective function
prob += 10 * A + 15 * B

# Constraints
prob += 2 * A + 1 * B <= 10   # Material X constraint
prob += 1 * A + 3 * B <= 15   # Material Y constraint

# Solve the problem
prob.solve()

# Print the results
print("Status:", LpStatus[prob.status])
print("Optimal production of A:", A.varValue)
print("Optimal production of B:", B.varValue)
print("Maximum profit:", value(prob.objective))
```

In this code you see how we formulate the problem using `pulp` the `LpProblem` is where we define it as a maximization problem we create `LpVariable` for our variables and assign our objective function and the constraints we then run the solver and its a black box and tada you get an answer thats awesome right

Now you might be thinking okay thats a pretty basic example whats it like when things get complex it’s gonna be like a whole can of worms opened but trust me with pulp it is fairly okay

Let's crank up the complexity a notch what if we have multiple resources say materials X Y and Z and we also have several different products and what if some resources can be used in multiple production lines and we also wanna take care of transport cost that adds to the equation now the model can grow exponentially with variables and constraints and sometimes you just stare at the problem and question your life choices but here it is still ok with pulp

Lets say we have three products A B and C that each needs materials X Y and Z and have different transportation cost from 2 different warehouses and now with different profit margin it gets a little harder to calculate by hand because of all different combinations

```python
from pulp import *

# Define sets
products = ["A", "B", "C"]
materials = ["X", "Y", "Z"]
warehouses = ["W1", "W2"]

# Data
profit_margins = {"A": 10, "B": 15, "C": 20}
resource_requirements = {
    ("A", "X"): 2, ("A", "Y"): 1, ("A", "Z"): 0,
    ("B", "X"): 1, ("B", "Y"): 3, ("B", "Z"): 1,
    ("C", "X"): 0, ("C", "Y"): 2, ("C", "Z"): 2,
}
resource_availability = {"X": 10, "Y": 15, "Z": 12}
transport_costs = {
    ("A", "W1"): 1, ("A", "W2"): 2,
    ("B", "W1"): 2, ("B", "W2"): 1,
    ("C", "W1"): 3, ("C", "W2"): 2
}

# Create the LP problem
prob = LpProblem("Complex_Factory_Production", LpMaximize)

# Decision Variables
production_vars = LpVariable.dicts("Production", [(p, w) for p in products for w in warehouses], lowBound=0, cat='Integer')

# Objective function
prob += lpSum(profit_margins[p] * production_vars[(p, w)] - transport_costs[(p, w)] * production_vars[(p, w)]
for p in products for w in warehouses )

# Resource constraints
for m in materials:
    prob += lpSum(resource_requirements[(p, m)] * production_vars[(p, w)] for p in products for w in warehouses) <= resource_availability[m]

# Solve the problem
prob.solve()

# Print the results
print("Status:", LpStatus[prob.status])
for p in products:
    for w in warehouses:
        print(f"Production of {p} from {w}: {production_vars[(p,w)].varValue}")
print("Maximum profit:", value(prob.objective))
```

In this code you see we create sets of our products materials and warehouses and use these to create variables now we are no longer creating single variables instead we create dictionaries of them then we simply define the objective function and add the constraint for resource usage

One last point some resources can be used by multiple departments we are now creating more complex resource constraints but with pulp you can structure your problem with sets like you see here and it makes it all much easier to model it really is

When I was dealing with the cloud resource allocation problem I was facing a similar challenge I had different types of VMs different types of storage and needed to consider not just cost but also performance I was using linear programming to figure out how to allocate resources to different users while also meeting their SLAs

Now here's the real trick with pulp its not just about getting a solution it's about understanding your constraints and tweaking them sometimes the model will give you what looks like an optimal solution but when you dig into it you might discover its not really feasible or it doesn’t match the requirements of the business so you might need to add or change the constraints I needed to fine tune my constraints to find a balance between cost and user performance

So lets give you an example now with a constraint that we have a certain ratio of different products for example we need to produce at least half of the amount of product A than of B now this might be some real life requirement

```python
from pulp import *

# Create the LP problem
prob = LpProblem("Ratio_Constraint", LpMaximize)

# Decision Variables
A = LpVariable("A", lowBound=0, cat='Integer')
B = LpVariable("B", lowBound=0, cat='Integer')

# Objective function
prob += 10 * A + 15 * B

# Resource Constraints
prob += 2 * A + 1 * B <= 10   # Material X constraint
prob += 1 * A + 3 * B <= 15   # Material Y constraint

#Ratio constraint
prob += A >= 0.5 * B

# Solve the problem
prob.solve()

# Print the results
print("Status:", LpStatus[prob.status])
print("Optimal production of A:", A.varValue)
print("Optimal production of B:", B.varValue)
print("Maximum profit:", value(prob.objective))
```

This code is just slightly changed with a new constraint added now we have a minimum ratio of product A vs B you can play around with these constraints and see how the solution changes

Now here is the funny part one day I had a problem that had way to many variables and for 5 hours my solution was stuck in optimizing but when it finally came out I looked at it and all it did was a 5% improvement in the objective function I have never wanted to pull my hair out so bad in my life as I did that day optimization sometimes its funny like that its not always a huge improvement

To be fair if you want a deeper dive into linear programming you could look at books like "Introduction to Linear Optimization" by Dimitris Bertsimas and John Tsitsiklis or "Linear Programming" by Vasek Chvatal they go into the underlying math and the theory behind the algorithms plus they have better examples I recommend it for better understanding but pulp really abstracts away all the complexity of the problem

So yeah that's my experience with linear programming optimization using pulp Its a tool I heavily use in my day to day it handles surprisingly complex optimization problems and if you learn it well it can be a really useful tool in your own toolbox
