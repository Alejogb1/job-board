---
title: "How to Mathematical modeling from gurobipy to pyomo: How to enumerate over a set in pyomo?"
date: "2024-12-15"
id: "how-to-mathematical-modeling-from-gurobipy-to-pyomo-how-to-enumerate-over-a-set-in-pyomo"
---

so, you're looking at moving from gurobipy to pyomo, and the sticking point seems to be how you handle set enumeration, which, yeah, i get it, it's a common spot where these two libraries differ. i've been there, trust me. i started out doing operations research stuff for a small logistics firm back in the early 2000's and gurobi was, for a time, my best friend, it was the speed demon back then, but then i needed more flexibility, and that's where pyomo came in. so, let me break it down based on my own switchover experiences and the common sticking points.

the thing is, gurobipy does a lot of the set handling implicitly. when you define variables or constraints, it often assumes you're iterating over a set you've pre-defined, without needing to be super verbose about it. it just works, and it's quick. pyomo is a bit more explicit. you need to define the sets, index them, and then refer to those indices in your model. it's more boilerplate at first glance, but it makes things way more traceable and flexible in the long run. plus, you get to define specific indexing mechanisms, you'll see the beauty of it, later on.

let's think of a simple scenario: imagine you are modelling production planning for different products in a factory. you want to define production variables for each product, so you need to iterate through that 'products' set. in gurobipy, this would seem like an afterthought, you define a dictionary, and then that is it. whereas, in pyomo, it's explicit.

here's an example. let's say you want to model the production quantities of three products. we'll create a basic constraint that says production of each product must be non-negative.

here is how you would do it in pyomo:

```python
from pyomo.environ import *

# defining your model
model = ConcreteModel()

# defining the sets
model.products = Set(initialize=['product_a', 'product_b', 'product_c'])

# variables using the sets
model.production = Var(model.products, within=NonNegativeReals)

# here are the constraints (just to show you how to use the set)
def constraint_rule(model, product):
    return model.production[product] >= 0

model.production_non_negative = Constraint(model.products, rule=constraint_rule)

# this model is now ready for objective functions and more complex constraints
```

in this example, `model.products` is our set, we explicitly define the `products` set using `Set(initialize=['product_a', 'product_b', 'product_c'])`. the variables are indexed using this set, shown in `model.production = Var(model.products, within=NonNegativeReals)`, and when we define the constraint `model.production_non_negative`, we specify that `model.products` dictates how many instances of the constraint should exist, effectively making it an enumeration. if you were to print the constraint, it would show something like `production_non_negative[product_a]`, `production_non_negative[product_b]`, and `production_non_negative[product_c]`. it's direct and it lets you reason about all the elements of the set that are involved. if you have the model defined in a more abstract way, it makes things even easier to understand later on.

it may seem like an extra step compared to gurobi, but trust me it will save your debugging time if you ever have a big model.

let's consider a slightly more complex case, imagine you have to model transportation between different cities, and you have a cost associated with each route. you will have two sets in this situation: `cities` and `routes` which is built using `cities`.

```python
from pyomo.environ import *

model = ConcreteModel()

# define the cities
model.cities = Set(initialize=['city_a', 'city_b', 'city_c'])

# derive the routes from the cities, every combination is a route
model.routes = Set(initialize=model.cities * model.cities, dimen=2) # all combinations of routes

# transportation cost between cities
model.transport_cost = Param(model.routes, initialize={
    ('city_a', 'city_b'): 10,
    ('city_a', 'city_c'): 15,
    ('city_b', 'city_a'): 12,
    ('city_b', 'city_c'): 18,
    ('city_c', 'city_a'): 20,
    ('city_c', 'city_b'): 14,
})

# model transportation variable
model.transportation_amount = Var(model.routes, within=NonNegativeReals)

# let's create a dummy constraint, just to show the routes
def route_constraint_rule(model, city_from, city_to):
    return model.transportation_amount[city_from, city_to] >= 0

model.route_constraints = Constraint(model.routes, rule=route_constraint_rule)
```

this snippet defines a `routes` set, using the cartesian product of cities. this `model.routes` set can be indexed using a tuple, representing the origin and destination city. the transportation variable, `model.transportation_amount` is also indexed using this set, as it's a variable specific for a given origin and destination. if you were to print the parameters it would be like `transport_cost[city_a,city_b]=10` and so on. the key takeaway is: you have explicit control over the index sets and their dimensions, making it easier to deal with complex scenarios. in a very large scale, this kind of explicit indexing will make your code more understandable when you have to debug, it will be much more complex if you don't handle it properly from the start. i remember once, i was coding some complex logistics optimization model with lots of routes, and not indexing properly from the start, and i ended up with a spiderweb of variables and parameters and i was not able to debug it for two whole days, you would think "i am so smart i can do it like this", yeah, no, it was a mess.

now, let's move on, what about dynamic sets? what if your sets depend on your data, meaning you read the data from a file and then create the sets? it is a very common scenario. let's simulate this. let's say you're modeling a network of stores and warehouses. your set of stores, and warehouses will vary according to the data you have.

```python
from pyomo.environ import *

# this simulates reading data from a file
store_data = ['store_1', 'store_2', 'store_3', 'store_4']
warehouse_data = ['warehouse_a', 'warehouse_b']
# you would usually read this from your data

model = ConcreteModel()

# building sets dynamically
model.stores = Set(initialize=store_data)
model.warehouses = Set(initialize=warehouse_data)

# variables (using the sets)
model.shipment_amount = Var(model.stores, model.warehouses, within=NonNegativeReals)

# a dummy constraint to show the indexes, all shipments should be greater or equal to 0
def shipment_constraint_rule(model, store, warehouse):
    return model.shipment_amount[store, warehouse] >= 0

model.shipment_constraints = Constraint(model.stores, model.warehouses, rule=shipment_constraint_rule)

# imagine adding more complex constraints and objective functions here
```

in this example, we dynamically create sets `model.stores` and `model.warehouses` from a list of strings, usually read from a data file in a real situation. notice how the variables `model.shipment_amount` are indexed by two sets simultaneously. pyomo allows you to flexibly create sets from any iterable, it can be a list of strings, a dictionary, or even the results of some computation. this dynamic nature is a big advantage for those who work with data sets that change often, you can literally build all your model using a database. it makes it much more data-driven and less hardcoded which is good. also note that you can use pandas if you wish, it will make loading the data very easy and you can easily format it before using it to build the sets.

to wrap it all up, the key to set enumeration in pyomo is to explicitly define your sets and then use them to index your variables, parameters, and constraints. pyomo might seem more verbose than gurobi, but this explicitness brings more clarity and flexibility and allows for complex problems to be more easily modelled, especially for complex large models that are common in industrial and operations research contexts. while gurobi is quick at solving, pyomo makes the modeling easier, and that makes debugging easier, and that makes your life better. i remember one professor saying "i'd rather spend one day modeling and one hour debugging, than one hour modeling and one day debugging", and you know, i agree with it. plus, it plays very well with other python libraries and that is key when building complex model workflows.

i can recommend some literature to understand this better. for a general background on mathematical optimization, look for "linear programming and network flows" by m. s. bazaraa et al., it's a classic. then for pyomo itself i really recommend "pyomo -- optimization modeling in python" by william e. hart, carl d. laird, john-paul watson, david l. woodruff, it has all the information you will ever need, and they have good examples and tutorials. plus they explain all these topics with great detail. they are a great way to deep dive into it, and learn the best practices for modeling.

and one last thing, i heard gurobi is getting into knitting patterns, because it always seems to find the optimal thread count, i'll leave it here, don't tell them i said that.
