---
title: "How to do Optimization with a constraint on a different vector to the objective function?"
date: "2024-12-15"
id: "how-to-do-optimization-with-a-constraint-on-a-different-vector-to-the-objective-function"
---

alright, so you’re tackling optimization with a constraint that's not directly on the variables you're trying to minimize or maximize, yeah? i’ve been there, it's a common pain point, and frankly, it's one of those situations where the devil is in the details of how you set it up. let's walk through this as i’ve had my share of late nights staring at similar problems.

first, let’s break this down into what’s really going on here. you have a cost function (or objective function), that depends on one set of variables, let's call them `x`. this could be anything, like model weights in a neural network, parameters for a simulation, or literally anything you're tweaking to reach a 'best' value.

but now the issue pops in. you also have a constraint, but that constraint is on another set of variables, say `y`, that are not necessarily the same as `x`. and these `y` variables are somehow related to your `x` variables. this is where the party gets complicated, not impossible, but it requires a little more thinking than your simple textbook optimization example.

i had this thing pop up during a project i did back in my phd days. it was about optimizing the trajectory of a robotic arm, the `x` vector, to achieve some goal but the physical limitations of the motors, the `y` vector, imposed constraints. The motors had a power limit that, if exceeded, could literally melt the circuitry! obviously, i couldn't just optimize the arm’s movement without also keeping the motor's power within safe levels. this meant we had an objective (the arm's path) and a constraint (the motors’ power) on different sets of variables linked through complex physical simulation. it was such a mess i nearly quit and went into farming. (just kidding, sort of, the thought was there)

so how do we address this from a more general standpoint?

the core of the solution is finding a way to *relate* your `x` and `y` variables. that relationship can be anything from a direct algebraic equation to a complex numerical simulation. but, if there is a relationship between them, we can think of `y` as a function of `x`, this can be expressed as `y = f(x)`. once you have this, you can rewrite the constraints in terms of just `x`.

for example, suppose your objective function is something simple like a sum of squares:

```python
import numpy as np

def objective_function(x):
    return np.sum(x**2)
```

and suppose the constraint variables 'y' are computed from 'x' by a function:

```python
def constraint_function(x):
    #example of some arbitrary transformation of x into y
    return np.sin(x) + 0.5
```

let’s say your constraint is that the sum of `y` must not exceed a certain value, `c`. so, we have sum(y) <= c, we are now set to formulate our optimization:

```python
from scipy.optimize import minimize

def combined_constraint(x, constraint_limit):
   #this function returns the violation of the constraint, it's 0 or negative
   #if the constraint is fulfilled
    y = constraint_function(x)
    return constraint_limit - np.sum(y)


def optimization_problem(x0, constraint_limit):
  #set the constraint as a tuple of function and arguments
    cons = ({'type':'ineq', 'fun': combined_constraint, 'args':(constraint_limit,)})
   #call scipy minimizer, set an initial guess and the constraint
    result = minimize(objective_function, x0, method='SLSQP', constraints=cons)
    return result
    

if __name__ == '__main__':
    initial_guess = np.array([0.5, 0.5])
    limit = 1.0 #our constraint limit
    results = optimization_problem(initial_guess, limit)
    print(results)
```

in the above example, we use `scipy.optimize.minimize` with the 'slsqp' method, which is a good general purpose constrained optimizer. the constraint is implemented in the `combined_constraint` function and passed as an inequality using a dictionary in the constraint. we make sure to pass the constraint limit to the constraint function. what this function returns is that must be positive or zero if the constraint is not violated.

now, this approach is straightforward when you have an analytical form of the relation between x and y like in this case, which is nice when it works. but what if your relationship, `y = f(x)`, is not so simple or not available analytically?

this is where you might need to consider a numerical approximation or a simulation. i dealt with this during a project where we were optimizing the design of a chemical reactor, where we had a cost function based on the produced yield and a constraint on the internal temperature, but the reactor’s temperature was derived from a full-blown computational fluid dynamics (cfd) simulation given the reactor's dimensions, which were represented by `x`, the `y`s were the temperatures. you can't just get a `y` with a simple formula, so instead, i had to make use of an auxiliary simulation or function which gives back an approximate solution to the 'y'. it was slow but it worked.

here’s a simplified idea of how to put together a simulation inside your optimization, for instance, assume you have a cfd simulation that you run given parameters `x` and that returns `y`, we wrap it inside a python function:

```python
import numpy as np
from scipy.optimize import minimize

#this function simulates a cfd calculation
def cfd_simulation(x):
    #simplified simulation function based on x parameters that
    #should return the `y` vector
    return x * np.array([1.2,0.5]) + 0.3

def objective_function(x):
    return np.sum(x**2)

def combined_constraint(x, constraint_limit):
   #this function returns the violation of the constraint
    y = cfd_simulation(x)
    return constraint_limit - np.sum(y)


def optimization_problem(x0, constraint_limit):
  #set the constraint as a tuple of function and arguments
    cons = ({'type':'ineq', 'fun': combined_constraint, 'args':(constraint_limit,)})
   #call scipy minimizer, set an initial guess and the constraint
    result = minimize(objective_function, x0, method='SLSQP', constraints=cons)
    return result
    

if __name__ == '__main__':
    initial_guess = np.array([0.5, 0.5])
    limit = 1.0 #our constraint limit
    results = optimization_problem(initial_guess, limit)
    print(results)

```
in the code above we replaced the simple `constraint_function` with `cfd_simulation`, and now the optimization goes through the `cfd_simulation`, a computationally demanding operation that provides the `y` used to evaluate the constraint. if your simulations are expensive, it is a good idea to make sure that your constraint function does not compute it every time it is called but keeps a cached version or implements a fast approximator for the cfd simulation.

another challenge is that sometimes, `f(x)` can be discontinuous or non-differentiable. this is not a problem in general, depending on what optimizer are you using. for instance, the `slsqp` method, as shown in the examples, will work as it does not need the gradients. but others might fail to converge or give strange solutions. in such cases, you might need to explore derivative-free methods, like evolutionary algorithms, which are more robust in these situations. a book i found very useful here is “numerical optimization” by jorge nocedal, this should be a bible to you if you are dealing with complex non-linear optimization problems.

to deal with non-analytical functions you can also consider surrogate optimization. the main concept of surrogate optimization is to build an easy-to-compute approximation or a surrogate function. the surrogate function is built using the information computed previously from the optimization loop. for example, a response surface or an approximation with a gaussian process could be used for that purpose. this could reduce drastically your computational cost in scenarios where the relationship between `x` and `y` is obtained through an expensive simulation.

in essence, the solution to your problem is not a simple code snippet but a whole procedure. the approach requires that you properly formulate the relationship between your objective function's variables `x` and your constraint's variables `y`, whether through a direct calculation or a numerical simulation, and finally, select the appropriate optimizer and parameters depending on the characteristics of the problem at hand. it’s an iterative process, and often a combination of good understanding of the underlying physics and a bit of trial and error to nail down the specifics, and this is how i usually approach these kinds of problems, and i hope it can help you in your journey.
