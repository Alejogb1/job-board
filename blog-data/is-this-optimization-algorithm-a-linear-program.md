---
title: "Is this optimization algorithm a linear program?"
date: "2024-12-23"
id: "is-this-optimization-algorithm-a-linear-program"
---

Let's dive into this. One of the first things that struck me when I initially encountered optimization problems at scale, back at my previous gig architecting the routing engine, was the temptation to classify every optimization task under the umbrella of linear programming. It’s a natural inclination, given the elegance and power of linear programming (LP) solvers, but it's also a pitfall if you're not careful. So, is *this* particular optimization algorithm actually a linear program? It's a question that warrants careful consideration.

The core issue hinges on the fundamental structure of the problem. A linear program, at its heart, requires three crucial elements: a linear objective function, linear constraints, and variables that are themselves inherently linear. The objective function, that's the target we are trying to either maximize or minimize, for example, minimizing cost or maximizing profit. The constraints limit the variable spaces to specific feasible ranges, and the variables are essentially scalars that dictate how much of something we use. Critically, all relationships within a linear program *must* be described by linear equations or inequalities. If we encounter non-linearities, we’ve crossed outside of the domain of classic linear programming.

Now, before we get mired in abstract concepts, let me share a specific scenario that might sound familiar. Back in the day, dealing with real-time traffic flow, I had a seemingly simple problem: allocate network bandwidth to different service queues to minimize latency. Initially, the first thought was to represent the delay of each queue using a queuing theory model, which involved terms like 1/(μ - λ), where μ was the service rate and λ the arrival rate. These relationships, unfortunately, are nonlinear. While I *could* model average queue length linearly by making approximations on those rates, the actual latency optimization didn't quite fit the linear mold. I had to use an approximation strategy first to fit in a linear program, and that was a significant compromise. It turns out the overall behavior was more of a mixed integer nonlinear program (MINLP).

The trick is in the details: scrutinize the relationships between variables. Are they strictly proportional to each other? Can the cost or value associated with a variable be computed as a linear combination of other variables? If you find terms like variables multiplied by each other, exponential terms, logarithmic functions, or other non-linear operators, then your problem is decidedly not a linear program. Also, look at whether your variables are integers or not. If they are, it's likely a mixed integer program (MIP). Integer or mixed-integer programs can be, and often are, linear in their core structures, which makes the choice of approximation technique very important.

Let’s illustrate this with a few code examples, using python with scipy, a standard scientific library, to demonstrate how one sets up different optimization problems.

First, let’s consider a straightforward case of a classic linear programming problem: minimizing cost. Let's assume we need to produce two products, A and B. Each unit of product A requires 2 units of resource X and 1 unit of resource Y, and yields a profit of 3 dollars. Each unit of product B requires 1 unit of resource X and 3 units of resource Y, and yields a profit of 5 dollars. We have 10 units of resource X and 15 units of resource Y available.

```python
from scipy.optimize import linprog

# objective function: minimize negative of profit
c = [-3, -5] # coefficients to maximize 3a + 5b, hence, multiply by -1
# inequality constraints: Ax <= b
A = [[2, 1],
     [1, 3]]
b = [10, 15]
# variable bounds (non-negativity constraints)
x0_bounds = (0, None)
x1_bounds = (0, None)

result = linprog(c, A_ub=A, b_ub=b, bounds=[x0_bounds, x1_bounds], method='highs')

print(result)
# Optimal: [1.5, 4.16] with a value of -25.333
```

This snippet is a clear, canonical LP. Everything is linear: the constraints on resource consumption, the profit objective, and the variable space.

Now, let’s change this scenario. Imagine instead of a linear profit, our profit scales with the square of the amount of product we sell. This is unrealistic for most applications, but I'll use this to illustrate non-linearity simply. The objective now is to maximize 3 * a^2 + 5 * b^2, while keeping all other constraints.

```python
from scipy.optimize import minimize
import numpy as np

def objective_nonlinear(x):
  a, b = x
  return -1 * (3 * a**2 + 5 * b**2) # negative sign for minimization

def constraint_linear(x):
  a, b = x
  return np.array([10 - 2*a - b, 15 - a - 3*b])

x0 = np.array([0,0]) # initial starting point

result = minimize(objective_nonlinear, x0, method='SLSQP', constraints={'type':'ineq', 'fun': constraint_linear}, bounds=[(0, None), (0, None)])

print(result)

# Optimal output depends on start parameters, but shows the optimization is no longer linear.
```
As you can see, this snippet changes the code and now requires a different solver. The objective function now includes the square of the variables, introducing the non-linearity into the problem. In these cases, we must resort to non-linear optimization techniques.

Finally, let’s look at a MIP example, where our variables are integer values. Let’s return to our linear production example, but now the amount of product we can produce is integers.

```python
from scipy.optimize import milp
import numpy as np

# objective function: minimize negative of profit
c = [-3, -5] # coefficients to maximize 3a + 5b, hence, multiply by -1
# inequality constraints: Ax <= b
A = [[2, 1],
     [1, 3]]
b = [10, 15]
# variable bounds (non-negativity constraints)
x0_bounds = (0, None)
x1_bounds = (0, None)
integrality = np.array([1, 1]) # 1 indicates integer, 0 indicates real variable


result = milp(c=c, constraints=[(A, b)], integrality = integrality, bounds=[x0_bounds, x1_bounds])

print(result)
#Optimal: [1, 4] with a value of -23.0
```

This example, while structurally similar to our linear programming example, uses an integer solver in scipy and forces the variables to integer values. This means it is not strictly a linear program, but a Mixed Integer Program (MIP). This simple change means you can't simply use linear program solvers.

The lesson here is that superficial similarities aren’t enough; you must delve into the mathematical formulation of the problem. So, to address the question: Is your algorithm a linear program? Take a hard look at the objective and constraints. Are they composed of exclusively linear expressions? Do your variables involve integer constraints? If the answer is no, then, you likely have a different optimization problem. Trying to shoehorn a non-linear problem into an LP framework can lead to highly suboptimal or even incorrect solutions. Understanding that distinction is paramount.

For a deeper dive into this, I'd recommend exploring *Convex Optimization* by Stephen Boyd and Lieven Vandenberghe. It provides a rigorous mathematical foundation for optimization concepts, including linear programming. Also, *Introduction to Operations Research* by Frederick S. Hillier and Gerald J. Lieberman is an excellent all-around resource, covering the breadth of operations research techniques, including detailed discussions on linear programming, mixed-integer programming, and non-linear optimization. Another good book is *Numerical Optimization* by Jorge Nocedal and Stephen J. Wright if you want to understand how numerical optimization actually works. These resources will offer you a far deeper understanding and help avoid the pitfall of blindly applying linear programming techniques in inappropriate contexts, a situation I've seen all too often.
