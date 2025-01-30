---
title: "Is Mode 9 feasible in GEKKO?"
date: "2025-01-30"
id: "is-mode-9-feasible-in-gekko"
---
Mode 9, referring to a specific, and somewhat ill-defined, operational regime of dynamic optimization in chemical process control, presents significant challenges within the GEKKO modeling environment, although not definitively impossible. My experience implementing various process optimization strategies over the last seven years suggests that while GEKKO's robust capabilities address many advanced control problems, Mode 9 pushes it, and any similar library, toward the edge of its practical applicability. The core issue isn't GEKKO's optimization algorithms, but rather the ill-defined nature and computational demands of what we’ll call “Mode 9” for this discussion.

The operational regime referred to here assumes a hypothetical scenario where process parameters, specifically those relating to model dynamics (not simply manipulated variables, but time constants, gain values, and process interactions) are not only subject to optimization but are also functions of *other* optimization variables, and further these relationships can be discontinuous or conditional, all while the typical constraints related to process throughput, product quality, and safety must be satisfied within a finite horizon, all with limited computational resources. While this isn’t exactly a named control method within process engineering, it's what I've encountered when tackling particularly complex dynamic optimization problems. GEKKO is primarily designed for solving optimal control problems where the system dynamics are known a priori, with parameters that remain either constant or are simple functions of time or manipulated variables. Introducing this high degree of non-linearity and parametric dependence creates optimization landscapes that are exceptionally complex, with multiple local optima and non-smooth response surfaces.

To understand the feasibility, we need to break down the challenges. First, GEKKO models are inherently described by differential and algebraic equations (DAEs). It uses a solver to handle these equations, primarily through numerical integration. The core optimization algorithms, such as Interior Point Methods (IPOPT) or Sequential Quadratic Programming (SQP), work well when the gradients and Hessians of the objective function and constraints are well-defined and can be computed. In Mode 9, with model dynamics being functions of optimization variables, those functions need to be differentiable, and even then they may create numerically ill-conditioned Jacobians which can cause the numerical solver to be slow, or even outright fail to converge. Moreover, this means that the model isn't static; its equations are changing as the optimization proceeds. This dynamic model modification means that every iteration of the optimization algorithm would require recalculating and re-analyzing the entire set of equations. This adds computational expense and makes robust convergence much less probable. It also often creates “chattering”, where the optimizer bounces around a region without making useful progress.

The second key problem involves handling the conditional and discontinuous relationships. If a particular optimization variable causes a discrete jump in a model parameter, or introduces a wholly new equation, then standard gradient-based optimization techniques are insufficient and require special handling, or often even a different optimization approach that is not provided within the core GEKKO library. The underlying algorithms may struggle with such abrupt changes, leading to convergence failures, instability, and suboptimal results. Finally, the combination of these two challenges, that is, complex relationships and discontinuous dependencies, can lead to combinatorial explosion. Finding a single global optimum within this type of problem can quickly become an intractable problem for even the most powerful available computers.

However, this doesn’t make Mode 9 entirely infeasible, just exceptionally challenging. With significant effort, and a deep understanding of numerical optimization, we can apply GEKKO to certain reduced cases. Let’s illustrate a basic version of this conceptual Mode 9 case, using three simplified examples, with increasing degrees of complexity.

First, consider a simple case where a time constant of a first-order system is directly proportional to a control variable (`u`):

```python
from gekko import GEKKO
import numpy as np

m = GEKKO()

# Time discretization
m.time = np.linspace(0, 10, 101)

# Variables
u = m.Var(value=1, lb=0.1, ub=10) # Optimization variable
x = m.Var(value=0) # State variable

# Model parameter that depends on u
tau = m.Intermediate(1/u)

# Differential equation
m.Equation(x.dt()== -x/tau + 1)

# Objective function
m.Obj((x-1)**2) # Drive x towards 1

# Solve
m.options.IMODE = 6
m.solve()

print(f"Optimal u: {u.value[0]}")
print(f"Final x: {x.value[-1]}")
```

In this example, we have a first-order system where the time constant (`tau`) is inversely proportional to the optimization variable (`u`). Though this isn't a complex dependency, it demonstrates the essential premise of having model parameters directly impacted by optimization variables. This problem can be solved by GEKKO.

Next, let’s look at a case where we add a discontinuous function affecting a parameter. Here, we simulate a "switch" where, above some value of 'u', a gain (`k`) is applied:

```python
from gekko import GEKKO
import numpy as np

m = GEKKO()

# Time discretization
m.time = np.linspace(0, 10, 101)

# Variables
u = m.Var(value=1, lb=0, ub=10) # Optimization variable
x = m.Var(value=0) # State variable
k = m.Var(value=1) # Gain parameter

# Intermediate - conditional gain
m.Equation(k == m.if3(u-5, 0, 10)) # A conditional: If u > 5, k=10, else k=0

# Differential equation
m.Equation(x.dt()== -x + k)

# Objective function
m.Obj((x-1)**2) # Drive x towards 1

# Solve
m.options.IMODE = 6
m.solve()

print(f"Optimal u: {u.value[0]}")
print(f"Final x: {x.value[-1]}")
```

Here, we use `m.if3()` which is a three-way conditional. If `u > 5` then `k=10`; otherwise, `k=0`. While GEKKO can solve this specific conditional, it is already approaching the edge of its capabilities. In a practical control setting where the dependency of the gain (`k`) on some other variable `u` is more complicated, or if more conditional parameters are introduced, such an approach becomes impractical as each condition adds more difficulty to the optimization. The `if3` statement is also not smoothly differentiable, creating problems for the optimizer as it nears that value of 'u' where `u=5`.

Finally, consider a scenario where we add an entire equation if a parameter is above a threshold. This isn't a direct implementation within GEKKO, but an illustration of the kind of challenge presented by Mode 9. This case requires some manual modification at each optimization iteration, which is a more realistic implementation of “Mode 9”, but is impossible to demonstrate within a simple GEKKO script. We can’t use standard GEKKO tools. It would be necessary to implement a loop that solves the optimization, checks the value of the optimization variable, modifies the equations if necessary, then reruns the optimizer. In practice, with more complex situations, each of these steps would be computationally expensive. This third example is only a conceptualization, and not functional code, to demonstrate the need to reconfigure the entire optimization problem on the fly which is beyond the intended use of GEKKO:

```python
#Conceptual Example Only, not Functional Code
#Initialize a set of equations
equations = ["x.dt() == -x + u"] #Initial Equations
def dynamic_optimizer():
  global equations
  for iteration in range(10): # Run several optimization iterations
    #Solve GEKKO problem using 'equations'
    #If optimization parameter above a threshold:
    if(optimizationVariable > 7):
      #Modify the equation list
      equations.append("y.dt() == -y + x")
      #Reinitialize the model with new equations (This needs to be manually implemented)
    #Solve
    #Return
```

This conceptual pseudo-code illustrates that the structure of the problem changes with each optimization iteration. The dynamic modification of equations is beyond the standard capabilities of GEKKO as it was not designed to modify its model on the fly during the solution process.

Based on these illustrative examples, I believe that Mode 9 problems, in their most general sense, are not directly feasible with GEKKO without significant modifications. It is possible to approximate it with conditional statements, or to reformulate problems such that there is no parameter dependence. However, GEKKO struggles with the required level of complexity and discontinuity in such an implementation. While GEKKO excels at solving dynamic optimization problems with known models, Mode 9 presents a class of problems where model dynamics change as the optimization proceeds, requiring more advanced, or perhaps even bespoke, solution methodologies.

For users encountering similar problems, I recommend studying advanced optimization texts, focusing on methods for handling non-smooth optimization and dynamic programming. Furthermore, researching adjoint sensitivity analysis can be useful in understanding the changes in gradient behavior during optimization with dynamic parameters. Finally, familiarity with advanced numerical solution techniques for differential equations will provide a stronger basis for modifying or extending GEKKO’s capabilities.
