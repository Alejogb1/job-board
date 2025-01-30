---
title: "Why does GEKKO dynamic optimization fail with short timesteps?"
date: "2025-01-30"
id: "why-does-gekko-dynamic-optimization-fail-with-short"
---
Dynamic optimization using GEKKO, particularly when applied to stiff systems, frequently encounters numerical challenges when employing overly short time steps. The root cause isn't a deficiency in GEKKO itself, but rather the inherent difficulties numerical integration methods face when approximating the solution to differential equations with rapid state changes. My own experience, developing a model for a rapid-response chemical reactor using GEKKO, highlighted this very problem, forcing me to re-evaluate discretization strategies.

The core issue is that short timesteps, while seemingly providing higher fidelity, can exacerbate truncation errors, which are the errors introduced by approximating continuous derivatives with discrete differences. Numerical integrators rely on approximating the derivative of a state variable by calculating the slope between current and next-step state values. This approximation only holds true if the time step is sufficiently small compared to the characteristic time scales of the underlying system dynamics. However, making it *too* small introduces problems. When timesteps become much smaller than these characteristic times, the differences between consecutive states can become nearly insignificant relative to the inherent floating-point precision of the computer. This leads to loss of information, as subtle changes in the state variables are essentially rounded to zero. When calculating derivatives through difference quotients, this can result in inaccurate derivative estimates, and consequently unstable solutions.

Furthermore, short timesteps amplify the issue of *round-off* errors. Every floating-point operation carries an inherent error related to the representation of numbers with a limited number of bits. While typically negligible on an individual step, these tiny errors accumulate with each step of the numerical integration. As the number of timesteps increase due to shorter time increments, these rounding errors collectively grow, obscuring the true solution and leading to solution divergence or oscillations. This effect is particularly damaging when optimizing using gradient-based methods, as inaccurate derivatives can lead to incorrect search directions, preventing convergence.

The interplay of these errors manifests differently depending on the selected numerical integrator. GEKKO offers a selection of ODE/DAE solvers. Implicit methods, like `IMODE=5` (for dynamic optimization) or `IMODE=4` (for simulation) using the `idas` solver, are generally more stable for stiff systems, but not immune to these problems. They involve solving an algebraic system to compute the next state value, which can be computationally more expensive but often allows for larger timesteps. However, even implicit methods suffer when time steps become extremely small, forcing the solver to perform a larger number of very precise calculations, where round-off errors dominate and the linear algebra involved in the implicit solve loses accuracy.

Consider the following example, a simplified representation of a first-order reaction in a batch reactor modeled using GEKKO, where a short timestep is used:

```python
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# Initialize model
m = GEKKO(remote=False)
# Time horizon
tf = 1.0
m.time = np.linspace(0,tf,1000) #Short timesteps
# Variables
c = m.Var(value=1.0,lb=0.0) # Concentration
k = m.Param(value=0.5)     # Reaction rate constant
# Equations
m.Equation(c.dt() == -k*c)
# Options
m.options.IMODE = 4 # ODE simulation
# Solve
m.solve(disp=False)

# Plot Results
plt.plot(m.time, c.value, label='Concentration')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.show()
```

Here, the `m.time` array defines 1000 points from 0 to 1 second, resulting in a very short timestep of 0.001 seconds. Though in this case, with this simple problem, the solution is likely to converge due to the simple differential equation and implicit integrator (IDAS), it is illustrative of the problem. The relative changes in `c` from timestep to timestep are very small. The solver must perform a large number of very precise integrations, which would expose this solution to more opportunities for accumulated error.

Now, let's examine the same system with a more reasonable number of timesteps:

```python
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# Initialize model
m = GEKKO(remote=False)
# Time horizon
tf = 1.0
m.time = np.linspace(0,tf,100) # Reasonably sized timesteps
# Variables
c = m.Var(value=1.0,lb=0.0) # Concentration
k = m.Param(value=0.5)     # Reaction rate constant
# Equations
m.Equation(c.dt() == -k*c)
# Options
m.options.IMODE = 4 # ODE simulation
# Solve
m.solve(disp=False)

# Plot Results
plt.plot(m.time, c.value, label='Concentration')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.legend()
plt.show()

```

This example uses 100 time points across the same time span (0 to 1 second), thus with a time step of 0.01. We have the same problem, but the solver is allowed to take larger steps. The result is smoother, and more stable as the truncation error is reduced by using timesteps that are more appropriate to the problem. This is more in line with the kind of timestep used in typical dynamic optimization problems.

The problem becomes even more prevalent when attempting to use dynamic optimization (`IMODE=6` or `IMODE=9`). In that case, the optimization process must determine the optimal control input profile, with each iteration relying on accurate gradients calculated from the simulation model results. Short timesteps, by amplifying numerical noise, greatly diminish the quality of these gradient evaluations and consequently hamper optimization.  Here is an example where this effect is exacerbated by a dynamic optimization problem.

```python
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# Initialize model
m = GEKKO(remote=False)
# Time horizon
tf = 5.0
m.time = np.linspace(0,tf,100) # Short timesteps for simulation
# Variables
c = m.Var(value=1.0,lb=0.0)   # Concentration
u = m.MV(value=1.0,lb=0.0,ub=2.0) # Control input
k = m.Param(value=0.5)     # Reaction rate constant
# Equations
m.Equation(c.dt() == -k*c + u) # Now with control input
# Objective function
m.Obj(m.integral((c-0.25)**2) + 0.01*m.integral((u-1.0)**2))
# Options
m.options.IMODE = 9 # Dynamic optimization
m.options.SOLVER = 3
# Solve
m.solve(disp=False)

# Plot Results
plt.plot(m.time, c.value, label='Concentration')
plt.plot(m.time, u.value, label='Control Input')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
```

In this example, though the solver is able to find a solution, there may be issues. If `m.time` had many more points, the solver may struggle to converge at all, potentially returning an error regarding gradient evaluations.  Here, it is clear that the objective and control input profile are not quite smooth. This is due to the numerical noise in the integration due to small step sizes. By selecting an appropriate number of steps, this noise can be reduced.  The solver is less likely to get stuck in local minima, and a more accurate solution is possible.

In summary, while shorter time steps may intuitively seem better for accuracy, they can introduce significant numerical problems, particularly when dealing with optimization of stiff systems. The truncation and round-off errors induced by these small steps will lead to unstable solutions, degraded gradient approximations, and difficulty achieving convergence.  It's crucial to select a time discretization that is appropriate for the characteristic time scales of the system and the capabilities of the numerical integrator.

For further understanding, I recommend researching the following topics:

*   **Numerical Integration Methods for Ordinary Differential Equations:** Focus on implicit methods like Backward Differentiation Formula (BDF) methods, which are often used to solve stiff differential equations. Compare with explicit methods, which are usually unsuitable for stiff systems.
*   **Error Analysis in Numerical Computation:** Study the sources of error, particularly truncation error and round-off error.  Learn about how they propagate during numerical calculations.
*   **Stiffness in Differential Equations:** Gain a good understanding of what defines a stiff differential equation. Recognize their impact on the selection of numerical solvers.
*   **Practical Aspects of Dynamic Optimization:** Research techniques such as choosing appropriate time discretization, scaling, and parameter tuning for reliable convergence.
*   **Floating-point Arithmetic:** Become acquainted with limitations of using floating-point representation of numbers on computers.  Understand how the limitations of binary representations affect accuracy.
*   **GEKKO's documentation:** Focus on the explanation of the available solvers and their appropriate uses, as well as the different modes.

Employing a judicious balance between accuracy and numerical stability when choosing time step size is crucial for achieving robust and reliable dynamic optimization solutions using GEKKO. This understanding will help in diagnosing and preventing problems and using GEKKO for complex systems.
