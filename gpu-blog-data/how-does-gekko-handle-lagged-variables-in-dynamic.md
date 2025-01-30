---
title: "How does GEKKO handle lagged variables in dynamic optimization objectives?"
date: "2025-01-30"
id: "how-does-gekko-handle-lagged-variables-in-dynamic"
---
The handling of lagged variables in GEKKO’s dynamic optimization is primarily achieved through the use of differential algebraic equations (DAEs) and the inherent time discretization within its solver. Unlike static optimization where variables represent fixed values, dynamic problems involve time-dependent variables and their derivatives. GEKKO doesn't inherently implement a direct "lag" operator in the traditional sense; instead, lags are represented by the system's state variables and their historical values, accessed through indexing in the time dimension. My experience in developing control strategies for chemical reactors using GEKKO has consistently revealed this approach.

Fundamentally, GEKKO solves dynamic optimization problems by discretizing the continuous time domain into a finite number of time points. Within each time interval, the state variables are approximated by algebraic variables and the derivatives are discretized using numerical methods, often backward difference formulas (BDF). This discretization transforms the original set of DAEs into a large, sparse system of nonlinear equations. When you need to represent a variable at a previous time point, you essentially access the previously calculated state variable, facilitated by GEKKO's implicit temporal indexing. This is not a literal 'lag' operation; it's the retrieval of a state value associated with a prior time step in the discretized solution.

For instance, if you have a system where a feed flow rate affects a downstream reactor temperature after a time delay, the delayed effect is modeled using intermediate state variables and not through some isolated time delay operator. You define equations governing the rate of change of these intermediate states and their relation to the final state. Consider a simplified example of a heat exchanger where the temperature of the outgoing fluid at time *t* depends on the inlet fluid temperature at *t - τ*, where *τ* is the time delay. In GEKKO, I wouldn't directly introduce a 'lag(T_in, τ)' function. Instead, I’d utilize either a distributed-parameter model with a discretization along the spatial coordinate, or employ a series of consecutive well-mixed tanks that approximate the plug flow behavior which gives rise to a time lag.

The accuracy of the representation of lagged effects relies on the discretization resolution. A finer time grid (i.e., smaller time step) provides a better approximation of the continuous system behavior and also leads to a more accurate representation of how lagged values influence the dynamics. If the time delay is large relative to the time step, it may require using a substantial number of intermediate steps, and could increase computation time.

Here’s an example illustrating the concept using a simple dynamic model:

```python
from gekko import GEKKO
import numpy as np

# Initialize GEKKO model
m = GEKKO()

# Time horizon and steps
n = 101
tf = 10
m.time = np.linspace(0, tf, n)

# Define variables
x = m.Var(value=0, name='x')
u = m.Var(value=1, lb=0, ub=1, name='u') # control variable
lag_x = m.Var(value=0, name='lag_x')  # lagged variable

# Define parameters
tau = 2  # lag time constant

# Define time-dependent relationship
m.Equation(lag_x.dt()==(x-lag_x)/tau)  # introduce a lag via an ODE

# Model equation
m.Equation(x.dt() == -x + u)

# Objective: Minimize deviation from target value of 1.0
m.Obj(m.integral((x-1)**2))

# Solver setup
m.options.IMODE = 6 # Dynamic optimization
m.options.SOLVER = 3 # IPOPT solver
m.solve(disp=False)

print(f'Final x value: {x.value[-1]}')
```
In this script, `x` is the main state variable, influenced by a control input `u`.  The variable `lag_x` isn’t a direct lag of x; rather, it's the solution of a differential equation which results in `lag_x` being an exponentially lagged approximation of `x` with characteristic lag time of 'tau'. This example shows how a delayed effect can be introduced through a differential equation rather than accessing previous values directly with an index. This allows GEKKO to manage it within the solution process.

Let's explore a slightly more complex situation involving a time delay applied to the control variable:

```python
from gekko import GEKKO
import numpy as np

# Initialize GEKKO model
m = GEKKO()

# Time horizon and steps
n = 101
tf = 10
m.time = np.linspace(0, tf, n)

# Define variables
x = m.Var(value=0, name='x')
u = m.Var(value=1, lb=0, ub=1, name='u')
u_delayed = m.Var(value=1, name = 'u_delayed')  #Delayed control input variable

# Define parameters
tau = 2 # lag time constant

# Introduce the lag through a first-order ODE
m.Equation(tau*u_delayed.dt()== u - u_delayed)

# Model equation
m.Equation(x.dt() == -x + u_delayed)

# Objective: Minimize the deviation of x from 1
m.Obj(m.integral((x-1)**2))

# Solver setup
m.options.IMODE = 6 # Dynamic optimization
m.options.SOLVER = 3 # IPOPT solver
m.solve(disp=False)

print(f'Final x value: {x.value[-1]}')
```
Here, `u_delayed` doesn't represent a directly indexed delayed input. Instead, the differential equation ensures that `u_delayed` evolves as an approximation of the lagged value of `u`, again with a time constant of ‘tau’.  This method effectively approximates the lag of the control variable affecting the dynamics of `x`. The use of a first order ODE effectively acts as a low pass filter and introduces the lag. The time constant `tau` governs the time scale of the delay.

Finally, consider a more direct approach, using past values directly:

```python
from gekko import GEKKO
import numpy as np

# Initialize GEKKO model
m = GEKKO()

# Time horizon and steps
n = 101
tf = 10
m.time = np.linspace(0, tf, n)

# Define variables
x = m.Var(value=0, name='x')
u = m.Var(value=1, lb=0, ub=1, name='u')

# Define parameters
tau_steps = 20 # lag time in number of steps

# Initialize the delayed u values.  
u_past = np.ones(n)

# define a variable to access prior value of u for the lag
u_delayed = m.Var(value=1, name='u_delayed')

#Introduce the time lag
for i in range(tau_steps, n):
    u_past[i] = u.value[i-tau_steps]
m.Equation(u_delayed == m.Intermediate(u_past))

# Model equation
m.Equation(x.dt() == -x + u_delayed)

# Objective: Minimize the deviation of x from 1
m.Obj(m.integral((x-1)**2))

# Solver setup
m.options.IMODE = 6 # Dynamic optimization
m.options.SOLVER = 3 # IPOPT solver
m.solve(disp=False)

print(f'Final x value: {x.value[-1]}')
```

In this example, a time-shifted value of `u` is explicitly generated and assigned to `u_delayed` to introduce the lag. This approach requires manipulating a numpy array outside the optimization and passing this to an intermediate variable within GEKKO for use in the model equations. Note that the manipulation of the `u_past` array occurs prior to solving and thus the time shifting is pre-calculated and cannot be part of the optimization process. This approach is less flexible than the ODE-based approach and only useful when a fixed known delay is required.
In summary, GEKKO does not have a dedicated 'lag' operator. Instead, lagged effects are incorporated through a combination of state variables, discretized derivatives, and time indexing of past values when needed. Differential equations that relate variables using a characteristic lag time is a common way to approximate lagged variable behaviour. The choice of method depends on the nature of the lag and the desired accuracy of the solution.  

For further information on time series modeling and numerical integration techniques, I suggest reviewing resources covering the basics of numerical solution of differential equations, and the theory behind discretization methods, especially those used in DAE solvers. Additionally, reading documentation related to model building and DAE integration methods would benefit a user.
