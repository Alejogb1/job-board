---
title: "Can GEKKO find an optimal solution for a moon lander that doesn't land at zero velocity?"
date: "2025-01-26"
id: "can-gekko-find-an-optimal-solution-for-a-moon-lander-that-doesnt-land-at-zero-velocity"
---

The challenge of achieving a non-zero final velocity landing for a lunar module using GEKKO requires reframing the traditional optimal control problem. Specifically, instead of minimizing fuel consumption to reach a velocity of zero at the lunar surface, we aim to minimize fuel while reaching a *target* non-zero velocity. This necessitates adjusting the objective function and potentially the constraints within the GEKKO model. My experience with similar trajectory optimization problems suggests several key considerations.

The core issue lies in understanding that GEKKO, by its nature, seeks to minimize a defined cost function subject to given constraints. When tasked with a traditional landing, where the end velocity is zero, the model's objective is typically to minimize fuel consumption, which naturally leads to a zero-velocity end state because further consumption would be deemed unnecessary to maintain an already achieved zero-velocity state. To achieve a target, non-zero final velocity, the optimization must be incentivized to reach *that* specific velocity instead.

First, we need a clear understanding of the mathematical model for the lunar lander. This often involves a simplified model of the lander's motion, neglecting factors like aerodynamic drag (which is negligible on the moon) but accounting for gravity and thrust. Let's consider the following simplified system in one dimension (vertical motion):

*   **State variables:**
    *   `y(t)`: Altitude (position above the lunar surface)
    *   `v(t)`: Vertical velocity (positive upwards, negative downwards)
    *   `m(t)`: Mass of the lander

*   **Control variable:**
    *   `u(t)`: Thrust level (between 0 and maximum thrust)

*   **Parameters:**
    *   `g`: Lunar gravity
    *   `Tmax`: Maximum thrust
    *   `isp`: Specific impulse (linked to fuel consumption)

The system's dynamics are governed by differential equations:

1.  `dy/dt = v`
2.  `dv/dt = -g + (u * Tmax) / m`
3.  `dm/dt = -(u * Tmax) / (g * isp)`

The objective function, to minimize fuel consumption can be indirectly achieved by minimizing the time integral of thrust rate; which has a one-to-one relationship with fuel consumed. This is often easier to implement than a complex fuel model, and effectively seeks a solution with the least thrust input necessary to achieve the target conditions.

Now, we can implement this in GEKKO.

**Code Example 1: Traditional Zero-Velocity Landing (Baseline)**

```python
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# Parameters
g = 1.62  # Lunar gravity
Tmax = 10000  # Maximum thrust
isp = 300  # Specific impulse
tf = 50  # Final time
n = 51  # Number of time points

# Initial Conditions
y0 = 1000
v0 = 0
m0 = 1000

m = GEKKO()
m.time = np.linspace(0, tf, n)
y = m.Var(value=y0, name='y')
v = m.Var(value=v0, name='v')
mass = m.Var(value=m0, lb=1, name='mass')
u = m.Var(value=0, lb=0, ub=1, name='u')

m.Equation(y.dt() == v)
m.Equation(v.dt() == -g + (u * Tmax) / mass)
m.Equation(mass.dt() == -(u * Tmax) / (g * isp))

# Boundary conditions for traditional landing:
m.Equation(y[-1] == 0)  # Touchdown
m.Equation(v[-1] == 0)  # Zero velocity at touchdown
# Objective function is minimization of thrust:
m.Minimize(m.integral(u))

m.options.IMODE = 6
m.solve(disp=False)

# Plotting results (simplified)
plt.figure()
plt.subplot(3,1,1)
plt.plot(m.time, y.value)
plt.ylabel('Altitude (m)')
plt.subplot(3,1,2)
plt.plot(m.time,v.value)
plt.ylabel('Velocity (m/s)')
plt.subplot(3,1,3)
plt.plot(m.time,u.value)
plt.ylabel('Thrust (0-1)')
plt.xlabel('Time (s)')
plt.show()
```

This first example sets up a basic GEKKO model for a lunar lander. It defines the state variables (altitude, velocity, mass) and a control variable (thrust level). Boundary conditions enforce a zero altitude and zero velocity at the end of the simulation (`m.time[-1]`). The objective is to minimize the thrust integral, effectively minimizing fuel.

**Code Example 2: Achieving a Non-Zero Target Velocity**

```python
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# Parameters (same as above)
g = 1.62  # Lunar gravity
Tmax = 10000  # Maximum thrust
isp = 300  # Specific impulse
tf = 50  # Final time
n = 51  # Number of time points

# Initial Conditions
y0 = 1000
v0 = 0
m0 = 1000

target_v = -2  # Target velocity (negative = downward)

m = GEKKO()
m.time = np.linspace(0, tf, n)
y = m.Var(value=y0, name='y')
v = m.Var(value=v0, name='v')
mass = m.Var(value=m0, lb=1, name='mass')
u = m.Var(value=0, lb=0, ub=1, name='u')

m.Equation(y.dt() == v)
m.Equation(v.dt() == -g + (u * Tmax) / mass)
m.Equation(mass.dt() == -(u * Tmax) / (g * isp))

# Boundary conditions modified:
m.Equation(y[-1] == 0)  # Touchdown
m.Equation(v[-1] == target_v)  # Non-zero velocity at touchdown

m.Minimize(m.integral(u))

m.options.IMODE = 6
m.solve(disp=False)

# Plotting results (simplified)
plt.figure()
plt.subplot(3,1,1)
plt.plot(m.time, y.value)
plt.ylabel('Altitude (m)')
plt.subplot(3,1,2)
plt.plot(m.time,v.value)
plt.ylabel('Velocity (m/s)')
plt.subplot(3,1,3)
plt.plot(m.time,u.value)
plt.ylabel('Thrust (0-1)')
plt.xlabel('Time (s)')
plt.show()
```

Here, we've modified the boundary condition by setting `m.Equation(v[-1] == target_v)`, where `target_v` is a non-zero, negative value. The optimization process will now seek to minimize fuel consumption, but *while* hitting that desired final velocity. This example shows that the core of modification lies in adjusting terminal constraints.

**Code Example 3: Exploring a Range of Velocities**

```python
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# Parameters (same as above)
g = 1.62
Tmax = 10000
isp = 300
tf = 50
n = 51

# Initial Conditions
y0 = 1000
v0 = 0
m0 = 1000

target_velocities = [-1,-2,-3,-4] # Range of target velocities

plt.figure()
for idx,target_v in enumerate(target_velocities):
  m = GEKKO()
  m.time = np.linspace(0, tf, n)
  y = m.Var(value=y0, name='y')
  v = m.Var(value=v0, name='v')
  mass = m.Var(value=m0, lb=1, name='mass')
  u = m.Var(value=0, lb=0, ub=1, name='u')

  m.Equation(y.dt() == v)
  m.Equation(v.dt() == -g + (u * Tmax) / mass)
  m.Equation(mass.dt() == -(u * Tmax) / (g * isp))

  m.Equation(y[-1] == 0)
  m.Equation(v[-1] == target_v)

  m.Minimize(m.integral(u))

  m.options.IMODE = 6
  m.solve(disp=False)
  plt.subplot(len(target_velocities),3,idx*3+1)
  plt.plot(m.time, y.value)
  plt.ylabel('Altitude (m)')
  plt.subplot(len(target_velocities),3,idx*3+2)
  plt.plot(m.time,v.value)
  plt.ylabel('Velocity (m/s)')
  plt.subplot(len(target_velocities),3,idx*3+3)
  plt.plot(m.time,u.value)
  plt.ylabel('Thrust (0-1)')
  plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()
```

This example expands the concept by iterating through several target velocities. This highlights that changing the required final velocity directly impacts the trajectory. As the target velocity increases in magnitude, more thrust is required for deceleration closer to the ground and the overall time to achieve the same altitude is reduced. This allows for better sensitivity analysis.

For further learning, I recommend exploring the following resources. Regarding textbooks, consider works on Optimal Control Theory, especially those covering numerical methods for solving trajectory optimization problems.  Texts on aerospace engineering often provide specific insights into modeling lander dynamics. Additionally, research papers focusing on trajectory optimization for spacecraft landing can offer different perspectives and advanced techniques. For practical exploration, consider documentation and tutorials on the GEKKO package, specifically on dynamic optimization and constraint programming, as well as other optimization libraries. Finally, studying case studies of similar control problems, in particular those involving non-trivial boundary conditions, will offer further experience.
