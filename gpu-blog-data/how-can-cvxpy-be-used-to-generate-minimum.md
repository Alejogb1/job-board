---
title: "How can CVXPY be used to generate minimum jerk trajectories?"
date: "2025-01-30"
id: "how-can-cvxpy-be-used-to-generate-minimum"
---
Minimum jerk trajectory generation, fundamental in robotics and animation, lends itself well to convex optimization given its inherent mathematical formulation. My experience in developing motion planning for a 7-DOF manipulator highlighted the efficiency and flexibility CVXPY provides in solving these problems. The core idea rests on minimizing the integral of the squared third derivative (jerk) of the position with respect to time, subject to constraints on position, velocity, and acceleration. This allows us to produce smooth, natural-looking movements.

The fundamental principle behind minimum jerk trajectory generation lies in finding a function, typically representing position over time, that minimizes the integral of its squared jerk (the rate of change of acceleration). Mathematically, given a time interval from *t₀* to *tf*, position *p(t)*, velocity *v(t)* (the first derivative of *p(t)*), acceleration *a(t)* (the second derivative), and jerk *j(t)* (the third derivative), we aim to minimize:

∫<sub>*t₀*</sub><sup>*tf*</sup> *j(t)*<sup>2</sup> dt

subject to initial and final conditions on position, velocity, and sometimes acceleration. The direct solution to this is often analytically difficult, especially with added constraints. CVXPY, a Python-embedded modeling language for convex optimization, provides a robust framework for expressing and solving such problems numerically. We formulate the problem as a constrained optimization, where decision variables are typically the coefficients of a polynomial that describes the trajectory, or the discrete position, velocity, and acceleration at a series of time points.

To effectively use CVXPY, we discretize the time domain into *N* equally spaced points, *t₁, t₂, ..., tₙ*, and represent the position, velocity, and acceleration at each point as variables. The jerk is then approximated using finite differences. Minimizing the sum of the squared differences in acceleration becomes equivalent to minimizing the integral of the squared jerk.

Let's illustrate this with a concrete example. Imagine a simple one-dimensional movement where we want to transition a manipulator's end-effector from an initial position, velocity, and acceleration to a final position, velocity, and acceleration. We use a fifth-order polynomial, the lowest order for achieving independent constraints on position, velocity, and acceleration at two time points.
```python
import cvxpy as cp
import numpy as np

def generate_min_jerk_trajectory_polynomial(t0, tf, p0, v0, a0, pf, vf, af, n_points):
    """Generates minimum jerk trajectory using polynomial coefficients."""
    T = tf - t0
    t_vals = np.linspace(t0, tf, n_points)

    # Decision Variables: Polynomial coefficients
    a = cp.Variable((6))

    # Time vector for constraint construction
    T_matrix = np.array([
        [1, t0, t0**2, t0**3, t0**4, t0**5],  # position
        [0, 1, 2*t0, 3*t0**2, 4*t0**3, 5*t0**4],  # velocity
        [0, 0, 2, 6*t0, 12*t0**2, 20*t0**3],  # acceleration
        [1, tf, tf**2, tf**3, tf**4, tf**5],  # position
        [0, 1, 2*tf, 3*tf**2, 4*tf**3, 5*tf**4],  # velocity
        [0, 0, 2, 6*tf, 12*tf**2, 20*tf**3]  # acceleration
    ])

    # Constraints: Initial and final states
    constraints = [
      T_matrix @ a == np.array([p0, v0, a0, pf, vf, af])
      ]

    # Objective: Minimize sum of squared jerk
    Jerk_Matrix = np.zeros((n_points, 6))
    for i, t in enumerate(t_vals):
        Jerk_Matrix[i] = [0,0,0, 6, 24*t, 60*t**2] # the third derivative of a polynomial

    objective = cp.Minimize(cp.sum_squares(Jerk_Matrix @ a))
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    p_vals = np.array([sum(a.value * np.array([1,t,t**2,t**3,t**4,t**5])) for t in t_vals])
    v_vals = np.array([sum(a.value * np.array([0,1,2*t,3*t**2,4*t**3,5*t**4])) for t in t_vals])
    a_vals = np.array([sum(a.value * np.array([0,0,2,6*t,12*t**2,20*t**3])) for t in t_vals])

    return t_vals, p_vals, v_vals, a_vals

# Example usage
t0, tf = 0, 10
p0, v0, a0 = 0, 0, 0
pf, vf, af = 5, 0, 0
n_points = 100

time, position, velocity, acceleration = generate_min_jerk_trajectory_polynomial(t0, tf, p0, v0, a0, pf, vf, af, n_points)

# Print or visualize these result
# print (f"Time:{time}")
# print (f"Position:{position}")
# print (f"Velocity:{velocity}")
# print (f"Acceleration:{acceleration}")

```
In this example, the code formulates the minimum jerk problem by parameterizing the trajectory with a fifth order polynomial. We declare the polynomial coefficients as decision variables. We constrain the values of the position, velocity and acceleration at the initial and final times through linear constraints. We then minimize the sum of squares of the (discrete) jerk values, obtained by using the third derivative of the polynomial evaluated over the time domain.

A second approach uses discrete variables for position, velocity and acceleration at the time sample points. We approximate the derivatives using finite difference approximations.
```python
import cvxpy as cp
import numpy as np

def generate_min_jerk_trajectory_discrete(t0, tf, p0, v0, a0, pf, vf, af, n_points):
  """Generates minimum jerk trajectory using discrete position, velocity, acceleration variables."""
  T = tf - t0
  t_vals = np.linspace(t0, tf, n_points)
  dt = T/(n_points -1)

  # Decision Variables: Position, velocity, and acceleration at each time step
  p = cp.Variable((n_points))
  v = cp.Variable((n_points))
  a = cp.Variable((n_points))

  # Finite differences to calculate velocity and acceleration and jerk
  j = cp.Variable(n_points-2)


  # Constraints: Initial and final states
  constraints = [
      p[0] == p0,
      v[0] == v0,
      a[0] == a0,
      p[-1] == pf,
      v[-1] == vf,
      a[-1] == af,
      v[1:] == v[:-1] + dt * a[:-1],  # forward Euler
      p[1:] == p[:-1] + dt * v[:-1],   # forward Euler
      j == (a[2:] - 2*a[1:-1] + a[:-2])/ dt   # approximation of discrete jerk
  ]

  # Objective: Minimize sum of squared jerk
  objective = cp.Minimize(cp.sum_squares(j))

  problem = cp.Problem(objective, constraints)
  problem.solve()

  return t_vals, p.value, v.value, a.value


# Example usage
t0, tf = 0, 10
p0, v0, a0 = 0, 0, 0
pf, vf, af = 5, 0, 0
n_points = 100

time, position, velocity, acceleration = generate_min_jerk_trajectory_discrete(t0, tf, p0, v0, a0, pf, vf, af, n_points)

# Print or visualize these result
# print (f"Time:{time}")
# print (f"Position:{position}")
# print (f"Velocity:{velocity}")
# print (f"Acceleration:{acceleration}")
```
In this instance, the code utilizes position, velocity and acceleration as discrete variables. Instead of polynomial coefficients, the trajectories are described as a sequence of these state variables, connected by finite difference equations. The jerk is then formulated using finite difference approximations of the acceleration's changes.

Lastly, an example shows how to include intermediate waypoints that act as both constraints and variables.
```python
import cvxpy as cp
import numpy as np

def generate_min_jerk_trajectory_waypoints(t0, tf, p0, v0, a0, waypoints, vf, af, n_points):
  """Generates minimum jerk trajectory with intermediate waypoints."""
  T = tf - t0
  n_waypoints = len(waypoints)
  t_vals = np.linspace(t0, tf, n_points)
  dt = T/(n_points -1)

  # Decision Variables: Position, velocity, and acceleration at each time step
  p = cp.Variable((n_points))
  v = cp.Variable((n_points))
  a = cp.Variable((n_points))

  # Finite differences to calculate velocity and acceleration and jerk
  j = cp.Variable(n_points-2)

  # Constraint: initial condition
  constraints = [
      p[0] == p0,
      v[0] == v0,
      a[0] == a0,
      v[-1] == vf,
      a[-1] == af,
      v[1:] == v[:-1] + dt * a[:-1],  # forward Euler
      p[1:] == p[:-1] + dt * v[:-1],   # forward Euler
      j == (a[2:] - 2*a[1:-1] + a[:-2])/ dt   # approximation of discrete jerk
  ]

  # Intermediate Waypoint constraint:
  waypoint_indices = np.linspace(0,n_points-1, n_waypoints + 2, dtype=int)[1:-1]
  constraints += [p[idx] == val for idx, val in zip(waypoint_indices, waypoints)]


  # Objective: Minimize sum of squared jerk
  objective = cp.Minimize(cp.sum_squares(j))

  problem = cp.Problem(objective, constraints)
  problem.solve()

  return t_vals, p.value, v.value, a.value


# Example usage
t0, tf = 0, 10
p0, v0, a0 = 0, 0, 0
waypoints = [2,3]
vf, af = 0, 0
n_points = 100

time, position, velocity, acceleration = generate_min_jerk_trajectory_waypoints(t0, tf, p0, v0, a0, waypoints, vf, af, n_points)


# Print or visualize these result
# print (f"Time:{time}")
# print (f"Position:{position}")
# print (f"Velocity:{velocity}")
# print (f"Acceleration:{acceleration}")

```
This final example builds upon the second approach but introduces a list of intermediate waypoints. These waypoints are specified as points the trajectory should pass through at certain time instants. The code defines them by explicitly constraining the position at the corresponding time indices. It showcases how CVXPY's expressive syntax can seamlessly incorporate additional constraints.

For further study, I would recommend exploring academic papers on trajectory optimization, specifically focusing on minimum jerk approaches. Texts covering convex optimization are beneficial, particularly those addressing practical implementations. Additionally, the official CVXPY documentation provides extensive examples and details for more advanced use. Finally, research into numerical methods, especially finite difference methods for approximating derivatives, is worthwhile for deeper understanding. These resources will allow one to construct more complex and tailored trajectory generation algorithms.
