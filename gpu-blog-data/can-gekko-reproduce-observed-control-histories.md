---
title: "Can Gekko reproduce observed control histories?"
date: "2025-01-30"
id: "can-gekko-reproduce-observed-control-histories"
---
The accurate reproduction of observed control histories is a critical, and often challenging, aspect of validating dynamic process models using tools like Gekko. While Gekko is capable of solving dynamic optimization problems, achieving a perfect match between model predictions and real-world control data is rarely straightforward. Factors such as model fidelity, measurement noise, and unmodeled disturbances introduce discrepancies that must be carefully addressed. Through my work with process control simulations and optimization, I've encountered situations where replicating observed behaviors requires nuanced handling of both the control scheme and the inherent limitations of the modeling framework.

Gekko, at its core, relies on a discretized time representation of continuous systems. This numerical approximation means that the solver will only approach the observed control trajectory, not precisely replicate it unless the model perfectly captures the underlying dynamics and all external influences. Reproducing control histories is thus less about mimicking the exact values, and more about accurately characterizing the system’s response to a particular control strategy. When I have attempted to match existing control data using Gekko, I've found it helpful to distinguish between two main approaches: parameter estimation and explicit control implementation. In parameter estimation, the goal is to adjust model parameters to make the model's open-loop behavior consistent with observed data under a known control history. In explicit control implementation, the control policy is encoded as part of the model and evaluated directly using historical inputs to evaluate if it produces output data that reflects observed data.

In parameter estimation, an objective function typically penalizes deviations between the model's predictions and the measured outputs, while also ensuring the model parameters remain reasonable. This approach does not force the model to exactly match observed data on a time-point-by-time-point basis, rather it tries to identify parameters which minimize error over the entire historical dataset. It attempts to find a balance between fidelity to the data and a realistic physical model, where discrepancies may also be caused by disturbances not captured in the model.

Explicit control implementation attempts to simulate the real-world control loop as closely as possible. This usually involves setting known historical control values into the Gekko model as inputs and then comparing the simulation results to the recorded process response. Any deviation implies discrepancies between model behavior and reality. This strategy can also serve as a diagnostic tool; for example, if a model accurately replicates the open-loop response but struggles when a controller is introduced, this may suggest deficiencies in our understanding of the actual control algorithm in use, or possible errors in the controller’s tuning parameters. Furthermore, if we can accurately reproduce the observed control data using the controller simulation, it then demonstrates the validity of the model as a testbed to evaluate control performance.

The level of success in replicating control histories hinges significantly on the accuracy of the model. Over-simplified models cannot be expected to exhibit complex behavior, and high precision should be reserved for areas of interest. When I have encountered this difficulty, it has proven useful to explore various levels of model detail and to prioritize areas where accuracy is most critical.

Here are a few code examples with detailed explanations:

**Example 1: Simple Parameter Estimation**

This example demonstrates basic parameter estimation for a first-order system, where the aim is to match model predictions to a time series of recorded data. This assumes the control history has been recorded, and is known.

```python
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# Time points and observed data.
t = np.linspace(0, 10, 51)
y_observed = np.array([0.000, 0.121, 0.224, 0.310, 0.379, 0.435, 0.480, 0.514, 0.539, 0.558,
                  0.573, 0.583, 0.591, 0.596, 0.599, 0.600, 0.600, 0.599, 0.596, 0.591,
                  0.583, 0.573, 0.558, 0.539, 0.514, 0.480, 0.435, 0.379, 0.310, 0.224,
                  0.121, 0.000, -0.121, -0.224, -0.310, -0.379, -0.435, -0.480, -0.514,
                 -0.539, -0.558, -0.573, -0.583, -0.591, -0.596, -0.599, -0.600, -0.600,
                 -0.599, -0.596, -0.591])

# Initial guess for the parameter.
tau_guess = 5.0

# Initialize Gekko model.
m = GEKKO(remote=False)

# Time array
m.time = t

# Model parameter to be optimized
tau = m.FV(value=tau_guess)
tau.STATUS = 1 # Allow parameter to be adjusted
u = m.Param(value=1.0)

# State Variable
y = m.Var(value=0.0)

# Differential Equation
m.Equation(tau * y.dt() + y == u)

# Objective function - Squared error with observed data.
m.Obj((y-y_observed)**2)

# Solver options
m.options.IMODE=5 # Dynamic Estimation
m.options.SOLVER = 3 #IPOPT
m.solve(disp=False)

# Plot results
plt.plot(t,y.value, label='Model Prediction')
plt.plot(t, y_observed, 'ro', label='Observed Data')
plt.legend()
plt.xlabel('Time')
plt.ylabel('y')
plt.show()

print(f"Estimated time constant: {tau.value[0]:.3f}")
```

In this example, the time constant (`tau`) of a first-order system is adjusted to match the observed output (`y_observed`). The `m.Obj` function defines the error, which is the squared difference between the model’s predictions and the observed values at each time point. `m.FV` is a manipulated variable, which may be adjusted by the solver. By adjusting `tau`, the model attempts to best reproduce the observed data. This is an example of parameter estimation, where we are not explicitly inputting a control trajectory, but are rather adjusting model parameters to match observed responses.

**Example 2: Explicit Control Input**

This example shows how to directly use recorded control data to drive a model. This may be useful to directly evaluate how a model performs against a specific control trajectory.

```python
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# Time points and observed data.
t = np.linspace(0, 10, 51)
u_observed = np.array([0.000, 0.121, 0.224, 0.310, 0.379, 0.435, 0.480, 0.514, 0.539, 0.558,
                  0.573, 0.583, 0.591, 0.596, 0.599, 0.600, 0.600, 0.599, 0.596, 0.591,
                  0.583, 0.573, 0.558, 0.539, 0.514, 0.480, 0.435, 0.379, 0.310, 0.224,
                  0.121, 0.000, -0.121, -0.224, -0.310, -0.379, -0.435, -0.480, -0.514,
                 -0.539, -0.558, -0.573, -0.583, -0.591, -0.596, -0.599, -0.600, -0.600,
                 -0.599, -0.596, -0.591])

# Initial guess for the model parameter.
tau = 3.0

# Initialize Gekko model.
m = GEKKO(remote=False)

# Time array
m.time = t

# Model parameter.
tau = m.Param(value=tau)

# Manipulated Variable, defined as historical data
u = m.Param(value=u_observed)

# State Variable
y = m.Var(value=0.0)

# Differential Equation
m.Equation(tau * y.dt() + y == u)

# Solver options
m.options.IMODE=4 # Simulation
m.solve(disp=False)

# Plot results
plt.plot(t,y.value, label='Model Prediction')
plt.xlabel('Time')
plt.ylabel('y')
plt.plot(t, u_observed, 'r-', label="Observed Control Data")
plt.legend()
plt.show()
```

In this example, `u_observed` represents the recorded control input, which is defined as a `Param` within Gekko. The simulation is then performed using this known input trajectory. The model output, `y` is then evaluated.  Deviations between the model's output and the observed process response can be used to diagnose inaccuracies in the model's parameter (`tau`) or structural assumptions.

**Example 3: Implementing a Control Algorithm**

This example simulates a simple proportional control algorithm to see if the model’s behavior matches observations.

```python
from gekko import GEKKO
import numpy as np
import matplotlib.pyplot as plt

# Time points
t = np.linspace(0, 10, 51)

# Initial guess for the model parameter.
tau = 3.0
Kc = 0.5 # Controller gain

# Initialize Gekko model.
m = GEKKO(remote=False)

# Time array
m.time = t

# Model parameter.
tau = m.Param(value=tau)
Kc = m.Param(value=Kc)

# Manipulated Variable
u = m.Var(value=0.0)

# State Variable
y = m.Var(value=0.0)
y_sp = m.Param(value=0.6 * np.ones_like(t))

# Controller logic
m.Equation(u == Kc*(y_sp - y))
m.Equation(tau * y.dt() + y == u)

# Solver options
m.options.IMODE=4
m.solve(disp=False)

# Plot results
plt.plot(t,y.value, label='Model Output')
plt.plot(t, u.value, 'r--', label='Control Effort')
plt.plot(t, y_sp.value, 'k--', label="Setpoint")
plt.legend()
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
```

In this example, the control input `u` is not directly provided, rather it is determined by a proportional control law based on the error between the model’s state variable `y` and a setpoint `y_sp`.  The control effort calculated by this equation is used to drive the underlying dynamics of the model.  Comparing the simulated control action `u` and resulting state `y` to equivalent recorded data is used to evaluate model performance. This can help to identify if the model reproduces the closed loop dynamics of the system.

Regarding further resources, several publications provide comprehensive treatments on system identification and dynamic modeling. "System Identification" by Ljung and "Process Dynamics and Control" by Seborg et al. are excellent references for the theoretical background. Additionally, documentation for the Python libraries like `scipy.optimize` and `scikit-learn` can provide valuable insights into parameter estimation techniques. Practical experience is often the best teacher; applying these techniques to different process problems will develop familiarity with the challenges and approaches.
