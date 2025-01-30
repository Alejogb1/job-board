---
title: "How can numerical methods be used to find optimal solutions?"
date: "2025-01-30"
id: "how-can-numerical-methods-be-used-to-find"
---
The core principle of using numerical methods for optimization lies in iteratively approximating the solution to a problem where an analytical, closed-form solution is either impossible or computationally expensive to obtain. My experience developing robotics control systems has consistently brought me up against such challenges, forcing me to rely on numerical techniques extensively. The essence of optimization in this context is to find input parameters that minimize or maximize an objective function – this could be anything from minimizing energy consumption to maximizing the speed of a robotic arm.

Numerical optimization methods can be broadly categorized into gradient-based and derivative-free techniques. Gradient-based methods, such as gradient descent and its variants (e.g., Adam, RMSprop), leverage the gradient of the objective function to guide the search for an optimum. These methods are efficient when gradients can be calculated, either analytically or through numerical approximation. Derivative-free methods, like simulated annealing or genetic algorithms, are employed when gradients are unavailable or too noisy, often sacrificing efficiency for generality. I’ve found that the appropriate choice depends heavily on the specific problem and the available computational resources. For instance, during the development of a complex motion-planning algorithm, I had to use a derivative-free optimization routine, because the cost function involved collisions whose analytical derivatives were intractable, to find an initial, decent solution. Then, I would use a more efficient, gradient-based method starting from that decent solution to quickly fine-tune to a local minimum.

The implementation process typically involves these steps: first, defining the objective function and deciding whether it is a minimization or maximization problem. For maximization, it’s standard practice to negate the function so that a minimization algorithm can be used. Second, one must establish appropriate bounds or constraints on the parameters that will be optimized, often ensuring stability or physical validity. Third, you select a numerical method suited to the problem characteristics and implement it. Fourth, one must tune parameters and perform numerical convergence checks. Finally, the found solution has to be tested in the intended context to verify its quality.

Here are three examples to illustrate various aspects of optimization, based on problems I’ve faced:

**Example 1: Unconstrained Optimization with Gradient Descent**

Consider the simple problem of fitting a polynomial to a set of data points. Suppose you have a set of x and y values, and you want to determine the coefficients of a quadratic equation (y = ax² + bx + c) that best approximates the data. The objective function to be minimized here is the sum of the squared errors between the predicted y-values from the equation and the observed y-values. Gradient descent is a good method for this type of problem because the gradient is easily computed.

```python
import numpy as np

def objective_function(params, x, y):
    """Calculates the sum of squared errors for a quadratic."""
    a, b, c = params
    y_predicted = a * x**2 + b * x + c
    return np.sum((y_predicted - y)**2)

def gradient_descent(x, y, learning_rate=0.01, iterations=1000, initial_params=None):
    """Performs gradient descent to minimize the objective_function."""
    if initial_params is None:
      params = np.random.rand(3)  # Initial guess for a, b, c
    else:
      params = np.array(initial_params)
    
    for _ in range(iterations):
        gradient_a = 2 * np.sum((params[0] * x**2 + params[1] * x + params[2] - y) * x**2)
        gradient_b = 2 * np.sum((params[0] * x**2 + params[1] * x + params[2] - y) * x)
        gradient_c = 2 * np.sum(params[0] * x**2 + params[1] * x + params[2] - y)
        gradient = np.array([gradient_a, gradient_b, gradient_c])
        params = params - learning_rate * gradient
    return params

# Example data
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([3, 7, 13, 21, 31]) # approximates y=x^2 + 2x

# Execute Gradient Descent
optimized_params = gradient_descent(x_data, y_data)
print("Optimized coefficients (a, b, c):", optimized_params)
```
This code demonstrates a basic gradient descent implementation to fit a quadratic polynomial to data. The `objective_function` calculates the error, and the `gradient_descent` method iteratively adjusts the parameters. It calculates numerical approximations of the gradient by using a finite-difference approximation, which is simple to understand and implement, but can be less accurate than analytical derivatives. The method starts with a random guess for the parameters (or using a pre-defined starting point), and repeatedly updates those parameters according to the negative of the gradient. This continues until a convergence criterion is met. It should be clear, of course, that the gradient calculation can be improved by using numpy operations to make it more efficient and concise.

**Example 2: Constrained Optimization with Penalty Methods**

During the design of a robotic manipulator, I needed to optimize the joint angles for a specific end-effector position, while respecting joint limits. This is a constrained optimization problem. Penalty methods were a useful technique for converting this constrained problem to an unconstrained one. The key idea is to augment the objective function with penalty terms that increase in value as a constraint is violated.

```python
import numpy as np
from scipy.optimize import minimize

def objective_function_constrained(params, target_pos, robot_model, penalty_factor=1000):
    """Calculates a penalized error function considering joint limits."""
    
    end_effector_pos = robot_model.forward_kinematics(params) # a dummy forward kinematics model
    error = np.linalg.norm(end_effector_pos - target_pos) # euclidean distance
    
    # Penalty terms for joint limits (example: limits between 0 and pi rad)
    penalty = 0
    for param in params:
      if param < 0:
          penalty += penalty_factor * (0 - param)**2
      if param > np.pi:
          penalty += penalty_factor * (param - np.pi)**2
      
    return error + penalty

# Dummy forward kinematics robot model
class DummyRobot:
  def __init__(self, num_joints):
    self.num_joints = num_joints

  def forward_kinematics(self, joints):
    """A placeholder to make a forward kinematics computation"""
    x = sum(np.cos(joints))
    y = sum(np.sin(joints))
    return np.array([x,y])
    
# Example usage
num_joints = 3
robot = DummyRobot(num_joints)
target_position = np.array([1, 1])

# Initial random guess for the joint angles
initial_joint_angles = np.random.rand(num_joints)

# Using the minimize function from scipy with SLSQP (Sequential Least Squares Programming) method to optimize
result = minimize(objective_function_constrained, initial_joint_angles, args=(target_position, robot), method='SLSQP')
print("Optimal joint angles:", result.x)
```
Here, the `objective_function_constrained` incorporates a squared penalty for parameters going beyond a predefined range, which effectively guides the optimizer towards solutions that respect the constraints. A simple dummy robot model is used as an abstraction of real-world forward kinematic computations. Instead of manually implementing a minimization algorithm, we use SciPy's `minimize` function with the `SLSQP` method, which is adept at solving constrained non-linear optimization problems, which was not discussed yet and is outside of the scope of this response. The user of such a method typically has to tune the penalty factor to achieve a balance between the target error and constraint compliance.

**Example 3: Derivative-Free Optimization with the Nelder-Mead method**

In another instance, I was attempting to tune the parameters of a controller for a complex dynamical system. The cost function involved was a black box, meaning no closed-form representation was available. For such cases, derivative-free methods are suitable. The Nelder-Mead algorithm, a simplex method, was effective here because it did not require gradients.

```python
import numpy as np
from scipy.optimize import minimize

def black_box_objective(params, system_simulation, target_state):
  """Simulates the system and returns an error with respect to the target state."""
  # The system simulation here is a black box, so we define a dummy model.
  # Normally this would involve complex simulations and numerical integration steps.
  
  simulated_state = system_simulation.run(params) # run simulation
  return np.linalg.norm(simulated_state - target_state)

class DummySystemSimulator():
  """A dummy system simulator for testing purposes"""
  def __init__(self):
    pass
  def run(self, params):
    """Dummy simulation method"""
    state = sum(params) # return a dummy state based on the controller params
    return np.array(state)
  
# Example usage
dummy_system = DummySystemSimulator()
target_state = np.array(10)

# Initial guess for controller params
initial_params = np.random.rand(3)

# Optimize controller parameters using Nelder-Mead
result = minimize(black_box_objective, initial_params, args=(dummy_system, target_state), method='Nelder-Mead')
print("Optimal controller parameters:", result.x)
```

In this scenario, the `black_box_objective` interacts with a hypothetical `system_simulation` to evaluate the objective function. A `DummySystemSimulator` class provides a placeholder for simulation steps. Due to the lack of gradient, the `Nelder-Mead` algorithm is used by SciPy's `minimize` to search for the parameters that minimize the error. I found that these methods, although not as efficient as gradient methods, have proven useful in situations where derivatives are difficult to compute. The efficiency of derivative-free optimization also depends greatly on the starting point; however, a reasonable starting point can be obtained from experience, physical insights, or another, less efficient method.

For more information, I recommend consulting books on numerical optimization, such as "Numerical Optimization" by Nocedal and Wright, or "Practical Optimization" by Gill, Murray, and Wright. Also, the documentation provided by libraries such as SciPy (especially scipy.optimize), as well as courses on optimization methods from various universities can provide in-depth understanding and practical implementation guidelines. I’ve found the theoretical explanations and examples in these resources to be invaluable when applying these methods in my work.
