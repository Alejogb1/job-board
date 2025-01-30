---
title: "How accurate are solutions to differential equations obtained using DeepXDE?"
date: "2025-01-30"
id: "how-accurate-are-solutions-to-differential-equations-obtained"
---
The accuracy of solutions to differential equations obtained using DeepXDE is fundamentally tied to the chosen neural network architecture, the training data provided, and the specific nuances of the differential equation itself; they are, by their nature, approximate solutions. Having spent considerable time developing physics-informed neural networks (PINNs) with DeepXDE for various fluid dynamics problems, I've observed that while the framework offers a powerful methodology for tackling complex systems, achieving high accuracy requires careful consideration of these interconnected factors, and cannot be assumed.

DeepXDE, in essence, frames the solution of differential equations as an optimization problem. Rather than directly solving the equation analytically or numerically through methods like finite difference or finite element, it leverages neural networks to learn the function that satisfies the given equation. This approach inherently introduces approximation errors. The neural network, with its finite number of parameters, can only represent a function within a limited space. Further, training such a network relies on numerical optimization algorithms that converge to a minimum, but this minimum isn’t guaranteed to be a global one representing the true solution.

The primary source of approximation stems from the neural network architecture. The depth and width of the network determine its capacity to learn complex patterns. A network that is too shallow or narrow may lack the representational power to capture the intricacies of the solution, resulting in significant errors. Conversely, a network that is excessively large can lead to overfitting, capturing noise in the training data and failing to generalize well to the entire solution domain. Furthermore, the choice of activation functions within the network influences the type of functions it can approximate. Common activation functions like ReLU might struggle with highly oscillatory solutions or those with sharp discontinuities. This necessitates architectural experimentation to identify networks suitable for the specific problem.

The training process itself contributes substantially to the solution's accuracy. PINNs, like those constructed with DeepXDE, minimize a loss function that comprises two main components: the residual of the differential equation and boundary conditions. The accuracy hinges heavily on the number and distribution of training points within the domain and along the boundaries. A sparse training dataset can lead to inaccurate solutions, particularly in regions far from the training points. Moreover, the formulation of the loss function is crucial. Balancing the loss terms associated with the differential equation and the boundary conditions is not trivial and requires careful tuning. An improperly weighted loss function can result in a solution that is correct in some regions but highly inaccurate in others.

DeepXDE uses automatic differentiation to calculate derivatives needed in the differential equations, which introduces a minor numerical error, depending on the underlying library used (e.g., TensorFlow or PyTorch). However, this error is often negligible compared to the approximation error of the neural network itself. Further complexity is added by the nature of the differential equation, whether linear or nonlinear, and the presence of singularities or discontinuities in the solution itself. These can challenge the neural network’s ability to converge and yield accurate solutions.

Below are code examples demonstrating these complexities and potential solutions using DeepXDE:

**Example 1: The Poisson Equation with Insufficient Training Data**

```python
import deepxde as dde
import numpy as np

# Define the geometry
geom = dde.geometry.Rectangle([0, 0], [1, 1])

# Define the differential equation (Poisson)
def poisson_eq(x, u):
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    du_yy = dde.grad.hessian(u, x, i=1, j=1)
    return -(du_xx + du_yy) - np.pi**2*2*np.sin(np.pi*x[:, 0])*np.sin(np.pi*x[:, 1]) # RHS to match the sin solution

# Define the boundary conditions
def boundary(x, on_boundary):
    return on_boundary

bc = dde.DirichletBC(geom, lambda x: 0 , boundary)

#Define the domain and boundary points
data = dde.data.PDE(geom, poisson_eq, bc, num_domain = 200, num_boundary= 100)

# Define the neural network
layer_size = [2] + [32] * 3 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# Define the model
model = dde.Model(data, net)

# Compile and train
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=1000)
```
This code sets up the Poisson equation and solves it using a basic network. With *num_domain* only at 200, and three hidden layers of 32 neurons, the solution would be inaccurate especially in the center of the domain compared to the true solution, illustrating the effects of sparse training points. Increasing this number and potentially the network size will improve accuracy.

**Example 2: The Heat Equation with Time-Dependent Solution**

```python
import deepxde as dde
import numpy as np

# Define the geometry (time domain is from 0 to 1)
geom = dde.geometry.Interval(0, 1)
timedomain = dde.geometry.TimeDomain(0,1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Define the differential equation (heat equation)
def heat_eq(x, u):
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    return du_t - du_xx

# Define the boundary conditions and initial condition.
def boundary(x, on_boundary):
    return on_boundary and np.isclose(x[0], 0.0) or np.isclose(x[0], 1.0)

def initial(x, on_initial):
    return on_initial and np.isclose(x[1], 0.0)

bc = dde.DirichletBC(geomtime, lambda x: 0, boundary)
ic = dde.IC(geomtime, lambda x: np.sin(np.pi*x[:, 0:1]), initial)

#Define the domain and boundary points
data = dde.data.TimePDE(geomtime, heat_eq, [bc, ic], num_domain=500, num_initial=100, num_boundary=100)

# Define the neural network
layer_size = [2] + [32] * 4 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# Define the model
model = dde.Model(data, net)

# Compile and train
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=2000)
```
Here, we have the time-dependent heat equation. The choice of network architecture and training parameters becomes critical.  Using four layers may not be enough for more complex solutions. Also,  the performance of the network is highly sensitive to the number of points sampled across the spatial and temporal domains. The *num_domain, num_boundary, num_initial* parameters need careful calibration to capture the wave propagation behavior accurately. The initial condition and boundary conditions have to be imposed with particular care to give rise to the time dependent behavior.

**Example 3: A Nonlinear Equation with Overfitting**

```python
import deepxde as dde
import numpy as np

# Define the geometry (1D domain)
geom = dde.geometry.Interval(0, 1)

# Define the differential equation (a non-linear equation, Burgers equation)
def burger_eq(x, u):
    du_t = dde.grad.jacobian(u, x, i=0, j=1)
    du_x = dde.grad.jacobian(u, x, i=0, j=0)
    du_xx = dde.grad.hessian(u, x, i=0, j=0)
    return du_t + u*du_x - 0.1 * du_xx

# Boundary and initial condition definitions
def boundary(x, on_boundary):
    return on_boundary

def initial(x, on_initial):
    return on_initial

bc = dde.DirichletBC(geom, lambda x: 0, boundary)
ic = dde.IC(geom, lambda x: np.sin(2*np.pi*x[:, 0:1]), initial)

# Time domain definition
timedomain = dde.geometry.TimeDomain(0,1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

#Define the domain and boundary points
data = dde.data.TimePDE(geomtime, burger_eq, [bc, ic], num_domain=1000, num_initial=200, num_boundary=200)

# Define an overly complex neural network
layer_size = [2] + [64] * 6 + [1]
activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN(layer_size, activation, initializer)

# Define the model
model = dde.Model(data, net)

# Compile and train
model.compile("adam", lr=0.001)
losshistory, train_state = model.train(iterations=3000)
```
Here, we solve a nonlinear Burgers equation using an overly complex network which can potentially lead to overfitting. The high number of nodes and layers makes the network highly adaptable to the training data, potentially leading to poor generalization performance outside of the training points. This demonstrates the need to tune not only the training points, but also the model complexity itself. Techniques like regularization (L1 or L2 regularization), or dropout could improve generalization ability, although it was not demonstrated here for brevity.

**Resource Recommendations**

To enhance understanding of the concepts discussed and to improve the accuracy of DeepXDE-based solutions, I would recommend consulting resources focused on: numerical methods for partial differential equations (PDEs); foundational concepts in neural networks and deep learning, emphasizing both theoretical understanding and practical applications; and specific discussions on the nuances of physics-informed neural networks and best practices within the DeepXDE framework. Exploring publications that detail the convergence rates and error analysis for PINNs will also offer significant insight. Finally, experimenting with different network architectures, training algorithms, and loss function formulations is essential to developing a nuanced perspective on the accuracy limitations of these methods.
