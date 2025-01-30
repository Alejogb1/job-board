---
title: "How can a 1D heat equation be solved using PyTorch neural networks?"
date: "2025-01-30"
id: "how-can-a-1d-heat-equation-be-solved"
---
The inherent ability of neural networks to approximate complex functions, coupled with their parallel processing capabilities, makes them a suitable alternative to traditional numerical methods for solving partial differential equations (PDEs), including the 1D heat equation.  My experience with finite element methods and their limitations in handling complex geometries and boundary conditions led me to explore this approach.  Specifically, I found that employing Physics-Informed Neural Networks (PINNs) offers an elegant solution, effectively embedding the PDE directly into the loss function of the neural network.

The 1D heat equation, given by  ∂u/∂t = α ∂²u/∂x², where *u(x,t)* represents temperature, *x* represents spatial position, *t* represents time, and *α* is thermal diffusivity, describes the diffusion of heat over time.  Solving this equation analytically is often limited to simple boundary conditions and geometries.  A PINN approach circumvents these restrictions by learning the solution implicitly from the governing equation and boundary/initial conditions.

The core principle involves training a neural network to minimize the error between the network's prediction of *u(x,t)* and the actual solution, as dictated by the heat equation. This is achieved by constructing a loss function that comprises three components:

1. **PDE Loss:** This term measures the residual of the heat equation when the network's prediction is substituted.  Ideally, this residual should approach zero.

2. **Boundary Condition Loss:** This term accounts for the specified boundary conditions, ensuring the network’s predictions satisfy them.

3. **Initial Condition Loss:** This term enforces the initial temperature distribution at *t=0*.

The network is then trained using an optimizer (e.g., Adam) to minimize this composite loss function. The resulting trained network acts as an approximate solution to the 1D heat equation, providing temperature predictions at any spatial location and time within the domain of interest.

Let's consider three code examples illustrating varying complexities:


**Example 1: Simple Dirichlet Boundary Conditions**

This example demonstrates solving the heat equation with constant Dirichlet boundary conditions (fixed temperatures at both ends of the rod).

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define the neural network
class HeatEquationNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = nn.Sequential(*layers)

    def forward(self, x, t):
        input = torch.cat((x, t), dim=1)
        return self.net(input)


# Define the heat equation and boundary conditions
alpha = 0.1
def heat_equation(u, x, t):
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t - alpha * u_xx

# Training parameters
layers = [nn.Linear(2, 20), nn.Tanh(), nn.Linear(20, 20), nn.Tanh(), nn.Linear(20, 1)]
net = HeatEquationNet(layers)
optimizer = optim.Adam(net.parameters(), lr=0.001)
epochs = 10000


# Training loop (simplified for brevity)
for epoch in range(epochs):
    x = torch.rand(100, 1) * 1 #Spatial domain [0,1]
    t = torch.rand(100, 1) * 1 #Time domain [0,1]
    u = net(x,t)
    loss = torch.mean(heat_equation(u, x,t)**2) + torch.mean((u[:,0] - 0)**2) + torch.mean((u[:,-1] - 1)**2) # boundary conditions u(0,t) = 0, u(1,t) = 1
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

This code defines a simple neural network, the heat equation, and the loss function including Dirichlet boundary conditions at x=0 and x=1. The training loop iteratively minimizes the loss.  Note that this is a simplified version; a production-ready model would involve more sophisticated techniques like adaptive learning rates and more robust error handling.


**Example 2: Neumann Boundary Conditions**

This example incorporates Neumann boundary conditions (specified heat flux) instead of Dirichlet conditions.

```python
#... (Network definition and parameter setup from Example 1 remains the same) ...

# Modified heat equation function with Neumann boundary conditions
def heat_equation_neumann(u, x, t):
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    return u_t - alpha * u_xx

# Training loop modified for Neumann conditions
for epoch in range(epochs):
    x = torch.rand(100, 1) * 1
    t = torch.rand(100, 1) * 1
    u = net(x, t)
    loss = torch.mean(heat_equation_neumann(u, x, t)**2) + torch.mean((u_x[:, 0] - 0)**2) + torch.mean((u_x[:, -1] - 1)**2) #Neumann conditions: du/dx(0,t) = 0, du/dx(1,t) = 1
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

The key difference lies in how the boundary conditions are incorporated into the loss function.  Here, we calculate the spatial derivative using `torch.autograd.grad` and penalize deviations from the specified flux.

**Example 3: Incorporating an Initial Condition**

This example adds an initial temperature profile to the problem.

```python
#... (Network definition, parameter setup, and heat equation function from Example 1 or 2) ...

# Initial condition function
def initial_condition(x):
    return torch.sin(torch.pi * x)

# Modified training loop to include initial condition loss
for epoch in range(epochs):
    x = torch.rand(100, 1) * 1
    t = torch.rand(100, 1) * 1
    u = net(x, t)
    initial_x = torch.rand(100,1)
    initial_u = net(initial_x, torch.zeros(100,1)) #evaluate at t = 0
    loss = torch.mean(heat_equation(u, x, t)**2) + torch.mean((u[:, 0] - 0)**2) + torch.mean((u[:, -1] - 1)**2) + torch.mean((initial_u - initial_condition(initial_x))**2) #Added initial condition loss term
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
```

Here, we define an initial condition function (`initial_condition`) and add a term to the loss function that penalizes discrepancies between the network's prediction at *t=0* and this defined initial profile.


**Resource Recommendations:**

For deeper understanding, I recommend exploring textbooks on numerical methods for PDEs and introductory materials on neural networks and their applications in scientific computing.  Furthermore, specialized literature on Physics-Informed Neural Networks will prove invaluable.  Reviewing research articles on the application of PINNs to various PDEs will significantly broaden your understanding of this technique's versatility and limitations.  Finally,  familiarity with automatic differentiation libraries, such as Autograd (within PyTorch), is crucial.
