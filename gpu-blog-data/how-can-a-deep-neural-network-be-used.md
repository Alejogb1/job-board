---
title: "How can a deep neural network be used to solve a partial differential equation?"
date: "2025-01-30"
id: "how-can-a-deep-neural-network-be-used"
---
Partial differential equations (PDEs) often describe complex physical phenomena, and obtaining analytical solutions is frequently impossible. Over the past few years, I've explored using deep neural networks (DNNs) as function approximators to circumvent this analytical bottleneck, finding they offer a powerful, albeit nuanced, alternative to traditional numerical methods. The core idea revolves around representing the solution to the PDE as a neural network, then training this network to satisfy both the governing equation and the associated boundary conditions.

The primary strategy, which I have employed in multiple projects involving fluid dynamics and heat transfer simulations, is to formulate the problem as a minimization task. Instead of directly solving the PDE, we minimize a loss function that quantifies how well the neural network approximation adheres to the PDE and its boundary conditions. This requires calculating the derivatives of the neural network's output with respect to its inputs, which is readily achievable via automatic differentiation (AD) commonly provided by deep learning frameworks. The loss function typically has two main components: the residual loss and the boundary loss.

The residual loss, *L<sub>r</sub>*, measures the violation of the PDE within the domain. I construct this by feeding a sample of points within the domain into the neural network. I then compute the required derivatives of the network’s output (representing the approximated solution) using AD. These derivatives are substituted into the PDE. The result is ideally zero if the neural network exactly solves the PDE. The deviation from zero represents the loss. Formally, if the PDE can be expressed as F(x, ∂u/∂x, ∂²u/∂x², ...)=0 where *u* is the solution function and *x* is the domain variable, *L<sub>r</sub>*  is defined as  E[|F(x, ∂u/∂x, ∂²u/∂x², ...)|²], where E denotes the expectation over a set of training points *x*.

The boundary loss, *L<sub>b</sub>*, enforces the boundary conditions. This loss is calculated by evaluating the neural network’s output at locations on the boundary. This result is compared to the known boundary values. I've commonly used a squared error loss function, effectively penalizing deviations from the boundary values. For example, if *u(x<sub>b</sub>) = g(x<sub>b</sub>)* represents a boundary condition, where x<sub>b</sub> is a point on the boundary and *g(x<sub>b</sub>)*  is the known boundary value, then *L<sub>b</sub>* can be expressed as E[|u(x<sub>b</sub>) - g(x<sub>b</sub>)|²], where E is the expectation over a set of boundary points *x<sub>b</sub>*.

The total loss is the sum of these two losses, with weighting factors *λ<sub>r</sub>* and *λ<sub>b</sub>* controlling the relative emphasis on satisfying the PDE and boundary conditions, respectively: *L<sub>total</sub> = λ<sub>r</sub>L<sub>r</sub> + λ<sub>b</sub>L<sub>b</sub>*. The training process involves optimizing the weights of the neural network using gradient-based optimization techniques (e.g., Adam) to minimize this total loss. The trained network becomes a functional representation of the solution to the PDE.

Here are three specific examples using Python and PyTorch to illustrate different aspects of this process:

**Example 1: 1D Poisson Equation with Dirichlet Boundary Conditions**

Consider the 1D Poisson equation: -d²u/dx² = f(x), with boundary conditions u(0)=0 and u(1)=0. Assume a source term, f(x) = π²sin(πx), whose analytical solution is u(x) = sin(πx). This simple case will serve as a demonstration of the process.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

def poisson_residual(net, x):
  u = net(x)
  du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
  d2u_dx2 = torch.autograd.grad(du_dx, x, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0]
  f = torch.pi**2 * torch.sin(torch.pi * x)
  return -d2u_dx2 - f

def train(net, epochs, learning_rate, domain_pts, boundary_pts):
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        residual_loss = torch.mean(poisson_residual(net, domain_pts)**2)
        boundary_loss = torch.mean(net(boundary_pts)**2)
        total_loss = residual_loss + boundary_loss
        total_loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
          print(f"Epoch: {epoch}, Loss: {total_loss.item()}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = NeuralNet(20).to(device)
    domain_pts = torch.rand(1000, 1, requires_grad=True, device=device)
    boundary_pts = torch.tensor([[0.0], [1.0]], requires_grad=True, device=device)

    train(net, 2000, 0.001, domain_pts, boundary_pts)

    # Test and visualize (omitted)
```

This example sets up a basic neural network to approximate the solution to the Poisson equation. The `poisson_residual` function calculates the loss based on the PDE. The boundary loss forces the solution towards zero at x=0 and x=1. The `train` function carries out the training loop, applying gradient descent to optimize the network.

**Example 2: 2D Laplace Equation with Mixed Boundary Conditions**

Consider the Laplace equation ∇²u = ∂²u/∂x² + ∂²u/∂y² = 0 over the square domain [0,1]x[0,1]. Suppose we have a Dirichlet condition u(x,0) = sin(πx) on the bottom boundary, and Neumann boundary conditions ∂u/∂n = 0 on the other three edges.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x):
      x = torch.tanh(self.fc1(x))
      x = torch.tanh(self.fc2(x))
      x = self.fc3(x)
      return x


def laplace_residual(net, xy):
    u = net(xy)
    du_dxy = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u_dx2 = torch.autograd.grad(du_dxy[:, 0], xy, grad_outputs=torch.ones_like(du_dxy[:, 0]), create_graph=True)[0][:, 0]
    d2u_dy2 = torch.autograd.grad(du_dxy[:, 1], xy, grad_outputs=torch.ones_like(du_dxy[:, 1]), create_graph=True)[0][:, 1]
    return d2u_dx2 + d2u_dy2

def train(net, epochs, learning_rate, domain_pts, boundary_pts):
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        residual_loss = torch.mean(laplace_residual(net, domain_pts)**2)
        boundary_loss = torch.mean((net(boundary_pts[:,:2]) - torch.sin(torch.pi * boundary_pts[:,0:1]))**2)
        neumann_loss = 0.0
        for i in [0,1,2]:
            side = boundary_pts[boundary_pts[:,2]==i][:,:2]
            if side.shape[0] > 0:
              u = net(side)
              du_dxy = torch.autograd.grad(u, side, grad_outputs=torch.ones_like(u), create_graph=True)[0]
              if i == 0: #left boundary
                neumann_loss += torch.mean(du_dxy[:, 0]**2)
              elif i == 1: #top boundary
                neumann_loss += torch.mean(du_dxy[:, 1]**2)
              elif i == 2: #right boundary
                neumann_loss += torch.mean(du_dxy[:, 0]**2)

        total_loss = residual_loss + boundary_loss + neumann_loss
        total_loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f"Epoch: {epoch}, Loss: {total_loss.item()}")


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = NeuralNet(20).to(device)
    domain_pts = torch.rand(1000, 2, requires_grad=True, device=device)
    #boundary points are created with a flag on which edge they belong to. 0-left, 1-top, 2-right
    boundary_x = torch.rand(200, 1, device=device)
    boundary_y = torch.zeros(200, 1, device=device)
    boundary_bottom = torch.cat([boundary_x, boundary_y, torch.full((200,1), 3, device=device)], dim=1)
    boundary_x = torch.zeros(200, 1, device=device)
    boundary_y = torch.rand(200, 1, device=device)
    boundary_left = torch.cat([boundary_x, boundary_y, torch.full((200,1), 0, device=device)], dim=1)
    boundary_x = torch.ones(200, 1, device=device)
    boundary_right = torch.cat([boundary_x, boundary_y, torch.full((200,1), 2, device=device)], dim=1)
    boundary_x = torch.rand(200, 1, device=device)
    boundary_y = torch.ones(200, 1, device=device)
    boundary_top = torch.cat([boundary_x, boundary_y, torch.full((200,1), 1, device=device)], dim=1)

    boundary_pts = torch.cat([boundary_bottom, boundary_left, boundary_top, boundary_right], dim=0)
    boundary_pts.requires_grad_(True)

    train(net, 2000, 0.001, domain_pts, boundary_pts)
```

This example expands the previous concept to two dimensions, and includes mixed boundary conditions. It requires calculating the Laplacian and the gradients used for the neumann condition. Notice the use of the additional column in the `boundary_pts` tensor, which helps identify which portion of the boundary it represents.

**Example 3: Time-Dependent Heat Equation**

Consider the 1D heat equation ∂u/∂t = α∂²u/∂x² with an initial condition u(x,0)=sin(πx) and boundary conditions u(0,t) = u(1,t) = 0. Assume  α=0.01.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNet(nn.Module):
    def __init__(self, hidden_size):
      super().__init__()
      self.fc1 = nn.Linear(2, hidden_size)
      self.fc2 = nn.Linear(hidden_size, hidden_size)
      self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, xt):
      xt = torch.tanh(self.fc1(xt))
      xt = torch.tanh(self.fc2(xt))
      xt = self.fc3(xt)
      return xt


def heat_residual(net, xt, alpha):
  u = net(xt)
  du_dt = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 1:2]
  du_dx = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True)[0][:, 0:1]
  d2u_dx2 = torch.autograd.grad(du_dx, xt, grad_outputs=torch.ones_like(du_dx), create_graph=True)[0][:, 0:1]
  return du_dt - alpha * d2u_dx2

def train(net, epochs, learning_rate, domain_pts, boundary_pts, initial_pts, alpha):
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        residual_loss = torch.mean(heat_residual(net, domain_pts, alpha)**2)
        boundary_loss = torch.mean(net(boundary_pts)**2)
        initial_loss = torch.mean((net(initial_pts) - torch.sin(torch.pi*initial_pts[:, 0:1]))**2)
        total_loss = residual_loss + boundary_loss + initial_loss
        total_loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
          print(f"Epoch: {epoch}, Loss: {total_loss.item()}")

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = NeuralNet(20).to(device)
    domain_pts = torch.rand(1000, 2, requires_grad=True, device=device)
    domain_pts[:,1] = domain_pts[:,1] # times ranging from 0->1
    boundary_x = torch.tensor([[0.0], [1.0]], device=device)
    boundary_t = torch.rand(2,1, device=device)
    boundary_pts = torch.cat([boundary_x, boundary_t], dim=1)
    boundary_pts.requires_grad_(True)

    initial_x = torch.rand(1000,1, device=device)
    initial_t = torch.zeros(1000,1, device=device)
    initial_pts = torch.cat([initial_x, initial_t], dim=1)
    initial_pts.requires_grad_(True)

    alpha = 0.01
    train(net, 2000, 0.001, domain_pts, boundary_pts, initial_pts, alpha)
```

This last example extends the methodology to a time-dependent PDE. It demonstrates that our neural network's input space can encompass multiple dimensions, and we use this to encode both spatial and temporal information.  The loss function is updated to include the initial condition loss.

These examples highlight the main aspects of using deep neural networks for solving PDEs. However, this is an ongoing research field and many refinements and extensions exist, like adaptative sampling, specific network architectures and more complex loss functions.

For a deeper understanding of the underlying concepts, I recommend consulting resources on numerical analysis, specifically on finite difference and finite element methods; these methods provide the traditional foundation that the presented neural network approaches seek to improve or augment. Additionally, material focused on gradient-based optimization, automatic differentiation, and practical implementation using deep learning libraries (like PyTorch) is invaluable. Reviewing literature on physics-informed neural networks (PINNs) can further deepen the understanding of this approach for PDE solving. The documentation of the deep learning framework chosen is also crucial for effective implementation.
