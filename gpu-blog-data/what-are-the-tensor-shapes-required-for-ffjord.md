---
title: "What are the tensor shapes required for FFJORD bijectors?"
date: "2025-01-30"
id: "what-are-the-tensor-shapes-required-for-ffjord"
---
Understanding the tensor shapes involved in Free-Form Jacobian of Reversible Dynamics (FFJORD) bijectors is crucial for successful implementation and manipulation of these complex normalizing flows. My experience developing and debugging probabilistic models with FFJORD has highlighted that correct shape management prevents runtime errors and ensures the integrity of both forward and inverse transformations.

FFJORD, in its essence, is a continuous normalizing flow (CNF). This means the transformation from a base distribution, often Gaussian, to a target distribution is defined through an ordinary differential equation (ODE) parameterized by a neural network. The bijector, in this context, performs the forward and inverse mapping between the base and target spaces. Crucially, the neural network, which dictates the vector field of the ODE, must operate on tensors of specific shapes to maintain consistency in the integration process.

The primary shape consideration revolves around the input and output of the neural network used to parameterize the vector field. This network, frequently referred to as the 'drift' or 'velocity' network, typically takes as input a tensor representing the current state of the flow (at a particular time ‘t’ along the trajectory) and a tensor representing the time ‘t’. It outputs a tensor that has the same shape as the input representing the velocity field at the given point and time. This velocity field is used by an ODE solver, which iteratively computes the forward and inverse transformations.

Specifically, let's denote the input tensor of the bijector as `x`. Assuming a `d`-dimensional problem, the shape of `x` will be `(batch_size, d)`. Here, `batch_size` represents the number of samples being processed simultaneously and `d` is the dimensionality of the data being transformed. This assumes that data is in a batch. The neural network's input, denoted as `z` (after some initial linear transformation), will also have a shape of `(batch_size, d)`. Time, `t`, is given as a scalar usually scaled to (0,1). The neural network concatenates `z` with a time vector `t` and uses this as the input. This input has shape `(batch_size, d+1)`. The time variable can also be represented as a time embedding with more than one dimension. The neural network’s output must be of the same shape as the input `z`, which is `(batch_size, d)`. The ODE solver requires that the output of the neural network match the input, as it uses the vector field to update the position at each iteration. This ensures consistency in the flow transformation. It is crucial to note that the time tensor doesn’t have its own batch size and it is passed along as a single value to the neural network.

The Jacobian determinant computation, central to normalizing flow likelihood estimation, also hinges on these shape conventions. To efficiently calculate the trace of the Jacobian, which is equivalent to the log-determinant for FFJORD, a scalar value, is returned for each batch element. Therefore, the Jacobian trace term has shape `(batch_size, 1)` or `(batch_size,)` after being squeezed.

Here are three code examples demonstrating different facets of shape requirements:

**Example 1: Basic FFJORD Setup with Correct Shapes**

```python
import torch
import torch.nn as nn

class VectorField(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d + 1, 64)
        self.linear2 = nn.Linear(64, d)
    def forward(self, z_t):
        z, t = z_t[:, :-1], z_t[:, -1:]  # Split input and time
        out = torch.relu(self.linear1(torch.cat((z,t),-1)))
        out = self.linear2(out)
        return out

class FFJORDBijector(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.vector_field = VectorField(d)
        self.d = d
    def forward(self, x):
        batch_size = x.shape[0]
        t = torch.linspace(0,1,10).to(x.device) # Example time steps
        integrated_x = torch.zeros_like(x) # Initialize output tensor
        integrated_logdet = torch.zeros(batch_size, device = x.device)
        for i in range(1,len(t)):
            delta_t = t[i] - t[i-1]
            z = x.clone() # Initialize start point at x
            z.requires_grad_(True)
            d_zt = self.vector_field(torch.cat((z,t[i-1] * torch.ones(batch_size, 1).to(x.device)),-1))

            with torch.set_grad_enabled(True):
              d_zt = self.vector_field(torch.cat((z,t[i-1] * torch.ones(batch_size, 1).to(x.device)),-1))
              trace = torch.autograd.grad(d_zt.sum(),z, create_graph=True, retain_graph=True)[0].diag().sum(-1)
            integrated_x = integrated_x + delta_t* d_zt
            integrated_logdet = integrated_logdet + delta_t * trace
        return integrated_x, integrated_logdet

# Example usage
d = 2 # 2-dimensional data
batch_size = 32
x = torch.randn(batch_size, d)
bijector = FFJORDBijector(d)
transformed_x, log_det = bijector(x)
print("Shape of transformed_x:", transformed_x.shape)
print("Shape of log_det:", log_det.shape) # log_det has shape (32,)
```

In this example, the `VectorField` network takes `z` of shape `(batch_size,d)` and the scalar time `t`, and outputs the velocity field of shape `(batch_size, d)`. The `FFJORDBijector` iterates over time, using the vector field to integrate and find the transformation and computes the log-determinant. The output shapes align with the initial discussion.

**Example 2: Illustrating Incorrect Shapes in the Vector Field**

```python
import torch
import torch.nn as nn

class VectorFieldIncorrect(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d + 1, 64)
        self.linear2 = nn.Linear(64, 1)  # Incorrect output dimension
    def forward(self, z_t):
        z, t = z_t[:, :-1], z_t[:, -1:]
        out = torch.relu(self.linear1(torch.cat((z,t),-1)))
        out = self.linear2(out)
        return out

class FFJORDBijectorIncorrect(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.vector_field = VectorFieldIncorrect(d)
        self.d = d
    def forward(self, x):
        batch_size = x.shape[0]
        t = torch.linspace(0,1,10).to(x.device)
        integrated_x = torch.zeros_like(x)
        integrated_logdet = torch.zeros(batch_size, device = x.device)
        for i in range(1,len(t)):
            delta_t = t[i] - t[i-1]
            z = x.clone()
            z.requires_grad_(True)
            d_zt = self.vector_field(torch.cat((z,t[i-1] * torch.ones(batch_size, 1).to(x.device)),-1))

            with torch.set_grad_enabled(True):
              d_zt = self.vector_field(torch.cat((z,t[i-1] * torch.ones(batch_size, 1).to(x.device)),-1))
              trace = torch.autograd.grad(d_zt.sum(),z, create_graph=True, retain_graph=True)[0].diag().sum(-1)
            integrated_x = integrated_x + delta_t* d_zt
            integrated_logdet = integrated_logdet + delta_t * trace
        return integrated_x, integrated_logdet


# Example usage
d = 2
batch_size = 32
x = torch.randn(batch_size, d)
bijector = FFJORDBijectorIncorrect(d)
try:
    transformed_x, log_det = bijector(x)
except RuntimeError as e:
    print("RuntimeError:", e)
```

Here, the `VectorFieldIncorrect` outputs a tensor of shape `(batch_size, 1)` instead of `(batch_size, d)`. Consequently, the code will throw an error while integrating the vector field. This example shows that a dimension mismatch leads to a failure. This also illustrates that the shape of the output from the network must match the dimension of the input being transformed.

**Example 3: Handling Batch Dimension Properly**

```python
import torch
import torch.nn as nn

class VectorField(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.linear1 = nn.Linear(d + 1, 64)
        self.linear2 = nn.Linear(64, d)
    def forward(self, z_t):
        z, t = z_t[:, :-1], z_t[:, -1:]
        out = torch.relu(self.linear1(torch.cat((z,t),-1)))
        out = self.linear2(out)
        return out

class FFJORDBijector(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.vector_field = VectorField(d)
        self.d = d
    def forward(self, x):
        batch_size = x.shape[0]
        t = torch.linspace(0,1,10).to(x.device)
        integrated_x = torch.zeros_like(x)
        integrated_logdet = torch.zeros(batch_size, device = x.device)
        for i in range(1,len(t)):
            delta_t = t[i] - t[i-1]
            z = x.clone()
            z.requires_grad_(True)
            d_zt = self.vector_field(torch.cat((z,t[i-1] * torch.ones(batch_size, 1).to(x.device)),-1))

            with torch.set_grad_enabled(True):
              d_zt = self.vector_field(torch.cat((z,t[i-1] * torch.ones(batch_size, 1).to(x.device)),-1))
              trace = torch.autograd.grad(d_zt.sum(),z, create_graph=True, retain_graph=True)[0].diag().sum(-1)
            integrated_x = integrated_x + delta_t* d_zt
            integrated_logdet = integrated_logdet + delta_t * trace
        return integrated_x, integrated_logdet


# Example usage
d = 2
batch_size_1 = 32
batch_size_2 = 64
x1 = torch.randn(batch_size_1, d)
x2 = torch.randn(batch_size_2, d)
bijector = FFJORDBijector(d)
transformed_x1, log_det1 = bijector(x1)
transformed_x2, log_det2 = bijector(x2)
print("Shape of transformed_x1:", transformed_x1.shape)
print("Shape of log_det1:", log_det1.shape)
print("Shape of transformed_x2:", transformed_x2.shape)
print("Shape of log_det2:", log_det2.shape)
```

This example shows the flexibility of the FFJORD. It can handle different batch sizes. The shapes of the tensors are correctly handled, irrespective of batch size, because of the use of batch dimension `0`. The output of the vector field always matches the shape of the input.

For further exploration and a deeper understanding, I would recommend reviewing resources that specifically address neural ordinary differential equations, continuous normalizing flows, and automatic differentiation techniques as applied to Jacobian computation. These areas provide the necessary theoretical background for effectively using FFJORD bijectors.  It is useful to also study examples that show implementations of FFJORD in Pytorch and TensorFlow.
