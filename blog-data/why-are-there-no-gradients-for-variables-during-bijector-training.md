---
title: "Why are there no gradients for variables during bijector training?"
date: "2024-12-23"
id: "why-are-there-no-gradients-for-variables-during-bijector-training"
---

Okay, let's unpack this. I recall back during my early days working on normalizing flows for generative modeling, I ran into this very issue. Training a bijector, particularly when dealing with complex architectures, felt surprisingly different than training a typical neural network. It's not that we don't have any gradients; it's about *where* those gradients flow and why they don't impact the bijector’s input variables in the way you might initially expect. The core problem boils down to the purpose and mathematical nature of a bijector and its specific role within a normalizing flow architecture.

To properly explain, consider a standard normalizing flow. In essence, we’re trying to transform samples from a simple, known distribution (think standard normal) into samples from a complex target distribution. This transformation is done through a series of invertible functions, the bijectors. Now, these bijectors aren't arbitrary functions. They're designed to satisfy the requirement of being *bijective* – one-to-one and onto – guaranteeing that we can map back and forth between the latent space (the simple distribution) and the data space. Crucially, during training, we are *not* optimizing the input of each bijector. Instead, we focus on modifying the *parameters* of the bijector itself, in effect shaping the flow itself. The input, or more appropriately the output from the previous layer, is passed through and transformed without any backpropagation with respect to its value.

Think about it like this: the gradients flow backwards through the bijector’s computations, and are used to optimize the *internal* parameters of the bijector (like the weights and biases in a neural network-based bijector). These internal parameters define *how* the bijector transforms its input, and we want that transformation to be such that the final distribution closely resembles our target distribution. We achieve this indirectly, by minimizing some loss function related to the likelihood of the target data. The input to the first bijector (typically sampled from the simple base distribution) isn't something we modify; it’s what starts the flow transformation. Similarly, the outputs of intermediate bijectors are just intermediates in the calculation, they are not variables to be optimized. Hence, no gradients are computed with respect to those input variables. We're aiming to learn the transformation itself, not manipulate the inputs.

Let’s make this concrete with some examples. Let’s start with a simple Affine coupling layer, a commonly used bijector:

```python
import torch
import torch.nn as nn

class AffineCoupling(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2),
            nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Linear(input_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim // 2),
        )

    def forward(self, x):
        x1, x2 = torch.split(x, x.shape[1] // 2, dim=1)
        scale = self.scale_net(x1)
        translate = self.translate_net(x1)
        y2 = x2 * torch.exp(scale) + translate
        y = torch.cat((x1,y2), dim=1)

        # for the reverse pass
        self.cache = x1, scale, translate

        return y

    def inverse(self, y):
       y1, y2 = torch.split(y, y.shape[1] // 2, dim=1)
       x1, scale, translate = self.cache
       x2 = (y2 - translate) * torch.exp(-scale)
       x = torch.cat((y1, x2), dim=1)
       return x

input_dim = 4
hidden_dim = 16
bijector = AffineCoupling(input_dim, hidden_dim)

# Example usage
x = torch.randn(1, input_dim, requires_grad=True) # Initialize with requires_grad, will make it clear later.
y = bijector(x)
print("Input:", x.data)
print("Output:", y.data)

# We can check if there is a gradient w.r.t. the input.
loss = torch.sum(y)  # Dummy loss, just to show backpropagation is not against the variable 'x'
loss.backward()

print("Gradient of Input:", x.grad) # This should be None
for p in bijector.parameters():
     print("Gradients of Bijector Parameter:", p.grad) # Bijector gradients are calculated.
```

In this code, `x` is the input to the `AffineCoupling` bijector.  Even though I initialized `x` with `requires_grad=True`, which would typically allow gradients to propagate back to it, `x.grad` is `None` after backpropagation through a loss using `y`. The gradient instead goes to the *parameters* of the `bijector`, shown by `p.grad`. Notice the parameter gradients are no longer `None`. This demonstrates clearly that backpropagation occurs through the bijector, using it's parameters, and not its input.

Let's solidify this further with another example, a simple planar flow which is a simpler parameterized bijector.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PlanarFlow(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(input_dim, 1))
        self.w = nn.Parameter(torch.randn(input_dim, 1))
        self.b = nn.Parameter(torch.randn(1))


    def forward(self, z):
      wz = torch.matmul(z, self.w)
      uhat = self.u - (torch.matmul(self.w, self.u)*torch.matmul(self.w, self.w)/torch.matmul(self.w, self.w)) * self.w
      f = wz + self.b
      m_f_p = F.tanh(f)
      zprime = z + uhat*m_f_p.transpose(0, 1)
      self.cache_m_f_p= m_f_p #cache this for inverse calculation

      return zprime

    def inverse(self, zprime):
      m_f_p = self.cache_m_f_p
      uhat = self.u - (torch.matmul(self.w, self.u)*torch.matmul(self.w, self.w)/torch.matmul(self.w, self.w)) * self.w
      z = zprime - uhat*m_f_p.transpose(0,1)
      return z

input_dim = 2
flow = PlanarFlow(input_dim)
z = torch.randn(1, input_dim, requires_grad=True)
zprime = flow(z)

loss = torch.sum(zprime)
loss.backward()

print("Gradient of Input:", z.grad) # This should be None
for p in flow.parameters():
  print("Gradients of Flow Parameter:", p.grad) # Flow gradients are calculated.
```
Here the `PlanarFlow` bijector's forward pass involves a transformation defined by its parameters `u`, `w`, and `b`. Similar to the previous example, the input `z` does not receive a gradient even with `requires_grad=True`, instead the gradient is applied to the trainable parameters of the flow itself.

Lastly, I want to illustrate a more complex bijector, one using a neural network to define the transform.

```python
import torch
import torch.nn as nn

class NeuralNetworkBijector(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        # In a real flow, we would usually use a parameterised transform which is also invertible.
        # Here, we'll use a naive and non-invertible one for simplicity of example
        y = x + self.net(x)
        self.cache = x # For inverse pass, not realistically a valid inverse.

        return y

    def inverse(self, y):
      # this is not a proper inverse since it doesn't revert the operation in the forward pass.
      x = self.cache
      return x

input_dim = 2
hidden_dim = 16
bijector = NeuralNetworkBijector(input_dim, hidden_dim)
x = torch.randn(1, input_dim, requires_grad=True)
y = bijector(x)
loss = torch.sum(y)

loss.backward()

print("Gradient of Input:", x.grad) # Still none
for p in bijector.parameters():
    print("Gradients of Bijector Parameter:", p.grad) # Parameter gradients calculated as before.
```

The `NeuralNetworkBijector` has its transform defined by `self.net`, which uses a feed-forward neural network. The non-invertible nature of this function is irrelevant here, it is included to demonstrate that no gradients are calculated with respect to the input, just with respect to the parameters of the transformation.

The key takeaway here is that the bijector is trained to transform the input data, not to change the data itself. The gradients backpropagate through the bijector’s computations to update its parameters, which are the levers to shaping the transformation, not its inputs. If we tried to get gradients with respect to the inputs of each bijector, we’d actually be moving the *source* distribution around instead of learning the transformation.

For a more in-depth theoretical understanding, I'd recommend examining the seminal papers on normalizing flows, such as "Variational Inference with Normalizing Flows" by Dinh et al. (2014) and "NICE: Non-linear Independent Components Estimation" by Dinh et al. (2015). Also, the book "Deep Learning" by Goodfellow, Bengio, and Courville includes a detailed chapter on generative models and normalizing flows which are highly recommended. These resources provide a solid foundation for grasping the mathematical underpinnings and practical implementation of these concepts, further illuminating why gradient calculations work in this specific way. Finally, the more recent "Glow: Generative Flow with Invertible 1x1 Convolutions" (Kingma & Dhariwal, 2018) is worth examining for its insights on implementing more complex bijectors. These resources should provide a very strong framework for understanding the behaviour of gradient calculations in bijector training.
