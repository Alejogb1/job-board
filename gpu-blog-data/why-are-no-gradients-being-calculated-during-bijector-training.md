---
title: "Why are no gradients being calculated during bijector training?"
date: "2025-01-26"
id: "why-are-no-gradients-being-calculated-during-bijector-training"
---

In my experience developing variational autoencoders (VAEs) employing normalizing flows, a frequent stumbling block arises when bijector parameters fail to update despite an end-to-end training process. This situation, where gradients appear to vanish or remain zero specifically for the bijector, typically stems from a disconnect between the computational graph involved in the loss calculation and the parameters the optimizer is attempting to adjust. The issue isn’t inherently related to the backpropagation algorithm itself, but rather how it is applied across the bijector’s operations.

The core problem lies in the requirement that a bijector's *forward* and *inverse* transformations must both be differentiable with respect to both their input *and* their own trainable parameters. When using standard deep learning frameworks, automatic differentiation calculates gradients based on a sequence of operations defined by the forward pass. If these operations do not explicitly register the dependency of the final loss on the bijector’s *trainable* parameters, the gradients for those parameters will be zero. This situation arises from a variety of implementation oversights, most commonly failing to properly involve these parameters within the computational graph that propagates back to the loss. Simply put, if the framework doesn’t “see” the parameters affecting the final loss, it won't calculate gradients for them.

Here’s how this manifests in a typical VAE with normalizing flows: the VAE encoder outputs latent variables, which are then passed through a chain of bijectors, such as planar flows or radial flows, before being decoded by the VAE decoder. During the training process, the VAE's loss function is calculated based on the decoded output and the input. Crucially, if the bijector’s own parameters, which determine the transformation applied at each flow layer, are not integrated into the loss calculation in a manner that facilitates backpropagation, their gradients will be zero. This commonly occurs when bijector parameters are initialized but are not correctly used during the transformation process. For instance, the trainable parameters may be present but not connected into the network flow. Another common error is to rely on hard-coded constants instead of the trainable parameters within the bijector’s internal logic, effectively bypassing the trainable parameters in the forward pass.

Below are a few concrete examples, illustrating typical problems and their solutions:

**Example 1: Missing dependency in parameter calculation**

This example shows a bijector implementation where the parameters, `scale` and `shift`, are initialized but not used correctly within the forward transformation, resulting in no gradient information.

```python
import torch
import torch.nn as nn

class IncorrectBijector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim)) # Trainable scale parameter
        self.shift = nn.Parameter(torch.zeros(dim)) # Trainable shift parameter

    def forward(self, x):
        # Incorrect use; parameters not incorporated
        return x + 1.0 # transformation is independent of the parameters.

    def inverse(self, y):
        return y - 1.0
    
    def log_abs_det_jacobian(self, x):
      # Jacobian determinant is 0 since forward pass is parameter independent
      return torch.zeros_like(x)
```

In this case, although `scale` and `shift` are declared as trainable parameters, the `forward` method applies a fixed transformation: `x + 1.0`, not incorporating the parameters at all. Consequently, the backpropagation process finds no functional dependence on these parameters, and their gradients will be zero. The determinant of the Jacobian, which is zero in this case, further emphasizes the lack of dependence on trainable parameters.

**Example 2: Correct implementation with parameters**

The next example shows a corrected version where the parameters are properly utilized in the forward and inverse transformations.

```python
import torch
import torch.nn as nn
import torch.distributions as dist

class CorrectBijector(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return self.scale * x + self.shift # Correct use of the parameters
    
    def inverse(self,y):
        return (y - self.shift) / self.scale
    
    def log_abs_det_jacobian(self, x):
      return torch.log(torch.abs(self.scale)).sum(-1)
```

Here, the `forward` method correctly applies the trainable `scale` and `shift` parameters, using the standard affine transformation. The gradient calculation will now correctly propagate through these parameters. The log of the absolute determinant of the Jacobian is also calculated based on `scale` ensuring the bijector can contribute to the overall training. The forward and inverse implementations work to ensure the forward method is invertible and thus a valid bijector.

**Example 3: Proper integration within a flow framework**

This final example demonstrates how the corrected bijector is used within a VAE and includes a check to ensure the parameters are indeed updating.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_sigma = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        mu = self.fc_mu(h)
        sigma = torch.exp(self.fc_sigma(h) / 2.0)
        return dist.Normal(mu, sigma)

class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, 128)
        self.fc2 = nn.Linear(128, output_dim)

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        output = self.fc2(h)
        return output


class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim, bijector_dims):
        super().__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.bijector = CorrectBijector(bijector_dims)  # Use the correct implementation
        self.decoder = Decoder(latent_dim, input_dim)
        self.prior = dist.Normal(0,1)
    def forward(self, x):
        q_z = self.encoder(x)
        z = q_z.rsample()
        z_transformed = self.bijector(z)
        x_recon = self.decoder(z_transformed)
        return x_recon, z, q_z
    
    def loss(self, x, x_recon, z, q_z):
      log_likelihood = -torch.nn.functional.mse_loss(x_recon, x, reduction='sum')
      kl_div = dist.kl_divergence(q_z, self.prior).sum(-1) # KL divergence of original gaussian
      log_det = self.bijector.log_abs_det_jacobian(z) # Correct Jacobian application

      return -log_likelihood + kl_div - log_det
# Setting up the model and optimizers
input_dim = 784
latent_dim = 32
bijector_dims = latent_dim
vae = VAE(input_dim, latent_dim, bijector_dims)

optimizer = optim.Adam(vae.parameters(), lr=1e-3)

# Sample input data
x_sample = torch.randn(1, input_dim)

# Training loop
for epoch in range(50):  
  optimizer.zero_grad()
  x_recon, z, q_z  = vae(x_sample)
  loss = vae.loss(x_sample, x_recon, z, q_z)
  loss.backward()
  optimizer.step()
  
  if epoch % 10 == 0:
    print(f"Epoch: {epoch}, Loss: {loss.item():.4f}, Bijector Scale: {vae.bijector.scale.mean().item():.4f}")
    
print("Final bijector scale: ", vae.bijector.scale)
```

This code exemplifies a basic VAE framework that incorporates the `CorrectBijector`. By properly calculating the Jacobian determinant and incorporating it into the overall loss, the optimizer now correctly updates the `scale` parameter. Running this code shows that the `vae.bijector.scale` values change significantly over training epochs, unlike in the original, broken, `IncorrectBijector` example. This shows how a missing component in the loss calculation can disrupt the training process.

To further clarify and avoid this common issue, consulting resources focusing on automatic differentiation in deep learning frameworks is very beneficial. Additionally, texts detailing the theoretical underpinnings of normalizing flows and their integration within variational inference can help identify issues early on. Examining code examples that use bijectors within frameworks like PyTorch or TensorFlow can help understand the correct practical implementation of flow layers. Also, specific guides on variational inference and understanding the role of the Jacobian are invaluable. These resources provide a deeper understanding and facilitate the debugging process. Finally, checking the gradients for the bijector parameters through a debugger or by inspecting the values directly (as above) is helpful to verify the behavior of the implementation.
