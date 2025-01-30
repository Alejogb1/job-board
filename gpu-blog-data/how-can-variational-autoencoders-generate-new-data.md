---
title: "How can variational autoencoders generate new data?"
date: "2025-01-30"
id: "how-can-variational-autoencoders-generate-new-data"
---
Variational autoencoders (VAEs) leverage the principles of probabilistic modeling to create a latent representation of input data, from which new, similar data points can be sampled. Unlike traditional autoencoders which learn a deterministic mapping from input to a compressed latent space, VAEs learn a probability distribution over that latent space. This stochastic nature is key to their generative capabilities.

My experience building generative models for medical image analysis highlighted the power of VAEs in creating plausible variations of patient scans, specifically to augment datasets. The traditional approach of deterministic encodings, where a single input mapped to a single point in latent space, failed to capture the natural variability inherent in the medical data. Consequently, new generated images lacked diversity and often appeared as mere replications of the training set. VAEs, by modeling the latent space as a probability distribution, allowed for much more nuanced generation.

The core concept of a VAE revolves around two neural networks: an encoder and a decoder. The encoder, instead of directly outputting a single latent vector, outputs parameters that define a probability distribution, typically a Gaussian, over the latent space. These parameters usually represent the mean (μ) and standard deviation (σ) of the Gaussian distribution. A latent vector is then sampled from this distribution, and is subsequently passed to the decoder, which attempts to reconstruct the original input.

The training process for a VAE involves two main loss components. Firstly, there is the reconstruction loss, which assesses how well the decoder can reconstruct the input from the sampled latent vector (typically, the L2 or Mean Squared Error). This drives the model towards learning meaningful latent representations. Secondly, there is the Kullback-Leibler (KL) divergence loss. The KL divergence measures how similar the learned latent distribution is to a predefined prior distribution, usually a standard Gaussian with zero mean and unit variance. This regularization term forces the learned distributions to be similar to the prior. By enforcing this similarity, VAEs ensure that the latent space is continuous and well-structured, allowing for meaningful interpolation and sampling. It is this stochastic sampling, coupled with the structured latent space enforced by KL divergence, that enables the VAE to generate novel data.

Once the VAE is trained, generating new data is conceptually straightforward. We sample a latent vector from the prior distribution, i.e., a standard Gaussian. This random vector is then fed into the decoder, which produces an output that resembles the training data. The quality of the generated data heavily relies on the quality of training, ensuring the encoder has effectively learned a meaningful latent representation of the training distribution.

Here are three code examples using Python and the PyTorch library that illustrate these concepts.

**Example 1: A Simple VAE Encoder Implementation**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_sigma = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        log_sigma = self.fc_sigma(h)
        return mu, log_sigma
```
*Commentary:* This code defines a basic encoder using PyTorch's neural network modules. It takes input data, passes it through a fully connected layer with ReLU activation, and then outputs two vectors representing the mean and the logarithm of the standard deviation of the latent distribution. Using log variance, or log sigma, can lead to more numerically stable training. The user would need to exponentiate this when sampling from the distribution later on.

**Example 2: Latent Sampling and Decoder Implementation**

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = F.relu(self.fc1(z))
        return torch.sigmoid(self.fc2(h)) # Sigmoid for normalized data.

def reparameterize(mu, log_sigma):
    std = torch.exp(0.5 * log_sigma)
    eps = torch.randn_like(std)
    return mu + eps * std


input_dim = 784  # Example for MNIST, 28x28
latent_dim = 20
hidden_dim = 512
output_dim = input_dim
encoder = Encoder(input_dim, hidden_dim, latent_dim)
decoder = Decoder(latent_dim, hidden_dim, output_dim)

x = torch.randn(1, input_dim) # Example input
mu, log_sigma = encoder(x)

z = reparameterize(mu, log_sigma) # Sampling from the learned distribution
reconstructed_x = decoder(z)
```
*Commentary:*  This code presents the decoder structure, taking a latent vector and mapping it back to the input space via linear layers. Notably, this snippet demonstrates the reparameterization trick, a vital component of VAE training. Rather than directly sampling from the learned Gaussian distribution (which is a non-differentiable operation), it samples from a unit Gaussian and then applies the mean and standard deviation parameters. This makes backpropagation possible during the training process. Additionally, the final decoder activation is a sigmoid. If dealing with non-normalized data, a more appropriate activation or no activation would be required. The example input x is generated with `torch.randn` as a placeholder.

**Example 3: Loss Function and Training Setup**

```python
import torch.optim as optim

def loss_function(reconstructed_x, x, mu, log_sigma):
    recon_loss = F.binary_cross_entropy(reconstructed_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
    return recon_loss + kl_div

optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)

# Simplified training loop. Actual loops would involve dataset iterators.
x = torch.randn(1, input_dim)
optimizer.zero_grad()
mu, log_sigma = encoder(x)
z = reparameterize(mu, log_sigma)
reconstructed_x = decoder(z)
loss = loss_function(reconstructed_x, x, mu, log_sigma)
loss.backward()
optimizer.step()

# Generation from prior.
z_prior = torch.randn(1,latent_dim) # Sample a prior distribution.
generated_x = decoder(z_prior)
```

*Commentary:* This demonstrates how to define the combined loss function including the reconstruction loss (binary cross-entropy) and the KL divergence. It includes a basic optimization step using Adam. Note that this example is simplified and does not include data loading and batching. Finally, it shows how to sample from the prior Gaussian distribution and generate new data using the trained decoder. For binary cross entropy, the data needs to be within the 0 and 1 range (thus the sigmoid in decoder)

In summary, the generative power of VAEs comes from the combination of encoding inputs into a structured, probabilistic latent space and then sampling from this space to reconstruct the data. The KL divergence loss is essential to ensure the continuous structure of the latent space.

For further understanding of VAEs, I recommend exploring resources that delve into Bayesian inference, especially variational inference, as these are the theoretical underpinnings of the model. Books covering Deep Learning that include sections on generative models would also be useful. Look for tutorials that emphasize the probabilistic interpretation of the VAE objective function. Practical implementation guides can enhance familiarity with code details, but an understanding of the underlying mathematical concepts will be beneficial for effective utilization of this architecture.
