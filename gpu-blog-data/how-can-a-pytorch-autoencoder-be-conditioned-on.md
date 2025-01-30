---
title: "How can a PyTorch autoencoder be conditioned on time?"
date: "2025-01-30"
id: "how-can-a-pytorch-autoencoder-be-conditioned-on"
---
A common challenge in sequential data modeling, specifically within the realm of autoencoders, lies in effectively incorporating temporal dependencies. Merely feeding time-series data as a single, unordered input vector to an autoencoder overlooks crucial temporal context. Conditioned autoencoders, however, offer a powerful mechanism to address this by explicitly including time information into both the encoding and decoding processes.

My experience with anomaly detection on sensor data streams highlighted the limitations of vanilla autoencoders. Their inability to discern between similar data points arriving at different times led to high false positive rates. Consequently, I investigated methods to explicitly condition the autoencoder on the timestamp. The most effective approach I found involved augmenting the input and latent representations with time embeddings, which allowed the network to learn temporal patterns more readily.

The core principle involves generating a time embedding that reflects the temporal information associated with each data point. This embedding can be a simple scalar value representing the time step, or a more complex high-dimensional vector derived using trigonometric functions, Fourier transforms, or learned embedding layers. Once generated, this embedding is incorporated into the autoencoder at different stages: the input layer and the latent space. This allows the model to learn an encoded representation that is influenced by the temporal context.

In the encoding path, the time embedding can be concatenated with the input data vector before passing through the encoder layers. This results in the encoder learning a representation that reflects the data *and* its temporal context. This ensures that the encoded representation captures the data within its specific temporal window. During the decoding stage, this time embedding must be available again to correctly reconstruct the input, thereby ensuring that temporal consistency is maintained. Typically, this involves a separate time embedding concatenated to the latent representation just before passing through decoder layers.

Below are three examples demonstrating different strategies for incorporating temporal conditioning within a PyTorch autoencoder:

**Example 1: Concatenation with Scalar Time Embedding**

This example utilizes a basic scalar representing the time step and concatenates it with the input and latent representations.

```python
import torch
import torch.nn as nn

class TimeConditionedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(TimeConditionedAutoencoder, self).__init__()
        self.input_dim = input_dim
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + 1, 128),  # +1 for scalar time
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 128), # +1 for scalar time
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x, t):
        # Time embedding (scalar)
        t = t.unsqueeze(1).float()  # Ensure time is float and has a dimension for concatenation

        # Encoding
        x_encoded = torch.cat((x,t), dim = 1) #Concatenate input with time
        z = self.encoder(x_encoded)

        #Decoding
        z_decoded = torch.cat((z,t), dim = 1) #Concatenate latent representation with time
        x_reconstructed = self.decoder(z_decoded)
        return x_reconstructed

#Example usage
input_dimension = 10
latent_dimension = 3
model = TimeConditionedAutoencoder(input_dimension, latent_dimension)
input_data = torch.randn(1, input_dimension) #Batch size 1
time_step = torch.tensor([5]) #Single time step
output = model(input_data,time_step)
print(output.shape)
```

In this implementation, the `t` variable represents the time step. It is unsqueezed to allow concatenation along the correct dimension (columns). Concatenation is performed on both the input and latent representations. Although simple, this method can prove surprisingly effective when the time is monotonic, such as a time series where each data point occurs later than the previous one.

**Example 2: Concatenation with Sinusoidal Time Embedding**

This example utilizes sinusoidal functions to generate a higher-dimensional time embedding which offers more flexibility.

```python
import torch
import torch.nn as nn
import math

def get_sinusoidal_embedding(time_step, embedding_dim):
    position = torch.arange(0, embedding_dim).float()
    div_term = torch.exp(position * (-math.log(10000.0) / embedding_dim))
    time_tensor = time_step.float().unsqueeze(1)
    embeddings = torch.cat((torch.sin(time_tensor * div_term), torch.cos(time_tensor * div_term)), dim=1)
    return embeddings

class SinusoidalTimeConditionedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, time_emb_dim):
        super(SinusoidalTimeConditionedAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.time_emb_dim = time_emb_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + time_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x, t):
      #Time Embedding
      time_embedding = get_sinusoidal_embedding(t, self.time_emb_dim // 2) #Half embedding dim for each sine and cosine
      # Encoding
      x_encoded = torch.cat((x,time_embedding), dim = 1)
      z = self.encoder(x_encoded)
      # Decoding
      z_decoded = torch.cat((z,time_embedding), dim = 1)
      x_reconstructed = self.decoder(z_decoded)
      return x_reconstructed

#Example Usage
input_dimension = 10
latent_dimension = 3
time_embedding_dim = 16
model = SinusoidalTimeConditionedAutoencoder(input_dimension, latent_dimension,time_embedding_dim)
input_data = torch.randn(1, input_dimension) #Batch size 1
time_step = torch.tensor([5]) #Single time step
output = model(input_data,time_step)
print(output.shape)
```

Here, `get_sinusoidal_embedding` produces a higher-dimensional representation using trigonometric functions.  The time is encoded into a fixed size vector that can be effectively learned by the subsequent layers. This is particularly useful when dealing with cyclical time dependencies, as the sinusoidal embedding captures temporal patterns regardless of the magnitude of the timestamp itself.

**Example 3: Learned Time Embedding with Linear Layer**

This final example shows a learned embedding using a linear layer to transform the time into an embedding.

```python
import torch
import torch.nn as nn

class LearnedTimeConditionedAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim, time_emb_dim):
        super(LearnedTimeConditionedAutoencoder, self).__init__()
        self.input_dim = input_dim
        self.time_emb_dim = time_emb_dim

        self.time_embedding = nn.Linear(1, time_emb_dim) #Single dim as time is a scalar

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim + time_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + time_emb_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x, t):
      #Time Embedding
      time_embedding = self.time_embedding(t.unsqueeze(1).float())
      #Encoding
      x_encoded = torch.cat((x, time_embedding), dim = 1)
      z = self.encoder(x_encoded)
      #Decoding
      z_decoded = torch.cat((z, time_embedding), dim = 1)
      x_reconstructed = self.decoder(z_decoded)
      return x_reconstructed

# Example usage
input_dimension = 10
latent_dimension = 3
time_embedding_dim = 16
model = LearnedTimeConditionedAutoencoder(input_dimension, latent_dimension, time_embedding_dim)
input_data = torch.randn(1, input_dimension)
time_step = torch.tensor([5]) # Single time step
output = model(input_data, time_step)
print(output.shape)

```

In this case, a linear layer is used as a learnable time embedding, allowing the model to learn the most suitable representation of time based on the task at hand. This can offer flexibility in the time-embedding and it can adapt itself more effectively to intricate temporal relationships within the dataset compared to fixed embeddings.

In addition to the code demonstrations above, effective implementation of conditioned autoencoders relies on careful consideration of several key areas: The choice of time embedding (scalar, sinusoidal, learned) is highly task-dependent and should be explored experimentally. The dimensionality of the time embedding should be large enough to capture significant temporal information without excessive computational overhead. The manner of concatenation (before input layer or directly into each layer) might also affect performance. Regularization techniques and dropout layers are often beneficial to prevent overfitting, particularly if the time-conditioning adds complexity. Furthermore, ensure time inputs are properly preprocessed and normalized in a manner that makes sense for the specific application.

For those seeking more comprehensive learning, I recommend exploring resources focused on temporal modeling using neural networks. Research papers that investigate sequential autoencoders with different types of conditioning mechanisms, focusing especially on time series data modeling. Books covering deep learning principles and their applications to sequence-based tasks will also provide the theoretical foundations for understanding these techniques. Finally, reviewing community forums and repositories containing similar model implementations can offer further practical guidance. These resources, when combined with empirical investigation, form the cornerstone of mastering conditioned autoencoders for time-aware data processing.
