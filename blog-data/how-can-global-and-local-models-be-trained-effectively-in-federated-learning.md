---
title: "How can global and local models be trained effectively in federated learning?"
date: "2024-12-23"
id: "how-can-global-and-local-models-be-trained-effectively-in-federated-learning"
---

,  I've spent a fair amount of time dealing with the intricacies of federated learning (FL), particularly when balancing the needs of global model convergence and the specific nuances of local data. It’s not just about throwing some algorithms at the problem; it requires a nuanced understanding of the trade-offs involved. The question of effectively training global and local models in FL is multi-faceted, and let me share some of what I've learned over the years, including some specific cases I encountered.

Initially, many assume that a basic Federated Averaging (FedAvg) approach will suffice. And it’s a good starting point. It involves each client training a local model on its private data and then sending the model updates (often weights) to a central server. This server then averages these updates to form a new global model, which is subsequently distributed back to the clients. This dance continues iteratively. However, in practice, we see that this strategy is not always optimal. What we often get is a global model that is good on average but does not perform exceptionally well on specific clients. Local models, on the other hand, tend to perform well on their client’s data, but they also tend to overfit. The crux of the matter is balancing the convergence of a globally relevant model with the specific needs of each client, and that’s where some more advanced strategies come into play.

One area we should delve into is the handling of non-i.i.d (non-independent and identically distributed) data, a common reality in federated settings. Imagine a scenario where you are training a model to predict user behavior; some clients may be predominantly from a particular demographic, while others might represent a very diverse group. In this situation, if we blindly aggregate the model updates, the global model will be biased towards the distribution that's dominant in the aggregation. We are essentially letting the average overpower the specifics.

To mitigate this, techniques like *FedProx* can be invaluable. FedProx modifies the local training objective by adding a proximal term that encourages the local model to stay close to the global model. This reduces the drift of local models during training, leading to more stable and consistent global model convergence. You could think of it like adding a gentle restraint on the local training, preventing it from diverging too far, but not restricting it completely, leaving it room to learn. Here’s a simple illustration of how one might modify the loss function in PyTorch using FedProx; it's a conceptual example rather than a directly runnable code:

```python
import torch
import torch.nn as nn
import torch.optim as optim

def fedprox_loss(local_model, global_model, data, labels, mu=0.1):
    """Calculates the FedProx loss."""
    criterion = nn.CrossEntropyLoss() # Using Cross-Entropy for example
    output = local_model(data)
    loss = criterion(output, labels)

    # Calculate proximal term
    proximal_term = 0.0
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        proximal_term += (torch.norm(local_param - global_param)**2)

    total_loss = loss + (mu/2) * proximal_term

    return total_loss


# Example usage
local_model = nn.Linear(10, 2) # Dummy model
global_model = nn.Linear(10, 2) # Dummy model
optimizer = optim.SGD(local_model.parameters(), lr=0.01)
data = torch.randn(32, 10)  # Dummy input data
labels = torch.randint(0, 2, (32,)) # Dummy labels
mu = 0.1 # Example proximity value


optimizer.zero_grad()
loss = fedprox_loss(local_model, global_model, data, labels, mu)
loss.backward()
optimizer.step()
```

This code modifies your loss function to be more than simply about classifying a client's data locally, but also taking into account how far the client’s model has drifted from the global model. This helps the local model stay aligned while allowing for local learning as well.

Another effective method is employing *personalized federated learning* (PFL). PFL recognizes that a single global model will not always be optimal for every client. So, instead of focusing purely on a single global model, the aim is to train a customized model for each client based on their specific data distribution while leveraging global knowledge. This can be approached in several ways. One involves training a shared representation and then adding personalized layers for each client. Another involves fine-tuning the global model using client-specific data.

I once worked on a project dealing with time-series data from various edge devices which monitored critical infrastructure. These devices had very different operational patterns. The idea was that, in that case, relying on a single global model would simply not deliver the level of precision we needed on each device. We had to move towards something more adaptive, and I implemented this personalized approach, by taking a shared base model and adding a client-specific dense layer. It worked surprisingly well.

Here is how you could do that conceptually:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import copy

class PersonalizedModel(nn.Module):
    def __init__(self, input_size, base_size, output_size):
        super().__init__()
        self.base_layer = nn.Linear(input_size, base_size)
        self.personalized_layer = nn.Linear(base_size, output_size)

    def forward(self, x):
        x = torch.relu(self.base_layer(x))
        x = self.personalized_layer(x)
        return x


# Example usage
input_size = 10
base_size = 32
output_size = 2
local_model = PersonalizedModel(input_size, base_size, output_size)

global_model = nn.Linear(input_size, output_size) # Dummy global model

optimizer = optim.SGD(local_model.parameters(), lr=0.01)
data = torch.randn(32, input_size)  # Dummy input data
labels = torch.randint(0, 2, (32,)) # Dummy labels


for epoch in range(2): # Example training loop
  optimizer.zero_grad()
  output = local_model(data)
  criterion = nn.CrossEntropyLoss()
  loss = criterion(output, labels)
  loss.backward()
  optimizer.step()

  # Update the base layer (not fully accurate here,
  # it’s conceptual and illustrative for a real system)
  for local_param, global_param in zip(local_model.base_layer.parameters(), global_model.parameters()):
      global_param.data.copy_(local_param.data) # Conceptual weight sync
```

In this snippet, the shared parameters from `global_model` are copied into the `local_model.base_layer` after local updates, effectively creating a personalized layer for the local model but utilizing global information within the `base_layer`.

Finally, *hierarchical federated learning* (HFL) is a valuable approach when dealing with federated networks that have a hierarchical structure. Think of it this way; instead of all clients communicating directly with a central server, they first form clusters and then each cluster has its own local aggregation server. These cluster-level servers then communicate with the global server. This reduces the communication overhead with the central server and promotes local model diversity within each cluster, because each cluster would be dealing with more homogeneous data.

Here’s a conceptual breakdown:
1. **Local Training:** Each client within a cluster performs its local training as normal.
2. **Cluster Aggregation:** The local server in the cluster aggregates the local updates, creating a local cluster model.
3. **Global Aggregation:** The global server aggregates these cluster-level models to create the new global model.

We employed a form of HFL when dealing with a network of sensor devices in various locations, where sensor readings tended to correlate strongly within a specific region but were rather different between regions. This allowed the regions to adapt faster to their local specifics, while still benefiting from global knowledge. While I can’t provide an exact code snippet that would simulate a hierarchical approach here due to the complexity of the implementation, I will tell you that the key here is not only to average but to introduce a mechanism to ensure that the global model would not wash away the clusters' specific learning – this could include using techniques like weighted averages or other clustering aware aggregations.

For those keen to explore these methods in further detail, I would strongly recommend diving into the *“Advances and Open Problems in Federated Learning”* paper by Peter Kairouz et al., as well as “Federated Learning” by Yang et al. as a comprehensive textbook on the field. Also, looking into the recent works coming from the *NeurIPS* and *ICML* conferences would also prove extremely useful as these venues often explore the latest innovations in the field.

In essence, the key to effective federated learning lies in acknowledging that local data is unique and devising training strategies that account for these differences, while converging to a good global model that generalises well.
