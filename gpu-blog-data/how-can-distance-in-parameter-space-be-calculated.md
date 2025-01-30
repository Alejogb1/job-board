---
title: "How can distance in parameter space be calculated for a PyTorch network?"
date: "2025-01-30"
id: "how-can-distance-in-parameter-space-be-calculated"
---
The core challenge in determining distance within a PyTorch network's parameter space arises from the high-dimensionality of the parameter vector and the non-Euclidean nature of the loss landscape it inhabits. Direct Euclidean distance, while computationally simple, often fails to capture meaningful differences in the network's behavior or performance. I've encountered this issue extensively while experimenting with various adversarial training techniques and model interpolation strategies. The standard L2 norm, commonly used for vector distances, treats all parameters equally which can be misleading when parameter sensitivity varies drastically. More sophisticated measures that incorporate the loss function and its derivatives frequently provide a more relevant notion of distance.

Calculating distance in parameter space can involve several approaches, ranging from simple vector norms to more complex metrics informed by the network's function and the curvature of the loss landscape. These distances allow us to, for instance, evaluate the degree of change a perturbation induces in a model, the similarity of different models, or progress during training. My experience suggests that the appropriate measure depends significantly on the task at hand. For a first approximation, Euclidean distance, calculated as the L2 norm of the difference between two flattened parameter vectors, serves as a reasonable starting point. However, it doesn't account for the model's sensitivity to individual parameters. A more nuanced approach involves considering the influence of parameters on the output, which can be approximated using the Fisher Information Matrix or other similar methods.

The parameter vector is obtained by flattening all trainable parameters within the network using `torch.nn.utils.parameters_to_vector()` which returns a single tensor representing all parameter values as a single vector. This allows us to calculate vector distances. Let me demonstrate this with some code.

**Code Example 1: Euclidean Distance**

```python
import torch
import torch.nn as nn

def euclidean_distance(model_a, model_b):
  """Calculates the Euclidean distance between two models' parameters.

  Args:
    model_a: A PyTorch model.
    model_b: A PyTorch model.

  Returns:
    The Euclidean distance as a float.
  """
  params_a = torch.nn.utils.parameters_to_vector(model_a.parameters())
  params_b = torch.nn.utils.parameters_to_vector(model_b.parameters())
  return torch.norm(params_a - params_b)

# Example usage
model_a = nn.Linear(10, 2)
model_b = nn.Linear(10, 2)
model_b.weight.data += torch.randn_like(model_b.weight) * 0.1  # Introduce small difference
distance = euclidean_distance(model_a, model_b)
print(f"Euclidean distance: {distance.item():.4f}")
```

In the example above, I define a function `euclidean_distance` that takes two PyTorch models as input. The `parameters_to_vector` function converts all trainable parameters in the models into flat vectors. The difference between these vectors is computed, and then the L2 norm, as a measure of the magnitude of the difference vector, is calculated using `torch.norm`. I introduced a small perturbation in `model_b`'s weights to showcase the change in distance. This method provides a basic distance measure but fails to account for the varying importance of different parameter groups or parameters within the model.

**Code Example 2: Weighted Euclidean Distance**

To mitigate the limitation of treating all parameters equally, we can incorporate a weight matrix that scales parameter changes based on their respective gradients or Fisher Information. Since the Fisher information calculation can be computationally intensive, let us approximate this with the gradients of model parameter with respect to an arbitrary random input.

```python
import torch
import torch.nn as nn
import torch.optim as optim

def weighted_euclidean_distance(model_a, model_b, input_data):
  """Calculates a weighted Euclidean distance based on parameter gradients.

  Args:
      model_a: A PyTorch model.
      model_b: A PyTorch model.
      input_data: An example input tensor for gradient calculations.

  Returns:
      The weighted Euclidean distance as a float.
  """
  params_a = torch.nn.utils.parameters_to_vector(model_a.parameters())
  params_b = torch.nn.utils.parameters_to_vector(model_b.parameters())

  criterion = nn.MSELoss()
  optimizer_a = optim.SGD(model_a.parameters(), lr=0.01)
  optimizer_b = optim.SGD(model_b.parameters(), lr=0.01)

  optimizer_a.zero_grad()
  loss_a = criterion(model_a(input_data), model_a(input_data)) # self comparison
  loss_a.backward()

  optimizer_b.zero_grad()
  loss_b = criterion(model_b(input_data), model_b(input_data)) # self comparison
  loss_b.backward()


  grad_a = torch.cat([p.grad.flatten() for p in model_a.parameters()])
  grad_b = torch.cat([p.grad.flatten() for p in model_b.parameters()])
  grad_weights = (grad_a + grad_b)/2 # average gradient used as weight

  weighted_diff = (params_a - params_b) * grad_weights
  return torch.norm(weighted_diff)

# Example usage
model_a = nn.Linear(10, 2)
model_b = nn.Linear(10, 2)
model_b.weight.data += torch.randn_like(model_b.weight) * 0.1 # Introduce small diff
input_data = torch.randn(1, 10) # Input for gradient calc
distance = weighted_euclidean_distance(model_a, model_b, input_data)
print(f"Weighted Euclidean distance: {distance.item():.4f}")
```

In this revised example, the `weighted_euclidean_distance` function now calculates gradients with respect to a random input. It calculates the gradients for model A and B with respect to themselves respectively (a simple approach for obtaining gradient). These gradients, averaged across models, serve as weights applied to the parameter differences before calculating the Euclidean norm. This gives more weight to changes in parameters that significantly impact the model's output during the gradient computation phase, potentially providing a more meaningful distance metric. Note, I am averaging two gradients for the weighting. A more accurate approach would be to compute a full Fisher matrix using a set of inputs, which is outside the scope of this response.

**Code Example 3: Cosine Similarity in Parameter Space**

Another valuable approach is to consider the cosine similarity between the parameter vectors rather than their difference. Cosine similarity captures the angle between the vectors and is less sensitive to the magnitudes of the individual parameters. This can be useful when the absolute magnitudes are less important than the relative directions of changes in parameter space.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def cosine_similarity_distance(model_a, model_b):
  """Calculates 1 - cosine similarity between two models' parameters.

  Args:
      model_a: A PyTorch model.
      model_b: A PyTorch model.

  Returns:
      1 - cosine similarity as a float (distance).
  """
  params_a = torch.nn.utils.parameters_to_vector(model_a.parameters())
  params_b = torch.nn.utils.parameters_to_vector(model_b.parameters())
  similarity = F.cosine_similarity(params_a, params_b, dim=0)
  return 1 - similarity

# Example usage
model_a = nn.Linear(10, 2)
model_b = nn.Linear(10, 2)
model_b.weight.data += torch.randn_like(model_b.weight) * 0.1  # Introduce small diff
distance = cosine_similarity_distance(model_a, model_b)
print(f"Cosine similarity distance: {distance.item():.4f}")
```

In the third example, the `cosine_similarity_distance` function computes the cosine similarity between the two parameter vectors and returns 1 minus the result. This value serves as a distance measure, as similarity (ranging from -1 to 1) is highest when vectors point in the same direction and lowest when they point in opposite directions. Subtracting from 1 reinterprets similarity as a distance. The advantage of cosine similarity is that it is robust to differences in the magnitude of the parameters.

For further exploration of this topic, I recommend consulting resources on numerical optimization and deep learning theory. Textbooks covering topics such as convex optimization and methods to estimate the curvature of loss function are invaluable. Additionally, research papers focusing on parameter space analysis in neural networks, specifically those addressing techniques like mode connectivity and flat minima, offer significant insights. Papers on the Fisher Information Matrix and related methods for approximating the Hessian can also be very informative. Studying the theoretical foundations of these concepts provides a robust framework for addressing this problem in more complex scenarios and understanding the implications of various choices for distance metrics.
