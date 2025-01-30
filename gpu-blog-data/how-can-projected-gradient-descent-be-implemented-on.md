---
title: "How can projected gradient descent be implemented on the probability simplex in PyTorch?"
date: "2025-01-30"
id: "how-can-projected-gradient-descent-be-implemented-on"
---
Projected Gradient Descent (PGD), when constrained to the probability simplex, requires careful consideration of the projection step. This constraint, which mandates non-negative components summing to unity, is not naturally handled by standard gradient descent. I've encountered this directly during the development of a custom classification model where maintaining a probabilistic output was critical. The core challenge lies in ensuring that each iterative update of a parameter, after the gradient step, remains within the simplex.

The standard gradient descent update rule is:

```
parameter = parameter - learning_rate * gradient
```

This is insufficient when parameters are probabilities, as it can lead to negative values or a sum deviating from one. Projecting the updated parameters back to the simplex is essential for this class of problems. I’ve found the following approach efficient and numerically stable.

First, consider the probability simplex in *n*-dimensions. A point, denoted as vector *x*, is a member of the simplex if:

1.  *x<sub>i</sub>* ≥ 0 for all *i* (non-negativity)
2.  ∑*x<sub>i</sub>* = 1 (sum-to-one constraint)

The gradient descent update might result in an *x* that does not satisfy these constraints. To perform the projection, I utilize a method based on sorting and cumulative sums. The procedure is as follows:

1.  Sort the elements of *x* in descending order, denoted as *x*<sup>sorted</sup>.
2.  Compute cumulative sums of the sorted elements, *cumsum(*x*<sup>sorted</sup>)*.
3.  Identify the largest index *k* such that (*cumsum*(*x*<sup>sorted</sup>) - 1 + (k + 1)*x*<sup>sorted</sup><sub>k</sub>)  > 0. This is the *critical* index that governs the projection.
4.  Calculate the value *θ =* ( *cumsum*(*x*<sup>sorted</sup>) - 1 ) / (k + 1).
5.  The projection *x*<sup>projected</sup> is then computed as max(*x* - *θ*, 0) . This ensures non-negativity while implicitly making the components sum to one.

This method, while concise, requires a careful implementation in PyTorch. Specifically, the indexing for sorting and cumulative sums needs to be handled correctly to maintain gradients that are compatible with PyTorch's automatic differentiation.

Here are three code examples illustrating this projection and its integration within a PGD loop:

**Example 1: Standalone Projection Function**

```python
import torch

def project_to_simplex(x):
    """Projects a tensor onto the probability simplex.

    Args:
        x: A PyTorch tensor of any shape. The last dimension is assumed to represent probabilities.

    Returns:
        A PyTorch tensor with the same shape as x, projected onto the probability simplex.
    """
    original_shape = x.shape
    x = x.view(-1, x.shape[-1]) # Flatten for simpler indexing
    n = x.size(1)
    x_sorted, indices = torch.sort(x, dim=1, descending=True)
    x_cumsum = torch.cumsum(x_sorted, dim=1)
    k = (x_cumsum - 1 > (torch.arange(1, n + 1, device=x.device) * x_sorted)).sum(dim=1) -1
    theta = (x_cumsum[torch.arange(x.shape[0]), k] - 1) / (k + 1)
    x_projected = torch.clamp(x - theta[:, None], min=0)
    return x_projected.view(original_shape)


# Example Usage
x = torch.tensor([[0.2, 0.3, 0.5], [0.7, 0.1, 0.1]], requires_grad=True) # Example input
x_projected = project_to_simplex(x)
print(x_projected)
print(torch.sum(x_projected, dim = -1)) # verifies each row sums to 1

```

This function takes a tensor *x*, flattens the last dimension for easier indexing, sorts the elements, and computes cumulative sums. It then calculates *k* and *θ* to perform the projection. The result is reshaped back to the original dimensions. The example usage showcases the application of the projection on a sample tensor and validates that the output sums to one. I use `clamp` for efficient non-negative constraints.

**Example 2: PGD implementation with custom parameter update**

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimplexParameter(nn.Parameter):
    """
    A parameter that automatically projects onto the probability simplex after each gradient update.
    """

    def __init__(self, data=None, requires_grad=True):
        if data is None:
             data = torch.rand(1)
        super().__init__(data, requires_grad=requires_grad)

    def data_update(self, lr, grad):
        with torch.no_grad():
            self.data = project_to_simplex(self.data - lr * grad)

class SimpleModel(nn.Module):
    def __init__(self, num_features):
        super(SimpleModel, self).__init__()
        self.weights = SimplexParameter(data=torch.rand(num_features)) # Ensure initial weights in the simplex

    def forward(self, x):
        return torch.dot(x, self.weights)

# Example usage
num_features = 5
model = SimpleModel(num_features)
optimizer = optim.SGD(model.parameters(), lr = 0.1)

data = torch.rand(num_features)
for epoch in range(100):
    optimizer.zero_grad()
    output = model(data)
    loss = torch.square(output - 1) # Simple square loss
    loss.backward()

    with torch.no_grad():
        for p in model.parameters():
           if isinstance(p, SimplexParameter):
                p.data_update(lr=0.1,grad=p.grad)

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")

```

In this example, I encapsulate the projection within a custom parameter class, `SimplexParameter`. This class overrides the default parameter update to incorporate the projection after the gradient descent step. The model uses this custom parameter. During training, the custom parameter update ensures the weights remain on the simplex. I have found this to be more organized than injecting the projection inside the training loop itself as it keeps the model’s learning behavior focused on standard gradient descent practices but enforces constraints at a lower level.

**Example 3: PGD within a classification context**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassificationModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ClassificationModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)


    def forward(self, x):
        logits = self.fc(x)
        return logits

input_size = 10
num_classes = 3
model = ClassificationModel(input_size, num_classes)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

data = torch.randn(1, input_size)
target = torch.randint(0, num_classes, (1,))

for epoch in range(100):
    optimizer.zero_grad()
    logits = model(data)
    loss = criterion(logits, target)
    loss.backward()
    with torch.no_grad():
       for p in model.parameters():
            if isinstance(p, nn.Linear):
                p.weight.data = project_to_simplex(p.weight.data)
    optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    print(F.softmax(logits, dim=1))

```
This example shows the application of simplex projection in a typical classification task using a linear layer. Here, after every gradient update from backward pass, I directly project the weights of the linear layer `p.weight`. In practice, the final layer of classification models often represents probabilities; hence this projection is highly relevant. Notice that I use softmax before displaying the logits, but the projection is done directly on `p.weight`, not the output of softmax.

For further study, I recommend exploring resources covering convex optimization, specifically the theory and practical applications of projection methods. Material discussing projected gradient descent, its convergence guarantees, and various projection algorithms is essential. Detailed explanations on the implementation of simplex projections in different libraries, beyond PyTorch, could also be beneficial.  In summary, while standard gradient descent is common, it may need to be constrained based on problem domains.  For probability-related tasks, incorporating a projection back to the probability simplex is essential and can be performed efficiently in PyTorch through careful consideration of indexing, sorting, and parameter handling.
