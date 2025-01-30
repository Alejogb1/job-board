---
title: "What causes 'NaN or Inf found in input tensor' errors during PyTorch training, specifically those logged by `torch.autograd.detect_anomaly`?"
date: "2025-01-30"
id: "what-causes-nan-or-inf-found-in-input"
---
The root cause of "NaN or Inf found in input tensor" errors during PyTorch training, as detected by `torch.autograd.detect_anomaly`, almost invariably stems from numerical instability within the computation graph.  My experience debugging such issues across numerous projects, from large-scale image classification to reinforcement learning environments, points to three primary sources: exploding gradients, division by zero, and the propagation of NaNs/Infs from upstream operations.  These aren't mutually exclusive; often, they interact in complex ways.

**1. Exploding Gradients:**  This is perhaps the most frequent culprit.  During backpropagation, gradients can become excessively large, exceeding the representable range of floating-point numbers.  This leads to `Inf` values which then propagate through the network, contaminating subsequent calculations and ultimately resulting in NaNs (Not a Number) due to operations like `Inf - Inf`.  Deep networks with many layers, inappropriate activation functions (e.g., unconstrained sigmoid variants), or poorly scaled data are particularly vulnerable.  Gradient clipping, weight normalization techniques, and careful selection of optimizers are effective mitigation strategies.

**2. Division by Zero:**  A seemingly obvious source of error, division by zero directly generates `Inf`.  This can occur explicitly in the model's forward pass, perhaps through a poorly designed calculation, or implicitly within the activation functions, especially those involving exponential operations where the denominator can become vanishingly small.  Identifying the exact location of the zero-division may require careful examination of the model's architecture and input data distributions.

**3. Propagation of NaNs/Infs:**  Once a NaN or Inf appears within the computation graph, its effects can cascade.  Any subsequent operation involving a NaN or Inf will likely produce another NaN or Inf, rapidly contaminating the entire tensor.  This makes pinpointing the initial source crucial.  Tools like `torch.autograd.detect_anomaly` are invaluable in this context, precisely because they highlight the point of first occurrence.  However, even with this, tracing back to the root cause often requires methodical debugging.

Let's illustrate these causes with code examples and subsequent analysis:

**Example 1: Exploding Gradients**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# A simple model prone to exploding gradients
model = nn.Sequential(
    nn.Linear(10, 100),
    nn.ReLU(),
    nn.Linear(100, 100),
    nn.ReLU(),
    nn.Linear(100, 1)
)

optimizer = optim.SGD(model.parameters(), lr=1.0) # High learning rate

# Sample data
x = torch.randn(1, 10)
y = torch.randn(1, 1)


for i in range(1000):
    with torch.autograd.detect_anomaly():
        optimizer.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        optimizer.step()
    print(f"Iteration: {i}, Loss: {loss.item()}")

```

In this example, the high learning rate (lr=1.0) in the SGD optimizer significantly increases the likelihood of exploding gradients.  The lack of weight normalization or gradient clipping exacerbates the issue.  The `torch.autograd.detect_anomaly()` context manager will pinpoint the layer where the instability begins.


**Example 2: Division by Zero**

```python
import torch
import torch.nn as nn

class DivByZeroModel(nn.Module):
    def forward(self, x):
        return x / (x - x)

model = DivByZeroModel()
x = torch.randn(10)

with torch.autograd.detect_anomaly():
    output = model(x)
    print(output)

```

This model deliberately introduces division by zero.  The `detect_anomaly` context will immediately flag the error at the line with the problematic division.  The core issue is the calculation `x - x`, which always results in zero, regardless of the input `x`.


**Example 3: NaN Propagation**

```python
import torch
import torch.nn as nn

class NaNPropagationModel(nn.Module):
    def forward(self, x):
        y = torch.where(x > 0, x, torch.tensor(float('nan')))  # Introduces NaN for negative values.
        return torch.log(y) # Log of NaN is still NaN


model = NaNPropagationModel()
x = torch.randn(10)

with torch.autograd.detect_anomaly():
    output = model(x)
    print(output)
```

This example showcases NaN propagation. The `torch.where` function introduces NaNs for negative input values, and the subsequent `torch.log` operation propagates the NaN to the output.  The `detect_anomaly` context will identify where the initial NaN is generated, helping to trace the source.


**Resource Recommendations:**

1.  PyTorch documentation: Thoroughly review the sections on automatic differentiation and optimization algorithms.
2.  Advanced optimization literature: Explore papers and books focusing on gradient clipping, weight normalization, and adaptive optimization methods.
3.  Debugging tools:  Familiarize yourself with PyTorch's debugging tools beyond `detect_anomaly`, including tensorboard for visualization and Jupyter notebooks for interactive inspection.


Addressing "NaN or Inf found in input tensor" errors requires a systematic approach.  Begin by using `torch.autograd.detect_anomaly` to pinpoint the location of the problem.  Then, carefully analyze the model's architecture, the activation functions employed, the data scaling and normalization procedures, and the choice of optimizer and learning rate.  By methodically investigating these aspects, you will be able to effectively diagnose and resolve these commonly encountered issues.
