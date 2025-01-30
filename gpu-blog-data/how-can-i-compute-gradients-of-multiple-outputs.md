---
title: "How can I compute gradients of multiple outputs with respect to each input in a PyTorch network?"
date: "2025-01-30"
id: "how-can-i-compute-gradients-of-multiple-outputs"
---
PyTorch's automatic differentiation engine, while straightforward for single-output loss functions, requires careful handling when gradients for multiple outputs, each potentially influencing downstream operations, are needed relative to the same set of inputs. This often arises in multi-task learning scenarios or when analyzing intermediate activations within a model. Directly calling `.backward()` on each output separately will overwrite previously computed gradients, rendering them unusable. My experience building a multi-objective reinforcement learning system highlighted this challenge; each sub-policy generated a distinct loss and I needed gradients of all these losses with respect to the shared policy parameters.

The core issue stems from the cumulative nature of `.backward()`. Each call performs backpropagation *from* a specific tensor, accumulating gradients within the computation graph. If multiple outputs are individually backpropagated, the gradient buffers for shared parameters only reflect the result of the *last* backward pass. This means that if output tensors `output_1` and `output_2` are both functions of parameters in a model `model`, calling `output_1.backward()` and then `output_2.backward()` effectively disregards the gradients of `output_1` relative to the model parameters. The second backward call overwrites the model parameter gradients that were computed during the first call.

The solution involves a combined backward pass using vector-Jacobian products. Instead of calling `.backward()` on each output independently, we first construct a vector `v` of the same size as the output. Then we invoke `backward` with the output tensor and `v`. This computes the Jacobian-vector product, or in other words the gradients of the linear combination (output dot v) rather than the gradients of each element of the output. For the specific case where we require gradients with respect to all components of a tensor output, we would ideally compute the Jacobian matrix, whose rows correspond to each output dimension. Rather than calculating that matrix, we compute Jacobian-vector products using unit vectors. Repeating that procedure we can extract each row of the Jacobian.

I will now demonstrate this with three code examples. The first will demonstrate the problem, while the second will show the correct implementation using Jacobian vector products, and the final one will showcase a more computationally efficient method.

**Example 1: Incorrect Backpropagation**

This example simulates a simple two-output model and demonstrates the issue of overwritten gradients.

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.linear(x))
        return self.linear2(x)

model = SimpleModel()
inputs = torch.randn(1, 5, requires_grad=True)
outputs = model(inputs)

# Simulate two outputs
output_1 = outputs[0, 0]
output_2 = outputs[0, 1]

# Incorrect backpropagation
output_1.backward(retain_graph=True)
grad_1 = model.linear.weight.grad.clone()
model.zero_grad()
output_2.backward()
grad_2 = model.linear.weight.grad.clone()

print("Gradient after output_1:", grad_1)
print("Gradient after output_2:", grad_2)
```

In this code, calling `output_1.backward(retain_graph=True)` populates the gradient buffers. However, the subsequent `output_2.backward()` overwrites the gradients from the first backward pass. As a result, `grad_2` contains only the derivatives related to the second output, and `grad_1` has been overwritten within the PyTorch system, even though we cloned it. I used the `retain_graph=True` argument to make the graph available for the second call to `backward()`.

**Example 2: Correct Backpropagation with Jacobian-Vector Products**

This example demonstrates the correct way to backpropagate with multiple outputs using Jacobian-vector products.

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.linear(x))
        return self.linear2(x)

model = SimpleModel()
inputs = torch.randn(1, 5, requires_grad=True)
outputs = model(inputs)
num_outputs = outputs.size(1)

all_grads = []

for i in range(num_outputs):
    model.zero_grad()
    unit_vector = torch.zeros_like(outputs)
    unit_vector[0,i] = 1.0
    outputs.backward(gradient=unit_vector, retain_graph=True)
    all_grads.append(model.linear.weight.grad.clone())
    
for i, grad in enumerate(all_grads):
    print(f"Gradient of output {i+1}:", grad)
```

Here, we iterate through each output. For every output, we create a `unit_vector`, which is a tensor of zeros with a `1.0` at the index corresponding to the output. When `backward` is called with this vector, the computed gradient will be the gradient of the chosen output element with respect to the inputs. The `retain_graph=True` argument is necessary to keep the computational graph in memory across calls to `backward()`. I clone the gradient tensor, as the values are overwritten with each call to `backward()`. This method returns a separate gradient corresponding to each output.

**Example 3: Simplified Computation**

The previous method involves iterating through each element. It is also possible to compute the complete Jacobian, which represents gradients of all outputs with respect to all inputs, using `torch.autograd.grad`. Note that `torch.autograd.grad` does *not* accumulate gradients into the `.grad` attribute of the parameters. Instead, it returns them directly.

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)
        self.linear2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.linear(x))
        return self.linear2(x)

model = SimpleModel()
inputs = torch.randn(1, 5, requires_grad=True)
outputs = model(inputs)

# Compute the Jacobian
jacobian = torch.autograd.grad(outputs, inputs, grad_outputs=torch.eye(2), create_graph=True)[0]

print("Jacobian:\n", jacobian)
```

In this method, we pass a tensor of ones for `grad_outputs`. This causes autograd to return the jacobian matrix. The element at location `[i,j]` represents how the value of the `i`-th output changes when changing the `j`-th input, multiplied by the loss. We use `torch.eye(2)` to obtain the Jacobian.

The `create_graph=True` option is included since it may be desired to compute second-order derivatives, although this is not shown in the example. We extract the Jacobian from the return value, which is a tuple of tensors with length equal to the number of specified inputs for which we compute the gradient, which in our case is `1`.

**Resource Recommendations**

For a deeper understanding of PyTorch's autograd engine, I recommend consulting the official PyTorch documentation on automatic differentiation, especially the section on advanced autograd functions. Articles on Jacobian-vector products and their applications in deep learning also provide valuable context. Furthermore, resources explaining the concept of the computational graph and its implications for backpropagation are extremely helpful. Studying examples of implementation in various applications is useful, including multi-task learning and gradient-based meta-learning. These resources, when combined, offer a comprehensive perspective on effectively using PyTorch for multi-output gradient computations. I would also suggest experimentation with different network structures to gain a practical understanding of the behavior of the autograd engine.
