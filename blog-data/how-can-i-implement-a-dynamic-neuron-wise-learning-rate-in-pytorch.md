---
title: "How can I implement a dynamic neuron-wise learning rate in PyTorch?"
date: "2024-12-23"
id: "how-can-i-implement-a-dynamic-neuron-wise-learning-rate-in-pytorch"
---

Alright, let's tackle dynamic per-neuron learning rates in PyTorch. This is a topic I’ve spent a fair amount of time with, especially during a project a few years back involving extremely sparse neural networks for recommender systems. The conventional approach of using a single global learning rate often falls short when dealing with such networks, where different neurons have wildly varying activation frequencies and gradient magnitudes. We need something more granular, and PyTorch thankfully gives us the tools to achieve this.

The core idea is straightforward: instead of applying the same learning rate to all parameters, we maintain a separate learning rate for each neuron's parameters. This allows neurons with infrequent updates, or those generating small gradients, to receive larger updates while those with frequent, large gradients can be updated more cautiously. It's analogous to using different sized hammers for different nails – you wouldn’t use a sledgehammer for a pin.

Implementing this effectively in PyTorch requires modifying the standard optimization procedure. The *Adam* or *SGD* optimizers, as they're typically used, don't offer this per-parameter granularity 'out of the box.' We need to introduce custom logic into the parameter update process. We won't modify the optimizers themselves directly as that can get complex quickly, instead, we will implement custom parameter groups within optimizers.

Let's illustrate this with a few code examples, starting with a basic approach and then getting progressively more sophisticated.

**Example 1: Using a Custom Parameter Group with Fixed Learning Rates Per Layer**

This first example shows how to define a custom parameter group for each layer in a model and assign a fixed, but different, learning rate to each layer. While not 'per-neuron', this is a foundational step towards it, and it introduces the concept of targeted learning rate application.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNet()
optimizer = optim.Adam(
    [
        {'params': model.fc1.parameters(), 'lr': 0.001},
        {'params': model.fc2.parameters(), 'lr': 0.0005},
        {'params': model.fc3.parameters(), 'lr': 0.0002}
    ]
)

# Example usage:
input_data = torch.randn(1, 10)
target = torch.randint(0, 2, (1,))
criterion = nn.CrossEntropyLoss()

optimizer.zero_grad()
output = model(input_data)
loss = criterion(output, target)
loss.backward()
optimizer.step()

```
Here, we grouped the layers using `params` and set fixed, different `lr` values. Each layer gets updated with its own specific learning rate. This works, but it's still coarse-grained.

**Example 2: Approximating Neuron-Wise LR by Scaling Based on L2 Norm of Weights**

Now, let's move closer to a per-neuron update. Here, we maintain a base learning rate, but scale it for each neuron based on the L2 norm of its input weights, approximating the magnitude of gradients it receives over time. This isn’t true tracking but a good heuristic and much less computationally expensive than true per-parameter adaptive methods.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNet()
base_lr = 0.001
optimizer = optim.Adam(model.parameters(), lr=base_lr)

def calculate_neuron_lrs(model, base_lr, scaling_factor=0.01):
    lrs = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weight_norm = torch.norm(module.weight, dim=1)
            per_neuron_lrs = base_lr * (1 + scaling_factor * weight_norm)
            lrs.append(
                {'params': module.parameters(),
                'lr': per_neuron_lrs}
                )

    return lrs

# Example Usage:
input_data = torch.randn(1, 10)
target = torch.randint(0, 2, (1,))
criterion = nn.CrossEntropyLoss()

optimizer.zero_grad()
output = model(input_data)
loss = criterion(output, target)
loss.backward()

# Apply our neuron-wise LRs:
lrs = calculate_neuron_lrs(model, base_lr)
optimizer.param_groups = lrs # Important step to override

optimizer.step()
```
Here, `calculate_neuron_lrs` iterates through the linear layers, calculates the L2 norm of input weights of each neuron and uses this to generate learning rates and these rates are applied before optimizer step. The scale factor is important to tune as too large a value will make this method unstable. The important step here is overwriting `optimizer.param_groups`. We are creating a new parameter groups dictionary with custom learning rates based on neuron weight norms. This makes the learning rates dynamic. This still has limitations, as it only measures weights not actual gradients received, which can be different, thus, a better way is required.

**Example 3: Tracking Gradient Statistics for Per-Neuron Learning Rates**

Finally, a more robust solution involves tracking gradient statistics, which is a much more accurate representation of what each neuron is doing during learning. In this example, I will demonstrate one method called AdaGrad, but many other strategies that track running means are also effective here.
```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = SimpleNet()
base_lr = 0.01
optimizer = optim.Adam(model.parameters(), lr=base_lr)
grad_squared_sums = {}
for name, param in model.named_parameters():
    grad_squared_sums[name] = torch.zeros_like(param)
epsilon = 1e-8 # For Numerical stability
def calculate_and_update_neuron_lrs(model,base_lr,optimizer, grad_squared_sums, epsilon):
    lrs = []
    for name, param in model.named_parameters():
         if param.grad is not None:
             grad = param.grad.detach()
             grad_squared_sums[name] = grad_squared_sums[name] + grad**2
             per_param_lrs = base_lr / (torch.sqrt(grad_squared_sums[name])+ epsilon)
             lrs.append({'params':param,'lr':per_param_lrs})
    return lrs

# Example Usage:
input_data = torch.randn(1, 10)
target = torch.randint(0, 2, (1,))
criterion = nn.CrossEntropyLoss()
optimizer.zero_grad()
output = model(input_data)
loss = criterion(output, target)
loss.backward()

lrs = calculate_and_update_neuron_lrs(model, base_lr,optimizer,grad_squared_sums,epsilon)
optimizer.param_groups = lrs
optimizer.step()

```
In this example, `grad_squared_sums` tracks the sum of squared gradients for each parameter. In `calculate_and_update_neuron_lrs` this is then used to compute per parameter learning rates similar to the Adaptive Gradient Algorithm (AdaGrad), and are used to override `optimizer.param_groups` before the update. This provides a proper method to track gradient magnitudes directly which is what is desirable for per neuron learning rate.

**Important Considerations and Further Reading**

Implementing dynamic learning rates at this level is not trivial, and certain practical considerations are required for successful implementation. These include:

*   **Computational Overhead:** Calculating and applying per-parameter or per-neuron learning rates adds computation time. The method chosen should balance accuracy and computational cost, and it may require some profiling to achieve this.
*   **Parameter Initialization:** The initial learning rates can have a large impact, specifically in per-parameter/per-neuron updates, especially when based on gradient norms. Initialising appropriately and careful learning rate schedule experiments are required.
*   **Stability:** Very large or very small learning rates, specific to a neuron, can affect training convergence, careful tuning and gradient clipping can be necessary.
* **Optimiser Choice:** While Adam can be used, in some cases it might be more beneficial to use a custom adaptive optimiser class that updates parameter learning rates directly in the step function.
*   **Regularization:** Techniques like weight decay and batch normalization are very important for any neural network, and can interact in complex ways with adaptive methods. These should also be considered carefully while implementing custom per neuron update rules.

For deeper exploration, I would recommend:

*   **"Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville.** This book provides a detailed treatment of various optimization algorithms and is a foundational text in the field.
*   **Papers on Adam, RMSProp, and AdaGrad:** These optimizers are key to understanding per-parameter learning rates. Understanding their original papers provides insights into the methodologies of how they function.
*   **Research papers on layer-wise learning rates or adaptive learning rates.** There is a vast amount of literature about learning rate schedules and adaptive methodologies, they provide insights into novel techniques.

I've presented a progression, and in practice, I've found that starting from the more basic fixed learning rates per layer, and gradually moving up to the gradient statistic based learning rates works effectively. Remember, each implementation should be carefully evaluated against your specific neural network architecture, task and computational budget. This method is not always required, but it will be important to have in the arsenal of techniques as models become more complex and data becomes less readily available.
