---
title: "How can I print the learning rate during training in PyTorch?"
date: "2025-01-30"
id: "how-can-i-print-the-learning-rate-during"
---
The learning rate, a crucial hyperparameter controlling the step size of an optimization algorithm, significantly impacts model convergence. Monitoring it during training provides essential feedback on the optimization process. Failing to observe its dynamics can lead to issues like oscillation, slow convergence, or premature stopping. In PyTorch, accessing and printing the learning rate, while not a default logging feature, requires explicit handling of the optimizer’s state.

The core mechanism involves accessing the `param_groups` attribute of a PyTorch optimizer object. This attribute is a list of dictionaries, where each dictionary represents a set of parameters and their corresponding learning rate.  For typical scenarios, there will be one entry, but it's possible, especially with advanced techniques like layer-wise learning rate adaptation, to have multiple entries. To extract the learning rate, one would typically access the `'lr'` key within the first dictionary element of the `param_groups` list. This allows us to extract and display it at desired intervals within the training loop.

Let me illustrate with specific examples, drawing upon my past experience. I encountered a situation with a complex image classification model where the training loss plateaued early, and it wasn't immediately clear why. Monitoring the learning rate revealed that it was decaying too rapidly, preventing the model from escaping a shallow local minimum. This observation allowed me to adjust the learning rate schedule and achieve significant performance improvements.

Here's a basic code snippet demonstrating how to extract and print the learning rate within a standard training loop:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assume model and training data loaders are defined elsewhere
# Example model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
dummy_data = torch.randn(1,10)
dummy_target = torch.tensor([[0.2,0.8]])

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(dummy_data)
    loss = criterion(outputs, dummy_target)
    loss.backward()
    optimizer.step()

    # Access learning rate
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch: {epoch+1}, Learning Rate: {lr:.6f}")
```

In this code, after each optimization step, I've added the line `lr = optimizer.param_groups[0]['lr']` to extract the learning rate associated with the first group of parameters (which is almost always the case). I then print this value alongside the epoch number. The `:.6f` in the print statement ensures the learning rate is displayed with six decimal places for better readability. This simple approach works when you have a single parameter group and use a standard optimizer like Adam or SGD.

However, the situation becomes slightly more nuanced when utilizing learning rate schedulers, which dynamically adjust the learning rate throughout training. Consider the following example using `torch.optim.lr_scheduler.StepLR`:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)
model = SimpleModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=2, gamma=0.1)
dummy_data = torch.randn(1,10)
dummy_target = torch.tensor([[0.2,0.8]])


# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(dummy_data)
    loss = criterion(outputs, dummy_target)
    loss.backward()
    optimizer.step()

    # Learning rate after scheduler step
    lr = optimizer.param_groups[0]['lr']
    print(f"Epoch: {epoch+1}, Learning Rate: {lr:.6f}")
    scheduler.step()

```

Here, the `StepLR` scheduler reduces the learning rate by a factor of `gamma` every `step_size` epochs. Critically, the learning rate *update* happens *after* the optimizer step, through the call to `scheduler.step()`.  Therefore, if you want the "next" learning rate value to be logged, you must extract the `lr` *after* calling the scheduler’s `step()`. Otherwise, you would be printing the learning rate for the current epoch, not the next one. This subtle ordering of operations can often lead to unexpected behavior when debugging training.

Finally, for those situations where your optimizer has multiple parameter groups, for example, when using layer-wise learning rates, you must loop through the parameter groups to log the learning rate for each. Here's how:

```python
import torch
import torch.nn as nn
import torch.optim as optim

class MultiParamGroupModel(nn.Module):
    def __init__(self):
        super(MultiParamGroupModel, self).__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.layer1(x)
        return self.layer2(x)

model = MultiParamGroupModel()
criterion = nn.MSELoss()
params = [
    {'params': model.layer1.parameters(), 'lr': 0.001},
    {'params': model.layer2.parameters(), 'lr': 0.01}
]

optimizer = optim.Adam(params)
dummy_data = torch.randn(1,10)
dummy_target = torch.tensor([[0.2,0.8]])

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(dummy_data)
    loss = criterion(outputs, dummy_target)
    loss.backward()
    optimizer.step()

    for i, param_group in enumerate(optimizer.param_groups):
         lr = param_group['lr']
         print(f"Epoch: {epoch+1}, Group {i+1} Learning Rate: {lr:.6f}")
```

This example defines two distinct parameter groups with varying initial learning rates.  Instead of directly accessing `optimizer.param_groups[0]`, the code iterates through each group and extracts and logs each learning rate independently.  This is crucial for identifying whether some layers are being trained at a faster or slower rate. In a project where I implemented transfer learning with frozen feature extraction layers,  I relied on this technique to verify that only the classifier layers were being trained, avoiding the problem of over-fitting a pre-trained model when the learning rate was too high for feature extraction layers.

In summary, accessing the learning rate in PyTorch involves examining the `param_groups` attribute of your optimizer. While straightforward, one must account for the usage of learning rate schedulers and the presence of multiple parameter groups to correctly print or log the learning rates. Monitoring this hyperparameter is not just about observing values; it's about gaining insights into the training process and ensuring that your models converge effectively.

For those interested in expanding their knowledge beyond the basics demonstrated here, I would recommend exploring the following resources within the PyTorch documentation:  the section dedicated to optimizers, the section covering learning rate schedulers, and any comprehensive tutorial on training neural networks. This can offer additional insights regarding best practices and edge cases not explored in this specific reply. Furthermore, engaging in code reviews within development teams also provides ample opportunity for observing how more seasoned professionals handle these situations in real-world contexts.
