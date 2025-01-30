---
title: "Why save the PyTorch optimizer state?"
date: "2025-01-30"
id: "why-save-the-pytorch-optimizer-state"
---
The primary justification for saving the PyTorch optimizer state lies in the need to resume model training from a specific point, often after interruptions or when experimenting with different training configurations.  A trained model’s weights are not the only information needed to continue learning; the optimizer's state, which encompasses variables like momentum buffers and learning rate adjustments, is equally critical for maintaining the integrity of the training process. Disregarding this state results in the optimizer restarting as if from initialization, potentially degrading performance and hindering convergence.

Specifically, gradient descent algorithms do not simply modify weights directly using the computed gradients. Instead, they incorporate historical gradient information, learning rate scheduling, and potentially other adaptive parameters that are tracked by the optimizer internally. Examples include momentum, which smooths updates and can accelerate training, and adaptive learning rate methods such as Adam, which maintain individual learning rates for each parameter. Resetting this state effectively erases the learning trajectory achieved thus far. Reverting to an untrained optimizer essentially throws away all information about previous weight changes and momentum, losing valuable accumulated information needed for future updates. Consequently, resuming training without the optimizer's state will often lead to worse results, such as longer training times or failure to converge.

Consider a situation where I've trained a deep neural network on a large image dataset for several epochs. Suddenly, the training process gets interrupted by a hardware failure. If I only saved the model weights, reloading that model and resuming training with a freshly initialized optimizer would be akin to forcing the model to relearn everything, ignoring all its prior training and the optimizer state. The momentum and adaptive learning rates are lost, requiring the training to start virtually from scratch. By saving the optimizer's state, I can continue the training process from where it left off, maintaining the benefit of previous iterations and significantly reducing wasted time and resources.

Here are three code examples demonstrating different aspects of saving and loading the optimizer state in PyTorch, along with commentary:

**Example 1: Basic Saving and Loading**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

# Initialize model and optimizer
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy training step
inputs = torch.randn(1, 10)
targets = torch.tensor([1])
criterion = nn.CrossEntropyLoss()

optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()

# Save the model and optimizer state
torch.save({
    'epoch': 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')

# Reset model and optimizer (for demonstration)
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Load the saved state
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

# Resume training
print(f"Resumed training at epoch: {epoch}, Loss: {loss}")
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()
print(f"Loss after resuming: {loss}")
```

*Commentary:* This example shows the basic process. We initialize a model and optimizer, perform a training step and then save their states using `torch.save`. We use a dictionary to save them alongside other information like the current epoch number and loss. Critically, the optimizer's state is stored separately via `optimizer.state_dict()`.  Then we re-initialize the model and optimizer to show loading the stored state, and resuming training. The printed loss demonstrates that the optimizer has correctly resumed learning from where we left off and continued to learn from the resumed training. The code explicitly shows the separation of concerns where saving the weights (model state) and optimizer state need to be done to continue training smoothly.

**Example 2: Saving with Learning Rate Schedulers**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Model and optimizer definitions (same as before)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Add a learning rate scheduler
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Dummy training loop
for epoch in range(10):
  inputs = torch.randn(1, 10)
  targets = torch.tensor([1])
  criterion = nn.CrossEntropyLoss()
  optimizer.zero_grad()
  outputs = model(inputs)
  loss = criterion(outputs, targets)
  loss.backward()
  optimizer.step()
  scheduler.step()

# Save the model, optimizer, and scheduler state
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'loss': loss,
}, 'checkpoint_scheduler.pth')

# Reset model, optimizer, and scheduler (for demonstration)
model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

# Load the saved state
checkpoint = torch.load('checkpoint_scheduler.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(f"Resumed training at epoch: {epoch}, Loss: {loss}")
for i in range(2):
  inputs = torch.randn(1, 10)
  targets = torch.tensor([1])
  criterion = nn.CrossEntropyLoss()
  optimizer.zero_grad()
  outputs = model(inputs)
  loss = criterion(outputs, targets)
  loss.backward()
  optimizer.step()
  scheduler.step()
  print(f"Loss after resuming: {loss}")
```

*Commentary:* This example extends the previous one to include a learning rate scheduler. Many real-world training scenarios utilize such schedulers to dynamically adjust the learning rate during training. Saving the state of the scheduler using `scheduler.state_dict()` is equally critical since it tracks the number of steps and the current learning rate value. Failing to save this state can cause a jump in the learning rate when training resumes and possibly instability. The resume section of code uses the saved information and the continuation is demonstrated by loss during the continuation.

**Example 3: Saving and Loading with Multiple Optimizers**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Model with two parts and two optimizers
class MultiPartNet(nn.Module):
    def __init__(self):
        super(MultiPartNet, self).__init__()
        self.part1 = nn.Linear(10, 5)
        self.part2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.part1(x)
        x = self.part2(x)
        return x

model = MultiPartNet()
optimizer1 = optim.Adam(model.part1.parameters(), lr=0.001)
optimizer2 = optim.SGD(model.part2.parameters(), lr=0.01)

# Dummy training loop
inputs = torch.randn(1, 10)
targets = torch.tensor([1])
criterion = nn.CrossEntropyLoss()

optimizer1.zero_grad()
optimizer2.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer1.step()
optimizer2.step()

# Save the model and both optimizer states
torch.save({
    'epoch': 1,
    'model_state_dict': model.state_dict(),
    'optimizer1_state_dict': optimizer1.state_dict(),
    'optimizer2_state_dict': optimizer2.state_dict(),
    'loss': loss
}, 'checkpoint_multi_opt.pth')

# Reset model and optimizers (for demonstration)
model = MultiPartNet()
optimizer1 = optim.Adam(model.part1.parameters(), lr=0.001)
optimizer2 = optim.SGD(model.part2.parameters(), lr=0.01)

# Load the saved state
checkpoint = torch.load('checkpoint_multi_opt.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer1.load_state_dict(checkpoint['optimizer1_state_dict'])
optimizer2.load_state_dict(checkpoint['optimizer2_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

print(f"Resumed training at epoch: {epoch}, Loss: {loss}")
optimizer1.zero_grad()
optimizer2.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer1.step()
optimizer2.step()
print(f"Loss after resuming: {loss}")
```

*Commentary:* This final example demonstrates saving multiple optimizer states when using different optimizers for distinct parts of the model. This occurs when training using a pre-trained model with freezing certain parameters or when one desires to control the learning of sub-networks in a different way. I have encountered situations where a large model’s main network used Adam and some newly added task-specific layers would use SGD, leading to this multiple optimizers set up. Each optimizer’s state is saved separately, and upon loading, the correct state is restored for each one using their respective names in a dictionary.

For further study on this area, the PyTorch documentation on saving and loading models and optimizers should be consulted. Additionally, the documentation on different optimizers (e.g., Adam, SGD) provides more details about their internal states.  Finally, research into best practices for deep learning training, such as learning rate scheduling, will also touch upon the need for correct checkpointing with optimizers. These resources will provide a deeper theoretical and practical understanding of the issues discussed.
