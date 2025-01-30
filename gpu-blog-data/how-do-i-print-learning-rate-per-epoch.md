---
title: "How do I print learning rate per epoch with a functional learning rate scheduler?"
date: "2025-01-30"
id: "how-do-i-print-learning-rate-per-epoch"
---
When utilizing a functional learning rate scheduler within a deep learning training loop, monitoring the learning rate at each epoch requires accessing the scheduler’s internal state rather than simply relying on its initial configuration. I’ve encountered this need frequently, particularly when experimenting with custom schedules or debugging training instability. The core challenge lies in the functional nature; the scheduler modifies the learning rate via a provided function, often without explicitly storing each value, necessitating active querying for the current rate.

The functional learning rate scheduler, typically seen in frameworks like PyTorch, operates by providing a function, rather than a stateful object, to the optimizer. This function takes the current epoch (or iteration step) and, optionally, other parameters, and returns the adjusted learning rate. Because the scheduler isn't an instance storing the history of past learning rates, we must directly ask for the learning rate *at that specific moment* within the training loop. This isn’t a property we access directly; we invoke the scheduler function.

The most straightforward approach involves accessing the optimizer’s internal learning rate parameters during the training process and passing the epoch number into the scheduler function. This strategy lets us know the computed learning rate *for that specific* epoch based on the current training state. We typically apply the calculated learning rate using a `lambda` function that wraps the scheduler and passes the current step, before each optimizer step. Critically, we record the value of this calculated rate *before* it is applied to the optimizer.

Here's how this is implemented in code. The following example uses a simplified cosine annealing schedule, which, for the purposes of clarity, is defined in a plain Python function.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def cosine_annealing_schedule(epoch, total_epochs, base_lr, min_lr):
  """Simplified Cosine Annealing Schedule."""
  progress = float(epoch) / float(total_epochs)
  cos_value = 0.5 * (1 + torch.cos(torch.tensor(torch.pi) * progress))
  return min_lr + (base_lr - min_lr) * cos_value

# Dummy Model and Dataset
model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)

#Hyperparameters
total_epochs=20
base_lr = 0.1
min_lr = 0.01

# Function using lambda to generate learning rate
lr_lambda = lambda epoch: cosine_annealing_schedule(epoch, total_epochs, base_lr, min_lr)

# Scheduler wraps a lambda function
scheduler = LambdaLR(optimizer, lr_lambda)

for epoch in range(total_epochs):

    # Obtain learning rate before any optimizer update
    current_lr = scheduler.optimizer.param_groups[0]['lr']

    print(f"Epoch: {epoch}, Learning Rate: {current_lr:.6f}")

    #Dummy Training Step
    inputs = torch.randn(32,10)
    labels = torch.randn(32,2)
    outputs = model(inputs)
    loss = nn.MSELoss()(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step() # Update the scheduler after the step.

```

In this first example, I define a simplified cosine annealing function and utilize a `LambdaLR` scheduler. The key to printing the learning rate is the line `current_lr = scheduler.optimizer.param_groups[0]['lr']`. This accesses the current learning rate from the optimizer's internal state before each training step. The `scheduler.step()` method updates the scheduler's internal parameters – and consequently the learning rate for the next step – after the weights have been updated by the optimizer.

Now, let's consider a scenario where we want to use a more complex custom schedule, potentially with a different learning rate for each parameter group within the optimizer. In this instance, the approach is similar, but we need to iterate through the optimizer's `param_groups`.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

def complex_schedule(epoch, base_lrs):
  """Custom learning rate scheduler."""
  lrs = []
  for base_lr in base_lrs:
      if epoch < 5:
          lrs.append(base_lr)
      elif epoch < 10:
          lrs.append(base_lr * 0.5)
      else:
          lrs.append(base_lr * 0.25)
  return lrs

model = nn.Linear(10,2)

#Example groups
params1 = list(model.parameters())[:1]
params2 = list(model.parameters())[1:]
base_lrs = [0.1, 0.05]

optimizer = optim.SGD([{'params': params1, 'lr': base_lrs[0]}, {'params': params2, 'lr':base_lrs[1]}], lr=0.1)

lr_lambda = lambda epoch: complex_schedule(epoch, base_lrs)
scheduler = LambdaLR(optimizer, lr_lambda)
total_epochs=15

for epoch in range(total_epochs):
    print(f"Epoch: {epoch}")
    for i, param_group in enumerate(optimizer.param_groups):
      current_lr = param_group['lr']
      print(f"  Param Group {i}, Learning Rate: {current_lr:.6f}")

    inputs = torch.randn(32,10)
    labels = torch.randn(32,2)
    outputs = model(inputs)
    loss = nn.MSELoss()(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

```

In this second code block, we create two parameter groups, each with its own learning rate, within the optimizer, and then assign a complex custom schedule that affects each parameter group differently. Within the training loop, the code iterates over each `param_group`, obtaining and printing the corresponding current learning rate. Again, this is accessed using the `param_group['lr']` field. This example highlights the flexibility of the `LambdaLR` scheduler and the need to carefully inspect optimizer structure in complex scenarios. The scheduler step is again called *after* the optimizer step.

Finally, let's modify the first example to show how the schedule is implemented using a custom class, instead of a function. Note that, in practice, creating the schedule as a Python function is usually preferred unless it needs to store some internal states.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR

class CosineAnnealingScheduler:
    def __init__(self, total_epochs, base_lr, min_lr):
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr

    def __call__(self, epoch):
        progress = float(epoch) / float(self.total_epochs)
        cos_value = 0.5 * (1 + torch.cos(torch.tensor(torch.pi) * progress))
        return self.min_lr + (self.base_lr - self.min_lr) * cos_value


model = nn.Linear(10, 2)
optimizer = optim.SGD(model.parameters(), lr=0.1)

total_epochs=20
base_lr = 0.1
min_lr = 0.01

custom_schedule = CosineAnnealingScheduler(total_epochs, base_lr, min_lr)

lr_lambda = lambda epoch: custom_schedule(epoch)
scheduler = LambdaLR(optimizer, lr_lambda)

for epoch in range(total_epochs):

    current_lr = optimizer.param_groups[0]['lr']

    print(f"Epoch: {epoch}, Learning Rate: {current_lr:.6f}")

    inputs = torch.randn(32,10)
    labels = torch.randn(32,2)
    outputs = model(inputs)
    loss = nn.MSELoss()(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
```

This example uses the same cosine annealing functionality as the first example, but encapsulates the functionality in a class. Here, the `__call__` method makes this class callable, enabling its use with `LambdaLR`. The extraction of the learning rate and the scheduler update proceed exactly as in the first example. This variant illustrates that the functional nature remains the same whether the schedule is a plain function or a callable object, and thus extracting the learning rate is implemented identically.

In summary, printing the learning rate per epoch with a functional learning rate scheduler requires obtaining the learning rate directly from the optimizer's parameter groups before the update step, since the scheduler does not implicitly store this information. The scheduler is updated via `step()` after the weights are updated in each optimizer step, to prepare the new learning rate for the next step. For complex cases, iterating through each parameter group is necessary. In addition, the schedule can be defined with Python functions or callable objects, without impacting the method of extracting and printing the current learning rate.

For further exploration of this topic, I recommend consulting the PyTorch documentation for `torch.optim.lr_scheduler.LambdaLR` and the `torch.optim.Optimizer` class. Additionally, various online deep learning tutorials often showcase different scheduler implementations and best practices, which can be helpful for practical applications. Examining the internal structure of optimizers through debugging techniques can also provide a deeper understanding. Finally, it can be useful to read research papers which utilize learning rate schedules; these often detail specific implementations or choices made during training.
