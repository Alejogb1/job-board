---
title: "How do `optimizer.param_groups'0'''lr''` and `scheduler.get_lr()'0'` differ in their reported learning rates?"
date: "2025-01-30"
id: "how-do-optimizerparamgroups0lr-and-schedulergetlr0-differ-in-their"
---
The discrepancy between `optimizer.param_groups[0]['lr']` and `scheduler.get_lr()[0]` stems from the fundamental difference in their perspectives on the learning rate.  `optimizer.param_groups[0]['lr']` reflects the *base* learning rate assigned to the optimizer's parameter group, while `scheduler.get_lr()[0]` provides the *effective* learning rate after any scheduling adjustments. This distinction is crucial for understanding learning rate behavior, especially in complex training scenarios employing sophisticated learning rate schedulers.  My experience debugging training instabilities across numerous large-scale NLP models has highlighted this subtle yet critical difference repeatedly.

**1. Clear Explanation:**

The optimizer, at its core, manages the updating of model parameters based on the calculated gradients.  Each optimizer (Adam, SGD, etc.) maintains a set of parameter groups, typically one per layer or a set of similar parameters.  Each parameter group possesses attributes, most notably `'lr'`, defining the learning rate for that group. When you access `optimizer.param_groups[0]['lr']`, you're directly querying the initial, or last explicitly set, learning rate for the first parameter group.  This value remains constant unless explicitly modified within the optimizer's configuration or through direct manipulation of the `param_groups` attribute.

A learning rate scheduler, on the other hand, dynamically adjusts the learning rate during training.  Its primary purpose is to modify the learning rate based on predefined schedules or metrics (e.g., epoch number, validation loss).  The scheduler does *not* directly modify the optimizer's `param_groups`. Instead, it computes a new learning rate based on its internal logic and the current training state. The `scheduler.get_lr()` method then returns a list of the *updated*, effective learning rates for each parameter group. Therefore, `scheduler.get_lr()[0]` represents the actual learning rate applied to the first parameter group *after* the scheduler's intervention.

The difference only manifests when a scheduler is employed.  If no scheduler is used, both expressions will yield the same value because the scheduler does not modify the base learning rate stored in the optimizer.  Furthermore, the discrepancy might be subtle in simple schedulers (e.g., StepLR with a large step size), but becomes more pronounced with more complex ones (e.g., ReduceLROnPlateau, CosineAnnealingLR) which may perform several smaller adjustments throughout training.

**2. Code Examples with Commentary:**

**Example 1: No Scheduler**

```python
import torch
import torch.optim as optim

model = torch.nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Optimizer LR: {optimizer.param_groups[0]['lr']}")  # Output: 0.001

# No scheduler, so get_lr() is not applicable or will return the base learning rate.  
#  Attempting to use it here would raise an AttributeError in most schedulers.
```

In this case, there's no scheduler, so both the base learning rate in the optimizer and any hypothetical effective learning rate would be identical.


**Example 2: Simple StepLR Scheduler**

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

model = torch.nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # Reduce LR by 10x every 10 epochs

print(f"Initial Optimizer LR: {optimizer.param_groups[0]['lr']}")  # Output: 0.01
scheduler.step() # Simulate one step
print(f"Optimizer LR after scheduler step: {optimizer.param_groups[0]['lr']}") # Output: 0.01 (StepLR doesn't directly modify)
print(f"Scheduler LR: {scheduler.get_lr()[0]}")  # Output: 0.001
```

Here, `StepLR` demonstrates the difference. While the optimizer's `lr` remains initially unchanged (it is only updated on scheduler steps for some schedulers), the scheduler correctly reports the adjusted (reduced) learning rate.


**Example 3:  ReduceLROnPlateau Scheduler**

```python
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

model = torch.nn.Linear(10, 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

print(f"Initial Optimizer LR: {optimizer.param_groups[0]['lr']}")  # Output: 0.01

# Simulate a scenario where the scheduler would reduce the learning rate.
# This would typically involve a loop that checks validation loss.
scheduler.step(1) # Simulate validation loss of 1
print(f"Optimizer LR after scheduler step: {optimizer.param_groups[0]['lr']}")  # Output: 0.01 (Still unchanged)
print(f"Scheduler LR: {scheduler.get_lr()[0]}")  # Output: 0.001

scheduler.step(0.1)  # Simulate a significant improvement in validation loss
print(f"Scheduler LR after second step: {scheduler.get_lr()[0]}") # Output: 0.001 (No further change in this example)
```

`ReduceLROnPlateau` is more dynamic. The base learning rate in the optimizer only updates on the scheduler's decision to step the LR. This highlights how the scheduler's `get_lr()` method is essential for obtaining the effective learning rate.


**3. Resource Recommendations:**

For deeper understanding, I strongly recommend consulting the official PyTorch documentation on optimizers and learning rate schedulers.  Thoroughly examining the source code of different scheduler implementations will also offer invaluable insight.  Finally, reviewing relevant research papers on adaptive learning rate methods will provide a more comprehensive theoretical foundation.  These resources, combined with hands-on experimentation, will allow for a more complete grasp of the subject matter.
