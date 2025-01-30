---
title: "How can I implement exponential moving average decay for PyTorch variables?"
date: "2025-01-30"
id: "how-can-i-implement-exponential-moving-average-decay"
---
Exponential moving average (EMA) decay, frequently employed in model training and optimization, provides a smoothed representation of a variable's evolution, thereby improving stability and generalization. I've utilized this technique extensively across various deep learning projects, particularly when dealing with noisy gradients or when a more robust estimate of a parameter's state is required. Implementing it correctly in PyTorch requires attention to detail, especially given PyTorch's dynamic graph construction.

The core concept of EMA involves updating a 'shadow' variable – an exponentially weighted average – alongside the primary training variable. This shadow variable doesn't directly participate in the backward pass but reflects a smoothed, historical view of the primary variable. The update rule typically takes this form:

*   `shadow_variable = decay * shadow_variable + (1 - decay) * variable`

where `decay` is a hyperparameter between 0 and 1, usually close to 1 (e.g., 0.999). A higher decay rate gives greater emphasis to the past. This averaging effect can be particularly valuable in scenarios like batch normalization, where smoothing statistics across batches is crucial, or for averaging model weights at the end of training, resulting in models that often generalize better.

The crucial aspect in PyTorch is ensuring that the shadow variable correctly tracks the training variable throughout the backpropagation process. This involves carefully maintaining and updating the shadow variable when the training variable is modified via an optimizer. In practice, this means you generally need to update the shadow variable *after* each optimizer step has been applied to the main variable.

I've found that naive implementations, particularly ones that try to calculate the shadow variable directly from gradients (which one does *not* do with EMA), lead to incorrect results, typically due to misaligned computations during the training loop. We operate on the parameter value itself, not the gradient.

Here are three different code examples I use regularly, each presenting different ways to integrate this into a PyTorch workflow:

**Example 1: Encapsulated EMA in a Class**

This approach encapsulates the EMA logic within a class, which helps maintain a clean separation of concerns and provides reusability.

```python
import torch

class EMA:
    def __init__(self, decay):
        self.decay = decay
        self.shadow = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def update(self, name, val):
        assert name in self.shadow, "Parameter not registered with EMA"
        self.shadow[name] = (
            self.decay * self.shadow[name] + (1 - self.decay) * val
        )

    def get(self, name):
        assert name in self.shadow, "Parameter not registered with EMA"
        return self.shadow[name]

    def copy_to(self, name, dest):
       assert name in self.shadow, "Parameter not registered with EMA"
       dest.data.copy_(self.shadow[name])
```

*   **Explanation**: The `EMA` class manages the shadow variables in a dictionary, keyed by the parameter name. The `register` method initializes the shadow value with a copy of the initial parameter, while the `update` method applies the moving average formula. The `get` method returns the stored shadow variable for a specified parameter name. The `copy_to` method copies the shadow value to another given tensor.

*   **Usage**: You would initialize an `EMA` object with a chosen decay value. Then, for each trainable parameter, you use the `register` method at the beginning of training and then, after each optimizer step, call `update`. When you need to use the EMA, you access the shadowed values using the `get` method or copy them back using the `copy_to` method. This method supports usage with a variable without making any changes to the training loop, but the variable's name must be tracked.

**Example 2: Integration within a Training Loop**

This example directly shows how EMA is often implemented within a typical PyTorch training loop, focusing on optimizer.step() and weight updates.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Example model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
decay = 0.999
ema_params = {}

# Initialization
for name, param in model.named_parameters():
    ema_params[name] = param.clone()

# Training loop
for epoch in range(10):
    for i in range(100): # Assume dataloader is being used here
        inputs = torch.randn(1, 10)
        targets = torch.randint(0, 2, (1,)).long()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()

        # EMA update
        with torch.no_grad():
            for name, param in model.named_parameters():
                ema_params[name] = (
                    decay * ema_params[name] + (1 - decay) * param
                )

# Copy EMA parameters back into model for evaluation or testing
with torch.no_grad():
    for name, param in model.named_parameters():
        param.copy_(ema_params[name])
```
*   **Explanation:** In this example, I iterate through the model's parameters, creating shadow variables of the same shape at initialization.  After every training step (optimizer.step()), I loop through the parameters again, calculating and assigning the updated shadow parameter. I use `torch.no_grad()` to prevent updating the shadow variable from impacting gradients within the primary model. Finally, I copy the EMA parameters back into the model parameters before final usage.

*   **Usage**: This code clearly shows the correct moment for EMA calculation (post-optimization) and includes a full loop example.  Note that the use of `named_parameters` makes the loop code more robust for use with more complex models. This is a straightforward, but fairly direct way of implementing EMA.

**Example 3: Using a context manager**

This method uses a context manager to handle the weight swap and copy back more safely.  It is the most complex but can also be the most convenient to use.

```python
import torch
import torch.nn as nn
import torch.optim as optim

class EMAManager:
    def __init__(self, model, decay):
      self.model = model
      self.decay = decay
      self.shadow = {}
      self._initial = {}
      self._is_active = False

    def _clone_params(self):
      for name, param in self.model.named_parameters():
        self.shadow[name] = param.clone()
        self._initial[name] = param.clone()


    def update(self):
       if not self._is_active:
          self._clone_params()
          self._is_active = True
       with torch.no_grad():
         for name, param in self.model.named_parameters():
           self.shadow[name] = (self.decay * self.shadow[name]) + ((1 - self.decay) * param)

    def get_shadow(self, name):
        return self.shadow[name]
    
    def swap_weights(self):
        if not self._is_active:
           self._clone_params()
        for name, param in self.model.named_parameters():
           self._initial[name] = param.clone() # save current weights
           param.data.copy_(self.shadow[name])

    def restore_weights(self):
        if not self._is_active:
           return
        for name, param in self.model.named_parameters():
           param.data.copy_(self._initial[name])

    def __enter__(self):
       self.swap_weights()
       return self

    def __exit__(self, exc_type, exc_val, exc_tb):
      self.restore_weights()

# Example model (same as before)
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)

model = SimpleNet()
optimizer = optim.Adam(model.parameters(), lr=0.01)
decay = 0.999
ema_manager = EMAManager(model, decay)

for epoch in range(10):
    for i in range(100):
        inputs = torch.randn(1, 10)
        targets = torch.randint(0, 2, (1,)).long()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.cross_entropy(outputs, targets)
        loss.backward()
        optimizer.step()
        ema_manager.update()

# At the end of training or during evaluation:
with ema_manager:
    # model is running with EMA values
    model.eval()
    # ... run tests or validation here ...
    model.train()

# Model automatically reverts to normal training weights upon exiting the context
# Can run regular training or validation with regular weights now.
```

*   **Explanation**:  The `EMAManager` now handles the EMA operations, saving initial parameter values upon the first update. `swap_weights` copies the EMA weights into the model before switching contexts (like evaluation), and `restore_weights` returns the model to its training state afterwards. The context manager allows using the EMA values only within specific blocks of code, without the need to swap weights manually. The context manager keeps the code clean and readable.

*   **Usage**: Here, you initialize the `EMAManager` with your model and the chosen decay. Call `update` after each optimization step. During evaluations, enter the context manager (the `with` statement) to run the model with EMA weights, which are then returned to training values when exiting the `with` statement.

**Resource Recommendations**

For a deeper theoretical understanding of exponential moving averages, I recommend consulting resources on signal processing and time series analysis. Texts covering numerical optimization often include detailed discussions on the role of moving averages in optimization algorithms. For more implementation focused guidance, look for blog posts and online discussions regarding deep learning training practices, specifically those addressing model smoothing and ensembling techniques. Additionally, exploring the documentation for optimizers and other deep learning libraries would provide further context.
