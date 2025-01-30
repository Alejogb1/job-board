---
title: "Why do calculated and logged losses differ in PyTorch Lightning?"
date: "2025-01-30"
id: "why-do-calculated-and-logged-losses-differ-in"
---
In PyTorch Lightning, discrepancies between calculated and logged losses during training often stem from a difference in *when* and *how* these losses are computed, particularly when using features like automatic optimization and distributed training. I've seen this misalignment cause significant debugging headaches, often leading to confusion regarding actual model performance. Here's a breakdown of the primary contributors:

**1. The Timing of Loss Computation and Logging:**

The core issue is that the loss value used for backpropagation (the "calculated loss") isn't always the same as the value being logged for monitoring (the "logged loss"). This difference arises from the specific lifecycle within a PyTorch Lightning training step.

* **Calculated Loss:** This loss is the output of your `training_step` function. It’s the direct result of passing your training batch through your model and applying your loss function. This value is immediately used for the backpropagation pass to update your model's weights using the chosen optimizer. It's a critical value, representing the error of the model on a specific batch.

* **Logged Loss:** The logged loss is the value you explicitly pass to `self.log` (or a similar logging function). This value is typically computed after the primary calculation and is often, but not always, the same as the calculated loss.  It's designed to be an aggregate value useful for monitoring training progress. Its computation and timing are entirely controlled by you within `training_step`.

The critical difference is in the potential for modifications or intermediate calculations between these two stages. Here's where discrepancies commonly originate:

    * **Averaging/Aggregation:** The calculated loss is usually a tensor resulting from a single batch. You might want to log an average over multiple batches. PyTorch Lightning provides several strategies to accumulate or average losses before logging them, frequently using the `on_step=True` or `on_epoch=True` arguments in `self.log`. Logging on a per-step basis often leads to different values than logging the aggregated loss on epoch completion.
    * **Normalization:** Loss functions themselves may introduce subtle differences. When using loss functions across multiple devices in distributed training, or when using a reduction scheme other than `mean`, the reported calculated loss may require additional processing before an accurate representative loss is logged. I once spent hours tracking down a discrepancy caused by a loss function that returned individual sample losses when used directly in the `training_step`, needing an explicit mean before logging.
    * **Logging Frequency:** If you’re logging only at the end of an epoch, the loss you’re logging is an average over the *entire* training data for that epoch. This contrasts directly with the single-batch calculated loss.
    * **Automatic Optimization:** When using automatic optimization, Lightning automatically handles the backward pass. However, `self.log` operates after optimization, which might mean that the logging is performed on a slightly different state of the model than the initial computation of the calculated loss.

**2. Code Examples and Commentary:**

I've prepared several scenarios to illustrate these points:

**Example 1: Basic Scenario - Alignment**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss) # Logging same value used for backward
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


#Dummy data for simplicity
X = torch.randn(32, 10)
Y = torch.randn(32, 1)
train_data = [(X,Y)] * 10

trainer = pl.Trainer(max_epochs=1)
model = SimpleModel()
trainer.fit(model, train_dataloaders=train_data)
```

*Commentary:* In this basic example, the loss value calculated by `F.mse_loss` is directly passed to `self.log`. Therefore, the logged loss should closely match the calculated loss as it is the same tensor instance without any alteration before logging. Here the logged loss represents batch loss as the default is `on_step=True`.

**Example 2: Accumulation of Losses - Difference in Values**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class AccuModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)
        self.epoch_loss = 0
        self.batches = 0

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.epoch_loss += loss
        self.batches += 1
        return loss

    def on_train_epoch_end(self):
      epoch_loss_avg = self.epoch_loss / self.batches
      self.log('train_loss', epoch_loss_avg)
      self.epoch_loss = 0
      self.batches = 0
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


X = torch.randn(32, 10)
Y = torch.randn(32, 1)
train_data = [(X,Y)] * 10

trainer = pl.Trainer(max_epochs=1)
model = AccuModel()
trainer.fit(model, train_dataloaders=train_data)
```

*Commentary:* Here, the calculated loss is still the batch loss resulting from `F.mse_loss`. However, the logged loss is the *average* loss over the entire epoch, calculated in `on_train_epoch_end`. The `training_step` method computes the loss but then just returns it. The `on_train_epoch_end` logs a different quantity (average loss of the current epoch). This introduces a significant difference; the calculated loss will change with each batch, while the logged loss will only appear at the end of each training epoch, representing the mean of all batch losses of the current epoch. This pattern is very common in practice to avoid noisy per-step logs.

**Example 3: Manual Optimization and Loss Scaling**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class ManualOptimModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

    def training_step(self, batch, batch_idx, optimizer_idx=None):
      x, y = batch
      y_hat = self(x)
      loss = F.mse_loss(y_hat,y)
      self.log('train_step_loss', loss, on_step=True, on_epoch=False)
      self.log('train_epoch_loss', loss, on_step=False, on_epoch=True)
      opt = self.optimizers()
      opt.zero_grad()
      self.manual_backward(loss)
      opt.step()

      return loss
    
    def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=0.001)

X = torch.randn(32, 10)
Y = torch.randn(32, 1)
train_data = [(X,Y)] * 10

trainer = pl.Trainer(max_epochs=1)
model = ManualOptimModel()
trainer.fit(model, train_dataloaders=train_data)
```

*Commentary:* This example uses manual optimization, highlighting that even in scenarios where the primary calculated loss is used for backpropagation, the logged loss can be customized. I explicitly log two different quantities: one is the per-step batch loss logged with `on_step=True` and `on_epoch=False`. The other is the epoch loss logged with `on_step=False` and `on_epoch=True`. With manual optimization, you are responsible for the backward call and optimizer step, allowing the additional logging of different quantities such as those that require explicit averaging.

**3. Resource Recommendations:**

For a deeper understanding, I recommend reviewing the following resources:

*   **The PyTorch Lightning Documentation:** Specifically, the sections on the training loop, logging, and distributed training strategies. The documentation clearly outlines the steps involved in each, and where hooks like `on_train_epoch_start`, `on_train_batch_start` and similar are placed during a training run. This can greatly enhance your understanding of the various moments in the loop.
*   **PyTorch Lightning Example Code:** Examine the provided examples that explicitly address loss calculation, logging, and manual optimization to understand best practices. There are many specific examples for both automatic and manual optimization that can prove to be valuable.
*   **The PyTorch API Documentation:** Familiarize yourself with the behavior of various loss functions (`torch.nn`) and optimizers (`torch.optim`) to understand how they might interact with PyTorch Lightning's training steps. Having a deep understanding of the various optimizers and loss functions can help to better diagnose a problem as it arises.

Understanding the separation of calculation and logging, especially with automatic optimization and distributed training, is crucial for maintaining accurate and interpretable training results in PyTorch Lightning. Careful consideration of the timing and method of logging will reduce such discrepancies.
