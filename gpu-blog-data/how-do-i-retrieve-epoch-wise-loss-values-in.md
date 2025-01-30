---
title: "How do I retrieve epoch-wise loss values in PyTorch Lightning?"
date: "2025-01-30"
id: "how-do-i-retrieve-epoch-wise-loss-values-in"
---
The fundamental mechanism for accessing epoch-wise loss values in PyTorch Lightning lies within the `LightningModule`'s lifecycle hooks and the training loop itself. Unlike raw PyTorch, which requires manual tracking, Lightning provides standardized callbacks and logging mechanisms designed for this purpose. My experience building a multi-modal image classification system, where monitoring training progress was paramount, has underscored the utility of this abstraction.

To clarify, PyTorch Lightning doesn't directly expose epoch-wise loss values through a single, readily available variable. Instead, these values are aggregated and logged by the framework. The key is understanding that loss calculations occur during training steps (e.g., `training_step`) and are subsequently accumulated and averaged across batches within an epoch. This accumulated and averaged value, often a scalar, is what we are ultimately trying to retrieve. Accessing these epoch summaries hinges on logging the loss during training and then extracting the logged values.

The most straightforward method involves using Lightning's automatic logging infrastructure in conjunction with the `log` method within the `training_step` of a `LightningModule`. When `self.log("train_loss", loss)` is invoked, Lightning will automatically collect and aggregate these values for each epoch, making them accessible post-training. We do not explicitly calculate epoch-wise averages ourselves. The system does it behind the scenes.

Let’s look at three specific scenarios and code examples, each illustrating different aspects of retrieving epoch-wise loss:

**Example 1: Basic Logging and Retrieval with `Trainer`**

This example showcases the most common scenario where we simply log a loss value, and then retrieve it post-training using the `Trainer`'s return value.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

class SimpleClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss) # Log the loss
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)

# Dummy data
X_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,))
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32)

model = SimpleClassifier()
trainer = pl.Trainer(max_epochs=3)
trainer.fit(model, train_loader)

# Retrieve logged metrics
metrics = trainer.logged_metrics
print(metrics)
```

In this snippet, within `training_step`, `self.log("train_loss", loss)` registers the loss value under the key "train_loss".  After training completes using `trainer.fit()`, `trainer.logged_metrics` contains a dictionary where each key represents the metric logged and its value is the final recorded result which is an average across the entire training period. Therefore, `metrics["train_loss"]` represents the average training loss *for the final epoch*.

It's essential to recognize that `trainer.logged_metrics` provides the final average value, not a per-epoch list. The framework aggregates the per-batch values to arrive at this single value. We need more involved logging to obtain epoch-specific details.

**Example 2: Per-Epoch Logging with Callbacks**

This example demonstrates how to use a callback to store per-epoch loss values.  This technique provides an accessible record of the loss for each training epoch.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

class EpochLossCallback(Callback):
    def __init__(self):
        super().__init__()
        self.epoch_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        logs = trainer.logged_metrics
        if "train_loss" in logs:
          self.epoch_losses.append(logs["train_loss"])

class SimpleClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)


X_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,))
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32)

model = SimpleClassifier()
epoch_loss_callback = EpochLossCallback()
trainer = pl.Trainer(max_epochs=3, callbacks=[epoch_loss_callback])
trainer.fit(model, train_loader)

print(epoch_loss_callback.epoch_losses)
```

Here, `EpochLossCallback` leverages the `on_train_epoch_end` hook. At the end of each training epoch, the callback accesses the training metrics (including the logged `train_loss`) via `trainer.logged_metrics` and stores the most recent epoch average.  This provides a list `epoch_loss_callback.epoch_losses`, each item of which corresponds to the average training loss for a particular epoch. This offers a better view than the final aggregated average.

This approach provides a clear advantage when visualizing loss curves or performing early stopping based on loss trends. It also demonstrates how callbacks can interact with the framework’s internal state.

**Example 3:  Accessing Aggregated Metrics during training**

This final example, while not directly retrieving values for *each* epoch, clarifies how to access aggregated metrics during training, within the training loop itself. This is helpful when one needs the average to log other metrics relative to it.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

class SimpleClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.cross_entropy(y_hat, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True) # Log on both steps and epoch
        current_loss = self.trainer.logged_metrics["train_loss"] #get the current aggregated train loss
        self.log("loss_plus_one", current_loss + 1, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.01)

X_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,))
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32)

model = SimpleClassifier()
trainer = pl.Trainer(max_epochs=3)
trainer.fit(model, train_loader)

metrics = trainer.logged_metrics
print(metrics)
```

Here, the key addition is `on_step=True, on_epoch=True` during the logging of `train_loss`. This specifies that the loss is logged for each step (batch) and then also aggregated across the epoch. Crucially, within the `training_step` method, I can immediately access the currently aggregated loss through `self.trainer.logged_metrics["train_loss"]`. Note that because we have logged the metric on both steps and epoch, the aggregated metric is updated throughout the training process. Therefore, if you need to perform per-step operations related to the aggregated loss, you must log on_step=True.

This example highlights a crucial fact: Lightning maintains a running average of logged metrics *within the training loop*. You can, in this way, access these average values during the epoch, but it's critical to be aware that these are averages to the *current point in the epoch* and not necessarily the final average for the epoch. Additionally, the aggregated value is *not* accessible outside the `training_step`, even if you log with `on_epoch=True`. It must be accessed with `trainer.logged_metrics`.

In conclusion, accessing epoch-wise loss values in PyTorch Lightning requires understanding the framework's logging system and the lifecycle hooks. The `self.log` method provides the basis for metric collection. To obtain per-epoch summaries, callbacks such as illustrated in example 2 are essential. Additionally, example 3 shows how to utilize the aggregated metric during training for intermediate calculations. For learning about advanced callback implementations and the nuances of different logging options, the official PyTorch Lightning documentation and examples, are invaluable resources. Experimentation with these methods will demonstrate the power of Lightning's abstraction for structured metric management.
