---
title: "Does PyTorch Lightning average metrics per epoch?"
date: "2025-01-30"
id: "does-pytorch-lightning-average-metrics-per-epoch"
---
PyTorch Lightning, by default, does *not* average metrics across an entire epoch.  Instead, it logs metrics at the end of each individual batch. This behavior is crucial to understand when interpreting training logs and designing effective monitoring strategies. My experience debugging a multi-GPU training script revealed this nuance quite forcefully; a seemingly stable training run exhibited significant batch-to-batch fluctuation in reported metrics, masking a subtle but critical instability in the model's gradient updates.  This necessitated a deep dive into Lightning's metric aggregation mechanisms.

**1.  Explanation of PyTorch Lightning's Metric Handling**

PyTorch Lightning utilizes the `LightningModule`'s `training_step`, `validation_step`, and `test_step` methods for calculating metrics.  Within these methods, metrics are typically logged using the `self.log()` method.  Crucially,  `self.log()` accepts a `prog_bar` argument, which determines whether the metric is displayed in the progress bar during training. This argument, however, doesn't dictate the averaging behavior. The default behavior is to log the metric's value *for the current batch*, not an accumulated average over the epoch.  While the progress bar might show a rolling average for visual convenience, the underlying logged data reflects the per-batch values.  This distinction is essential for accurate analysis and reproduction of results.

The averaging happens outside of the `LightningModule` within the `Trainer` class.  The `Trainer`'s logging callbacks, such as the default TensorBoard logger,  can aggregate the logged metrics, often computing epoch averages.  However, this aggregation is a post-processing step; the fundamental logging mechanism in PyTorch Lightning is batch-wise. This design decision allows for fine-grained monitoring and debugging.  Identifying anomalies within an epoch becomes significantly easier when per-batch metrics are available. It also enables more sophisticated logging and analysis, like plotting the metric progression during the epoch.  For instance, observing a consistent upward trend in a loss metric throughout an epoch might indicate a problematic learning rate scheduler or optimizer configuration.

**2. Code Examples Illustrating Metric Behavior**

The following examples demonstrate how metrics are logged and accessed in PyTorch Lightning, emphasizing the per-batch logging behavior.

**Example 1:  Default per-batch logging**

```python
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.accuracy = Accuracy()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss) # Per-batch logging
        acc = self.accuracy(y_hat, y)
        self.log('train_acc', acc, prog_bar=True) # Per-batch, shown on progress bar
        return loss

    # ... (rest of the LightningModule) ...

```
In this example, both `train_loss` and `train_acc` are logged per batch.  The progress bar might display a moving average for `train_acc`, but the underlying logs contain per-batch values.

**Example 2:  Manual epoch-level averaging**

While PyTorch Lightning doesn't automatically average metrics at the epoch level,  it's straightforward to implement this manually.

```python
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
import torch

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.accuracy = Accuracy()
        self.epoch_loss = 0.0
        self.epoch_acc = 0.0
        self.batch_count = 0

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.epoch_loss += loss.item()
        self.epoch_acc += acc.item()
        self.batch_count += 1
        return loss

    def training_epoch_end(self, outputs):
        avg_loss = self.epoch_loss / self.batch_count
        avg_acc = self.epoch_acc / self.batch_count
        self.log('train_epoch_loss', avg_loss)
        self.log('train_epoch_acc', avg_acc)
        self.epoch_loss = 0.0
        self.epoch_acc = 0.0
        self.batch_count = 0


    # ... (rest of the LightningModule) ...
```
Here, we explicitly accumulate the metrics throughout the epoch and calculate the average in `training_epoch_end`. This approach provides both per-batch and epoch-level metrics.


**Example 3: Using `reduce_fx` for metric reduction**

Certain metrics, particularly those from the `torchmetrics` library, allow for specifying a reduction function.

```python
import pytorch_lightning as pl
from torchmetrics import Accuracy
import torch

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.accuracy = Accuracy(reduce='mean') # Explicitly average across batch

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = self.accuracy(y_hat, y)
        self.log('train_acc', acc) # will log the batch-averaged metric
        return loss

    # ... (rest of the LightningModule) ...

```
This showcases how you can leverage `torchmetrics` to get batch-averaged metrics directly, though it's important to note this doesn't automatically average over the epoch.


**3. Resource Recommendations**

The official PyTorch Lightning documentation is your primary resource. Carefully review the sections on `LightningModule`, the `self.log()` method, and the available logging callbacks. Pay close attention to the `prog_bar`, `logger`, and `sync_dist` arguments of the `self.log()` function.  Consulting the documentation for the `torchmetrics` library is also vital, as it contains detailed explanations of various metrics and their reduction capabilities.  Finally, exploring examples and tutorials provided within the PyTorch Lightning community can offer practical insights and best practices.  Thorough testing and experimentation are crucial to confirming your understanding and adapting these techniques to your specific use case.
