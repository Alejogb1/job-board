---
title: "How to access all batch outputs at epoch end in PyTorch Lightning callbacks?"
date: "2025-01-30"
id: "how-to-access-all-batch-outputs-at-epoch"
---
My experience working with complex PyTorch Lightning training loops revealed a common need: accessing all batch outputs at the end of each training epoch. Standard callbacks typically operate on individual batches or aggregated metrics, not the raw, collected outputs. This requires a specific approach that leverages the internal mechanisms of PyTorch Lightning. The key is to modify the training step to collect batch outputs and then access them within a custom callback.

The default behavior in PyTorch Lightning does not inherently store batch outputs for later access. The `training_step` method calculates loss and other metrics, which are then accumulated by the trainer. Directly retrieving *all* raw outputs requires some intervention. My approach centers on three steps: 1) modifying the `training_step` to save outputs, 2) storing these outputs in a place accessible to a callback, and 3) defining a callback to retrieve and process them at epoch end.

**1. Modifying the `training_step` Method**

Initially, a standard `training_step` typically returns a loss tensor. To collect outputs, the `training_step` must also return the output alongside the loss. Here’s an example illustrating the necessary change:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        # Return both loss and output. Crucial step!
        return {'loss': loss, 'logits': logits}
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)
```

Here, the `training_step` now returns a dictionary containing both the calculated loss and the logits. The inclusion of the raw output from the model (`logits` in this instance) is the critical alteration. Without this, subsequent steps to store these outputs would be impossible. The loss, as before, is used for backpropagation by Lightning.

**2. Storing Batch Outputs in the Callback**

Once the `training_step` is returning the raw output, these need to be captured. The standard `on_train_batch_end` hook of `pytorch_lightning.Callback` is useful here. It provides access to the outputs from `training_step` for each batch. Crucially, I create a dictionary in the Lightning module to store outputs. This avoids creating a global variable, which is prone to errors with multi-process training.

```python
class OutputCollectorCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.epoch_outputs = [] # Reset for the new epoch

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.epoch_outputs.append(outputs)
```

In the `on_train_epoch_start` hook, I initialize `pl_module.epoch_outputs` with an empty list for each epoch, this prevents memory issues by avoiding accumulation across epochs. Within `on_train_batch_end`, I append the output dictionary returned by `training_step` to this list. This stores the output of each batch during training.

**3. Processing Outputs in a Dedicated Callback at Epoch End**

With the raw outputs now stored, a second callback accesses and processes them. The key here is the `on_train_epoch_end` hook. This provides the necessary access to the collected outputs once the epoch finishes.

```python
class OutputProcessingCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
         all_outputs = pl_module.epoch_outputs
         # Example processing: Collect all logits and calculate an aggregate metric
         all_logits = torch.cat([batch_output['logits'] for batch_output in all_outputs])
         avg_logits = torch.mean(all_logits)
         trainer.logger.log_metrics({'avg_logits': avg_logits}, step=trainer.current_epoch)
```

This `OutputProcessingCallback` accesses `pl_module.epoch_outputs` accumulated by `OutputCollectorCallback`. In this example, I am extracting all the logits tensors and concatenating them together, then calculating an average value. This processed data is then logged to the trainer's logger. The processing steps here are variable based on specific needs, but the mechanism to extract all outputs remains the same. Note, the `torch.cat` operation requires a list of tensors that are compatible, which might necessitate more care depending on the `training_step` output.

**Putting It All Together**

To implement this entire setup, you need to:

1.  Define the `SimpleModel` (or your equivalent). Ensure that it returns both the loss and model output in `training_step`.
2.  Include the `OutputCollectorCallback` and `OutputProcessingCallback` when initializing the `Trainer`.

Here is the complete minimal example:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class SimpleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
        self.epoch_outputs = [] #Initialize the list here
    def forward(self, x):
        return self.fc(x)
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        return {'loss': loss, 'logits': logits}
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

class OutputCollectorCallback(pl.Callback):
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.epoch_outputs = []

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.epoch_outputs.append(outputs)

class OutputProcessingCallback(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
         all_outputs = pl_module.epoch_outputs
         all_logits = torch.cat([batch_output['logits'] for batch_output in all_outputs])
         avg_logits = torch.mean(all_logits)
         trainer.logger.log_metrics({'avg_logits': avg_logits}, step=trainer.current_epoch)
if __name__ == '__main__':
    model = SimpleModel()
    trainer = pl.Trainer(max_epochs=2, callbacks=[OutputCollectorCallback(), OutputProcessingCallback()])
    train_dataloader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randint(0, 2, (100,))), batch_size=10)
    trainer.fit(model, train_dataloader)
```

This complete example demonstrates the modification of `training_step` for output collection, storage in a model attribute with a custom callback, and access within a separate callback at epoch end. The ability to access these outputs becomes particularly useful for tasks such as complex model analysis, visualization, or custom metric calculations that need to process all batch outputs together.

**Resource Recommendations**

To understand this implementation fully, reviewing the PyTorch Lightning documentation for callbacks is crucial. Specifically, focusing on the `Callback` class and its associated lifecycle methods will provide a thorough understanding of the system's mechanics. Examining the training loop documentation, particularly the section covering the `training_step`, will clarify the role and requirements of the function. Furthermore, delving into the `Trainer` documentation will allow deeper understanding of how callbacks are integrated into the training process. These documents contain detailed descriptions of the objects and methods used in this solution. Finally, understanding the basic workings of the PyTorch `torch.cat` command will help in modifying and applying this implementation in different situations.
