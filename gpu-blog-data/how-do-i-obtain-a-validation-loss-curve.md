---
title: "How do I obtain a validation loss curve per epoch in PyTorch Lightning?"
date: "2025-01-30"
id: "how-do-i-obtain-a-validation-loss-curve"
---
The core challenge in obtaining a validation loss curve per epoch within the PyTorch Lightning framework stems from the inherent asynchronous nature of its training loop and the need to explicitly log metrics during the validation phase.  My experience implementing and debugging complex deep learning models in PyTorch Lightning has shown that directly accessing the validation loss within the training loop is unreliable; instead, leveraging the `Trainer`'s logging capabilities and subsequently visualizing the logged data is the robust and recommended approach.


**1.  Clear Explanation**

PyTorch Lightning's `Trainer` manages the training process, including the validation loop.  It doesn't directly expose the validation loss as a readily accessible variable after each epoch.  Instead, the validation loss, along with any other metrics you define, needs to be logged using PyTorch Lightning's logging functionality.  This typically involves using the `self.log()` method within your `LightningModule`'s `validation_step`, `validation_epoch_end`, or similar callbacks.  The logged metrics are then automatically tracked by the `Trainer` and can be accessed post-training, for instance, through TensorBoard or by manually extracting data from the `Trainer`'s logger.  The key is to understand the separation of concerns: the `LightningModule` focuses on model definition and metric calculation, while the `Trainer` manages the training flow and logging.  Misunderstanding this separation often leads to incorrect attempts to directly access the loss within the training loop.


**2. Code Examples with Commentary**

**Example 1: Basic Validation Logging**

This example demonstrates the fundamental approach.  I've used this pattern in multiple projects where a simple validation loss curve was required for monitoring.

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10, 1)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)  # Log training loss
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss) # Log validation loss
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


model = MyModel()
trainer = pl.Trainer(max_epochs=10, logger=pl.loggers.TensorBoardLogger("logs/"))
trainer.fit(model, datamodule=your_datamodule)  # Requires a defined datamodule
```

This code logs both training and validation loss.  The `self.log()` method automatically handles the aggregation across batches and epochs for the `TensorBoardLogger`.


**Example 2:  Custom Metric Calculation within `validation_epoch_end`**

This scenario showcases a more complex scenario, reflecting my experience working with multi-faceted validation metrics.  Often, simple loss isn't enough, requiring custom aggregation:

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(pl.LightningModule):
    # ... (same __init__ and forward as Example 1) ...

    def validation_step(self, batch, batch_idx):
        # ... (same validation_step as Example 1, except return loss directly) ...
        return loss

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x for x in outputs]).mean()
        self.log('val_loss_avg', avg_loss)
        self.log('val_loss_final', avg_loss) #Example of logging the same metric with a different name.  Useful for TensorBoard.
        return


    # ... (same configure_optimizers as Example 1) ...

model = MyModel()
trainer = pl.Trainer(max_epochs=10, logger=pl.loggers.TensorBoardLogger("logs/"))
trainer.fit(model, datamodule=your_datamodule)
```

Here, the average validation loss is calculated across all batches within `validation_epoch_end`  before being logged. This is more efficient than logging individual batch losses and provides a clearer epoch-level metric.


**Example 3:  Handling Multiple Validation Datasets**

This example addresses a situation I encountered frequently where multiple validation datasets are used for different evaluation aspects.

```python
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(pl.LightningModule):
    # ... (same __init__ and forward as Example 1) ...

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log(f'val_loss_dataset_{dataloader_idx}', loss) # differentiate loss from different datasets.
        return loss

    def validation_epoch_end(self, outputs):
        pass # No need to aggregate here as self.log handles the aggregation across batches.

    # ... (same configure_optimizers as Example 1) ...


model = MyModel()
trainer = pl.Trainer(max_epochs=10, logger=pl.loggers.TensorBoardLogger("logs/"))
trainer.fit(model, datamodule=your_datamodule)
```

This approach utilizes `dataloader_idx` provided in the `validation_step` to differentiate logged metrics from different validation datasets within the `datamodule`.


**3. Resource Recommendations**

The official PyTorch Lightning documentation.  Thorough understanding of the `Trainer` and `LightningModule` classes is crucial.  Consult the documentation on logging and available loggers (TensorBoard, Weights & Biases, etc.).  Familiarize yourself with the various hooks available within the `LightningModule` for customizing the training and validation loops.  Explore examples showcasing advanced techniques like custom callbacks and metric calculations.  Pay close attention to the section explaining how to use and integrate different logging systems.  Finally, reviewing examples from the PyTorch Lightning GitHub repository often reveals practical solutions to common challenges.
