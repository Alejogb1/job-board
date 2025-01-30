---
title: "How do I check validation metrics in PyTorch Lightning?"
date: "2025-01-30"
id: "how-do-i-check-validation-metrics-in-pytorch"
---
Validation metrics in PyTorch Lightning are primarily managed and accessed within the `LightningModule` class, leveraging its inherent hooks and mechanisms for structured logging. The key fact is that these metrics are calculated *during* the validation step, not after, and are made available through the `validation_step`, `validation_epoch_end`, and the logger functionality. Misunderstanding this can lead to incorrect interpretations of performance and frustration with debugging. My experience in building deep learning models for medical image analysis highlighted the necessity of carefully structuring this process to ensure reliable tracking of key performance indicators such as Dice scores, sensitivity, and specificity. I’ll demonstrate how this works effectively based on best practices I’ve adopted over multiple projects.

**1. Understanding the Validation Process in PyTorch Lightning**

PyTorch Lightning automates the training and validation loops, abstracting away the boilerplate code typically associated with managing these processes in raw PyTorch. This automation is achieved through specific lifecycle methods within the `LightningModule`, such as `training_step`, `validation_step`, `test_step`, and their corresponding `*_epoch_end` methods.

The validation process fundamentally consists of two phases: the per-batch `validation_step` and the per-epoch `validation_epoch_end`. In `validation_step`, the input batch is processed, predictions are made, and metrics are *calculated on that batch*. These metrics are not automatically aggregated across batches. Instead, the results of this method are typically dictionaries. These are passed as arguments into the `validation_epoch_end` method. Here, we accumulate the results from individual batches, and ultimately, calculate summary metrics for the entire validation epoch.

The logging mechanism in PyTorch Lightning is crucial because it’s how metrics are recorded and monitored. Lightning integrates with various logging backends like TensorBoard or Wandb. You do not interact with the backend directly, but rather call `self.log` within the `LightningModule`. This method stores data and later transmits it to the specified logging system. This decouples metric recording from the specific implementation of different backends.

**2. Code Examples**

I'll now illustrate this with a series of examples building in complexity. Example 1 demonstrates the basic logging of a simple loss; Example 2 introduces the calculation of a single metric within the validation step, and the proper aggregation in `validation_epoch_end`. Example 3 tackles multiple metrics, including one that requires manual calculation across batches.

*   **Example 1: Logging Validation Loss**

    This first example shows a typical implementation for logging the validation loss, assuming you have computed a loss in the `validation_step`.

    ```python
    import torch
    import torch.nn as nn
    from pytorch_lightning import LightningModule, Trainer
    from torch.utils.data import DataLoader, TensorDataset

    class SimpleModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            self.loss_fn = nn.MSELoss()

        def forward(self, x):
            return self.linear(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)
            self.log('train_loss', loss, prog_bar=True)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)
            self.log('val_loss', loss, prog_bar=True)
            return loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
            return optimizer

    # Generate dummy data
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    dataset = TensorDataset(X, y)

    # Initialize model, trainer, and data loader
    model = SimpleModel()
    dataloader = DataLoader(dataset, batch_size=32)
    trainer = Trainer(max_epochs=3, accelerator="auto", devices=1)

    # Train the model, observing the logging
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)

    ```

    **Commentary:** Here, `self.log('val_loss', loss, prog_bar=True)` logs the validation loss for *each batch*. The `prog_bar=True` argument ensures that the validation loss is displayed in the progress bar during the validation phase. The key here is that there is no manual aggregation because this is a single metric computed using only the information from the current batch. This is why this metric will appear on a per batch basis in logs.
*   **Example 2: Calculating and Aggregating a Single Metric**

    This example demonstrates the calculation of a mean average precision (MAP) score, an appropriate metric for object detection and retrieval scenarios, where we are interested in the ranking quality of the predictions. While more complex, the core principal of metric accumulation and logging is consistent.

    ```python
    import torch
    import torch.nn as nn
    from pytorch_lightning import LightningModule, Trainer
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.metrics import average_precision_score

    class MAPModel(LightningModule):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            self.loss_fn = nn.MSELoss()

        def forward(self, x):
            return torch.sigmoid(self.linear(x)) # Output between 0 and 1

        def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)
            self.log('train_loss', loss)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)
            map_score = average_precision_score(y.cpu().numpy(), y_hat.detach().cpu().numpy())
            return {'val_loss':loss, 'val_map': map_score}


        def validation_epoch_end(self, outputs):
           avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
           avg_map  = sum([x['val_map'] for x in outputs])/len(outputs)
           self.log('val_loss', avg_loss, prog_bar=True)
           self.log('val_map', avg_map, prog_bar=True)

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
            return optimizer

    # Generate dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0,2, (100,1)).float() # Create binary labels
    dataset = TensorDataset(X, y)

    # Initialize model, trainer, and data loader
    model = MAPModel()
    dataloader = DataLoader(dataset, batch_size=32)
    trainer = Trainer(max_epochs=3, accelerator="auto", devices=1)

    # Train the model
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)

    ```

    **Commentary:** Here, `validation_step` calculates the mean average precision (MAP) using `sklearn`. The loss and the MAP score for the batch are returned as a dictionary. In `validation_epoch_end`, these batch-wise metrics are aggregated: the `avg_map` is the simple average across batches, and the `avg_loss` is the average of the loss across all validation batches. The accumulated metrics are then logged using `self.log`. The important distinction is that *average_precision_score* cannot be accumulated directly; it needs to be calculated per batch.
*   **Example 3: Accumulating Data for Metric Calculation**

    In this example, we will tackle a case where a single metric (Dice Score) requires knowledge of the true positives, false positives, and false negatives across all validation batches, and therefore it cannot be calculated directly within the step method, without accumulation first. We will assume this metric is more meaningful across the entire epoch.

    ```python
    import torch
    import torch.nn as nn
    from pytorch_lightning import LightningModule, Trainer
    from torch.utils.data import DataLoader, TensorDataset

    class DiceModel(LightningModule):
       def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)
            self.loss_fn = nn.BCEWithLogitsLoss()


       def forward(self, x):
          return self.linear(x)

       def training_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y)
            self.log('train_loss', loss)
            return loss


       def validation_step(self, batch, batch_idx):
         x, y = batch
         y_hat = self(x)
         preds = torch.sigmoid(y_hat) > 0.5
         return {'preds': preds.int(), 'labels': y.int()}


       def validation_epoch_end(self, outputs):
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_labels = torch.cat([x['labels'] for x in outputs])
        tp = torch.sum(all_preds & all_labels)
        fp = torch.sum(all_preds & (~all_labels))
        fn = torch.sum((~all_preds) & all_labels)

        dice = (2 * tp) / (2 * tp + fp + fn + 1e-8)  # Avoid division by zero
        self.log('val_dice', dice, prog_bar=True)


       def configure_optimizers(self):
           optimizer = torch.optim.Adam(self.parameters(), lr=0.02)
           return optimizer

    # Generate dummy data
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100, 1)).float()  # Binary Labels
    dataset = TensorDataset(X, y)

    # Initialize model, trainer, and data loader
    model = DiceModel()
    dataloader = DataLoader(dataset, batch_size=32)
    trainer = Trainer(max_epochs=3, accelerator="auto", devices=1)

    # Train the model
    trainer.fit(model, train_dataloaders=dataloader, val_dataloaders=dataloader)


    ```

    **Commentary:** This is probably the most nuanced example. The `validation_step` now only produces binary predictions, by thresholding the output logits, and returns the corresponding true labels. The `validation_epoch_end` gathers the predictions and labels across all batches. The true positives, false positives, and false negatives are then calculated. This is necessary because the Dice score requires aggregated counts, unlike the MAP score which can be averaged across batches. Finally the Dice score is calculated using these accumulated values, and then the result is logged.

**3. Resource Recommendations**

To further enhance understanding and effective implementation of validation metrics in PyTorch Lightning, I recommend delving deeper into several areas. I suggest starting with the official PyTorch Lightning documentation. Pay close attention to sections discussing `LightningModule`, particularly the `training_step`, `validation_step`, and `*_epoch_end` methods. The documentation also provides comprehensive details about logging and available logger integrations. Additionally, review the example notebooks and guides for specific use cases to observe diverse implementations. I find that studying implementations of models for image segmentation and object detection often serves as good, well-documented examples. Also, explore relevant tutorials on how to use specific logging backends like TensorBoard to visualize the logged metrics.

In addition, examining implementations of common deep learning architectures (e.g. those in TorchVision or TorchAudio) often yields further insights into best practices for defining validation metrics. Finally, studying discussions within the PyTorch Lightning community forums and repositories may reveal valuable tips and tricks and solutions to specific problems that may arise while working with the framework.
