---
title: "What's the difference between `on_validation_epoch_end` and `validation_epoch_end` in PyTorch Lightning?"
date: "2025-01-30"
id: "whats-the-difference-between-onvalidationepochend-and-validationepochend-in"
---
In PyTorch Lightning, both `on_validation_epoch_end` and `validation_epoch_end` methods are hooks triggered at the conclusion of a validation epoch; however, they serve distinct purposes and offer differing levels of access to the training loop. I’ve encountered scenarios where misusing these hooks led to subtle bugs in metric tracking and logging, solidifying my understanding of their nuances.

Fundamentally, `on_validation_epoch_end` is a *LightningModule hook*, and `validation_epoch_end` is a *validation loop hook*. This difference dictates the scope of the data they receive and the actions they should typically perform. The `validation_epoch_end` method is specifically designed to aggregate outputs from individual validation steps and is executed before any logging or metric computation on the epoch level. In contrast, `on_validation_epoch_end` is executed *after* these aggregations and logging operations, giving it visibility into the overall epoch-level metrics.

Specifically, `validation_epoch_end` takes a list of outputs, collected from every `validation_step` call within the epoch. This allows for custom aggregation logic. For instance, you might need to calculate a specialized metric that cannot be directly computed per batch. This is the place for operations like collecting model predictions for later analysis, or computing a metric that requires all the validation data. The return value from `validation_epoch_end` is not utilized by the PyTorch Lightning framework itself beyond potentially being used in logging. The framework handles the standard aggregations of things like loss using a built-in aggregator before this method is invoked.

`on_validation_epoch_end`, on the other hand, receives no direct outputs from the validation steps. Instead, it receives a dictionary of the *aggregated* logged metrics for the entire epoch and the entire validation data loader. This method is ideally suited for acting on these aggregated results, including performing operations like: logging additional epoch level metrics derived from the existing ones, saving model checkpoints based on validation performance, or triggering early stopping. This hook is also where things like custom logging callbacks should be performed, as access to the logged values is given here. Unlike `validation_epoch_end`, the return value of this method *is* important. It is used by some of Lightning's core functionality such as saving checkpoints, early stopping etc.

Let's illustrate these differences with code examples. Consider a scenario where we are training an image classifier and need to track accuracy, calculate a custom metric, and then also save the best model checkpoint based on validation accuracy.

**Example 1: Demonstrating `validation_epoch_end` for custom aggregation**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


class SimpleClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.example_input_array = torch.randn(1,input_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        return {'loss': loss, 'preds': preds, 'targets': y} #returns dictionary of results


    def validation_epoch_end(self, outputs):
      #Aggregate the model outputs
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        accuracy = (all_preds == all_targets).float().mean()
        self.log('val_accuracy_custom', accuracy) # logs the custom calculated accuracy metric

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
```

In this example, the `validation_step` returns a dictionary containing the loss, predictions, and targets for each batch. In `validation_epoch_end`, I aggregate these outputs to calculate the overall accuracy, which is logged as `val_accuracy_custom`. Note the return value of `validation_epoch_end` is ignored by lightning.

**Example 2: Demonstrating `on_validation_epoch_end` for epoch-level metric manipulation and checkpointing**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl


class SimpleClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.example_input_array = torch.randn(1,input_size)
        self.best_val_acc = 0.0


    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        accuracy = (preds == y).float().mean()
        self.log('val_accuracy', accuracy, on_epoch=True)
        return loss

    def on_validation_epoch_end(self):
        # access the logged metrics dictionary for the epoch
        metrics_dict = self.trainer.callback_metrics
        val_acc = metrics_dict['val_accuracy']
        if val_acc > self.best_val_acc:
          self.best_val_acc = val_acc
          print(f"Saving new best model as accuracy improved to: {val_acc}")
          self.trainer.save_checkpoint(f"best_model_{self.current_epoch}.ckpt") #save based on access to metrics
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
```

Here, the `validation_step` logs the accuracy *per batch* using `self.log(..., on_epoch=True)`. This is important as only these metrics are available in on `on_validation_epoch_end`. The `on_validation_epoch_end` method now retrieves the aggregated accuracy from `self.trainer.callback_metrics`, saves a checkpoint only when it improves and prints a message. It doesn't use the output of `validation_step`, but rather relies on the aggregated metrics that Lightning manages. Note that the return of this function is critical for the functioning of lightning's internal functionality. Here, I am not returning anything which implies that I'm not overriding the default behavior.

**Example 3: Using both hooks together for combined aggregation and logging**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

class SimpleClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()
        self.example_input_array = torch.randn(1,input_size)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        return {'loss': loss, 'preds': preds, 'targets': y}

    def validation_epoch_end(self, outputs):
      #Aggregate the model outputs
        all_preds = torch.cat([x['preds'] for x in outputs])
        all_targets = torch.cat([x['targets'] for x in outputs])
        accuracy = (all_preds == all_targets).float().mean()
        self.log('val_accuracy_custom', accuracy,on_epoch=True) # logs the custom calculated accuracy metric

    def on_validation_epoch_end(self):
        # access the logged metrics dictionary for the epoch
      metrics_dict = self.trainer.callback_metrics
      val_acc_custom = metrics_dict['val_accuracy_custom']
      print(f"Validation accuracy calculated with aggregation logic: {val_acc_custom}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
```

In this example, I'm showing how both hooks can be used in conjunction. In the `validation_epoch_end`, I calculate and log a custom accuracy score which is made available to the subsequent `on_validation_epoch_end` hook. The `on_validation_epoch_end` method then accesses this custom accuracy score to print a message. This is a useful use case as `on_validation_epoch_end` cannot perform any aggregation logic itself.

In summary, `validation_epoch_end` is for batch-output aggregation and custom metric calculations *before* any logging or framework operations. `on_validation_epoch_end` is for post-processing based on the epoch-level logged metrics, typically for logging new metrics derived from the existing ones, checkpointing, or early stopping procedures.

For further understanding, I recommend reviewing the official PyTorch Lightning documentation, particularly the sections on the training loop, and the hooks available in the LightningModule and the Trainer classes. Examining the source code for the `Trainer` and the `BaseTrainingTypePlugin` is also beneficial, as it highlights how these hooks are wired in the overall framework. Additionally, various example projects using PyTorch Lightning often demonstrate good practices for using these hooks. Specifically, focusing on examples that show custom logging implementations would help to see how the Trainer's callback_metrics dictionary is used.
