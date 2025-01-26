---
title: "How can I import TensorMetric in PyTorch Lightning?"
date: "2025-01-26"
id: "how-can-i-import-tensormetric-in-pytorch-lightning"
---

TensorMetric, while a useful tool for custom metric logging within PyTorch Lightning, is not a direct import. It’s a concept representing arbitrary scalar values you might wish to monitor throughout your training process, and it is integrated into PyTorch Lightning using its logging capabilities alongside user-defined computation logic. The framework itself does not provide a class or module named “TensorMetric.” I’ve encountered this frequently when building complex models with custom evaluation criteria. My workflow involves defining a metric, computing its value during training and validation steps, and using PyTorch Lightning’s logging mechanism to keep track of these metrics.

To implement what you’re likely seeking—i.e., a method to track custom, scalar metrics—you need to approach it from two sides: defining the calculation and properly using PyTorch Lightning’s `log` function. The calculation is purely dependent on the model you’re training and the specific metric you’re after. This can range from a simple accuracy score to something more intricate like Intersection-over-Union (IoU) for segmentation tasks. Once you have the computation, the `log` method provides the bridge into Lightning’s visualization and monitoring capabilities (e.g., TensorBoard, WandB).

The core of this process involves modifying your PyTorch Lightning Module subclass. Specifically, your `training_step`, `validation_step`, or even `test_step` method should contain the metric calculation and logging. I typically find that modularity is key here, so separating the metric computation into a dedicated function is advisable for readability and reusability. The following examples will illustrate this.

**Example 1: Implementing Simple Accuracy**

This example shows how to calculate and log accuracy within the `training_step`. The code assumes a classification task, using a hypothetical `predictions` tensor and a `targets` tensor, along with a simple helper function.

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

def calculate_accuracy(predictions, targets):
    """Calculates accuracy between predictions and targets."""
    predicted_classes = torch.argmax(predictions, dim=1)
    correct = (predicted_classes == targets).sum().float()
    accuracy = correct / targets.numel()
    return accuracy

class SimpleClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        accuracy = calculate_accuracy(outputs, targets)
        self.log('train_loss', loss)  # Logs training loss
        self.log('train_accuracy', accuracy) # Logs training accuracy
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
```

In this snippet, `calculate_accuracy` computes the accuracy. The `training_step` calculates the loss and accuracy and uses `self.log` to record them. The logged values will then be available in your chosen logger. Specifically, the string keys ‘train_loss’ and ‘train_accuracy’ specify what will appear in TensorBoard or WandB.

**Example 2: Logging Multiple Metrics in Validation**

Building upon the previous example, this example demonstrates logging multiple metrics (accuracy and a hypothetical precision) during the `validation_step`. This simulates a slightly more comprehensive monitoring approach.

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

def calculate_accuracy(predictions, targets):
    """Calculates accuracy between predictions and targets."""
    predicted_classes = torch.argmax(predictions, dim=1)
    correct = (predicted_classes == targets).sum().float()
    accuracy = correct / targets.numel()
    return accuracy

def calculate_precision(predictions, targets):
    """Hypothetical calculation of precision for demonstration."""
    predicted_classes = torch.argmax(predictions, dim=1)
    true_positives = ((predicted_classes == targets) & (targets == 1)).sum().float()
    predicted_positives = (predicted_classes == 1).sum().float()
    precision = true_positives / (predicted_positives + 1e-8)  # Adding a small constant for numerical stability
    return precision

class SimpleClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
         super().__init__()
         self.fc1 = nn.Linear(input_size, hidden_size)
         self.relu = nn.ReLU()
         self.fc2 = nn.Linear(hidden_size, num_classes)
         self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        accuracy = calculate_accuracy(outputs, targets)
        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        accuracy = calculate_accuracy(outputs, targets)
        precision = calculate_precision(outputs, targets) # Added precision calculation
        self.log('val_loss', loss) # Logs validation loss
        self.log('val_accuracy', accuracy) # Logs validation accuracy
        self.log('val_precision', precision) # Logs validation precision
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
```

Here, `validation_step` includes the calculation of both accuracy and precision. The crucial point is the use of `self.log` to record multiple metrics. This demonstrates that you are not limited to just one metric per step. The labels for the metrics such as “val_loss”, “val_accuracy”, and “val_precision” again determine how they appear in your logging dashboard.

**Example 3: Logging Average Metric Values Over Epoch**

The `self.log` method is not limited to logging values from a single step; it can also aggregate them over an entire epoch. For example, you might want to calculate the mean accuracy or precision across all validation steps. This approach allows tracking the average value of a metric over the full validation set.

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

def calculate_accuracy(predictions, targets):
    """Calculates accuracy between predictions and targets."""
    predicted_classes = torch.argmax(predictions, dim=1)
    correct = (predicted_classes == targets).sum().float()
    accuracy = correct / targets.numel()
    return accuracy

def calculate_precision(predictions, targets):
     """Hypothetical calculation of precision for demonstration."""
     predicted_classes = torch.argmax(predictions, dim=1)
     true_positives = ((predicted_classes == targets) & (targets == 1)).sum().float()
     predicted_positives = (predicted_classes == 1).sum().float()
     precision = true_positives / (predicted_positives + 1e-8)  # Adding a small constant for numerical stability
     return precision


class SimpleClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
         super().__init__()
         self.fc1 = nn.Linear(input_size, hidden_size)
         self.relu = nn.ReLU()
         self.fc2 = nn.Linear(hidden_size, num_classes)
         self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        accuracy = calculate_accuracy(outputs, targets)
        self.log('train_loss', loss, on_step=True, on_epoch=False) # logging loss only on step
        self.log('train_accuracy', accuracy, on_step=True, on_epoch=False) # logging accuracy only on step
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss_fn(outputs, targets)
        accuracy = calculate_accuracy(outputs, targets)
        precision = calculate_precision(outputs, targets)
        self.log('val_loss', loss, on_step=True, on_epoch=True) # logging loss on step and epoch
        self.log('val_accuracy', accuracy, on_step=True, on_epoch=True) # logging accuracy on step and epoch
        self.log('val_precision', precision, on_step=True, on_epoch=True) # logging precision on step and epoch
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
```

The key modification here is the addition of the `on_step` and `on_epoch` keyword arguments to `self.log`. By setting `on_step=True` and `on_epoch=False`, you are logging the metric only at the step level which means the metric won't be aggregated to the epoch level. Setting both to true, which is the default behavior, averages the logged values across all validation steps for each epoch and logs the mean metric value. This provides a summarized view at the end of each epoch.

**Resource Recommendations**

To deepen your understanding of this process, consult the official PyTorch Lightning documentation. Review the sections related to logging and callbacks. Understanding how callbacks work will give you further control over metric tracking by running custom code at different points in the training cycle. There are tutorials on model development within the framework and how to integrate a logger such as TensorBoard or WandB to visualize these metric logs, and an important area to investigate is the aggregation of logs as shown in the last example. I have personally found reviewing the source code of the `log` function to be helpful in understanding its mechanics as well. Lastly, looking at user examples for common metrics such as mean average precision (mAP) in segmentation tasks using the library is often beneficial as a further reference point.
