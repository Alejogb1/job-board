---
title: "How can I print PyTorch Lightning's accuracy and loss per epoch?"
date: "2025-01-30"
id: "how-can-i-print-pytorch-lightnings-accuracy-and"
---
During a recent project involving a large convolutional neural network for image classification, I faced a similar challenge: obtaining granular epoch-level metrics from PyTorch Lightning. While the library provides excellent tools for training, fine-grained logging required specific configuration and understanding of its internal mechanisms.

The key lies in PyTorch Lightning's `LightningModule`’s methods, primarily `training_step`, `validation_step`, and the associated logging within these methods. It is through these hooks that one can explicitly log metrics calculated during each training or validation iteration, which are then automatically aggregated across the entire epoch by the framework. Specifically, `self.log` is the function to achieve this. To obtain accuracy and loss per epoch, these values must be calculated within `*_step` and logged, thus enabling accurate epoch-level tracking when logging is configured correctly. It is not enough to calculate values at the beginning and end of the epoch outside of `*_step`, which will create confusion and erroneous results.

Let's examine three distinct examples, illustrating different levels of logging complexity and considerations for specific use-cases:

**Example 1: Basic Loss and Accuracy Logging During Training**

This example demonstrates the most fundamental implementation. Within the `training_step` method, we compute the loss and accuracy and immediately log them. PyTorch Lightning will accumulate these across training batches, then calculate average loss and accuracy at the end of each epoch, which will be available to callbacks and logging libraries.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from torchmetrics import Accuracy


class SimpleClassifier(pl.LightningModule):
    def __init__(self, input_size=10, hidden_size=64, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.accuracy = Accuracy(task="binary")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        return optimizer


if __name__ == '__main__':
    # Generate dummy data
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=32)

    model = SimpleClassifier()
    trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1)
    trainer.fit(model, train_loader)
```

In this snippet:
- `Accuracy` from `torchmetrics` is used for accuracy calculation. The `task="binary"` argument specifies that this is a binary classification task.
- `self.log` is called twice, once for the loss and once for the accuracy. Both metrics are logged at each step (`on_step=True`) and at each epoch end (`on_epoch=True`). The `prog_bar=True` flag makes the metrics visible in the training progress bar during training. Finally, `logger=True` makes it available to logger callbacks, like TensorBoard.
- The `configure_optimizers` method sets the optimizer.
- The main part generates dummy training data, sets up the `Trainer` class, and begins training. By setting `log_every_n_steps=1`, one will see progress at each training step.

**Example 2: Logging Validation Metrics Alongside Training Metrics**

This extends the previous example by logging metrics during the validation phase. This is crucial for monitoring model generalization during training.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from torchmetrics import Accuracy

class SimpleClassifier(pl.LightningModule):
    def __init__(self, input_size=10, hidden_size=64, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.accuracy = Accuracy(task="binary")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
      x, y = batch
      logits = self(x)
      loss = F.cross_entropy(logits, y)
      preds = torch.argmax(logits, dim=1)
      acc = self.accuracy(preds, y)

      self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
      self.log('val_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)

      return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        return optimizer

if __name__ == '__main__':
    # Generate dummy data
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=32)

    X_val = torch.randn(50, 10)
    y_val = torch.randint(0,2, (50,))
    val_data = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_data, batch_size=32)


    model = SimpleClassifier()
    trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1)
    trainer.fit(model, train_loader, val_loader)
```
The key addition here is the `validation_step` method. It operates analogously to `training_step` and logs the validation loss and accuracy, using the 'val\_loss' and 'val\_accuracy' keys. The `on_step=False` flag ensures logging only at the end of the epoch for validation metrics, which is preferable for validation metrics. Additionally, a validation dataloader `val_loader` is generated and passed to the `trainer.fit` function.

**Example 3: Implementing Step-Wise Accuracy Logging with `on_step` logging disabled**

Sometimes, particularly when you are looking at very granular metrics, it is not advisable to track progress at the step level, due to potentially noisy measures. This example disables step-wise accuracy reporting, providing only the epoch-level measure.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from torchmetrics import Accuracy


class SimpleClassifier(pl.LightningModule):
    def __init__(self, input_size=10, hidden_size=64, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.accuracy = Accuracy(task="binary")

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, y)

        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_accuracy', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
        return optimizer

if __name__ == '__main__':
    # Generate dummy data
    X_train = torch.randn(100, 10)
    y_train = torch.randint(0, 2, (100,))
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=32)

    X_val = torch.randn(50, 10)
    y_val = torch.randint(0, 2, (50,))
    val_data = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_data, batch_size=32)


    model = SimpleClassifier()
    trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1)
    trainer.fit(model, train_loader, val_loader)
```

The only modification compared to the second example is that during training the `train_accuracy` is only logged at the epoch level, by setting `on_step=False`. This prevents the progress bar from constantly updating with the current batch accuracy.

**Further Considerations and Recommendations**

Beyond these examples, several aspects of PyTorch Lightning logging require attention. First, careful consideration should be given to the metrics to log. When dealing with multi-class classification, using `torchmetrics.Accuracy` with `task="multiclass"` is required, potentially including an additional `num_classes` argument. One may also employ specialized metrics depending on the task, such as F1 scores or other domain-specific measures. Second, when using distributed training, logging mechanisms are slightly more complex, as the results from all the processors must be aggregated, which PyTorch Lightning handles implicitly. Third, be judicious in the frequency of step-level logging. Excessive logging can slow down the training process. For a large dataset, calculating a training set accuracy at the end of the epoch will also be slower and not representative of the current weights if calculated after each forward pass.

For further learning, I highly recommend reviewing the official PyTorch Lightning documentation. Specifically, the section on logging provides detailed explanations and examples, as does the documentation for the `torchmetrics` library. Additionally, examining example code within the `pytorch-lightning-bolts` repository can be informative for how more complex logging is handled in practice. Lastly, numerous research papers on specific neural network architectures include examples of how they handle metrics reporting and logging. These theoretical perspectives are useful for conceptualization. By utilizing these resources, and implementing the code above, one should be able to report loss and accuracy per epoch in a consistent, reproducible manner.
