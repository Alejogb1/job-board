---
title: "How to use `pytorch-lightning`'s `epoch_end` and `validation_epoch_end` methods effectively?"
date: "2025-01-30"
id: "how-to-use-pytorch-lightnings-epochend-and-validationepochend-methods"
---
In training deep learning models, meticulously tracking performance across epochs is paramount for effective hyperparameter tuning and model evaluation. `pytorch-lightning`, by design, offers granular control over this process via the `epoch_end` and `validation_epoch_end` hooks. I've seen firsthand, over numerous projects involving complex architectures, that understanding the nuances of these methods is crucial for avoiding common pitfalls related to metrics aggregation and efficient logging. These are not merely convenient callback functions; they’re pivotal for consolidating per-batch data into meaningful epoch-level summaries.

The key distinction lies in their execution context. `epoch_end` is triggered at the conclusion of *each* epoch, encompassing both training and, if present, validation phases. It operates on the outputs of all steps within that epoch. On the other hand, `validation_epoch_end` is invoked *solely* at the termination of the validation phase of the epoch. This implies that the output fed to `validation_epoch_end` originates from the `validation_step` calls, not the `training_step`. The primary purpose of these methods is not to calculate losses or gradients, as that is handled within `training_step` and `validation_step`, respectively. Instead, these methods are designed to aggregate and log metrics computed at the batch level, providing an epoch-wide overview of performance. They also allow for other actions such as logging images, histograms, or performing model checkpointing based on accumulated statistics, though this should be handled through dedicated callbacks for scalability reasons.

Let’s illustrate this with examples. Consider a basic image classification task where we’re measuring training and validation accuracy.

**Example 1: Basic Metric Aggregation**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from torchmetrics import Accuracy

class SimpleClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.train_accuracy(logits, y)
        self.log('train_loss', loss) # Batch level loss
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.val_accuracy(logits, y)
        self.log('val_loss', loss)
        return loss

    def epoch_end(self, outputs):
        # Aggregated after every epoch (train and validation)
        train_acc = self.train_accuracy.compute()
        self.log('train_epoch_accuracy', train_acc)
        self.train_accuracy.reset() # Important to reset for next epoch
    
    def validation_epoch_end(self, outputs):
         # Aggregated specifically after validation phase of each epoch
        val_acc = self.val_accuracy.compute()
        self.log('val_epoch_accuracy', val_acc)
        self.val_accuracy.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Generate dummy data
X_train = torch.randn(100, 10)
y_train = torch.randint(0, 3, (100,))
X_val = torch.randn(50, 10)
y_val = torch.randint(0, 3, (50,))

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_dataloader = DataLoader(train_dataset, batch_size=32)
val_dataloader = DataLoader(val_dataset, batch_size=32)


model = SimpleClassifier(input_size=10, hidden_size=32, num_classes=3)
trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1, enable_progress_bar=False)
trainer.fit(model, train_dataloader, val_dataloader)
```

In this example, `epoch_end` is used to compute the overall training accuracy for the epoch, and `validation_epoch_end` calculates the validation accuracy. Notice that the `Accuracy` metric objects are reset after each epoch to prevent information leak between epochs. Furthermore, observe the logging of 'train_loss' and 'val_loss' within `training_step` and `validation_step`, respectively. This represents batch-level logging, distinct from the epoch-level logging performed in `epoch_end` and `validation_epoch_end`. The `outputs` parameter of the methods is not being used because the metrics are tracked internally by `torchmetrics`, which is a best practice when possible.

**Example 2: Detailed Logging with Custom Outputs**

In situations where you need more detailed output at the batch level, you can also gather and process these outputs inside the training or validation step, and then use the  `epoch_end` and `validation_epoch_end` methods to aggregate and log them. Consider a scenario where we want to track the mean and standard deviation of the predicted logits during training and validation steps.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

class ClassifierWithOutputStats(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return {'logits': logits, 'loss': loss}

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('val_loss', loss)
        return {'logits': logits, 'loss': loss}

    def epoch_end(self, outputs):
        all_logits = torch.cat([x['logits'] for x in outputs]) #Gather across batches
        mean_logits = all_logits.mean()
        std_logits = all_logits.std()
        self.log('train_epoch_mean_logits', mean_logits)
        self.log('train_epoch_std_logits', std_logits)

    def validation_epoch_end(self, outputs):
        all_logits = torch.cat([x['logits'] for x in outputs])
        mean_logits = all_logits.mean()
        std_logits = all_logits.std()
        self.log('val_epoch_mean_logits', mean_logits)
        self.log('val_epoch_std_logits', std_logits)


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Generate dummy data (same as before)
X_train = torch.randn(100, 10)
y_train = torch.randint(0, 3, (100,))
X_val = torch.randn(50, 10)
y_val = torch.randint(0, 3, (50,))

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_dataloader = DataLoader(train_dataset, batch_size=32)
val_dataloader = DataLoader(val_dataset, batch_size=32)


model = ClassifierWithOutputStats(input_size=10, hidden_size=32, num_classes=3)
trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1, enable_progress_bar=False)
trainer.fit(model, train_dataloader, val_dataloader)
```

Here, we return a dictionary containing the logits and loss from both `training_step` and `validation_step`. In `epoch_end` and `validation_epoch_end`, we extract the logits from each batch output, concatenate them, and then compute the mean and standard deviation. This approach provides insight into the distribution of the predicted values across the entire epoch.

**Example 3: Conditional Logic and Early Exit**

The flexibility of these methods also extends to introducing conditional logic. Consider a scenario where you want to log a specific metric, only after a certain number of epochs have been trained.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from torchmetrics import Accuracy

class ConditionalLoggingClassifier(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.log_after_epoch = 3

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.train_accuracy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.val_accuracy(logits, y)
        self.log('val_loss', loss)
        return loss

    def epoch_end(self, outputs):
        if self.current_epoch >= self.log_after_epoch:
          train_acc = self.train_accuracy.compute()
          self.log('train_epoch_accuracy_cond', train_acc)
        self.train_accuracy.reset()
    
    def validation_epoch_end(self, outputs):
      if self.current_epoch >= self.log_after_epoch:
        val_acc = self.val_accuracy.compute()
        self.log('val_epoch_accuracy_cond', val_acc)
      self.val_accuracy.reset()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)

# Generate dummy data (same as before)
X_train = torch.randn(100, 10)
y_train = torch.randint(0, 3, (100,))
X_val = torch.randn(50, 10)
y_val = torch.randint(0, 3, (50,))

train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)

train_dataloader = DataLoader(train_dataset, batch_size=32)
val_dataloader = DataLoader(val_dataset, batch_size=32)


model = ConditionalLoggingClassifier(input_size=10, hidden_size=32, num_classes=3)
trainer = pl.Trainer(max_epochs=5, log_every_n_steps=1, enable_progress_bar=False)
trainer.fit(model, train_dataloader, val_dataloader)
```

In this specific example, metrics ('train_epoch_accuracy_cond' and 'val_epoch_accuracy_cond') are only logged after epoch 3, demonstrating that these methods can be made highly specific to the training context.

For further study, I would recommend exploring the official `pytorch-lightning` documentation, particularly the sections discussing metrics and callbacks. Consulting the source code directly for specific use cases, can also clarify how these methods interact internally. Understanding the design principles behind `torchmetrics` is also beneficial, since it's heavily used in conjunction with these methods. Finally, scrutinizing projects utilizing these methods within a realistic training context helps solidify their practical application. These methods are tools, but only when used correctly can we effectively tune and monitor model training.
