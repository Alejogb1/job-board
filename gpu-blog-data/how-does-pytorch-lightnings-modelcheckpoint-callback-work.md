---
title: "How does PyTorch Lightning's ModelCheckpoint callback work?"
date: "2025-01-26"
id: "how-does-pytorch-lightnings-modelcheckpoint-callback-work"
---

ModelCheckpoint in PyTorch Lightning operates by monitoring a specified metric during training, validation, or testing phases and saving model checkpoints based on whether the metric has improved according to a defined condition. This capability is crucial for preserving optimal model states during long or complex training processes, ensuring that one can retrieve the best-performing model rather than the final, potentially sub-optimal, model at the end of training. Over the years, my projects have benefited substantially from its efficient checkpointing system.

The core functionality of the `ModelCheckpoint` callback lies in its interaction with the `Trainer` instance. The `Trainer` is responsible for managing the training loop, and at each relevant step (e.g., end of an epoch), it triggers callbacks like `ModelCheckpoint`. The callback then assesses the current value of the monitored metric and, depending on the `mode` (e.g., `min`, `max`), determines if the current metric value represents an improvement. If so, it saves a checkpoint of the model’s state.

The process involves a few key components: the `filepath`, `monitor`, `mode`, `save_top_k`, `save_last`, and `period`.

*   `filepath` specifies the directory and filename pattern to use when saving model checkpoints. This pattern can include placeholders for epoch and validation/test metric values. For instance, `'{epoch}-{val_loss:.2f}'` creates filenames like `epoch=1-val_loss=0.45.ckpt`.
*   `monitor` designates the metric being observed, e.g., `val_loss`, `val_accuracy`. This metric must be logged by the model during its training or evaluation process.
*   `mode` dictates the condition for saving a checkpoint. Typically, it is set to `min` if the monitored metric is a loss or `max` for accuracy. This dictates if a lower or higher value of the monitored metric triggers a save.
*   `save_top_k` dictates how many of the best checkpoints will be retained. If set to `-1`, all checkpoints are saved. If set to `k` only the `k` best checkpoints will be saved.
*   `save_last` determines whether to save a checkpoint at the end of the training process, irrespective of whether it is the best checkpoint.
*   `period` determines the frequency of checking the monitored metric. A `period` of `1` will check the metric every training epoch or every validation/test batch when appropriate.

The checkpoint itself stores the model's state dictionary, optimizer's state dictionary, and any other relevant training information. This comprehensive save allows for complete resumption of training from any saved checkpoint. If a checkpoint already exists at the same filepath, the old checkpoint is overwritten or deleted based on `save_top_k` and the `mode`, allowing for management of disk usage.

Here are code examples illustrating the use of `ModelCheckpoint`:

**Example 1: Basic usage with validation loss monitoring**

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule

class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)

# Dummy data
X_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,))
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32)

X_val = torch.randn(50, 10)
y_val = torch.randint(0, 2, (50,))
val_dataset = TensorDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=32)

model = SimpleModel()
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    filepath='checkpoints/model-{epoch}-{val_loss:.2f}',
    save_top_k=3,
    save_last=True
)

trainer = Trainer(max_epochs=5, callbacks=[checkpoint_callback], default_root_dir='./checkpoints')
trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
```

In this instance, `ModelCheckpoint` monitors the `val_loss`. The mode is set to `min`, so the model’s state is saved when validation loss decreases. The `filepath` includes both epoch and validation loss in the filename, allowing easy identification of checkpoint importance. `save_top_k=3` ensures that only the top 3 checkpoints based on the lowest validation loss are preserved, while `save_last=True` saves the model’s final state.

**Example 2: Monitoring a custom metric with a custom file name**

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule
import numpy as np

class CustomModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        y_prob = F.softmax(y_hat, dim=1)
        y_pred = torch.argmax(y_prob, dim=1)
        accuracy = (y_pred == y).float().mean()
        self.log('val_accuracy', accuracy)
        return {'val_loss':loss, 'val_accuracy':accuracy}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# Dummy data
X_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,))
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32)

X_val = torch.randn(50, 10)
y_val = torch.randint(0, 2, (50,))
val_dataset = TensorDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=32)

model = CustomModel()
checkpoint_callback = ModelCheckpoint(
    monitor='val_accuracy',
    mode='max',
    filepath='best-model-{val_accuracy:.4f}',
    save_top_k=1,
    save_last = False
)


trainer = Trainer(max_epochs=5, callbacks=[checkpoint_callback])
trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)

```

Here, the callback tracks `val_accuracy`, using mode set to `max`, reflecting that an increasing accuracy value represents improvement. Note that the filename pattern does not include the epoch number. Also, only the absolute best model is saved and not the final model.

**Example 3: Saving checkpoint every N epochs**

```python
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning import LightningModule

class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


# Dummy data
X_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,))
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=32)

X_val = torch.randn(50, 10)
y_val = torch.randint(0, 2, (50,))
val_dataset = TensorDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=32)


model = SimpleModel()

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    filepath='every-other-epoch-{epoch}',
    save_top_k=-1,
    period=2,
    save_last=True

)

trainer = Trainer(max_epochs=10, callbacks=[checkpoint_callback])
trainer.fit(model, train_dataloader=train_dataloader, val_dataloaders=val_dataloader)
```

In this scenario, the `period` parameter is set to 2, which dictates the `ModelCheckpoint` to check the metric, and save if it improved, every two epochs.  `save_top_k=-1` now stores all of the checkpoints that improved.

For further exploration, I suggest consulting the PyTorch Lightning official documentation, which provides detailed explanations and numerous examples. Tutorials available on the PyTorch Lightning website offer more advanced implementations. Additionally, resources that cover best practices for training deep learning models, specifically dealing with model checkpointing, will enhance understanding.
