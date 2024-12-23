---
title: "How does pytorch_lightning's ModelCheckpoint callback function work?"
date: "2024-12-23"
id: "how-does-pytorchlightnings-modelcheckpoint-callback-function-work"
---

Alright, let's talk ModelCheckpoint in PyTorch Lightning. I’ve spent a fair bit of time working with it, mostly on projects ranging from complex NLP tasks to some equally intricate computer vision systems. And believe me, getting checkpointing just *right* is crucial, especially when you're training models for days, or even weeks. It's definitely saved my bacon a few times.

At its core, the `ModelCheckpoint` callback in `pytorch_lightning` is designed to automatically save model checkpoints during training based on specified conditions. Think of it as an intelligent automated save button that’s tied to your training performance and configuration. Instead of needing to manually keep track of epochs or specific metrics and implement saving logic yourself, you define what criteria should trigger a save, and the callback handles the rest, ensuring a consistent and reliable process.

The beauty of it is its configurability. You can save the *best* model based on a validation metric, or save every *n* epochs, or even save the last model after training finishes. You can even combine these. There are options for which files to keep after a save, which metric to monitor, and how to name your saved files. This flexibility is really what makes it so valuable, especially in large projects.

Now, let’s break down how it actually functions under the hood. When you instantiate a `ModelCheckpoint` callback, you’re setting up an internal state machine. Lightning will call its hooks at key points during the training loop, particularly after each validation step (if you’ve specified one) and at the end of each epoch. This is where the `ModelCheckpoint` callback comes into play, monitoring your chosen metric (usually validation loss or accuracy) and checking against its set conditions.

Specifically, within `pytorch_lightning`, the callback's `on_validation_epoch_end` method (or `on_train_epoch_end`, or `on_train_end` depending on your configuration) is where most of the logic lives. It checks the current metric value, determines if a new best metric has been found, and then calls the necessary save function or decides if older checkpoints should be removed, as per your settings. It's not a magic black box but a very deliberate and procedural process with clearly defined steps and hooks within the training loop. The callback manages file paths, ensuring that your checkpoint is saved in the designated directory and that the file names align with the epoch number, metric, or whatever configuration parameters you set.

Let’s dive into some code examples to illustrate how this works, using simplified but functional code snippets.

**Example 1: Saving the best model based on validation loss**

```python
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.linear(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
      return torch.optim.Adam(self.parameters())

#Dummy data for a training loop
train_dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10)
val_dataset = torch.utils.data.TensorDataset(torch.randn(50, 10), torch.randn(50, 1))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10)


model = MyModel()
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',
    mode='min',
    dirpath='./checkpoints',
    filename='best-model-{epoch}-{val_loss:.2f}',
    save_top_k=1,
    verbose = True
)

trainer = pl.Trainer(max_epochs=5, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_loader)
```
In this example, we initialize `ModelCheckpoint` to monitor `val_loss`, minimizing it. The `save_top_k=1` argument ensures only the best model will be retained. The `filename` parameter controls how the saved files are named, incorporating epoch and the validation loss.

**Example 2: Saving a checkpoint every *n* epochs**

```python
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class MyModel(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(10, 1)

  def training_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.linear(x)
    loss = torch.nn.functional.mse_loss(y_hat, y)
    self.log('train_loss', loss)
    return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.linear(x)
    loss = torch.nn.functional.mse_loss(y_hat, y)
    self.log('val_loss', loss)

  def configure_optimizers(self):
    return torch.optim.Adam(self.parameters())

#Dummy data for a training loop
train_dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10)
val_dataset = torch.utils.data.TensorDataset(torch.randn(50, 10), torch.randn(50, 1))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10)


model = MyModel()
checkpoint_callback = ModelCheckpoint(
    dirpath='./checkpoints_every_n',
    filename='epoch-{epoch}',
    every_n_epochs=2,
    save_top_k=-1, #important for not deleting other models
    verbose=True
)

trainer = pl.Trainer(max_epochs=6, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_loader)
```

Here, we use the `every_n_epochs` parameter to save a checkpoint every two epochs. The `save_top_k = -1` setting is critical; setting it to -1 saves every checkpoint, regardless of performance. When not set, `save_top_k` defaults to 1, which will only keep the single 'best' checkpoint.

**Example 3: Saving the last checkpoint after training**

```python
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

class MyModel(pl.LightningModule):
  def __init__(self):
    super().__init__()
    self.linear = torch.nn.Linear(10, 1)

  def training_step(self, batch, batch_idx):
      x, y = batch
      y_hat = self.linear(x)
      loss = torch.nn.functional.mse_loss(y_hat, y)
      self.log('train_loss', loss)
      return loss

  def validation_step(self, batch, batch_idx):
    x, y = batch
    y_hat = self.linear(x)
    loss = torch.nn.functional.mse_loss(y_hat, y)
    self.log('val_loss', loss)

  def configure_optimizers(self):
      return torch.optim.Adam(self.parameters())

#Dummy data for a training loop
train_dataset = torch.utils.data.TensorDataset(torch.randn(100, 10), torch.randn(100, 1))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=10)
val_dataset = torch.utils.data.TensorDataset(torch.randn(50, 10), torch.randn(50, 1))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10)

model = MyModel()
checkpoint_callback = ModelCheckpoint(
    dirpath='./last_checkpoint',
    filename='last-model',
    save_last=True,
    verbose=True
)

trainer = pl.Trainer(max_epochs=5, callbacks=[checkpoint_callback])
trainer.fit(model, train_loader, val_loader)
```

In this final example, the `save_last=True` parameter tells `ModelCheckpoint` to save the model after training is finished, irrespective of any performance metrics. This is incredibly useful if you are iterating on training hyperparameters, as you always have the model from the last training run.

For further exploration into checkpointing strategies, I recommend delving into the papers and books on deep learning and specifically on the practice of experimentation management. Specifically, “Deep Learning” by Goodfellow, Bengio, and Courville offers a very robust theoretical framework for understanding the mechanics of training, including a section on practical considerations like checkpointing. Similarly, "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Géron provides an accessible look at implementation and best practices in deep learning, though it may not directly address the specifics of PyTorch Lightning. Lastly, the PyTorch Lightning official documentation is a must-read for a complete understanding of all configuration options and internal mechanisms of callbacks, including `ModelCheckpoint`. It's essential to understand the intricacies of the framework you're using to get the most out of it. Remember to review the source code on GitHub as well; it provides the most authoritative information of how it operates.

Through personal experience, I can vouch that carefully implemented checkpointing using this callback can save considerable time and resources, and, more importantly, it will also save you from a lot of grief when your model is stuck in the middle of training for what seems like an eternity. It is a core aspect of the training process and deserves a closer look.
