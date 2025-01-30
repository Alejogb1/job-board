---
title: "How can PyTorch Lightning resume training from the last epoch using saved weights?"
date: "2025-01-30"
id: "how-can-pytorch-lightning-resume-training-from-the"
---
The critical aspect of resuming PyTorch Lightning training from a checkpoint lies not solely in loading the weights but also in correctly restoring the optimizer's state.  Failing to restore the optimizer's internal state – including momentum, learning rate scheduling, and potentially Adam's internal variables – will result in unpredictable and likely suboptimal training behavior.  My experience debugging such issues across numerous large-scale projects reinforces this point.  Overcoming this requires careful handling of the `Trainer` and `LightningModule` objects during the resumption process.

**1. Clear Explanation:**

PyTorch Lightning provides a streamlined mechanism for checkpointing and resuming training.  The `Trainer`'s `resume_from_checkpoint` argument allows specifying the path to a checkpoint file. This checkpoint typically contains the model's weights, the optimizer's state, and potentially other training metadata like the current epoch and step.  The process involves loading this checkpoint, instantiating the `LightningModule` (or loading it from a separate saved state), and then passing the checkpoint path to the `Trainer`'s `fit` method.  Critically, the internal logic of the `Trainer` manages the restoration process, ensuring that the optimizer's state is correctly loaded and training continues from where it left off.  However, inconsistencies can arise if the checkpoint format changes between PyTorch Lightning versions or if custom modifications to the `LightningModule` are not handled appropriately.

The success of this procedure relies on the structure of the saved checkpoint.  PyTorch Lightning automatically saves relevant information during training.  By default, the checkpoint includes the model's state_dict, the optimizer's state_dict, the epoch, and the global step.  Ensuring that the loaded checkpoint is compatible with the currently defined `LightningModule` is crucial. Changes to the model's architecture or hyperparameters between saving the checkpoint and attempting to resume training will almost certainly lead to errors.


**2. Code Examples with Commentary:**


**Example 1: Basic Resumption**

```python
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

class MyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Linear(5, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


# ... (Dataset and DataLoader definition would go here) ...

model = MyModel()
logger = TensorBoardLogger("tb_logs", name="my_model")
trainer = pl.Trainer(max_epochs=10, logger=logger, default_root_dir="checkpoints/")
trainer.fit(model, train_dataloader)

# Resume training from checkpoint. The checkpoint path must be correct.
resume_checkpoint_path = "checkpoints/my_model/version_0/checkpoints/epoch=9.ckpt" # Adapt path as needed
trainer = pl.Trainer(max_epochs=20, logger=logger, default_root_dir="checkpoints/", resume_from_checkpoint=resume_checkpoint_path)
trainer.fit(model, train_dataloader)
```

**Commentary:** This example demonstrates the simplest case. The `resume_from_checkpoint` argument is passed directly to the `Trainer` during initialization.  The `max_epochs` parameter is increased to continue training beyond the initial run.  The logger is reused for consistency in logging.  The crucial point here is that the `model` instantiation is identical between the initial training and the resumption.

**Example 2: Handling Custom Data Modules**

```python
# ... (MyModel remains the same) ...

class MyDataModule(pl.LightningDataModule):
    # ... (Data loading and preprocessing logic) ...

data_module = MyDataModule()
model = MyModel()
trainer = pl.Trainer(max_epochs=10, default_root_dir="checkpoints/")
trainer.fit(model, data_module)

# Resume training
resume_checkpoint_path = "checkpoints/epoch=9.ckpt" # Adapt path as needed
trainer = pl.Trainer(max_epochs=20, default_root_dir="checkpoints/", resume_from_checkpoint=resume_checkpoint_path)
trainer.fit(model, data_module)
```

**Commentary:** This illustrates resumption with a custom `LightningDataModule`. The data loading process is encapsulated within the `MyDataModule`, ensuring that data preparation remains consistent between training sessions.  This approach is essential for reproducibility and maintainability, particularly in complex projects.  The same `MyDataModule` instance is used during both the initial training and the resumption.


**Example 3:  Resuming with a Different Model Instance**

```python
# ... (MyModel remains the same) ...
# First Training Run:
model = MyModel()
trainer = pl.Trainer(max_epochs=5, default_root_dir="checkpoints/")
trainer.fit(model, train_dataloader)

# Resume training with a new instance of MyModel
model_resumed = MyModel() # A fresh instance!
resume_checkpoint_path = "checkpoints/epoch=4.ckpt" # Adapt path as needed
trainer = pl.Trainer(max_epochs=10, default_root_dir="checkpoints/", resume_from_checkpoint=resume_checkpoint_path)
trainer.fit(model_resumed, train_dataloader)

```

**Commentary:** This example showcases resuming training with a *new* instance of the `LightningModule`. This is perfectly valid provided the model architecture remains unchanged. PyTorch Lightning will load the weights from the checkpoint into the new model instance.  This highlights that the `LightningModule` instance itself isn't directly persisted; only its state is.  This approach is valuable for scenarios where the model needs to be recreated or if the original instance is no longer accessible.



**3. Resource Recommendations:**

The official PyTorch Lightning documentation provides comprehensive guides on training, checkpointing, and resuming.   Explore the sections on the `Trainer` class and its various arguments, particularly those concerning logging and checkpointing.   Furthermore, paying close attention to the API documentation for `LightningModule` will clarify the mechanisms involved in saving and restoring model states.  Finally, reviewing examples in the PyTorch Lightning GitHub repository often provides practical insights and solutions to common problems encountered during checkpoint management.  Consider studying example notebooks which demonstrate best practices for large-scale training and model persistence.
