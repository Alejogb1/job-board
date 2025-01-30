---
title: "How can a PyTorch model be extracted from a PyTorch Lightning model?"
date: "2025-01-30"
id: "how-can-a-pytorch-model-be-extracted-from"
---
PyTorch Lightning, while streamlining many aspects of model training, often obscures the direct access to the underlying PyTorch model instance. This separation is intentional, fostering cleaner training loops and improved reproducibility; however, situations inevitably arise where extracting the raw PyTorch model becomes necessary, for instance, for deployment without the Lightning framework or for very specific fine-tuning tasks outside of its controlled environment. My own experience working on a complex multi-modal learning project highlighted this issue when migrating a trained model to an embedded device with resource constraints and without the support of the complete PyTorch Lightning dependency.

The core concept involves understanding that a PyTorch Lightning `LightningModule` class is primarily a container and a specification for the model's training and validation procedures. The actual, trainable PyTorch model is an attribute of this `LightningModule`.  Accessing this attribute grants direct control over the core neural network's weights and architecture.  The specific attribute is conventionally named `model` (and in older versions it might be a custom name defined within the `__init__` of the `LightningModule`).  Therefore, the process of extraction revolves around obtaining an instantiated `LightningModule` class (typically after training), and accessing this attribute. It is crucial that the training process has completed (or a checkpoint is loaded) to have fully trained model weights.

The simplest extraction occurs when one already has a trained `LightningModule` object. Assume we have trained a classifier within a `LightningModule` class named `MyClassifierModule` and stored the trained instance in a variable called `trained_model`. The underlying PyTorch model is extracted with the code:

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

# Define a simple PyTorch model
class SimpleClassifier(nn.Module):
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

# Define a PyTorch Lightning Module
class MyClassifierModule(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.model = SimpleClassifier(input_size, hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
      return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Example training setup (replace with your actual training logic)
input_size = 10
hidden_size = 20
num_classes = 2
model_instance = MyClassifierModule(input_size, hidden_size, num_classes)
dummy_input = torch.rand(1,input_size)
dummy_target = torch.randint(0, num_classes, (1,))
dummy_batch = (dummy_input, dummy_target)

trainer = pl.Trainer(max_epochs=1, enable_progress_bar=False)
trainer.fit(model_instance, train_dataloaders=[dummy_batch])

# Extraction
extracted_model = model_instance.model

# Verify it is indeed a PyTorch model
print(type(extracted_model))

# Example usage
sample_input = torch.randn(1, input_size)
output = extracted_model(sample_input)
print("Output shape:", output.shape)
```

In this code, `model_instance.model` directly accesses the `SimpleClassifier` instance which was encapsulated inside `MyClassifierModule`. The type check (`type(extracted_model)`) verifies it is a `SimpleClassifier` instance (or whatever the underlying model was), demonstrating successful extraction. Then a simple forward pass verifies functionality. This straightforward method is applicable in scenarios where the `LightningModule` instance is already available, either from the training process or having been loaded via checkpoint.

However, a more common situation involves needing to extract the model from a checkpoint file.  PyTorch Lightning stores all module information in the checkpoint file, including the state dictionary of the PyTorch model. To achieve this, we must reconstruct an instance of the `LightningModule`, load the checkpoint state into it, and then retrieve the model from its `model` attribute.

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
import os

# (Same SimpleClassifier and MyClassifierModule as before)

class SimpleClassifier(nn.Module):
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

# Define a PyTorch Lightning Module
class MyClassifierModule(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.model = SimpleClassifier(input_size, hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
      return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Create dummy data
input_size = 10
hidden_size = 20
num_classes = 2

# Example training setup to create a checkpoint
model_instance = MyClassifierModule(input_size, hidden_size, num_classes)
dummy_input = torch.rand(1, input_size)
dummy_target = torch.randint(0, num_classes, (1,))
dummy_batch = (dummy_input, dummy_target)
trainer = pl.Trainer(max_epochs=1, default_root_dir='./tmp', enable_progress_bar=False) # temp output directory
trainer.fit(model_instance, train_dataloaders=[dummy_batch])
checkpoint_path = os.path.join(trainer.default_root_dir, 'lightning_logs', 'version_0', 'checkpoints', 'epoch=0-step=1.ckpt')


# Create a new instance of the LightningModule
model_instance_loaded = MyClassifierModule(input_size, hidden_size, num_classes)
# Load the checkpoint into this instance
checkpoint = torch.load(checkpoint_path)
model_instance_loaded.load_state_dict(checkpoint['state_dict'])

# Extract the model
extracted_model = model_instance_loaded.model

# Verify it's a PyTorch model and test it
print(type(extracted_model))
sample_input = torch.randn(1, input_size)
output = extracted_model(sample_input)
print("Output shape:", output.shape)

#cleanup temp directory
import shutil
shutil.rmtree('./tmp')
```
In this example, a checkpoint is created using the `Trainer`, and its path is saved.  A new `MyClassifierModule` instance is created (`model_instance_loaded`), and the state dictionary from the saved checkpoint file is loaded into it using `load_state_dict`. Accessing `model_instance_loaded.model` then successfully retrieves the underlying PyTorch model.  This approach is fundamental when deploying models from previously trained checkpoint files. It is important to note that `load_state_dict` expects a dictionary with keys matching the layers of the model, so the architecture must exactly match that during training, which is why we instantiate a new instance with the same `input_size, hidden_size, num_classes` parameters.

Finally, in certain situations, one might wish to extract the PyTorch model **before** training has concluded. Although less common, this may be useful for early visualization or for applying model distillation to a not-yet-fully-converged model. The approach is the same, accessing the attribute: `model` from the instantiated `LightningModule`. This can be done *before* initiating training or inside a callback during training. This highlights that the model is present from initialization within the LightningModule, even if it does not contain trained parameters. This is demonstrated in a small snippet below, building from the earlier snippets:
```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

# (Same SimpleClassifier and MyClassifierModule as before)
class SimpleClassifier(nn.Module):
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

# Define a PyTorch Lightning Module
class MyClassifierModule(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.model = SimpleClassifier(input_size, hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
      return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# Initializing the LightningModule (no training yet)
input_size = 10
hidden_size = 20
num_classes = 2
model_instance = MyClassifierModule(input_size, hidden_size, num_classes)

# Extract the model before training
extracted_model_before_training = model_instance.model

# Verify it's a PyTorch model, but with random weights
print("Type before training:", type(extracted_model_before_training))
sample_input = torch.randn(1, input_size)
output_before = extracted_model_before_training(sample_input)
print("Output shape before training:", output_before.shape)

# Example training setup
dummy_input = torch.rand(1, input_size)
dummy_target = torch.randint(0, num_classes, (1,))
dummy_batch = (dummy_input, dummy_target)

trainer = pl.Trainer(max_epochs=1, enable_progress_bar=False)
trainer.fit(model_instance, train_dataloaders=[dummy_batch])

# Extract the model after training
extracted_model_after_training = model_instance.model
print("Type after training:", type(extracted_model_after_training))
output_after = extracted_model_after_training(sample_input)
print("Output shape after training:", output_after.shape)
```
As shown above, this approach allows you to access the model prior to any training, albeit its parameters will be randomly initialized.  This is useful in various pre-training analysis or modification tasks. Subsequently, during or after training, the underlying model, accessed via the same `.model` attribute, now has trained parameters, as is confirmed by the altered output.

For further exploration of these concepts, I recommend reviewing the official PyTorch documentation regarding the `torch.nn.Module` class, along with the PyTorch Lightning documentation on `LightningModule` structure, and the checkpointing mechanism. Also, several blog posts are available that delve into PyTorch state dicts which might be informative.
