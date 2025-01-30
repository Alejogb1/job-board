---
title: "Why is pytorch_lightning missing the LightningModule attribute?"
date: "2025-01-30"
id: "why-is-pytorchlightning-missing-the-lightningmodule-attribute"
---
A common frustration for new PyTorch Lightning users arises when they attempt to access attributes defined directly within their `LightningModule` subclass, only to find them seemingly missing after initialization. This is not a bug, but a deliberate design choice related to how PyTorch Lightning manages and orchestrates training, validation, and testing processes. It’s tied to the framework's separation of concerns and its focus on managing model state and data flow within the training loop, rather than direct access.

The core issue is that PyTorch Lightning actively constructs and utilizes a `LightningModule` instance as a blueprint, not necessarily the *exact* object you've initialized. When you define your class like `class MyModel(pl.LightningModule): ...`, and then instantiate it as `model = MyModel(...)`, what you are actually passing to PyTorch Lightning’s `Trainer` is a reference to this *class*, not the instantiated `model` object directly. The `Trainer` then internally creates its own instance of your `MyModel` class, effectively decoupling the object you created from the one being used in the training process. This separation allows Lightning to control model initialization, loading from checkpoints, and management of distributed training across different devices without interfering with your original object.

Therefore, any attributes you set *after* initializing your model via `model = MyModel(...)` will only be present on that object and *not* on the internal instance being managed by PyTorch Lightning during the training loop. This includes any attributes you might have expected to persist such as a data loader reference, a learning rate, or other configuration information.

To properly transfer data into the training loop, PyTorch Lightning offers several established patterns, primarily using arguments to the `__init__` method, and overriding specific methods within the `LightningModule`, such as `training_step`, `validation_step`, and the configuration methods such as `configure_optimizers`. Instead of defining attributes that are intended to be passed to the `Trainer`, users should pass this information via arguments to the `__init__` method and initialize them within that method. Similarly, any training hyperparameters such as learning rate or batch size should be accessible via an attribute and properly utilized by the `configure_optimizers` method, which in turn is only called within PyTorch Lightning’s training loop. Any data loader is properly connected through implementing methods such as `train_dataloader`, `val_dataloader` and `test_dataloader`.

Here are three code examples to illustrate the problem and provide solutions:

**Example 1: The Incorrect Approach**

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class SimpleModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


model = SimpleModel(10, 20, 5)
model.learning_rate = 0.01  # Incorrect: attribute is only on local model
# Simulate some training data
train_data = [(torch.rand(1, 10), torch.rand(1, 5)) for _ in range(100)]
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4)
trainer = pl.Trainer(max_epochs=2)
trainer.fit(model, train_loader)


# Incorrect access of attribute (would raise AttributeError)
try:
    print(trainer.model.learning_rate)
except AttributeError:
    print("Error: 'learning_rate' attribute is not directly accessible")
```

In this code, `learning_rate` is assigned directly to the local `model` *object* after initialization. Because the `Trainer` creates its *own* instance of `SimpleModel`, that instance will not have this attribute. The code would therefore raise an error when trying to access the `learning_rate` from the trainer’s internal instance of the model.

**Example 2: The Correct Approach (Passing as `__init__` Parameter)**

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class SimpleModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

model = SimpleModel(10, 20, 5, learning_rate=0.01) # Correct: passed via __init__
# Simulate some training data
train_data = [(torch.rand(1, 10), torch.rand(1, 5)) for _ in range(100)]
train_loader = torch.utils.data.DataLoader(train_data, batch_size=4)

trainer = pl.Trainer(max_epochs=2)
trainer.fit(model, train_loader)
# Accessing learning rate through the trainers model object is now fine
print(trainer.model.learning_rate)
```

Here, `learning_rate` is passed as a parameter to the `__init__` method of the `SimpleModel`. The internal model instance created by the `Trainer` will now have the `learning_rate` attribute set. This approach ensures the data necessary for the model is passed from your instance of the model to the one being managed by Lightning.

**Example 3: Correctly Using `train_dataloader()`**

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl

class SimpleModel(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, learning_rate, train_data):
        super().__init__()
        self.learning_rate = learning_rate
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.train_data = train_data


    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=4)


# Simulate some training data
train_data = [(torch.rand(1, 10), torch.rand(1, 5)) for _ in range(100)]

model = SimpleModel(10, 20, 5, learning_rate=0.01, train_data=train_data)  # Correct: all needed information in init

trainer = pl.Trainer(max_epochs=2)
trainer.fit(model)
# Model has access to the `train_data` attribute
print(trainer.model.train_data)


```

In this version, the training data itself is also provided via the `__init__` method and the data loader is returned via the `train_dataloader()` method, meaning the internal lightning instance can access the training data directly.

In summary, the `LightningModule` attribute not being accessible after training is not an error but a consequence of how PyTorch Lightning manages model instances. All initialization data needs to be passed as arguments to the `__init__` method and any dataloaders should be accessed via the specific helper methods (e.g., `train_dataloader`). This ensures that the internally managed model instance by PyTorch Lightning is correctly configured with the necessary attributes, while keeping the user model and the managed training loop separate.

For resources, I would recommend spending time with the official PyTorch Lightning documentation, as well as exploring the various examples they provide on different modeling tasks. Additionally, there are numerous online tutorials and blog posts covering common patterns and best practices when using PyTorch Lightning. Experimenting with small-scale projects using the framework is also valuable, paying careful attention to attribute assignment and data loading strategies as a way to reinforce the conceptual differences between the initialized user model and the managed training object.
