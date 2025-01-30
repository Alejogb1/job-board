---
title: "Why does `pytorch-lightning.configure_optimizer` throw an AssertionError about param group type?"
date: "2025-01-30"
id: "why-does-pytorch-lightningconfigureoptimizer-throw-an-assertionerror-about-param"
---
The core issue stems from an inconsistency between how PyTorch Lightning expects optimizer parameter groups to be configured and how custom `configure_optimizers` methods sometimes inadvertently create them. Specifically, the assertion error arises when `pytorch-lightning.configure_optimizers` anticipates a specific structure for parameter groups within an optimizer (typically a list of dictionaries), and it encounters an instance where this structure is either missing, or of the incorrect type. This can occur frequently when a custom parameter tuning strategy, or a specific layer configuration needing different learning rates, is introduced directly into the model definition rather than adhering to Lightning’s prescribed approach within the `configure_optimizers` method.

Through several projects, I have encountered this error predominantly in scenarios involving complex parameter adjustments or when a developer attempts to handle optimizer creation outside of the intended Lightning flow. The problem is less about the optimization algorithm itself, but rather the format of the data specifying which parameters it should act upon and their associated settings like learning rates or weight decay.

The `configure_optimizers` method in PyTorch Lightning is meant to be a highly flexible interface for configuring multiple optimizers, learning rate schedulers, and parameter groups. However, it strictly expects the method to return one of the following formats:
1.  **Single Optimizer**: Returns an instance of `torch.optim.Optimizer`
2.  **Optimizer and Scheduler**: Returns a tuple containing an optimizer instance and a learning rate scheduler instance.
3.  **Multiple Optimizers/Schedulers**: Returns a list where each element is either an optimizer, or a dictionary containing keys like `optimizer` (required), and optionally `lr_scheduler`, `monitor`, and `frequency`.
4. **Parameter Group Specific**: When creating multiple parameter groups within one optimizer, the optimizer dictionary should include a `params` key containing the list of parameters, and other keys specific to parameter group such as a particular `lr`.

The assertion failure occurs when, under parameter group-specific optimization, this structure is not met, particularly if `params` are not supplied as a list, or if `lr` settings are not specified correctly within the dictionary context. This is crucial because the internal Lightning code uses indexing and specific attribute access that expects the `param_groups` to conform to a set, list-of-dictionaries format. Incorrectly structured information bypasses that expectation.

Let's illustrate this with code examples, highlighting the cases that commonly trigger the error:

**Code Example 1: Incorrect Parameter Group Specification**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer

class BadModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            [
               {'params': self.fc1.parameters(), 'lr': 1e-3},
               self.fc2.parameters() # Error here
            ]
        )
        return optimizer

model = BadModel()
trainer = Trainer(max_epochs=1)
dummy_input = torch.randn(1, 10)
dummy_target = torch.randn(1, 2)
trainer.fit(model, [[dummy_input, dummy_target]])
```

*Commentary:* In this example, the `configure_optimizers` method attempts to define two parameter groups for the Adam optimizer. The first group correctly specifies the parameters and learning rate as a dictionary, but the second group incorrectly passes the parameters directly without the required dictionary wrapper. The `AssertionError` arises because Lightning expects a list of dictionaries when parameter groups are involved and finds an `nn.Parameter` instance in place of a dictionary during processing of the second param_group. This highlights how failure to use a consistent dictionary format for all parameter groups triggers this error.

**Code Example 2:  Using `param_groups` directly within configure_optimizers**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer


class BadModel2(LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters()) # Initial optimizer setup
        # Attempting to manipulate param_groups directly after optimizer instance has been initialized.
        optimizer.param_groups[0]['lr'] = 1e-3 # Incorrect practice
        optimizer.param_groups.append({ 'params': self.fc2.parameters(), 'lr' : 1e-2}) # Also incorrect
        return optimizer

model2 = BadModel2()
trainer = Trainer(max_epochs=1)
dummy_input = torch.randn(1, 10)
dummy_target = torch.randn(1, 2)
trainer.fit(model2, [[dummy_input, dummy_target]])
```

*Commentary:* Here, the user initially creates a standard Adam optimizer with all the model's parameters. Then, the user attempts to modify the `param_groups` attribute of the optimizer outside the scope of an intended dictionary. While this may seem valid within standard PyTorch, Lightning’s internal logic requires parameter group configuration to occur within a dictionary based list. The attempt to modify and add to optimizer.param_groups outside of this context triggers the assertion.

**Code Example 3: Correct Parameter Group Specification (Mitigating the Error)**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from pytorch_lightning import LightningModule, Trainer


class GoodModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            [
                {'params': self.fc1.parameters(), 'lr': 1e-3},
                {'params': self.fc2.parameters(), 'lr': 1e-2}
            ]
        )
        return optimizer


model3 = GoodModel()
trainer = Trainer(max_epochs=1)
dummy_input = torch.randn(1, 10)
dummy_target = torch.randn(1, 2)
trainer.fit(model3, [[dummy_input, dummy_target]])
```

*Commentary:* This corrected version demonstrates the proper way to specify parameter groups in `configure_optimizers`. Each group, whether for `fc1` or `fc2`, is explicitly created as a dictionary containing a `params` key with parameter list and corresponding `lr` for the specific group. By using the appropriate list of dictionaries format, the PyTorch Lightning internal handling correctly parses these parameter groups.

In summary, the `AssertionError` arises from an incompatibility between the structure that PyTorch Lightning expects for optimizer parameter groups and the structure the developer actually provides within the `configure_optimizers` method. The root of the problem tends to stem from attempts to manipulate the optimizer's parameter groups after the optimizier instantiation or when a dictionary-based structure is not used for each group.

For further understanding, several resources have been invaluable to me during my projects.  The official PyTorch documentation offers in-depth explanations of the `torch.optim` package and its capabilities, particularly relating to parameter groups. The PyTorch Lightning documentation contains detailed descriptions of the `configure_optimizers` method and the various formats it supports. Additionally, carefully examining the numerous examples available within PyTorch Lightning’s GitHub repository can also offer clarification of best practices for parameter manipulation and optimizer configurations, particularly through usage examples. Lastly, the source code directly for `pytorch-lightning` offers deep insight into where and how assertions are checked and how `param_groups` are used.
