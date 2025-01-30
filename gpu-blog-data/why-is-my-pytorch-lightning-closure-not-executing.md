---
title: "Why is my PyTorch Lightning closure not executing?"
date: "2025-01-30"
id: "why-is-my-pytorch-lightning-closure-not-executing"
---
My experience with PyTorch Lightning has frequently highlighted subtle issues with closures, and the failure of a closure to execute typically stems from a misunderstanding of its context within the framework's training loop. Unlike standard Python functions, closures defined within a PyTorch Lightning `LightningModule` are subject to the orchestration of the training process. If a closure fails to execute as expected, it usually boils down to incorrect instantiation, misapplication of training signals, or the absence of necessary input to trigger the closure.

The core issue revolves around the interplay between PyTorch Lightning's automated training workflow and the stateful nature of closures. In a typical scenario, a closure might be intended for calculating a metric or performing a specific computation that depends on the intermediate results generated during training. This execution is not automatic; it requires explicit invocation, often as part of a hook or callback within the `LightningModule`. Consider closures designed to update an internal buffer or aggregate statistics: if not properly integrated within the training sequence, they will remain latent.

To elaborate further, PyTorch Lightning manages the data flow and computational graph across epochs and batches. Functions within a `LightningModule`, such as `training_step`, `validation_step`, or `test_step`, are executed during these processes. Closures, on the other hand, don’t inherently possess this kind of lifecycle. They are merely function objects with associated lexical environments; their execution must be explicitly requested, usually inside one of the lifecycle methods or within a callback. A failure to invoke the closure correctly is the most frequent culprit for an observed lack of execution. Closures also capture variables from their enclosing scope, and if those variables don't exist in the lifecycle execution environment, the closure either throws an error, fails silently, or returns unexpected results.

Here are three illustrative examples to demonstrate common causes of closures not executing within a PyTorch Lightning context:

**Example 1: Misplaced Closure Definition and Invocation**

The first example demonstrates a closure defined within the `__init__` method and then attempted to be invoked directly in the `training_step` without correct state management:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

class MisplacedClosureModel(pl.LightningModule):
    def __init__(self, input_size=10, hidden_size=20, output_size=1, learning_rate=1e-3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.learning_rate = learning_rate
        self.intermediate_values = []

        def capture_intermediate(x):
            self.intermediate_values.append(x.detach().cpu().numpy())
        
        self.capture_intermediate = capture_intermediate

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.capture_intermediate(self.fc1(x))  # Incorrect direct invocation
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


if __name__ == '__main__':
    # Sample Data
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=32)

    model = MisplacedClosureModel()
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model, train_loader)
    
    print(f"Captured Values: {len(model.intermediate_values)}")
```

In this example, the `capture_intermediate` closure is defined within `__init__`. The closure captures `self.intermediate_values`, which is expected to accumulate values across batches. The issue is that the direct invocation `self.capture_intermediate(self.fc1(x))` within `training_step` is conceptually incorrect. Whilst it will technically execute in each training step, the issue is that the intermediate value is not captured correctly because it is a side effect inside training step (a function that returns the loss, which is then used by the optimizer). Therefore, the values are captured, but their update is done on a different thread/process so that the internal state does not change between batches. This will mean that the length of the values printed at the end will be very different to what is expected. We need the closure to be a function that is used as an argument, but that also updates state as part of the computation, e.g. within a callback or metric.

**Example 2: Closure within a Metric for Correct Aggregation**

To solve this, a metric can be used, which will aggregate the values correctly through batches by acting on a copy of the tensor and then adding it back into state.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from torchmetrics import Metric
class CaptureIntermediateMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state("intermediate_values", default=[], dist_reduce_fx="cat")

    def update(self, x):
        self.intermediate_values.append(x.detach().cpu().numpy())

    def compute(self):
       return self.intermediate_values

class MetricClosureModel(pl.LightningModule):
    def __init__(self, input_size=10, hidden_size=20, output_size=1, learning_rate=1e-3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.learning_rate = learning_rate
        self.intermediate_metric = CaptureIntermediateMetric()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.intermediate_metric.update(self.fc1(x))
        self.log("train_loss",loss)
        return loss

    def on_train_epoch_end(self):
      values = self.intermediate_metric.compute()
      print(f"Captured Values: {len(values)}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == '__main__':
    # Sample Data
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=32)

    model = MetricClosureModel()
    trainer = pl.Trainer(max_epochs=2)
    trainer.fit(model, train_loader)
```

Here, `CaptureIntermediateMetric` is a metric which is able to update an internal state based on the inputs given. This allows the intermediate values to be properly collected between batches and epochs, and the aggregated value to be returned. The crucial change here is that we use the update method within `training_step` and then calculate the final result within `on_train_epoch_end`. This is the key idea to successfully implementing closures in pytorch lightning.

**Example 3: Closure in a Callback**

This example shows how to implement the same logic as above, but with a callback instead of a metric. It is useful if you need to control aspects of the training based on an output of the computation, instead of just collecting statistics.

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl

class CaptureIntermediateCallback(pl.Callback):
    def __init__(self):
        super().__init__()
        self.intermediate_values = []
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        x,y = batch
        intermediate_value = pl_module.fc1(x)
        self.intermediate_values.append(intermediate_value.detach().cpu().numpy())
    
    def on_train_epoch_end(self, trainer, pl_module):
      print(f"Captured Values: {len(self.intermediate_values)}")


class CallbackClosureModel(pl.LightningModule):
    def __init__(self, input_size=10, hidden_size=20, output_size=1, learning_rate=1e-3):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.learning_rate = learning_rate

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.mse_loss(y_hat, y)
        self.log("train_loss",loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

if __name__ == '__main__':
    # Sample Data
    X = torch.randn(100, 10)
    y = torch.randn(100, 1)
    train_dataset = TensorDataset(X, y)
    train_loader = DataLoader(train_dataset, batch_size=32)
    
    callback = CaptureIntermediateCallback()

    model = CallbackClosureModel()
    trainer = pl.Trainer(max_epochs=2, callbacks=[callback])
    trainer.fit(model, train_loader)
```

The `CaptureIntermediateCallback` uses the same idea as above, by capturing the intermediate value using the hook `on_train_batch_end`, and then printing the collected results at `on_train_epoch_end`.

In summary, debugging PyTorch Lightning closures typically requires a shift from thinking of them as standalone Python functions to entities that are controlled by the framework’s lifecycle. The key is ensuring the closure's execution is explicitly integrated into the training or evaluation flow using metrics or callbacks. Incorrect direct invocation within `training_step` or incorrect scope capture are the main reasons for the observed failure of the closure to run.

For further study, I suggest reviewing the PyTorch Lightning documentation on `LightningModule` lifecycle hooks and callbacks. Detailed investigation of custom metrics through the `torchmetrics` library can also prove useful. Understanding how these components interact with the automated training loop will make debugging these subtle issues far more straightforward. The core documentation for PyTorch Lightning and the documentation for `torchmetrics` will be very beneficial.
