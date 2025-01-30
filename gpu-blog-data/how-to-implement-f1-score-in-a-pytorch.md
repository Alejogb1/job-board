---
title: "How to implement F1 score in a PyTorch Ignite custom metric using scikit-learn?"
date: "2025-01-30"
id: "how-to-implement-f1-score-in-a-pytorch"
---
The F1 score, a harmonic mean of precision and recall, is frequently employed in evaluating binary or multi-class classification models, particularly when class imbalance is present. Integrating it into a PyTorch Ignite training loop requires a custom metric implementation that leverages the established functionality of scikit-learn for computation.

Fundamentally, PyTorch Ignite’s metric system is designed for accumulating batches of model predictions and targets, then performing a calculation to derive a scalar value representing the metric’s performance. This approach necessitates that we adapt scikit-learn's F1 score calculation to operate within this iterative batch-wise accumulation framework. Scikit-learn’s `f1_score` function operates on complete sets of predicted and true labels; thus, the challenge is to progressively accumulate these values from each batch in PyTorch Ignite's `metrics.Metric` class, before invoking the `f1_score` function during the computation stage.

The approach I've found effective involves creating a class that inherits from `ignite.metrics.Metric`. This class manages the storage of accumulated predicted labels and true labels during the execution of an Ignite engine. The update method, which is called for each batch of data, appends the predictions and targets to these storage variables. The compute method, executed after all batch data has been processed, then uses scikit-learn's `f1_score` to compute the overall F1 score from these accumulated labels.

Here's a code example demonstrating a custom metric for binary classification:

```python
import torch
import numpy as np
from ignite.metrics import Metric
from sklearn.metrics import f1_score

class BinaryF1Score(Metric):
    def __init__(self, output_transform=lambda x: x, average='binary'):
        self._average = average
        super().__init__(output_transform=output_transform)

    def reset(self):
        self._preds = []
        self._targets = []

    def update(self, output):
        y_pred, y = output # Assume output is a tuple of (predictions, targets)
        y_pred_binary = torch.round(torch.sigmoid(y_pred)).int().cpu().numpy() # Convert to binary
        y_binary = y.int().cpu().numpy() # Convert to binary

        self._preds.append(y_pred_binary)
        self._targets.append(y_binary)

    def compute(self):
         if not self._preds:
            return np.nan # Handle edge case of no predictions/targets

         all_preds = np.concatenate(self._preds)
         all_targets = np.concatenate(self._targets)

         return f1_score(all_targets, all_preds, average=self._average)

```

In this example, the `__init__` method accepts an optional `average` parameter, allowing for binary, micro, or macro F1 score calculations (which is the same setting with scikit-learn `f1_score`). The `reset` method is invoked at the beginning of an epoch or evaluation, ensuring that the accumulated values are cleared.  The `update` method processes each batch output, transforming the raw model predictions into binary values through a sigmoid activation followed by rounding. Both predictions and ground truth targets are moved to the CPU and converted to NumPy arrays before appending to the internal lists. Finally, the `compute` method concatenates all accumulated predictions and targets into complete arrays and then invokes `f1_score` from scikit-learn to obtain the final F1 score. It’s important to note that an early return of `np.nan` handles the edge case where no predictions have been made (e.g., during first evaluation prior to training), which prevents errors during metric computation. This is essential for robust training pipelines.

For multi-class classification, a slightly modified implementation is required:

```python
class MultiClassF1Score(Metric):
    def __init__(self, output_transform=lambda x: x, average='macro'):
        self._average = average
        super().__init__(output_transform=output_transform)

    def reset(self):
        self._preds = []
        self._targets = []

    def update(self, output):
        y_pred, y = output # Assume output is a tuple of (logits, targets)
        y_pred_classes = torch.argmax(y_pred, dim=1).cpu().numpy()  # Get predicted class indices
        y_classes = y.cpu().numpy()

        self._preds.append(y_pred_classes)
        self._targets.append(y_classes)

    def compute(self):
          if not self._preds:
             return np.nan

          all_preds = np.concatenate(self._preds)
          all_targets = np.concatenate(self._targets)

          return f1_score(all_targets, all_preds, average=self._average)
```

The multi-class version follows the same architecture as the binary version, but differs in how predictions are handled. Rather than applying a sigmoid and rounding, the predicted classes are determined by `torch.argmax` along the class dimension from the model's raw output logits. As before, both predicted and true class labels are converted to CPU NumPy arrays before appending to internal data structures. The `compute` method remains largely the same as its binary counterpart, passing these class labels directly into `f1_score`.

Lastly, it is beneficial to see an example using `output_transform`, demonstrating how the metric can be integrated into Ignite more seamlessly:

```python
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.metrics import MeanAbsoluteError
from ignite.handlers import EarlyStopping
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

# Dummy Data/Model Setup for demonstration
X = torch.rand(100, 10)
y = torch.randint(0, 2, (100,))
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32)
model = nn.Linear(10, 1)  # Binary Classifier
optimizer = Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCEWithLogitsLoss()

def training_step(engine, batch):
    model.train()
    x, y = batch
    y_pred = model(x)
    loss = loss_fn(y_pred, y.float().unsqueeze(1)) # Make target a float and same shape with y_pred
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

def transform_output(output):
  return output[0], output[1].squeeze() # squeeze y to remove dim of 1

trainer = Engine(training_step)
f1_metric = BinaryF1Score(output_transform=transform_output) # Use defined transform

f1_metric.attach(trainer, "f1_score")
mae_metric = MeanAbsoluteError()
mae_metric.attach(trainer, "mae")

@trainer.on(Events.EPOCH_COMPLETED)
def log_metrics(engine):
    metrics = engine.state.metrics
    f1_score = metrics['f1_score']
    mae_score = metrics['mae']
    print(f"Epoch: {engine.state.epoch}, F1 Score: {f1_score}, MAE: {mae_score}")

def score_function(engine):
  return engine.state.metrics['f1_score']

early_stopping_handler = EarlyStopping(patience=3, score_function=score_function, trainer=trainer)

trainer.add_event_handler(Events.COMPLETED, early_stopping_handler)
trainer.run(dataloader, max_epochs=50)
```

Here, `transform_output` preprocesses the output from the training step to be compatible with our defined `BinaryF1Score`.  The `BinaryF1Score` is instantiated, and it’s output transform is set using the `transform_output` function.  When the F1 score is attached to the trainer using `attach`, the transform specified in the F1 metric constructor is invoked on the output from the training step, extracting and returning the desired parts before sending them to the update method in the F1 score metric.  This allows for more flexible integration with various models and training procedures.

To enhance learning in this area, I suggest consulting the PyTorch Ignite official documentation, specifically regarding metric implementation and engine usage. Reviewing the scikit-learn documentation, particularly the `sklearn.metrics.f1_score` function, will offer a deeper comprehension of how the F1 score is computed and the different averaging options. Study material focused on the concepts of precision, recall, and the F1 score itself would provide a good theoretical foundation, ensuring that metrics are employed thoughtfully in training and evaluation. Also, examining common software engineering patterns employed when writing classes will be beneficial to understanding these metric implementations.
