---
title: "How do I implement early stopping in PyTorch?"
date: "2025-01-30"
id: "how-do-i-implement-early-stopping-in-pytorch"
---
Early stopping is a crucial technique in training neural networks to mitigate overfitting and improve generalization performance. I've encountered its necessity numerous times during model development, particularly when dealing with complex datasets and deep architectures. The core idea is to monitor a chosen validation metric during training, ceasing the process when that metric shows no further improvement, or even starts to degrade. This prevents the model from continuing to learn the nuances of the training set that don’t translate well to unseen data.

Implementing early stopping in PyTorch fundamentally involves keeping track of the best observed validation score and patience. Patience refers to the number of epochs or iterations the model is allowed to train despite no improvement in the validation metric before training is halted. The validation score is typically a metric like validation loss, accuracy, or area under the ROC curve, depending on the specific problem. The process requires periodic validation during training and a conditional check if the metric has improved relative to the best score encountered so far.

Here's a conceptual outline: during each epoch (or iteration if using gradient descent), validate the current model using the chosen validation set. Calculate the chosen metric. Compare this new metric against the best metric observed previously. If the new metric surpasses the previous best, update the best score, reset the patience counter, and save the current model’s parameters. If the new metric does not improve, decrement the patience counter. If patience reaches zero, stop training.

My first implementation employed a simple class-based approach, providing flexibility to control the validation metric, patience, and save path.

```python
import torch
import numpy as np
import os

class EarlyStopper:
    def __init__(self, patience=10, min_delta=0, save_path='best_model.pth', metric='loss'):
        self.patience = patience
        self.min_delta = min_delta
        self.save_path = save_path
        self.metric = metric
        self.counter = 0
        self.best_metric = None
        self.early_stop = False

    def _metric_improved(self, new_metric):
      if self.best_metric is None:
        return True

      if self.metric == 'loss':
        return new_metric < self.best_metric - self.min_delta
      elif self.metric == 'accuracy' or self.metric == 'auc':
        return new_metric > self.best_metric + self.min_delta
      else:
        raise ValueError("Metric must be loss, accuracy, or auc")


    def __call__(self, new_metric, model):
        if self._metric_improved(new_metric):
            self.best_metric = new_metric
            self.counter = 0
            torch.save(model.state_dict(), self.save_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


```

In the above `EarlyStopper` class, the `__init__` method initializes the patience, `min_delta` (minimal improvement to consider an improvement), save path, metric to evaluate and the counter. The `_metric_improved` method checks for improvement based on whether it is the first validation or the metric is better than the previously saved metric. The `__call__` method checks for improvement, saves the model, and increments the counter. If patience is exhausted the early_stop flag is set, allowing the user to check whether training should cease.

This simple class allowed me to experiment with varying patience and delta values, helping determine the most appropriate stopping point for different model architectures and dataset sizes. This approach offers clarity, but lacks the integration of external metric computation. It depends on providing the new metric as input.

My second approach focuses on integration within the training loop, embedding the logic directly, eliminating the need for a separate class. This strategy directly uses a validation loss function from `torch.nn`.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# Dummy Dataset
X_train = torch.randn(1000, 10)
y_train = torch.randint(0, 2, (1000,))
X_val = torch.randn(200, 10)
y_val = torch.randint(0, 2, (200,))

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


# Dummy Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = SimpleModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

patience = 5
best_val_loss = float('inf')
patience_counter = 0
save_path = 'best_model_embedded.pth'

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs.squeeze(), targets.float())
      loss.backward()
      optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
      for inputs, targets in val_loader:
        outputs = model(inputs)
        val_loss += criterion(outputs.squeeze(), targets.float()).item()
    val_loss /= len(val_loader)

    print(f'Epoch: {epoch+1}, Val Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), save_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break
```

In this second code block, early stopping is implemented directly within the training loop. This reduces the boilerplate needed for the early stopping logic by directly calculating the loss in the evaluation phase. It has increased efficiency because it calculates and compares the validation loss directly within the training loop. This integration simplifies the script but may make adjustments more complex. I have used dummy data, a dummy model, and a loss function to show a complete minimal example.

My final implementation leverages the `torchmetrics` library, allowing for a more modular and flexible metric handling.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from torchmetrics import Accuracy

# Dummy Dataset
X_train = torch.randn(1000, 10)
y_train = torch.randint(0, 2, (1000,))
X_val = torch.randn(200, 10)
y_val = torch.randint(0, 2, (200,))

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Dummy Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

model = SimpleModel()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

patience = 5
best_val_accuracy = 0.0
patience_counter = 0
save_path = 'best_model_torchmetrics.pth'
accuracy = Accuracy(task='binary')

num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs.squeeze(), targets.float())
      loss.backward()
      optimizer.step()


    model.eval()
    with torch.no_grad():
      for inputs, targets in val_loader:
         outputs = model(inputs)
         accuracy.update(outputs.squeeze(), targets)
    val_accuracy = accuracy.compute()
    accuracy.reset()


    print(f'Epoch: {epoch+1}, Val Accuracy: {val_accuracy:.4f}')

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        patience_counter = 0
        torch.save(model.state_dict(), save_path)
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping!")
            break
```

This example showcases the power of `torchmetrics` to calculate accuracy. A new `torchmetrics.Accuracy` object has been created to calculate accuracy and use it as the metric for early stopping. This approach offers flexibility and reduces the need to implement custom metric calculation.

For further exploration of these topics, I recommend reviewing the documentation of PyTorch related to training loops, model saving, and also the official documentation for `torchmetrics`. Additionally, research into best practices regarding hyperparameter tuning, especially patience selection, can substantially impact the efficacy of the implemented early stopping mechanisms. Books and articles focused on the practicalities of deep learning can also provide guidance on adapting these techniques to a wider range of problems.
