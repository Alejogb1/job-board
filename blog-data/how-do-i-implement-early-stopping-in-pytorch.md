---
title: "How do I implement early stopping in PyTorch?"
date: "2024-12-23"
id: "how-do-i-implement-early-stopping-in-pytorch"
---

Alright, let's talk about early stopping in PyTorch. I’ve definitely seen my share of models overfit, and implementing early stopping effectively is a cornerstone of robust model training. It’s not just about preventing overfitting; it’s also about saving computational resources and time. Instead of letting a model train for a fixed, often arbitrary number of epochs, early stopping monitors the model’s performance on a validation set and halts training when performance plateaus or starts to degrade. This is a core technique in machine learning for optimizing resource usage and ensuring generalizability.

The basic idea is simple: we track a chosen metric on our validation set (like loss or accuracy). If that metric doesn't improve for a certain number of epochs – what we often call the “patience” – we stop training. Now, while conceptually straightforward, implementation can vary slightly, and it's worth exploring a robust approach. I’ve found that implementing early stopping as a class, as opposed to a few haphazard functions, allows for greater flexibility, reuse, and a more structured setup.

First, we’ll build the core logic. We need to keep track of the best validation score we've seen, the number of epochs since that best score, and the patience. We’ll also need to save the model's weights associated with that best score. This means we'll need to define a class that encapsulates these details. Here is the first code snippet illustrating this structure:

```python
import torch
import copy

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

    def load_checkpoint(self, model):
      '''Loads model when validation loss decrease.'''
      model.load_state_dict(torch.load(self.path))
```

This `EarlyStopping` class handles the core logic for us. The `__init__` method sets up the instance variables, `__call__` evaluates whether to stop or not, the `save_checkpoint` method saves the best model parameters seen so far, and `load_checkpoint` reloads that state. Note that we use a negative of the validation loss for scoring purposes, as generally we want to maximize accuracy or some similar measure, while minimizing the loss.

Now, let’s see how we’d use this within a typical training loop. Here’s where I’ve found it’s crucial to keep the training loop concise and readable, and the early stopping logic cleanly separated. I’ve included a simplified training loop to show this in context:

```python
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Generate some dummy data
X_train = torch.randn(100, 10)
y_train = torch.randint(0, 2, (100,))
X_val = torch.randn(50, 10)
y_val = torch.randint(0, 2, (50,))

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_data, batch_size=10)
val_loader = DataLoader(val_data, batch_size=10)

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 100

# Initialize early stopping
early_stopping = EarlyStopping(patience=10, verbose=True, path='best_model.pt')

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
             outputs = model(batch_X)
             loss = criterion(outputs, batch_y)
             val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f'Epoch: {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

    early_stopping(avg_val_loss, model)
    if early_stopping.early_stop:
        print("Early stopping triggered.")
        break

print("Loading best model weights...")
early_stopping.load_checkpoint(model)
```

In this example, the `EarlyStopping` class is instantiated and used within the loop.  The critical part is the call to `early_stopping(avg_val_loss, model)` after each epoch's validation, passing the validation loss and the current model.  The logic inside the `EarlyStopping` instance determines if training should stop. Note the model weights at `best_model.pt` are automatically loaded with `early_stopping.load_checkpoint(model)` after the training has terminated.

Finally, I’ve always found it beneficial to make the early stopping criteria adaptable. Perhaps you want to monitor a different metric, or adjust your patience based on the dataset. It would be ideal to have an option to set a custom function for calculating the score being tracked, and we can add an option to switch between minimizing or maximizing the chosen metric, such as `metric_mode` in `EarlyStopping`:

```python
import torch
import copy

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print,
                 metric_mode = 'min', score_function = None):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
            metric_mode (str): Whether to maximize or minimize metric being tracked.
                            Default: 'min'
            score_function (function): optional custom score function. Defaults to negative loss.
                            Default: None
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.metric_mode = metric_mode
        self.score_function = score_function

    def __call__(self, val_metric, model):
        if self.score_function is None:
          if self.metric_mode == 'min':
            score = -val_metric
          elif self.metric_mode == 'max':
            score = val_metric
        else:
          score = self.score_function(val_metric)


        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_metric, model)
            self.counter = 0

    def save_checkpoint(self, val_metric, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation metric improved ({self.val_loss_min:.6f} --> {val_metric:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_metric

    def load_checkpoint(self, model):
      '''Loads model when validation loss decrease.'''
      model.load_state_dict(torch.load(self.path))
```

The modified class allows one to pass a `metric_mode` during initialization (`'min'` or `'max'`), allowing for early stopping based on accuracy (in which we look for an increase) or loss (in which we look for a decrease). More flexibly, a custom scoring function `score_function` can be passed as well, allowing further fine-tuning of the early stopping criteria.

For resources, I highly recommend exploring the relevant chapters in *Deep Learning* by Goodfellow, Bengio, and Courville (MIT Press, 2016), which goes into detail on overfitting and regularization techniques, including early stopping. Also, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Aurélien Géron (O'Reilly Media, 2019) provides practical guidance on implementing and using these concepts. Finally, reviewing publications on related topics at conferences like NeurIPS, ICML, and ICLR can offer a deeper, more research-oriented perspective. Implementing early stopping this way, with a bit of careful planning, will certainly be a benefit in any of your machine learning projects.
