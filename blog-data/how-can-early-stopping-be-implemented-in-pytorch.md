---
title: "How can early stopping be implemented in PyTorch?"
date: "2024-12-23"
id: "how-can-early-stopping-be-implemented-in-pytorch"
---

Alright, let's tackle early stopping in PyTorch. It’s a topic I’ve spent quite a bit of time with, especially back when I was training a rather complex sequence-to-sequence model for natural language processing. I recall that model would happily overfit if given the chance, making early stopping a critical component to its overall success. Essentially, early stopping is a regularization technique, not unlike dropout or weight decay, that aims to prevent overfitting by halting training when the model’s performance on a validation set starts to degrade. The underlying idea is simple: we monitor a chosen metric (like validation loss or accuracy) and stop training when that metric stops improving or, even worse, starts getting worse.

From my experience, a naïve implementation of early stopping can introduce as many problems as it solves, mainly if not handled carefully. Therefore, let’s get down to implementation details and a couple of examples to illustrate the point.

First, we need a way to track the best performance so far and keep a record of how many epochs have passed since then without any improvement. We will maintain a `patience` parameter; this represents how many epochs we will let pass without an improvement before stopping. We will also have a `min_delta` parameter to make sure small fluctuations don’t trigger our early stopping.

Here's the first snippet, which demonstrates a relatively straightforward implementation:

```python
import torch

class EarlyStopper:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_validation_loss = float('inf')
        self.early_stop = False

    def __call__(self, validation_loss):
        if validation_loss < self.best_validation_loss - self.min_delta:
            self.best_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

#Example use:
# early_stopper = EarlyStopper(patience=5, min_delta=0.001)
# for epoch in range(num_epochs):
#   ... train the model ...
#   validation_loss = ... calculate the validation loss ...
#   early_stopper(validation_loss)
#   if early_stopper.early_stop:
#      print("Early stopping triggered.")
#      break
```

In this initial example, we initialize an `EarlyStopper` object with parameters `patience` and `min_delta`, `best_validation_loss` set to infinity, and a counter initialized to zero. Every call to the object during the training loop checks if the `validation_loss` is lower than our best recorded `validation_loss`, with a buffer of `min_delta` for small differences. If so, we reset the counter. Otherwise, we increment it, and if it exceeds `patience`, we trigger `early_stop` which we can check in the training loop to stop the training.

Now, what about saving the best model? It's not just about stopping; we usually want the model weights at the point where performance was at its peak. Let’s expand our `EarlyStopper` class to handle that, introducing saving and loading functions. Here's the second snippet that addresses it:

```python
import torch
import os

class EarlyStopperWithCheckpoint:
    def __init__(self, patience=7, min_delta=0, path="checkpoint.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_validation_loss = float('inf')
        self.early_stop = False
        self.path = path

    def __call__(self, validation_loss, model):
        if validation_loss < self.best_validation_loss - self.min_delta:
            self.best_validation_loss = validation_loss
            self.counter = 0
            torch.save(model.state_dict(), self.path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))
        return model

    def check_checkpoint(self):
        return os.path.exists(self.path)

#Example use:
# early_stopper = EarlyStopperWithCheckpoint(patience=5, min_delta=0.001, path="best_model.pth")
# for epoch in range(num_epochs):
#   ... train the model ...
#   validation_loss = ... calculate the validation loss ...
#   early_stopper(validation_loss, model)
#   if early_stopper.early_stop:
#      print("Early stopping triggered, loading best model from checkpoint.")
#      if early_stopper.check_checkpoint():
#         model = early_stopper.load_checkpoint(model)
#      break
```

This enhanced version now saves the model's state dictionary using `torch.save` when a new best loss is reached, and offers a `load_checkpoint` method to load those saved parameters back into the model. Additionally, the function `check_checkpoint` helps to make sure the checkpoint exists prior to attempting to load it.

Finally, let's consider a scenario with a more complex metric, one where we track validation accuracy, instead of the loss. Furthermore, lets introduce an optional mode that can look for improvement on either a *min* or *max* metric, depending on the one we track. This can be implemented using the following code snippet:

```python
import torch
import os

class EarlyStopperWithMode:
    def __init__(self, patience=7, min_delta=0, mode="min", path="checkpoint.pth"):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
        self.path = path
        self.mode = mode

    def __call__(self, metric_value, model):
      if self.mode == "min":
        if metric_value < self.best_value - self.min_delta:
          self.best_value = metric_value
          self.counter = 0
          torch.save(model.state_dict(), self.path)
        else:
          self.counter += 1
          if self.counter >= self.patience:
            self.early_stop = True
      elif self.mode == "max":
        if metric_value > self.best_value + self.min_delta:
          self.best_value = metric_value
          self.counter = 0
          torch.save(model.state_dict(), self.path)
        else:
          self.counter += 1
          if self.counter >= self.patience:
              self.early_stop = True
    
    def load_checkpoint(self, model):
        model.load_state_dict(torch.load(self.path))
        return model

    def check_checkpoint(self):
        return os.path.exists(self.path)

# Example usage when maximizing validation accuracy:
# early_stopper = EarlyStopperWithMode(patience=5, min_delta=0.001, mode="max", path="best_acc_model.pth")
# for epoch in range(num_epochs):
#   ... train the model ...
#   validation_accuracy = ... calculate the validation accuracy ...
#   early_stopper(validation_accuracy, model)
#   if early_stopper.early_stop:
#     print("Early stopping triggered, loading best model from checkpoint.")
#     if early_stopper.check_checkpoint():
#         model = early_stopper.load_checkpoint(model)
#     break

```

In this enhanced version of our `EarlyStopper` class, the `mode` variable specifies if the monitored metric is of the minimum or maximum type and allows to choose accordingly how to update the best metric value. This allows a greater flexibility, to cover different use cases.

Now, for some recommendations, if you are serious about developing solid models, I would suggest to take a look at "Deep Learning" by Ian Goodfellow, Yoshua Bengio, and Aaron Courville for comprehensive coverage on regularization techniques, including early stopping. For more practical insights, I recommend "Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow" by Aurélien Géron, which is rich with specific examples and best practices.

From my experience, a critical detail is how frequently you evaluate performance and how often you update the early stopping procedure itself. Using a very high `patience` might lead you to overfit anyway, while very low values will potentially make you stop too early, without reaching a stable optimum. Similarly, a small `min_delta` can make early stopping very sensitive to minor fluctuation of the monitored metric, while a very large value for `min_delta` might make your procedure insensitive to the metric itself. You should also try not to base your early stopping decision on metrics that tend to be noisy such as the validation loss at very early stages of training. All of this should be fine-tuned depending on the task and the model itself. You need to ensure this procedure is implemented to address the specific requirements of your problem.
