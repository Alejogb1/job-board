---
title: "How to compute and print training accuracy in the PyTorch QuickStart tutorial?"
date: "2025-01-30"
id: "how-to-compute-and-print-training-accuracy-in"
---
The PyTorch Quickstart tutorial, while comprehensive for foundational learning, omits direct calculation and reporting of training accuracy within its primary training loop. This omission, while focusing on core mechanics, necessitates an understanding of how accuracy is derived from model predictions and ground truth labels during the training process. I’ve frequently encountered this gap during initial project setup, particularly when debugging and performance tracking requires this information. The issue primarily revolves around transforming raw model outputs into predicted classes and comparing those to the actual labels.

Essentially, accuracy during training is calculated by determining how often the model's highest probability prediction for a given input matches the true class label for that input across a batch of training data. This process involves a few critical steps within the training loop: forward propagation to get model outputs, conversion of these outputs into predicted class indices (typically using an argmax operation), and finally, a comparison with the known labels to accumulate a running average of accuracy. Let’s explore these steps concretely, incorporating example code to clarify each point.

Firstly, consider the basic training loop as outlined in the PyTorch quickstart guide. It typically involves a forward pass, calculation of loss using a criterion function, backpropagation, and finally, parameter updates through an optimizer. To introduce accuracy calculations, we must interject a few lines into the forward propagation step. This can be illustrated through a simple example that assumes a classification task with 10 classes:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assume a simple model and data loader (defined elsewhere for brevity)
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(784, 10) # Example input size for MNIST-like data

    def forward(self, x):
        return self.fc(x)


model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
# Assume train_dataloader is pre-populated with training data.
# train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

def train_step(model, criterion, optimizer, train_dataloader):
    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    train_loss, correct = 0, 0
    for batch, (X, y) in enumerate(train_dataloader):
        # Forward Propagation
        pred = model(X)
        loss = criterion(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    train_loss /= num_batches
    correct /= size
    print(f"Training Accuracy: {(100*correct):>0.1f}%, Loss: {train_loss:>8f}")

# Example usage within a training loop (assuming epoch number)
num_epochs = 10
for epoch in range(num_epochs):
    train_step(model, criterion, optimizer, train_dataloader)
```

In the revised code above, the `train_step` function now calculates accuracy. Key changes include the addition of the line `correct += (pred.argmax(1) == y).type(torch.float).sum().item()`. This performs the critical comparison. `pred` holds the output of the model, which represents raw logit scores for each class. `pred.argmax(1)` takes the index of the maximum logit score along the class dimension (axis 1), providing the predicted class for each sample within the batch. This is then directly compared with the target labels `y`. The result of this comparison is a tensor of booleans (True if the prediction matches the target, False otherwise). By converting this to a float tensor using `type(torch.float)` and summing the resulting tensor, we get the count of correct predictions for the current batch. This correct count is then accumulated in the `correct` variable. Finally, at the end of the epoch, both the accumulated loss and the accuracy are calculated by dividing `train_loss` and `correct` by number of batches or dataset size and then printed to standard output.

A second illustrative example focuses on scenarios where models are evaluated after every training batch, providing more granular insights into training dynamics. This requires slight modification of how accuracy is calculated and stored for output:

```python
def train_step_batch_accuracy(model, criterion, optimizer, train_dataloader):
  size = len(train_dataloader.dataset)
  num_batches = len(train_dataloader)
  train_loss = 0
  correct = 0
  for batch, (X, y) in enumerate(train_dataloader):

      # Forward Propagation
      pred = model(X)
      loss = criterion(pred,y)

      # Backpropagation
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      batch_correct = (pred.argmax(1) == y).type(torch.float).sum().item()
      correct += batch_correct
      print(f"Batch: {batch+1}/{num_batches}, Loss: {loss.item():>8f}, Batch Accuracy: {(100*(batch_correct / X.size(0))):>0.1f}%")


  train_loss /= num_batches
  correct /= size
  print(f"Training Accuracy: {(100*correct):>0.1f}%, Loss: {train_loss:>8f}")


# Example usage within a training loop
num_epochs = 10
for epoch in range(num_epochs):
    train_step_batch_accuracy(model, criterion, optimizer, train_dataloader)
```

Here, the primary modification is calculating and printing batch-level accuracy during the inner loop itself, by dividing `batch_correct` by `X.size(0)` - the size of batch being processed. This allows us to track the performance with every batch, as opposed to only at the end of the epoch. This is especially useful in diagnosing training instabilities or if training dataset is relatively small.

Finally, let's consider a scenario where the model's output is not simply class scores, but instead, a probability distribution over classes. This requires an adjustment in how predictions are extracted from the output, but the core concept remains the same. Suppose we introduce a softmax function as the last layer of our network to produce a probability distribution. The code would then be adjusted as follows:

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class SimpleModelWithSoftmax(nn.Module):
    def __init__(self):
        super(SimpleModelWithSoftmax, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        x = self.fc(x)
        return F.softmax(x, dim=1) #Softmax to output a probability distribution

model = SimpleModelWithSoftmax()
optimizer = optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


def train_step_softmax(model, criterion, optimizer, train_dataloader):
    size = len(train_dataloader.dataset)
    num_batches = len(train_dataloader)
    train_loss, correct = 0, 0

    for batch, (X, y) in enumerate(train_dataloader):
      pred = model(X)
      loss = criterion(torch.log(pred), y)  #Need log of the softmax output with CrossEntropyLoss

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      train_loss += loss.item()
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()


    train_loss /= num_batches
    correct /= size
    print(f"Training Accuracy: {(100*correct):>0.1f}%, Loss: {train_loss:>8f}")


num_epochs = 10
for epoch in range(num_epochs):
    train_step_softmax(model, criterion, optimizer, train_dataloader)
```

Here, we wrap the raw output of the linear layer with a softmax function in the `forward()` method. Because the cross entropy loss accepts logits rather than probabilities, the loss function is adjusted in the train_step to accept `torch.log(pred)` rather than simply `pred`. Crucially however, the accuracy calculation remains unchanged: we still take `argmax` on the output to determine predicted class label. It is important to note the stability issues of softmax/log-softmax calculation with PyTorch's `nn.CrossEntropyLoss`. For production, `nn.CrossEntropyLoss` can directly accept the unnormalized logits (i.e. no softmax), therefore the softmax layer can be eliminated to improve training and calculation stability.

For further understanding, I would recommend studying the following resources: The official PyTorch documentation, especially the sections covering `torch.nn` modules and `torch.optim`. Textbooks and tutorials on deep learning frequently explain the mathematical underpinnings of classification and accuracy metrics. Finally, exploring open source machine learning projects often provides concrete examples of different approaches and nuances associated with accuracy calculation across varying model architectures. These resources are crucial to solidify your understanding of the concepts explored here. The official tutorial provides details on `CrossEntropyLoss`.
