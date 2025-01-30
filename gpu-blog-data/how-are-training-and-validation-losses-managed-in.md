---
title: "How are training and validation losses managed in PyTorch?"
date: "2025-01-30"
id: "how-are-training-and-validation-losses-managed-in"
---
The crucial difference between training and validation loss lies in their purpose: training loss guides model parameter updates via backpropagation, while validation loss provides an unbiased measure of a model’s generalization performance. I've spent the better part of a decade developing and deploying deep learning models, so I understand intimately how critical it is to manage these two loss signals effectively. They aren't just numbers; they tell a story about how our model is learning and whether it's likely to perform well on unseen data.

In PyTorch, the training process primarily involves iterating through training data in batches, calculating the loss, computing gradients, and updating model parameters. This back-and-forth between data, loss function, and optimizer is what reduces the training loss, hopefully converging towards a point where the model performs well on the seen data. However, constantly decreasing training loss isn’t always a positive sign. A model trained too well only on the training dataset risks overfitting, which means it might perform poorly on new examples. That's where the validation set and its corresponding validation loss become essential. We specifically withhold a subset of data, called the validation set, and then evaluate the model on it after each training epoch or after a certain number of training steps. The loss we observe on this unseen data is the validation loss and serves as a proxy for unseen data. The ideal behavior is that both training and validation loss decrease together, implying the model is learning meaningful patterns and not just memorizing the training set.

Now, let's look at how this unfolds in PyTorch code, starting with a basic setup for a classification problem.

**Example 1: Basic Training and Validation Loop**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Sample data and model (replace with your actual data and model)
X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000,))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(20, 2)
    def forward(self, x):
        return self.linear(x)

model = SimpleClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_step(model, dataloader, criterion, optimizer):
  model.train()
  total_loss = 0
  for inputs, labels in dataloader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
  return total_loss / len(dataloader)


def validation_step(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad(): # Disable gradient tracking for validation
      for inputs, labels in dataloader:
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          total_loss += loss.item()
    return total_loss / len(dataloader)


epochs = 10
for epoch in range(epochs):
    train_loss = train_step(model, train_loader, criterion, optimizer)
    val_loss = validation_step(model, val_loader, criterion)

    print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
```

This first example outlines the essential structure. Within each epoch, we iterate over the training set using the `train_step` function, and calculate the loss, and update the model parameters using the optimizer. Then we iterate through the validation set in `validation_step`. Critically, during validation, `model.eval()` puts the model in evaluation mode, disabling dropout and other training-specific behavior, and `torch.no_grad()` prevents the calculation and tracking of gradients, saving computation. Without `torch.no_grad`, validation loss would be slower, and would not represent the model's generalization ability. We typically track and record both losses across epochs.

**Example 2: Incorporating Early Stopping**

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import numpy as np


# Sample data and model (replace with your actual data and model)
X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000,))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(20, 2)
    def forward(self, x):
        return self.linear(x)

model = SimpleClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_step(model, dataloader, criterion, optimizer):
  model.train()
  total_loss = 0
  for inputs, labels in dataloader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
  return total_loss / len(dataloader)


def validation_step(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad(): # Disable gradient tracking for validation
      for inputs, labels in dataloader:
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          total_loss += loss.item()
    return total_loss / len(dataloader)



epochs = 100
patience = 10
best_val_loss = np.inf
epochs_no_improve = 0

for epoch in range(epochs):
    train_loss = train_step(model, train_loader, criterion, optimizer)
    val_loss = validation_step(model, val_loader, criterion)

    print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
      best_val_loss = val_loss
      epochs_no_improve = 0
      torch.save(model.state_dict(), 'best_model.pth') #Save if validation loss improves
    else:
      epochs_no_improve+=1
    
    if epochs_no_improve > patience:
      print('Early stopping triggered')
      break

model.load_state_dict(torch.load('best_model.pth')) #load the model weights from the epoch with the lowest validation loss.
```
Here, early stopping has been implemented to halt training if the validation loss plateaus or increases for a certain number of epochs ('patience'). I’ve found this prevents overfitting. The best model parameters according to validation loss are saved during training, which are then loaded after early stopping. It saves the optimal model state rather than the final model state and is quite effective in practice.

**Example 3: Using TensorBoard for Visualization**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter


# Sample data and model (replace with your actual data and model)
X = torch.randn(1000, 20)
y = torch.randint(0, 2, (1000,))
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_data = TensorDataset(X_train, y_train)
val_data = TensorDataset(X_val, y_val)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32)

class SimpleClassifier(nn.Module):
    def __init__(self):
        super(SimpleClassifier, self).__init__()
        self.linear = nn.Linear(20, 2)
    def forward(self, x):
        return self.linear(x)

model = SimpleClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_step(model, dataloader, criterion, optimizer, writer, epoch):
  model.train()
  total_loss = 0
  for inputs, labels in dataloader:
      optimizer.zero_grad()
      outputs = model(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
  avg_loss = total_loss / len(dataloader)
  writer.add_scalar('training_loss', avg_loss, epoch) #Log training loss
  return avg_loss


def validation_step(model, dataloader, criterion, writer, epoch):
    model.eval()
    total_loss = 0
    with torch.no_grad(): # Disable gradient tracking for validation
      for inputs, labels in dataloader:
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          total_loss += loss.item()
    avg_loss = total_loss / len(dataloader)
    writer.add_scalar('validation_loss', avg_loss, epoch) #Log validation loss
    return avg_loss


writer = SummaryWriter('runs/experiment1') # Log to this path
epochs = 10

for epoch in range(epochs):
    train_loss = train_step(model, train_loader, criterion, optimizer, writer, epoch)
    val_loss = validation_step(model, val_loader, criterion, writer, epoch)

    print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
writer.close()
```
In this third example, I've integrated TensorBoard to visualize the losses over epochs. The `SummaryWriter` object logs training and validation losses at each epoch, which can then be viewed in TensorBoard by running `tensorboard --logdir=runs`. Monitoring the loss trends in a visual format is invaluable during development for spotting overfitting, underfitting, or other unusual behaviours.

Effective management of training and validation losses, coupled with practices like early stopping and visualization, represents the core of deep learning model development, ensuring that models not only learn from data but also generalize to new data. Additional resources that would be valuable in a real-world setting would include: *Deep Learning with Python* by Chollet, *Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow* by Geron, and *Programming PyTorch for Deep Learning* by Carbone. These texts would provide foundational knowledge as well as specific applications of model training and evaluation, going far beyond the scope of a simple response.
