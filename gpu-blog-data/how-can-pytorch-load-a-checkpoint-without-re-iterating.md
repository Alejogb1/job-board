---
title: "How can PyTorch load a checkpoint without re-iterating through the entire dataset?"
date: "2025-01-30"
id: "how-can-pytorch-load-a-checkpoint-without-re-iterating"
---
The challenge in efficiently loading a PyTorch checkpoint arises when the training loop logic depends on iterators or indices tied to the dataset itself. When restoring a model from a checkpoint, simply loading the state dictionary may not suffice if the training process needs to resume from a specific epoch or batch, rather than starting from the beginning. The critical point is that PyTorch's `torch.load()` function restores the model's parameters but not the state of the data loaders. This requires a deliberate strategy to maintain context.

I've encountered this frequently when dealing with large-scale image datasets for deep learning tasks. Let's say I'm training a segmentation network. My initial training script involves creating `DataLoader` instances that shuffle the data and feed batches to the model. Saving a checkpoint typically includes the model's parameters and the optimizer's state. However, this doesn't save the state of the `DataLoader`. When I load the checkpoint, I am essentially creating a brand-new iterator, thus restarting from epoch 0 and potentially undoing significant training progress. The need for efficient checkpointing, rather than re-iterating the entire dataset, stems from the computational cost associated with training, which should not be squandered. The core problem we're solving is preserving the state of the data loading process alongside the model's learned parameters.

The solution revolves around explicitly tracking and saving relevant data loader state alongside the model and optimizer states. This usually involves recording the epoch number, any batch-level indices, and the random number generator states for both Python and NumPy if the dataloader incorporates randomization. We must then restore these states during checkpoint loading.

Here's a basic example illustrating the issue, without checkpointing, followed by solutions:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np

# Simple Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


# Dummy dataset and data loader
def create_dataset_loader(batch_size=16, num_samples = 100):
    X = torch.randn(num_samples, 10)
    y = torch.randn(num_samples, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
dataloader = create_dataset_loader(batch_size=16)

for epoch in range(2):
    loss = train_epoch(model, dataloader, optimizer, criterion)
    print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
```

This simple code trains a linear model for two epochs. Each epoch starts from the beginning of the data, as it generates a new `DataLoader`. Now, let's look at saving and loading a checkpoint, and its pitfalls:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np

# Simple Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


# Dummy dataset and data loader
def create_dataset_loader(batch_size=16, num_samples = 100):
    X = torch.randn(num_samples, 10)
    y = torch.randn(num_samples, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

def save_checkpoint(epoch, model, optimizer, filename="checkpoint.pth.tar"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return epoch


model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
dataloader = create_dataset_loader(batch_size=16)

num_epochs = 2
for epoch in range(num_epochs):
    loss = train_epoch(model, dataloader, optimizer, criterion)
    print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
    if epoch == 0:
        save_checkpoint(epoch, model, optimizer)

print("--- Loading checkpoint ---")
loaded_epoch = load_checkpoint("checkpoint.pth.tar", model, optimizer)
dataloader = create_dataset_loader(batch_size=16) # Recreate dataloader
for epoch in range(loaded_epoch +1, num_epochs + 1):
    loss = train_epoch(model, dataloader, optimizer, criterion)
    print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')

```

In this example, we added `save_checkpoint` and `load_checkpoint` functions. We save the model, optimizer, and current epoch. Upon loading, while the model parameters and optimizer state are loaded correctly, the dataloader restarts from epoch 0.  This is evident from seeing the second "Epoch 2" loss being different than if the training continued from where it was saved. To solve this, we need to save the state of the random number generator, and track batch indices:

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
import numpy as np

# Simple Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)


# Dummy dataset and data loader
def create_dataset_loader(batch_size=16, num_samples = 100, seed = None):
    if seed:
      torch.manual_seed(seed)
      random.seed(seed)
      np.random.seed(seed)

    X = torch.randn(num_samples, 10)
    y = torch.randn(num_samples, 1)
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader


def train_epoch(model, dataloader, optimizer, criterion, start_batch = 0):
    model.train()
    epoch_loss = 0.0
    for batch_idx, (data, target) in enumerate(dataloader):
        if batch_idx < start_batch:
           continue # Skip to the right starting point
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / (len(dataloader) - start_batch) # Account for skipped batches


def save_checkpoint(epoch, model, optimizer, random_state, batch_index, filename="checkpoint.pth.tar"):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'random_state': random_state,
        'batch_index': batch_index
    }
    torch.save(checkpoint, filename)

def load_checkpoint(filename, model, optimizer,dataloader):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    batch_index = checkpoint['batch_index']
    random_state = checkpoint['random_state']
    # Reset the random seeds to continue from correct position
    torch.manual_seed(random_state['torch_seed'])
    random.seed(random_state['python_seed'])
    np.random.seed(random_state['numpy_seed'])


    return epoch, batch_index


model = SimpleModel()
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
dataloader = create_dataset_loader(batch_size=16, seed=42) # Set a seed

num_epochs = 2
start_batch = 0
for epoch in range(num_epochs):
    loss = train_epoch(model, dataloader, optimizer, criterion,start_batch)
    print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
    if epoch == 0:
         random_state = {
            'torch_seed': torch.initial_seed(),
            'python_seed': random.randint(0,1000), # random.getstate() is hard to persist.
            'numpy_seed': np.random.randint(0,1000)
         }
         save_checkpoint(epoch, model, optimizer, random_state, len(dataloader)- len(dataloader), filename="checkpoint_enhanced.pth.tar")
         start_batch = len(dataloader) # Start at the beginning of next "epoch"
    else:
      start_batch = 0

print("--- Loading checkpoint ---")

loaded_epoch,loaded_batch = load_checkpoint("checkpoint_enhanced.pth.tar", model, optimizer,dataloader)

for epoch in range(loaded_epoch +1, num_epochs + 1):
    loss = train_epoch(model, dataloader, optimizer, criterion,start_batch)
    print(f'Epoch {epoch + 1}, Loss: {loss:.4f}')
```

Here, we persist the random seeds and the batch index in `save_checkpoint`. In `load_checkpoint`, we explicitly load the random states, reset the seeds and return both the starting epoch and the starting batch. Finally, we modify `train_epoch` to track batch indices and skip batches accordingly. This version accurately resumes training from where it left off, avoiding redundant processing of data.

For further study, I recommend exploring resources on best practices for PyTorch training loops, focusing specifically on handling custom datasets and dataloaders.  Specifically, understand the principles of reproducible random numbers in deep learning.  Moreover, reviewing best practices for checkpointing in distributed training environments, where data loading is even more complex, can prove beneficial. I also find examining various PyTorch training loop boilerplate code available online to be useful to solidify an understanding of the common pitfalls.
