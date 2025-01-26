---
title: "How do batch size, epochs, and learning rate affect training with DistributedDataParallel?"
date: "2025-01-26"
id: "how-do-batch-size-epochs-and-learning-rate-affect-training-with-distributeddataparallel"
---

DistributedDataParallel (DDP), as I've observed through numerous projects scaling deep learning models, introduces unique considerations regarding batch size, epochs, and learning rate compared to single-GPU training. These parameters, fundamental to the training process, interact differently when data and computation are distributed across multiple devices. My experience focuses primarily on PyTorch and similar frameworks, so the examples and discussions will reflect that environment.

**Understanding the Core Impacts**

The key difference lies in the implicit aggregation of gradients when using DDP. In a single-GPU setting, the optimizer directly processes gradients calculated from a single batch of data. However, with DDP, each process (typically corresponding to a GPU) computes gradients on a portion of the total batch. These gradients are then synchronized and averaged across all processes *before* the optimizer updates the model's weights. This introduces a critical dependency between the effective batch size and the number of participating devices, profoundly affecting the optimal learning rate and epoch settings.

**Batch Size: Local vs. Global**

When we refer to batch size in a DDP context, it's essential to distinguish between the *local batch size* and the *global batch size*. The local batch size is the number of data samples each GPU processes before gradients are synchronized. The global batch size is the aggregate number of data samples processed during each gradient update, which is equivalent to the local batch size multiplied by the number of processes/GPUs. For example, if you are training on four GPUs with a local batch size of 32, your global batch size is 128.

A critical error I often see is when developers use the same local batch size with multiple GPUs as they used on a single GPU setup. This effectively increases the global batch size, which, if not adjusted for, can lead to suboptimal training, usually resulting in poor convergence, instability or both. It is generally accepted that increasing global batch size allows training to complete in less time, as more samples are processed in each update. However, with an unadjusted learning rate, this can reduce the accuracy of the final result as a greater change in weights occurs each update. To obtain comparable accuracy as a smaller batch size would, the learning rate needs to be tuned to compensate, typically requiring a slightly larger learning rate to maintain training stability and avoid local minima. The relationship between batch size and learning rate is not a trivial one, and often requires a hyperparameter search.

**Epochs: Data Coverage with Distributed Training**

An epoch, representing a full pass through the training data, behaves consistently regardless of whether you are using a single GPU or DDP. The number of epochs still governs how many times your model views the entire dataset. However, I’ve noted that with the larger effective batch size often used in DDP, the model may converge more rapidly, meaning fewer epochs may be needed to reach an optimal point. This is because with each parameter update more global data is considered. Careful monitoring of your validation loss is needed when scaling the number of devices, to avoid overfitting and over training, which becomes more likely. Furthermore, the actual data passed to a training process may vary epoch to epoch. If using a distributed sampler, a distinct subset of the dataset may be allocated to each node in a distributed training configuration, or the data is partitioned differently. The shuffle behaviour across nodes is also not the same as single GPU training, as the dataset is not shuffled as a whole. Therefore, you should be familiar with the sampler you’re using and how to reproduce the shuffled batches of data for training if you wish to do so.

**Learning Rate: Compensation for Increased Batch Size**

The learning rate is arguably the most sensitive parameter when transitioning to DDP. It directly controls the magnitude of weight updates during backpropagation. When you increase the global batch size through distributed training, you're essentially using an approximation of the true gradient that is averaged across more data samples. This leads to a more stable, yet potentially less variable update, as the larger batch means less gradient variance between updates. As mentioned, simply carrying over a learning rate that works for a small batch size to an equivalent scenario with DDP will often yield suboptimal results due to increased global batch size.

The “linear scaling rule” is often mentioned, where you multiply the original learning rate by the number of devices (e.g., doubling the learning rate for two GPUs, quadrupling for four, etc.). While this is often a useful heuristic to begin searching for an appropriate learning rate, it should not be blindly applied without experimentation. I have experienced scenarios where applying the linear scaling rule required further fine-tuning.

**Code Examples and Commentary**

The following examples illustrate how to configure batch size, learning rate and distributed data parallelism in PyTorch:

**Example 1: Basic DDP Setup**

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def train(rank, world_size, local_batch_size, learning_rate, epochs):
    dist.init_process_group(backend='nccl', init_method='env://',
                            rank=rank, world_size=world_size)

    model = nn.Linear(10, 2).to(rank) # Simplified model for illustration
    model = DDP(model, device_ids=[rank])

    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        for _ in range(10): #Simplified loop, typically would iterate over a DataLoader
            input_data = torch.randn(local_batch_size, 10).to(rank)
            target_data = torch.randint(0, 2, (local_batch_size,)).to(rank)
            optimizer.zero_grad()
            output = model(input_data)
            loss = nn.functional.cross_entropy(output, target_data)
            loss.backward()
            optimizer.step()
        print(f"Rank: {rank}, Epoch: {epoch}, Loss: {loss.item()}")

    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_batch_size = 32
    learning_rate = 0.01 # Base learning rate for a single GPU setup. Should be adjusted for number of GPUs
    epochs = 5

    train(rank, world_size, local_batch_size, learning_rate, epochs)
```

This example demonstrates a basic DDP setup. Key points to observe are `dist.init_process_group`, `DDP(model, device_ids=[rank])`, and the usage of `rank` to assign a device to each process. The data is also transferred to the rank-associated device for loss calculations, a process that must be mirrored across all devices. The training loop shows a very simplified setup for illustration, but the loss and other gradient calculations are performed on the individual device assigned to each process. The `optimizer` synchronizes gradients after backpropagation.

**Example 2: Modified Learning Rate**

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def train(rank, world_size, local_batch_size, learning_rate, epochs):
    dist.init_process_group(backend='nccl', init_method='env://',
                            rank=rank, world_size=world_size)

    model = nn.Linear(10, 2).to(rank)
    model = DDP(model, device_ids=[rank])

    #Adjusted LR based on the number of devices
    adjusted_lr = learning_rate * world_size
    optimizer = optim.SGD(model.parameters(), lr=adjusted_lr)

    for epoch in range(epochs):
        for _ in range(10):
            input_data = torch.randn(local_batch_size, 10).to(rank)
            target_data = torch.randint(0, 2, (local_batch_size,)).to(rank)
            optimizer.zero_grad()
            output = model(input_data)
            loss = nn.functional.cross_entropy(output, target_data)
            loss.backward()
            optimizer.step()
        print(f"Rank: {rank}, Epoch: {epoch}, Loss: {loss.item()}")

    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    local_batch_size = 32
    learning_rate = 0.01  # Base learning rate for a single GPU setup
    epochs = 5

    train(rank, world_size, local_batch_size, learning_rate, epochs)
```

In this modification, I've introduced the `adjusted_lr = learning_rate * world_size` line. While not always optimal, this demonstrates the linear scaling rule approach to learning rate adjustment as a starting point. This ensures that with more devices, the learning rate scales proportionally.

**Example 3: Adjusted Batch Size**

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
import os

def train(rank, world_size, local_batch_size, learning_rate, epochs):
    dist.init_process_group(backend='nccl', init_method='env://',
                            rank=rank, world_size=world_size)

    model = nn.Linear(10, 2).to(rank)
    model = DDP(model, device_ids=[rank])

    adjusted_lr = learning_rate * world_size
    optimizer = optim.SGD(model.parameters(), lr=adjusted_lr)

    for epoch in range(epochs):
        for _ in range(10):
            input_data = torch.randn(local_batch_size, 10).to(rank)
            target_data = torch.randint(0, 2, (local_batch_size,)).to(rank)
            optimizer.zero_grad()
            output = model(input_data)
            loss = nn.functional.cross_entropy(output, target_data)
            loss.backward()
            optimizer.step()
        print(f"Rank: {rank}, Epoch: {epoch}, Loss: {loss.item()}")

    dist.destroy_process_group()

if __name__ == '__main__':
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    #Adjust the local batch size to ensure a constant global batch size
    local_batch_size = 32 #If you were to train on one device
    global_batch_size = 128
    local_batch_size = int(global_batch_size / world_size)

    learning_rate = 0.01  # Base learning rate for a single GPU setup
    epochs = 5

    train(rank, world_size, local_batch_size, learning_rate, epochs)

```

In this third modification, I’ve adjusted the `local_batch_size` to maintain a `global_batch_size` of 128. This ensures that the total number of samples used in each global update stays consistent when scaling the number of devices involved in the training process.

**Resource Recommendations**

For further understanding of these concepts and best practices, I recommend the following. Refer to documentation from PyTorch itself, particularly for DDP and distributed sampling, which provides precise details. The original papers on large-batch training should also be consulted to grasp the theoretical underpinnings of learning rate scaling.  Lastly, explore tutorials covering practical distributed training workflows from machine learning blogs and forums. These sources will offer varying perspectives on adapting these parameters to diverse models and datasets.
