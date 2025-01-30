---
title: "How can PyTorch code be run on multiple GPUs?"
date: "2025-01-30"
id: "how-can-pytorch-code-be-run-on-multiple"
---
Distributed training in PyTorch, while significantly accelerating model training for large datasets and complex architectures, introduces complexities not encountered in single-GPU scenarios. The core issue revolves around data parallelism and model parallelism, and how PyTorch manages data synchronization and gradient aggregation across multiple processing units. Over the years, I've implemented several distributed training workflows, and the specific approach chosen depends heavily on the hardware resources available and the nature of the model being trained. This response focuses primarily on data parallelism, which is the most common case.

The fundamental concept is splitting the training dataset into batches, and distributing these batches across multiple GPUs. Each GPU performs a forward pass, calculating the loss, and then a backward pass to compute gradients. The challenge lies in aggregating these gradients across all GPUs before applying the optimizer to update the model's parameters. PyTorch provides different modules for managing this distributed training, including the `torch.nn.DataParallel` (DP) module, the `torch.nn.parallel.DistributedDataParallel` (DDP) module, and using custom approaches with `torch.distributed`. Choosing the appropriate method is crucial for achieving optimal performance and avoiding common pitfalls.

`torch.nn.DataParallel` offers the simplest entry point to multi-GPU training but comes with significant performance limitations in most practical settings. DP replicates the model onto each GPU and distributes data batches. The model parameter updates happen on the main GPU and then they are sent back to all the other GPUs. The primary downside is the single-threaded nature of the gradient aggregation and the model updates on the main GPU (GPU 0). This creates a bottleneck, especially with large models or high GPU counts. I have found that the overhead of transferring parameter updates back and forth can negate the performance gains when using DP for anything more complex than toy examples. It’s best for smaller projects or quick experimentation but for any serious research or industrial model development, it will fall short in many cases.

`torch.nn.parallel.DistributedDataParallel` is preferred for most production-level distributed training scenarios. Unlike `DataParallel`, DDP replicates the model and the optimizer parameters on each GPU. Each GPU computes local gradients, and then these gradients are synchronized across all GPUs using `torch.distributed` backends (NCCL is commonly used for NVIDIA GPUs). Parameter updates are performed locally on each GPU, using the synchronized gradients and thus removing the bottleneck of DataParallel. This parallelization of computation provides significant performance benefits. Correct configuration of the distributed environment is crucial when using DDP, as the training script needs to be launched on all the worker nodes/GPUs which requires a robust communication protocol for gradient synchronization. Launching distributed training with `torch.distributed.launch` or equivalent tools and proper initialization of the distributed environment is the key to achieving stable and scalable distributed training.

Here are a few code snippets to illustrate different approaches to multi-GPU training with commentary on their usage and expected performance. The focus will be on how data is loaded and distributed and the differences between DP and DDP.

**Example 1: DataParallel**

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Dummy Model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# Device Selection
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Assigns the single gpu if available to device
model = SimpleModel().to(device)

# DataParallel
if torch.cuda.device_count() > 1:
    print("Using DataParallel")
    model = nn.DataParallel(model)
# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Dummy Data
input_data = torch.randn(64, 10).to(device)
target_data = torch.randn(64, 1).to(device)

# Training Loop
optimizer.zero_grad()
output = model(input_data)
loss = nn.MSELoss()(output, target_data)
loss.backward()
optimizer.step()
print("Loss",loss.item())
```

This initial example demonstrates the simplicity of `DataParallel`. If multiple GPUs are detected, the model is wrapped with `nn.DataParallel`, which automatically handles data distribution. The remainder of the code is very similar to a standard single-GPU setup. However, the inefficiencies discussed earlier become apparent with more complex tasks. It is imperative to have the data reside on the device where the model resides to avoid excessive data movement and the corresponding overhead. Here the model and data are both on device, and this applies to the other examples.

**Example 2: DistributedDataParallel with `torch.distributed.launch`**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import os
from torch.nn.parallel import DistributedDataParallel

# Model Definition as in Example 1
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

# Initialize Distributed Process Environment
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def train_ddp(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")
    model = SimpleModel().to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    input_data = torch.randn(64, 10).to(device)
    target_data = torch.randn(64, 1).to(device)
    
    optimizer.zero_grad()
    output = model(input_data)
    loss = nn.MSELoss()(output, target_data)
    loss.backward()
    optimizer.step()
    
    print(f"Rank: {rank}, Loss: {loss.item()}")
    cleanup()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size>1:
      torch.multiprocessing.spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)
    else:
      print("Please specify more than 1 GPUs for DDP training, using Example 1 instead.")
```

This example illustrates the correct way of utilizing `DistributedDataParallel`. This script is designed to be launched using `torch.distributed.launch`. `setup` is responsible for initializing the distributed communication and assigning the `rank` of each node. Each process gets its own `rank` and the model is instantiated on different devices. The `DistributedDataParallel` wrapper then takes care of communication during the backward pass. Unlike `DataParallel`, gradients are synchronized and optimizer updates are performed locally on each GPU resulting in much improved scaling to higher number of GPUs. The script utilizes torch.multiprocessing to launch a separate process for each GPU. Launching DDP using multiprocessing this way ensures that each process has its own Python interpreter for more optimized execution. The `cleanup()` method destroys the process group created during the setup phase.

**Example 3:  Simplified DDP with `torch.distributed` and no launch**

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import os

# Model Definition as in Example 1
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def initialize_distributed(backend='nccl'):
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return False, None, None
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    dist.init_process_group(backend=backend)
    return True, rank, world_size

def train_ddp_simple():
    is_distributed, rank, world_size = initialize_distributed()
    
    if not is_distributed:
      print ("Not in distributed environment using single gpu training. Please run with torchrun (for DDP) or equivalent launcher")
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      model = SimpleModel().to(device)
      optimizer = optim.SGD(model.parameters(), lr=0.01)

      input_data = torch.randn(64, 10).to(device)
      target_data = torch.randn(64, 1).to(device)
    
      optimizer.zero_grad()
      output = model(input_data)
      loss = nn.MSELoss()(output, target_data)
      loss.backward()
      optimizer.step()
      print("Loss",loss.item())
      return

    device = torch.device(f"cuda:{rank}")
    model = SimpleModel().to(device)
    model = DistributedDataParallel(model, device_ids=[rank])
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    
    input_data = torch.randn(64, 10).to(device)
    target_data = torch.randn(64, 1).to(device)
    
    optimizer.zero_grad()
    output = model(input_data)
    loss = nn.MSELoss()(output, target_data)
    loss.backward()
    optimizer.step()
    print(f"Rank: {rank}, Loss: {loss.item()}")
    dist.destroy_process_group()


if __name__ == "__main__":
    train_ddp_simple()
```

Example 3 presents a simplified DDP approach where the script does not use `torch.multiprocessing.spawn` or `torch.distributed.launch`. Instead, it relies on environment variables (`RANK`, `WORLD_SIZE`) set by a compatible launcher (`torchrun`) to initialize the distributed environment and train the model in a distributed fashion. If the environment variables are not detected, the code will default to single GPU training. This example uses `torch.distributed` functions directly, checking for distributed context. This approach simplifies launching DDP, and it works well in most production-level systems. This script is ideal for integrating within an infrastructure that provides launchers to execute scripts in a distributed manner.

For further exploration into distributed training, I recommend studying resources detailing the `torch.utils.data.distributed.DistributedSampler` and its use in data loading within a distributed environment. It is also worthwhile to investigate the practical usage of different `torch.distributed` backends, as well as their implications on different hardware platforms. Additionally, researching gradient accumulation strategies will be useful for handling memory constraints with large models or large batch sizes. Understanding best practices for distributed training initialization and synchronization will be very beneficial when dealing with production-level systems. Furthermore, consider exploring techniques like mixed-precision training in the context of distributed setups, and examine the nuances of parameter aggregation, which are critical for optimizing the training process of complex deep learning models.
