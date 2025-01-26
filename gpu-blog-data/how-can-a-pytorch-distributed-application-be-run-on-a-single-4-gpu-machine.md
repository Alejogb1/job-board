---
title: "How can a PyTorch distributed application be run on a single 4-GPU machine?"
date: "2025-01-26"
id: "how-can-a-pytorch-distributed-application-be-run-on-a-single-4-gpu-machine"
---

Utilizing a single multi-GPU machine for distributed PyTorch training leverages the `torch.distributed` package, simulating a multi-node environment locally. The core concept is to treat each GPU on the machine as a separate process, allowing for parallel computation and data loading. This approach provides significant acceleration over single-GPU training when the model and dataset are large enough to saturate a single device, without the complexity of managing inter-machine communication. I have personally implemented this in several projects involving large language model fine-tuning, and I've found understanding the process group management and data distribution nuances crucial for success.

The `torch.distributed` package relies on initializing a communication backend and creating a process group. In a single-machine, multi-GPU scenario, typically the `nccl` backend is the most efficient, especially for Nvidia GPUs. The initialization involves setting a unique rank for each process, representing its identifier within the group, along with the total number of processes and a shared address and port for communication. Each process corresponds to one GPU. Subsequently, we need to ensure that each process operates on a non-overlapping portion of the training data to avoid redundancy, often through data partitioning or sharding techniques. This partitioning, combined with proper gradient synchronization between processes after each training batch, is essential for correct distributed training. Failure to properly partition data or synchronize gradients will lead to either incorrect training or undefined behavior.

Crucially, models need to be replicated across each GPU process. This is generally accomplished using `DistributedDataParallel`, which wraps the original model. `DistributedDataParallel` automatically handles the distribution of gradients and the model parameter updates across all devices, simplifying the distribution and synchronization procedures. This is vital. The alternative to using `DistributedDataParallel` would involve manual gradient synchronization and is far less robust. Finally, we launch multiple python processes, each using a dedicated GPU, and ensure they execute the identical training script but with different process rank configurations.

Here's the first code example illustrating the fundamental setup:

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_function(rank, world_size):
    setup(rank, world_size)

    # Simple Model
    model = nn.Linear(10, 2).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Dummy Data
    inputs = torch.randn(32, 10).to(rank)
    labels = torch.randint(0, 2, (32,)).to(rank)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(10): # Small training loop for illustration
        optimizer.zero_grad()
        outputs = ddp_model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f"Process {rank}: Training Finished")
    cleanup()


def main(world_size):
    mp.spawn(train_function,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPUs detected, exiting.")
    else:
         main(world_size)
```

This code defines a `setup` function for initializing the distributed process group, specifying 'localhost' as the master address and a fixed port. `cleanup` ensures that the process group is destroyed upon completion. The `train_function` encompasses the core training loop, taking the current process's rank and the total number of processes as parameters. The `DistributedDataParallel` wrapper around the model is critical; without it, the model wouldn't be replicated and gradients wouldn't synchronize. `mp.spawn` launches a distinct training process for each available GPU. This fundamental example highlights the critical steps needed for distributed setup and model handling, although a fully fledged solution would incorporate data loading and partitioning.

Here’s a second example demonstrating how to handle data partitioning with `DistributedSampler`. This utilizes a PyTorch dataset:

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class SimpleDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_function(rank, world_size):
    setup(rank, world_size)

    # Model
    model = nn.Linear(10, 2).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Dataset and Distributed Sampler
    dataset = SimpleDataset(size=1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(10):
      sampler.set_epoch(_) # essential for shuffling in distributed mode
      for inputs, labels in dataloader:
          inputs = inputs.to(rank)
          labels = labels.to(rank)
          optimizer.zero_grad()
          outputs = ddp_model(inputs)
          loss = loss_fn(outputs, labels)
          loss.backward()
          optimizer.step()

    print(f"Process {rank}: Training Finished")
    cleanup()

def main(world_size):
    mp.spawn(train_function,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPUs detected, exiting.")
    else:
        main(world_size)
```

The primary addition here is the inclusion of a `SimpleDataset` and a `DistributedSampler`. The `DistributedSampler` ensures that each process receives a non-overlapping subset of the data. The critical line, `sampler.set_epoch(_)` within the training loop, shuffles the data differently across epochs, which is crucial for proper convergence when data is distributed across processes. Failing to reset the sampler each epoch can result in the same mini-batches being used for each process across multiple epochs, impacting convergence.

The final example shows how to save and load models correctly in a distributed setting.

```python
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

class SimpleDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 10)
        self.labels = torch.randint(0, 2, (size,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_function(rank, world_size):
    setup(rank, world_size)

    # Model
    model = nn.Linear(10, 2).to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # Dataset and Distributed Sampler
    dataset = SimpleDataset(size=1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(10):
      sampler.set_epoch(_) # essential for shuffling in distributed mode
      for inputs, labels in dataloader:
          inputs = inputs.to(rank)
          labels = labels.to(rank)
          optimizer.zero_grad()
          outputs = ddp_model(inputs)
          loss = loss_fn(outputs, labels)
          loss.backward()
          optimizer.step()

    # Saving
    if rank == 0:
        torch.save(ddp_model.module.state_dict(), "model.pth")

    print(f"Process {rank}: Training Finished")
    cleanup()

def load_and_inference(rank, world_size):
    setup(rank, world_size)
    model = nn.Linear(10, 2).to(rank)
    # Model loading must be done after process group is initialized
    model_state = torch.load('model.pth')
    model.load_state_dict(model_state)
    inputs = torch.randn(1, 10).to(rank)
    with torch.no_grad():
      output = model(inputs)
      print(f"Process {rank}: Output {output}")
    cleanup()


def main(world_size):
    mp.spawn(train_function,
             args=(world_size,),
             nprocs=world_size,
             join=True)
    # Loading must be done after training is complete
    mp.spawn(load_and_inference,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("No GPUs detected, exiting.")
    else:
        main(world_size)
```

Here, I only save the model's `state_dict` from the process with rank 0 (the main process), because all processes have identical models, and it is redundant to save them all.  I save the `module` attribute of the DDP model; this attribute stores the original unwrapped model, and it's state dictionary can be used to load it using a standard  model object (not a DDP wrapped object). When loading the model for inference, I load it after initializing process groups, and perform inference. This demonstrates the approach to saving and loading in a distributed environment, where only one model needs to be saved.

For resources on the topic, I'd recommend examining PyTorch's official documentation concerning distributed training, specifically looking at the sections dealing with `torch.distributed`, `DistributedDataParallel`, and `DistributedSampler`. The documentation of the PyTorch framework itself, particularly its tutorial section, also offer valuable guidance. Books like "Deep Learning with PyTorch" also include discussions on distributed training, which provides more of a theoretical overview. Further useful information can be found in the documentation for the `torch.multiprocessing` module to understand process handling.
