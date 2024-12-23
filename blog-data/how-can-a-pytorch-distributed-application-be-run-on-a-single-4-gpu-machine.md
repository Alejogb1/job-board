---
title: "How can a PyTorch distributed application be run on a single 4-GPU machine?"
date: "2024-12-23"
id: "how-can-a-pytorch-distributed-application-be-run-on-a-single-4-gpu-machine"
---

Let's unpack this. Back in the days when I was optimizing a particularly large-scale image classification model, I ran headfirst into this very issue. Training on a single machine with multiple gpus, especially with a framework like pytorch, can seem initially straightforward, but there are nuanced aspects that can dramatically affect performance and correctness. The crux of it lies in how pytorch's distributed data parallel (ddp) or other parallelization strategies interact with single-machine multi-gpu setups.

The fundamental challenge is configuring the distributed environment correctly when everything is physically within one box. We don't have the complexities of network latency and inter-node communication, but we still need to emulate the distributed infrastructure that ddp expects. Essentially, ddp is built on the idea of having multiple processes, each operating on a subset of data and then communicating gradients across processes. Even on a single machine, this process separation is still necessary, and that’s where it begins to diverge from more straightforward, single-gpu training scripts.

First, understand that pytorch employs a multi-processing approach for distributed training, even when all processes are on one machine. We leverage the `torch.distributed` package to handle process group management and gradient synchronization. In practice, we use functions like `torch.distributed.init_process_group`, `torch.distributed.barrier`, and `torch.distributed.all_reduce` (implicitly within `ddp`). These aren't merely abstract concepts; they directly map to how data is handled and computations are synchronized. In our case, the local machine effectively becomes multiple "nodes."

To initialize the environment properly, we need to define a few crucial parameters. The most important are `rank` (the identifier of the current process), `world_size` (the total number of processes), and the communication backend (typically `nccl` for gpu operations).

Here’s a minimal snippet of how to achieve this initialization, for example using the `spawn` method from `torch.multiprocessing`:

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

def setup(rank, world_size):
    dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)

def cleanup():
    dist.destroy_process_group()


def train_process(rank, world_size):
  setup(rank, world_size)

  model = nn.Linear(10, 2).cuda(rank)
  ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

  optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

  data = torch.randn(100,10).cuda(rank)
  target = torch.randn(100,2).cuda(rank)

  for epoch in range(10):
    optimizer.zero_grad()
    output = ddp_model(data)
    loss = torch.nn.functional.mse_loss(output, target)
    loss.backward()
    optimizer.step()

    if rank == 0:
      print(f"Epoch: {epoch} Loss: {loss.item()}")
  cleanup()

if __name__ == "__main__":
  world_size = 4 # number of gpus in your machine
  mp.spawn(train_process,
            args=(world_size,),
            nprocs=world_size,
            join=True)
```

This snippet demonstrates how we can utilize `torch.multiprocessing.spawn` to create four processes, each one corresponding to one gpu. The crucial part is `device_ids=[rank]`, specifying that each process operates on its unique gpu. The `init_process_group` is configured with `backend='nccl'` for optimal gpu communication, and the `env://` method relies on environment variables to determine the process mapping. Note also, data needs to be on the correct cuda device (`data.cuda(rank)`). This approach is flexible and integrates well with many platforms.

However, sometimes you might be dealing with an older environment, or want to avoid environment variable setup. In those cases, the `init_method="tcp://..."` might be more suitable. Here is an example of how to initialize distributed training with the `tcp://` init method on a single machine:

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import os

def setup_tcp(rank, world_size, port="12355"):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = port

  dist.init_process_group(backend='nccl', init_method='tcp://localhost:' + port, world_size=world_size, rank=rank)

def cleanup():
    dist.destroy_process_group()


def train_process_tcp(rank, world_size):
  setup_tcp(rank, world_size)
  model = nn.Linear(10, 2).cuda(rank)
  ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

  optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

  data = torch.randn(100,10).cuda(rank)
  target = torch.randn(100,2).cuda(rank)

  for epoch in range(10):
      optimizer.zero_grad()
      output = ddp_model(data)
      loss = torch.nn.functional.mse_loss(output, target)
      loss.backward()
      optimizer.step()

      if rank == 0:
        print(f"Epoch: {epoch} Loss: {loss.item()}")
  cleanup()


if __name__ == "__main__":
  world_size = 4
  mp.spawn(train_process_tcp,
            args=(world_size,),
            nprocs=world_size,
            join=True)
```

In this example, I've explicitly set the master address and port using environment variables. This setup is slightly more verbose, but can be useful in situations where the environment variables are not automatically set. The core functionality however, remains the same - launching separate processes, initializing the distributed group and wrapping the model in ddp.

It's also worth emphasizing that the data must be distributed to each process before training starts, either using a custom data loader or by utilizing `torch.utils.data.distributed.DistributedSampler`. The sampler will ensure that each process gets a unique chunk of the dataset. Let's see a working example with a custom dataset using that sampler:

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import os

class DummyDataset(Dataset):
    def __init__(self, size):
        self.data = torch.randn(size, 10)
        self.target = torch.randn(size, 2)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

def setup_tcp(rank, world_size, port="12355"):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = port

  dist.init_process_group(backend='nccl', init_method='tcp://localhost:' + port, world_size=world_size, rank=rank)

def cleanup():
    dist.destroy_process_group()

def train_process_sampler(rank, world_size):
    setup_tcp(rank, world_size)
    dataset = DummyDataset(size=1000)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=10, sampler=sampler)

    model = nn.Linear(10, 2).cuda(rank)
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    for epoch in range(10):
        for data, target in dataloader:
            data = data.cuda(rank)
            target = target.cuda(rank)

            optimizer.zero_grad()
            output = ddp_model(data)
            loss = torch.nn.functional.mse_loss(output, target)
            loss.backward()
            optimizer.step()

        if rank == 0:
          print(f"Epoch: {epoch} Loss: {loss.item()}")
    cleanup()

if __name__ == "__main__":
  world_size = 4
  mp.spawn(train_process_sampler,
           args=(world_size,),
           nprocs=world_size,
           join=True)
```
In this enhanced version, `DistributedSampler` plays a vital role. Each process only sees its assigned subset of the full dataset, ensuring each GPU operates on different data and that model convergence isn’t affected by all the GPUs running the exact same data in each batch.

It is also crucial to note the communication overhead between these processes, even on the same machine. Techniques like gradient accumulation (if your batch size isn't large enough for effective parallelism), mixed precision training, and efficient data loading pipelines can be employed to mitigate this effect.

For resources, I'd strongly recommend delving into the Pytorch documentation directly, particularly the sections on distributed training. Also, the book "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann is exceptionally thorough and walks you through best practices. While not directly focusing on single-machine multi-gpu scenarios, the "Effective Parallel Programming" by Michael McCool, Arch D. Robison, and James Reinders is an excellent source to gain an in-depth view of parallel programming concepts and strategies that are fundamental in these scenarios. Understanding these theoretical underpinnings will help you make informed decisions as you navigate more complex distributed workloads.
