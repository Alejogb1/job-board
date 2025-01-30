---
title: "Is the `torchrun` command required for distributed training and, if so, how is it installed?"
date: "2025-01-30"
id: "is-the-torchrun-command-required-for-distributed-training"
---
Distributed training in PyTorch, particularly when utilizing multiple machines, frequently involves complexities beyond single-machine setups. While the core PyTorch library handles much of the computational logic, coordinating processes across nodes requires an additional layer of orchestration. The `torchrun` command is that critical layer, acting as the launch utility for distributed applications in a variety of circumstances. However, its necessity depends specifically on the distributed backend chosen and the training environment.

The core problem `torchrun` addresses is the initialization and management of distributed processes across multiple machines. When working on a single machine with multiple GPUs, one can utilize the `torch.distributed.launch` utility provided by PyTorch, a more straightforward approach. However, this method becomes inadequate once the training task needs to span multiple machines due to the added challenges of inter-process communication and network discovery. Here's where `torchrun` becomes a pivotal component. It’s designed to abstract the intricate details of network configuration, ensuring seamless and coordinated communication between processes on different machines. These processes must synchronize their gradient updates and model parameters during training, a task that requires robust and reliable communication infrastructure.

When employing the widely used `torch.distributed` package, especially with backends such as Gloo or NCCL, `torchrun` provides the mechanism to launch the training script on every designated machine within the distributed environment. It manages the initial handshake, ensures all processes share a global rank, and facilitates consistent data exchange and gradient updates during the training phase. Without `torchrun` (or a similar multi-machine launch tool), manual coordination via bash scripts and environment variables becomes necessary – an approach that is both prone to errors and cumbersome. While not strictly *required* in the sense that a minimal distributed training might work on a single node with `torch.distributed.launch`, `torchrun` is unequivocally essential for scaling to multi-machine distributed training using `torch.distributed`.

Now, regarding its installation, `torchrun` is not a standalone package in PyPI. Instead, it is included as an executable tool within the PyTorch installation itself. Thus, installing PyTorch using either `pip` or `conda` automatically installs `torchrun`. Specific installation instructions vary depending on your operating system and preferred package manager. For example, a common `pip` installation command targeting CUDA would look like:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

In a conda environment, the equivalent command might be:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```
These installation approaches include `torchrun` implicitly; no further install step is necessary. To confirm installation, execute `torchrun --version` from the command line. The output should show the PyTorch version. If it reports that command is not found, then you need to examine the PyTorch setup, verify the path, or try a reinstall. It should be accessible globally if everything was configured correctly during installation.

Let's illustrate with some code examples. The first example will demonstrate a rudimentary distributed setup and showcase how `torchrun` is utilized. Suppose we have a simple training script named `train.py`. Within that script, we initialize the distributed backend, retrieve our process rank, and perform some basic operations:

```python
# train.py
import torch
import torch.distributed as dist
import os

def train():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"Process {rank} of {world_size} reporting in!")

    # Simulate some data processing
    local_data = torch.ones(10) * (rank + 1) # different values for demonstration

    # All-reduce operation to get sums from each rank
    output_tensor = torch.empty(10)
    dist.all_reduce(local_data, op=dist.ReduceOp.SUM, out=output_tensor)

    if rank == 0:
        print(f"Combined data from all ranks: {output_tensor}")

    dist.destroy_process_group()

if __name__ == "__main__":
    train()

```

This script demonstrates a basic distributed setup. The `dist.init_process_group(backend="nccl")` initializes the distributed environment, specifying the NCCL backend which is efficient for GPU training.  The rank identifies each process, and all processes are synchronized. This setup requires `torchrun`.

To run this on two machines using NCCL with, for example, 4 GPUs per machine, we would use the following command on each machine. Note that hostnames/IP addresses are examples:

```bash
torchrun --nnodes=2 --nproc_per_node=4 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=host1:29400  train.py
```

Here:
* `--nnodes=2`: Specifies that we're using 2 nodes/machines.
* `--nproc_per_node=4`: Specifies four processes/GPUs per node.
* `--rdzv_id=123`: A unique identifier for the distributed training run; this must be the same for all participating machines
* `--rdzv_backend=c10d`: Uses the c10d rendezvous mechanism
* `--rdzv_endpoint=host1:29400`: The network address for the rendezvous process, which can be one of the host’s IPs. It is important to have the port 29400 free on the host machine. `host1` must be accessible by `host2` as well, so it may be an internal IP address on the cluster.

Note that this command needs to be identical across all machines, with the exception of possible environment variable setups. This single command orchestrates the distributed launch, setting up the rank, world size, and communication pathways between all 8 processes.

A second example extends this to a more practical scenario involving a PyTorch model:

```python
# model_train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import random

def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


def train(rank, world_size):
  setup_ddp(rank, world_size)

  model = SimpleModel().to(rank)
  model = DDP(model, device_ids=[rank])
  optimizer = optim.SGD(model.parameters(), lr=0.01)
  loss_fn = nn.MSELoss()

  data_input = torch.randn(10, 10).to(rank)
  label_output = torch.randn(10, 1).to(rank)


  optimizer.zero_grad()
  output = model(data_input)
  loss = loss_fn(output, label_output)
  loss.backward()
  optimizer.step()

  print(f"Process {rank}: Loss {loss.item()}")
  dist.destroy_process_group()

if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    train(local_rank, world_size)
```

Here, we’re creating a basic linear model using DistributedDataParallel (DDP). The `setup_ddp` function sets up DDP using environment variables, which work fine when we use `torchrun`. The core training loop remains largely the same but leverages DDP to distribute the model across devices. To run this with four processes on a single node using `torchrun`, we use:

```bash
torchrun --nproc_per_node=4 model_train.py
```

A crucial point is that `torchrun` sets environment variables such as `LOCAL_RANK` and `WORLD_SIZE` automatically. This is different compared to the `torch.distributed.launch` way of using environment variables. Without `torchrun`, this setup would require explicit management of these variables, greatly complicating the deployment process.

Finally, let us consider a third example using multiple nodes with explicit node ranks for clarity (this is usually handled automatically but helps to show the process), and again focusing on the use of `torchrun`:

```python
# complex_train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import random
import socket
import sys


def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = 'host1' # can also be IP
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

class ComplexModel(nn.Module):
  def __init__(self):
      super().__init__()
      self.linear1 = nn.Linear(10, 20)
      self.relu = nn.ReLU()
      self.linear2 = nn.Linear(20, 1)

  def forward(self, x):
      x = self.linear1(x)
      x = self.relu(x)
      x = self.linear2(x)
      return x



def train(rank, world_size):
    setup_ddp(rank, world_size)
    model = ComplexModel().to(rank)
    model = DDP(model, device_ids=[rank])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    data_input = torch.randn(64, 10).to(rank)
    label_output = torch.randn(64, 1).to(rank)

    optimizer.zero_grad()
    output = model(data_input)
    loss = loss_fn(output, label_output)
    loss.backward()
    optimizer.step()

    print(f"Process on node rank {rank}: Loss {loss.item()}")

    dist.destroy_process_group()


if __name__ == "__main__":
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    train(local_rank, world_size)

```

This example introduces a somewhat more complex model with multiple layers. Again, DDP is used to distribute the model. The `MASTER_ADDR` is set manually to `host1`, which is necessary as the node setup information is now needed explicitly, since it is not within the same process group. Now, for a two node setup, the following would be needed. Assume four processes per node. On node 1, we would execute:

```bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=host1:29400  complex_train.py
```
On node 2:
```bash
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 --rdzv_id=123 --rdzv_backend=c10d --rdzv_endpoint=host1:29400  complex_train.py
```
Note that node ranks start from 0. The `rdzv_id` and `rdzv_endpoint` must be identical to ensure communication. It's also crucial that each host can communicate to `host1` on port 29400.

In conclusion, while PyTorch can perform distributed training on a single machine without `torchrun`, it is not the preferred approach when scaling to multiple machines due to the complexities of inter-process communication. `torchrun` seamlessly handles these complexities making distributed training with `torch.distributed` much easier. Finally, in practice, `torchrun` is installed automatically along with PyTorch. For further reading on distributed training methodologies, consult the official PyTorch documentation. In addition, investigate resources that cover distributed computing paradigms within a machine learning context. Resources from Nvidia regarding NCCL can also provide insight into backend specifics.
