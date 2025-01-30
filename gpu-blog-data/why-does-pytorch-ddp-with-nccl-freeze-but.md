---
title: "Why does PyTorch DDP with NCCL freeze, but Gloo works?"
date: "2025-01-30"
id: "why-does-pytorch-ddp-with-nccl-freeze-but"
---
Deep within the intricacies of distributed training, a perplexing scenario often emerges: PyTorch's Distributed Data Parallel (DDP) implementation using the NVIDIA Collective Communications Library (NCCL) seemingly halts progress, while the same application executes flawlessly using the Gloo backend. This divergence typically isn’t a matter of inherent superiority of one backend over the other; rather, it originates from subtle differences in their operational characteristics, and how those interact with the underlying hardware and network configuration. I’ve personally navigated this issue on numerous occasions, troubleshooting large-scale language models, and the root cause usually boils down to a few consistent culprits.

Firstly, NCCL, designed explicitly for high-performance GPU-to-GPU communication, is exceptionally sensitive to network configuration. It assumes a closely coupled environment, typically within a single machine or across machines with high-bandwidth, low-latency interconnects, such as NVLink or InfiniBand. Unlike Gloo, which is designed for flexibility and can work effectively across Ethernet, NCCL expects a relatively seamless path between participating GPUs. If this path is compromised by, for instance, incorrect IP address assignments, network interface misconfigurations, or firewall interference, NCCL communications can stall indefinitely, producing what appears to be a complete freeze. Gloo, on the other hand, tends to be more forgiving, employing a less restrictive communication paradigm, often utilizing TCP, which allows for easier traversal of standard network infrastructure. This tolerance makes Gloo a more robust, albeit slower, alternative when dealing with less-than-ideal network setups.

Another key distinction lies in the handling of the underlying synchronization and communication primitives. NCCL relies on optimized, asynchronous operations, meaning the application initiates a communication operation but does not necessarily wait for its completion before proceeding. If a subsequent computation step or synchronization point depends on the outcome of an NCCL operation that has yet to fully materialize, the application will inevitably stall. This can happen when a particular process on a machine hasn't finished sending data and the program advances to the point where all processes need to synchronize using the result. Gloo generally defaults to synchronous operations, offering a more predictable flow, at the cost of performance. Synchronous operations simplify debugging because errors reveal themselves more directly. Asynchronous operations in NCCL, however, require diligent error handling which is often overlooked.

Furthermore, hardware discrepancies can also contribute. NCCL performs optimally within a homogeneous environment, where all nodes possess identical GPUs. When different GPU models or configurations are mixed within the distributed setup, subtle incompatibilities in driver versions, CUDA toolkit versions, or NCCL versions can create communication hiccups. These issues rarely occur with Gloo due to the abstraction it introduces over the hardware and its wider compatibility with the CPU and less specialized networks. Gloo's flexibility extends to handling diverse hardware situations with better overall stability.

To illustrate these points, let's consider a basic PyTorch DDP example. Initially, it's crucial to verify the distributed setup. If NCCL freezes, this check will quickly uncover basic network issues:

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size, backend):
    import os
    os.environ['MASTER_ADDR'] = 'localhost'  # Specify address for master node
    os.environ['MASTER_PORT'] = '12355'     # Specify port for master node
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def ddp_example(rank, world_size, backend):
    setup(rank, world_size, backend)
    tensor = torch.ones(1)
    if rank == 0:
      print(f"Rank {rank} initial tensor: {tensor}")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    if rank == 0:
      print(f"Rank {rank} final tensor: {tensor}")
    cleanup()

if __name__ == "__main__":
    world_size = 4
    backend_choice = "nccl"  # or "gloo"
    mp.spawn(ddp_example,
             args=(world_size, backend_choice),
             nprocs=world_size,
             join=True)
```

This code defines a simple all-reduce operation. We set up a process group using the chosen backend (NCCL or Gloo), initiate a tensor, and perform the all-reduce. If we execute the above code with `backend_choice = "nccl"` and it freezes, one should first verify if the following is true for all the process group members:

1.  All GPUs in use should have the correct drivers.
2.  NCCL version is compatible with the installed CUDA toolkit and PyTorch version.
3.  The network is setup such that all ranks can communicate on the provided master address/port.

This verification will likely reveal issues.

Next, a common oversight is the proper handling of asynchronous collectives. This example, while simplified, showcases the need for proper synchronization. Here, each process is summing its tensor. If asynchronous communications are used, we cannot guarantee that each process receives updated tensor.

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def setup(rank, world_size, backend):
    import os
    os.environ['MASTER_ADDR'] = 'localhost'  # Specify address for master node
    os.environ['MASTER_PORT'] = '12355'     # Specify port for master node
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

def ddp_async_example(rank, world_size, backend):
    setup(rank, world_size, backend)
    tensor = torch.ones(1)
    if rank == 0:
      print(f"Rank {rank} initial tensor: {tensor}")
    work = dist.all_reduce(tensor, op=dist.ReduceOp.SUM, async_op=True)
    work.wait()  # Ensure operations are complete
    if rank == 0:
      print(f"Rank {rank} final tensor: {tensor}")
    cleanup()

if __name__ == "__main__":
    world_size = 4
    backend_choice = "nccl"  # or "gloo"
    mp.spawn(ddp_async_example,
             args=(world_size, backend_choice),
             nprocs=world_size,
             join=True)
```

The key here is `async_op=True` and the `work.wait()`. This small change guarantees that the all-reduce is completed before moving on. It's imperative when using NCCL to explicitly handle these operations and wait for completion if required. Without `work.wait()`, this code could easily result in a hang because the print statement relies on the output of the all_reduce function.

Finally, complex model training often involves more intricacies, particularly when working with mixed-precision training or custom communication patterns. Consider a scenario where model parameters are distributed across GPUs. The gradient aggregation, a core part of DDP, could become problematic if asynchronous gradients are used without careful synchronization:

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

def setup(rank, world_size, backend):
    import os
    os.environ['MASTER_ADDR'] = 'localhost'  # Specify address for master node
    os.environ['MASTER_PORT'] = '12355'     # Specify port for master node
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

def ddp_model_example(rank, world_size, backend):
    setup(rank, world_size, backend)
    model = SimpleModel()
    model.to(rank) #Move model to the device of the rank
    ddp_model = nn.parallel.DistributedDataParallel(model, device_ids=[rank]) #Make model DDP
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)
    inputs = torch.randn(20, 10).to(rank) #Move input to the rank's device
    labels = torch.randn(20, 1).to(rank) #Move label to the rank's device
    optimizer.zero_grad() #Reset the gradients
    outputs = ddp_model(inputs) #Forward pass
    loss = nn.MSELoss()(outputs, labels) #Compute loss
    loss.backward() #Backward pass and computes gradient for each of the model's parameters.
    optimizer.step() #Updates the model's weights

    if rank == 0:
      print(f"Rank {rank} loss: {loss.item()}")
    cleanup()

if __name__ == "__main__":
    world_size = 4
    backend_choice = "nccl"  # or "gloo"
    mp.spawn(ddp_model_example,
             args=(world_size, backend_choice),
             nprocs=world_size,
             join=True)
```

This demonstrates the DDP module, a core part of distributed training. It also ensures the model and its inputs are correctly placed on the appropriate device of each rank. Proper handling of the model initialization, the move of model and input data to the specific device of each rank is essential for proper functioning. If NCCL hangs here, there are generally two causes: Either the network isn't setup correctly or the underlying parameters aren't on the same devices as the rank.

In summary, when NCCL freezes while Gloo works, it's rarely a matter of inherent fault in NCCL. It's generally attributable to the specific sensitivity of NCCL to network setup, proper synchronization of its asynchronous operations and correct parameter settings. Gloo's more forgiving nature masks some of these underlying complexities, which when not attended to with NCCL, lead to freezes. When debugging DDP with NCCL, systematically verify your network configuration, asynchronous communication handling, and hardware compatibility.

For further exploration, consult the official PyTorch documentation on distributed training, delve into the specifics of NCCL through the NVIDIA developer resources, and study distributed programming concepts from academic literature and online courses. Additionally, the PyTorch source code itself serves as a valuable resource for understanding the inner workings of the DDP module and its interactions with various backends. Thorough examination of these will greatly improve understanding of the underlying issues related to the distributed training.
