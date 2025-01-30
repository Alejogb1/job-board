---
title: "Why is `local_rank` 0 in DDP when I have CUDA visible device 2?"
date: "2025-01-30"
id: "why-is-localrank-0-in-ddp-when-i"
---
The observation of `local_rank` being 0 despite CUDA visible device 2 being present stems from a misunderstanding of how the `local_rank` variable interacts with the underlying CUDA device assignment within a DistributedDataParallel (DDP) training setup.  My experience working on large-scale NLP models across multiple GPU clusters has repeatedly highlighted this point: `local_rank` identifies the process's rank *within* a node, not its CUDA device ID.

**1. Explanation:**

The `local_rank` variable, commonly used in PyTorch's DDP, represents the rank of the current process within a single machine (node).  When distributing a training job across multiple GPUs within a single node (multi-GPU training), each process receives a unique `local_rank` starting from 0.  This is independent of the CUDA device the process uses;  the mapping between `local_rank` and CUDA device is handled externally, typically by the process launcher (e.g., `torchrun`, `python -m torch.distributed.launch`).

The CUDA visible devices, determined by environment variables like `CUDA_VISIBLE_DEVICES`, control which GPUs are accessible to the entire system. Setting `CUDA_VISIBLE_DEVICES=2` restricts the system to only utilize GPU with ID 2.  However, this setting does not directly influence the assignment of GPUs to individual DDP processes within a node.  If you have a multi-process DDP training on a single node, even with `CUDA_VISIBLE_DEVICES=2`, the processes are assigned GPUs sequentially (or based on a specified strategy by the launcher) *from the visible devices*.  If your process is the first one, you will always receive `local_rank=0`, regardless of the actual GPU ID assigned.

Therefore, observing `local_rank=0` with `CUDA_VISIBLE_DEVICES=2` indicates your process is the first process launched within the node and it's assigned to one of the visible devices (in this case, only device 2). The seemingly contradictory behavior arises from a conflation between process rank within the node (`local_rank`) and the physical GPU ID used by the process.

**2. Code Examples and Commentary:**

**Example 1: Single-node, multi-GPU training using `torchrun`:**

```python
import torch
import torch.distributed as dist
import os

# Assuming CUDA_VISIBLE_DEVICES=2 is set beforehand

dist.init_process_group("nccl")  # Or "gloo" for CPU
local_rank = dist.get_rank()
world_size = dist.get_world_size()

print(f"Local Rank: {local_rank}, World Size: {world_size}, CUDA Visible Devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Accessing the appropriate GPU using torch.cuda.set_device(local_rank) is crucial
torch.cuda.set_device(local_rank)  #This is usually handled by torchrun

# ...Rest of your DDP training code...
```
*Commentary:* This example utilizes `torchrun` which automatically handles GPU assignment based on `local_rank`. It shows the critical step of setting the device using `torch.cuda.set_device(local_rank)`.  Even with `CUDA_VISIBLE_DEVICES=2`, if you launch this script with `torchrun --nproc_per_node=2 your_script.py`, two processes will be created, each having a different `local_rank` (0 and 1), but both using GPU 2 (if available).

**Example 2: Manual process launching with environment variables:**

```python
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import os

def run(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355' # Choose a free port

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    local_rank = dist.get_rank()
    print(f"Local Rank: {local_rank}, World Size: {world_size}, CUDA Visible Devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    # Manual GPU assignment based on local_rank
    torch.cuda.set_device(local_rank)

    # ... Rest of your training code ...

    dist.destroy_process_group()

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    world_size = 2  #Number of processes to launch
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
```
*Commentary:* This explicitly demonstrates manual process launching.  The `CUDA_VISIBLE_DEVICES` environment variable is set before launching processes.  Despite this, `local_rank` still reflects the process rank (0, 1, etc.). Notice that explicit GPU assignment is managed within each process, not dictated solely by the environment variable.  This highlights the decoupling of device management and process rank.

**Example 3: Incorrect GPU assignment leading to errors:**

```python
import torch
import torch.distributed as dist
import os

dist.init_process_group("nccl")
local_rank = dist.get_rank()
print(f"Local Rank: {local_rank}, CUDA Visible Devices: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# INCORRECT:  This will lead to errors if local_rank > 0 and only one GPU is visible.
torch.cuda.set_device(2)  # Hardcoded device ID, ignoring local_rank

# ...Rest of your DDP training code (will likely fail)...
```

*Commentary:*  This example showcases a common mistake.  Hardcoding the device ID (`torch.cuda.set_device(2)`) ignores the `local_rank` and the intended distribution. If `local_rank` is 1 or greater, this will lead to runtime errors as the process attempts to use a device it doesn't have access to.  Correct handling requires dynamic device selection based on `local_rank`.

**3. Resource Recommendations:**

PyTorch DistributedDataParallel documentation.  Advanced PyTorch tutorials focusing on distributed training.  Relevant chapters in distributed computing textbooks.


In summary, `local_rank` signifies the process's position within a node, while CUDA visible devices dictate the system's accessible GPUs.  Their relationship is indirect: the process launcher maps `local_rank` to a CUDA device, usually sequentially from the visible devices.  Correctly managing this mapping within the code, often involving `torch.cuda.set_device(local_rank)`, is crucial for successful multi-GPU training with DDP.  Misunderstanding this leads to errors and inefficient resource utilization.
