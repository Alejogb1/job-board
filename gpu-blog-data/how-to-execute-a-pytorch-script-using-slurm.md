---
title: "How to execute a PyTorch script using Slurm?"
date: "2025-01-30"
id: "how-to-execute-a-pytorch-script-using-slurm"
---
The core challenge in executing PyTorch scripts with Slurm lies in efficiently managing the distributed nature of PyTorch's data parallelism and the resource allocation capabilities of Slurm.  My experience optimizing deep learning workloads on HPC clusters extensively involved addressing this very issue.  Failure to properly configure Slurm job scripts often results in suboptimal performance, or even complete job failure, particularly when dealing with large models and datasets.  Effective execution hinges on correctly specifying the number of nodes, GPUs, and the inter-process communication (IPC) mechanism.

**1. Clear Explanation:**

Slurm is a workload manager designed for high-performance computing environments. It allows users to submit jobs specifying resource requirements (CPU cores, memory, GPUs) and dependencies. PyTorch, on the other hand, offers powerful tools for distributed training, allowing the training process to be spread across multiple GPUs and nodes.  The key to successful integration is to bridge the gap between Slurm's resource allocation and PyTorch's distributed training mechanisms. This is achieved primarily through the Slurm job script, which controls the environment and execution of the PyTorch script.  The script must accurately reflect the hardware requirements of the PyTorch application and correctly initiate the distributed training process.  Incorrect specification of the number of tasks, nodes, or GPUs will lead to errors or severely hampered performance.  Further, careful consideration of the chosen communication backend (e.g., Gloo, NCCL) is critical for achieving optimal inter-process communication speed.

The typical workflow involves creating a Slurm job script (typically `.sbatch`) which defines the job's resource requests and then executes the PyTorch training script.  The PyTorch script itself utilizes PyTorch's distributed data parallel functionalities (e.g., `torch.distributed.launch`). The Slurm script orchestrates the launch of multiple PyTorch processes across the allocated nodes, ensuring efficient communication and data distribution among them. This involves careful management of environment variables, particularly those that define the process rank and world size, crucial for identifying the role of each process within the distributed training framework.  One common pitfall is neglecting to set the appropriate environment variables, leading to processes failing to communicate or coordinate correctly.


**2. Code Examples with Commentary:**

**Example 1: Single Node, Multiple GPUs:**

```bash
#!/bin/bash
#SBATCH --job-name=pytorch_single_node
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=10
#SBATCH --gpus=4
#SBATCH --mem=64G
#SBATCH --partition=gpu_partition  #Replace with your partition name

export MASTER_ADDR=$(hostname)
export MASTER_PORT=12355

srun -n 4 python train.py
```

* **Commentary:** This script requests 4 tasks (processes) on a single node, each with 10 CPUs and access to one of the four GPUs. `MASTER_ADDR` and `MASTER_PORT` are crucial for process communication within the PyTorch distributed training setup.  The `srun` command launches 4 instances of `train.py`.  The `train.py` script must accordingly utilize `torch.distributed.launch` to handle the distributed training.  This example assumes a GPU partition is available.


**Example 2: Multiple Nodes, Multiple GPUs:**

```bash
#!/bin/bash
#SBATCH --job-name=pytorch_multi_node
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=10
#SBATCH --gpus-per-node=2
#SBATCH --mem=64G
#SBATCH --partition=gpu_partition
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n1)
export MASTER_PORT=12355

srun -n 4 python train.py
```

* **Commentary:** This script requests 2 nodes, each with 2 tasks (4 tasks total). Each task is allocated 10 CPUs and one GPU.  The `MASTER_ADDR` is now obtained using `scontrol` to get the hostname of the first node in the allocation.  The `--output` and `--error` directives redirect stdout and stderr to files named according to the job ID, facilitating debugging.  Again, `train.py` must be appropriately configured for distributed training.


**Example 3:  Using `torch.distributed.launch` (train.py):**

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

#Simplified Model and Training Loop
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(10, 1)

    def forward(self, x):
        return self.linear(x)

if __name__ == "__main__":
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['SLURM_NTASKS'])
    setup(rank, world_size)
    model = MyModel()
    # Distributed training using DDP
    model = nn.parallel.DistributedDataParallel(model)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    #Training loop (simplified)
    for epoch in range(10):
        # data loading and training step
        pass
    cleanup()
```

* **Commentary:** This `train.py` script demonstrates the use of `torch.distributed.launch` within a PyTorch application.  It sets up the distributed process group using NCCL, a high-performance communication backend suitable for GPUs. The `setup` and `cleanup` functions handle initialization and finalization of the distributed process group.  Error handling and more robust training logic would be incorporated in a production setting.  Crucially, the rank and world size are obtained from Slurm environment variables.


**3. Resource Recommendations:**

For a more in-depth understanding of Slurm, consult the official Slurm documentation.  For advanced distributed training techniques within PyTorch, explore the PyTorch distributed training tutorials.  Finally,  a strong grasp of Linux command-line tools and shell scripting will significantly enhance your ability to manage and debug Slurm jobs.  Understanding concepts like process management and environment variable manipulation are essential.
