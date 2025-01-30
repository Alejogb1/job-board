---
title: "How can I safely run two instances of a script using torch.distributed.run?"
date: "2025-01-30"
id: "how-can-i-safely-run-two-instances-of"
---
`torch.distributed.run` simplifies distributed training in PyTorch, but its default behavior can be problematic when launching multiple instances of the same script. Without proper safeguards, you risk race conditions, overlapping port allocations, and unpredictable training behavior, effectively negating the benefits of distributed execution. The crux lies in ensuring each process has a distinct rendezvous point and local environment. This means carefully controlling rank assignment, master address, and port numbers. Let me explain how I've handled this in projects and provide some practical examples.

The primary challenge stems from `torch.distributed.run`’s reliance on environment variables to configure distributed setups. Each instance of the script, if launched concurrently without adjustments, would default to the same rank (often 0), master address, and port. Consequently, they would all attempt to communicate with each other instead of forming distinct groups, causing synchronization failures and leading to corrupted results or outright crashes.

My approach involves several key steps executed before invoking `torch.distributed.run`:

1.  **Dynamic Rank Assignment:** I never rely on the default rank assignment. Instead, I use unique identifiers, like process IDs or index assignments derived from the environment (e.g., SLURM task IDs), to ensure each launched instance has a unique rank within its distributed group. This prevents conflicts during initialization of the distributed process group.

2.  **Unique Master Address and Port:** I avoid hardcoding the master address (the IP address of the rank 0 process) and the port number. Instead, I generate them dynamically. For example, I often allocate a unique port range per experiment or job and select an unused port within that range based on the process index. Similarly, if working within a cluster, I might use the job’s allocated network interface and derive the master address from the allocated node's IP address, coupled with the determined port.

3.  **Explicit Configuration of `torch.distributed.run`:** Using Python’s subprocess module, I explicitly set environment variables like `MASTER_ADDR`, `MASTER_PORT`, `RANK`, and `WORLD_SIZE` before invoking `torch.distributed.run`. This ensures that each instance of the script operates within its designated environment, effectively treating them as independent distributed training sessions.

Let’s examine some code examples that encapsulate this methodology:

**Example 1: Two Independent Training Scripts on the Same Node**

This example demonstrates launching two instances of a script on the same machine for local testing.

```python
import subprocess
import os
import socket
import random
import torch

def find_free_port():
    with socket.socket() as sock:
        sock.bind(('', 0))
        return sock.getsockname()[1]

def launch_distributed_training(script_path, rank, world_size, master_port):
    master_addr = "127.0.0.1"
    env = os.environ.copy()
    env["MASTER_ADDR"] = master_addr
    env["MASTER_PORT"] = str(master_port)
    env["RANK"] = str(rank)
    env["WORLD_SIZE"] = str(world_size)
    env["CUDA_VISIBLE_DEVICES"] = str(rank)  # For local CUDA setup
    subprocess.run(["torchrun", "--nnodes=1", "--nproc_per_node=1", script_path], env=env)



if __name__ == "__main__":
    script_to_run = "train_script.py" #Replace with your training script path
    num_instances = 2
    base_port = 12000
    for instance_rank in range(num_instances):
        master_port = base_port + instance_rank # Generate unique port
        launch_distributed_training(script_to_run, instance_rank, 1, master_port)
```

*   **`find_free_port()`:**  This utility function, relying on Python’s `socket` module, dynamically finds an available port for binding.
*   **`launch_distributed_training()`:** This function encapsulates the core process. It takes the script path, the specific rank within a particular instance, and the master port, constructing environment variables prior to calling `torch.distributed.run`. It sets `CUDA_VISIBLE_DEVICES` to the rank for simple single GPU per process testing on a local machine.
*   **The `if __name__ == "__main__":` block:** This section configures the settings and triggers multiple instances. Each instance receives a unique rank and port, ensuring distinct communication.

**Example 2: Handling Multiple Nodes with a Slurm Cluster**

This example illustrates how to launch two instances of training scripts on a Slurm cluster.

```python
import subprocess
import os
import socket
import random
import torch

def get_slurm_node_ip():
  try:
    return socket.gethostbyname(socket.gethostname())
  except socket.gaierror:
    print ("Warning, cannot retrieve the node ip, please make sure the node has the hostname set")
    return None

def find_free_port():
    with socket.socket() as sock:
        sock.bind(('', 0))
        return sock.getsockname()[1]

def launch_slurm_distributed_training(script_path, instance_id, world_size, master_port, slurm_task_id, slurm_ntasks_per_node):
    master_addr = get_slurm_node_ip()
    if master_addr is None:
      return # stop if the IP address could not be resolved
    env = os.environ.copy()
    rank = (slurm_task_id % slurm_ntasks_per_node)
    env["MASTER_ADDR"] = master_addr
    env["MASTER_PORT"] = str(master_port)
    env["RANK"] = str(rank)
    env["WORLD_SIZE"] = str(world_size)
    env["SLURM_LOCALID"] = str(rank)
    env["SLURM_PROCID"] = str(slurm_task_id)
    env["SLURM_NTASKS_PER_NODE"] = str(slurm_ntasks_per_node)
    subprocess.run(["torchrun", f"--nnodes={slurm_ntasks_per_node}", f"--nproc_per_node=1", script_path], env=env)

if __name__ == "__main__":
    script_to_run = "train_script.py" # Replace with your training script path
    num_instances = 2
    base_port = 12000
    slurm_ntasks_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))

    for instance_id in range(num_instances):
        master_port = base_port + instance_id
        slurm_task_id = int(os.environ["SLURM_PROCID"]) # Get the task id from environment
        launch_slurm_distributed_training(script_to_run, instance_id, slurm_ntasks_per_node, master_port, slurm_task_id, slurm_ntasks_per_node)

```

*   **`get_slurm_node_ip()`:** This utility function attempts to retrieve the local IP of the slurm node.
*   **`launch_slurm_distributed_training()`:** This adapted function incorporates Slurm environment variables such as `SLURM_PROCID` and `SLURM_NTASKS_PER_NODE` to assign unique ranks within each instance’s node.
*   **The `if __name__ == "__main__":` block:** This block retrieves environment variables related to SLURM and generates the master port and launches training instances. It assumes that the `SLURM_PROCID` and `SLURM_NTASKS_PER_NODE` environment variables are present.

**Example 3: Encapsulating Training Logic**

This example shows how your training script `train_script.py` should be setup.

```python
import torch
import torch.distributed as dist
import os


def setup_distributed():
    if dist.is_available() and dist.is_initialized():
        return # Skip if already initialized
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ["MASTER_PORT"]
    dist.init_process_group(backend="nccl", init_method="tcp://" + master_addr + ":" + master_port,
                             world_size=world_size, rank=rank)


def train():
    setup_distributed()
    print(f"Rank: {dist.get_rank()} is running on host: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
    # Your training logic should go here, using torch.distributed constructs
    if dist.get_rank() == 0:
      print("This is only printed by rank zero")
    dist.destroy_process_group()

if __name__ == "__main__":
    train()
```

*   **`setup_distributed()`:** This crucial function, within the training script, retrieves the assigned rank, master address, master port, and world size from the environment variables and initializes the distributed process group using `dist.init_process_group()`. This ensures correct process communication.
*   **`train()`:** This is where your training logic goes. The rank is used to print the hostname, and an example of a condition where only the rank 0 process prints a message to demonstrate that each process is isolated from others.
*   **The `if __name__ == "__main__":` block:** This starts the training script and calls the training logic function.

These examples serve as a foundation. The precise implementation might require adaptation based on your specific cluster setup, resource manager, and the intricacies of your training task. However, the core principle remains constant: explicit control over environment variables and the use of dynamic resource allocation are imperative for safely launching multiple instances of a script with `torch.distributed.run`.

For further information and best practices I recommend consulting the official PyTorch documentation on distributed training. You will also find insightful examples and tutorials on the PyTorch website itself and in publications from the deep learning community that covers distributed training best practices. Moreover, resources that focus on SLURM documentation will provide valuable insights if running jobs in a SLURM cluster environment.
