---
title: "How do MASTER_ADDR and MASTER_PORT affect PyTorch DDP?"
date: "2025-01-26"
id: "how-do-masteraddr-and-masterport-affect-pytorch-ddp"
---

The core function of `MASTER_ADDR` and `MASTER_PORT` in PyTorch's Distributed Data Parallel (DDP) lies in establishing the initial rendezvous point for distributed processes. These environment variables, typically set before launching a DDP training script, designate the network address and port on which the *master* process (rank 0) will listen for connections from all other participating processes. Without a precisely defined master address and port, DDP processes cannot locate and communicate with each other, thus failing to initiate distributed training.

DDP leverages a process-group-based communication backend, often using Gloo or NCCL for inter-process communication. The master process acts as a central coordinator, collecting information about network topologies and providing it to other processes. This information exchange is essential for setting up the necessary distributed infrastructure, including the building of communication primitives for efficient gradient aggregation and model parameter updates. Therefore, the master address and port, respectively specified by `MASTER_ADDR` and `MASTER_PORT`, establish a single, agreed-upon point for every training process to contact the coordinator and initialize the distributed setting.

The `MASTER_ADDR` variable specifies the IP address or hostname of the machine hosting the master process. In a multi-node environment, this will be the address of the machine where the process with rank 0 will execute. For single-machine multi-process training, `MASTER_ADDR` often points to `localhost` or `127.0.0.1`, representing the local machine. Using the loopback address works efficiently because all processes are running within the same host. Incorrectly specifying the master address, especially in multi-node configurations, will cause worker processes to be unable to join the process group, ultimately preventing distributed training.

Similarly, `MASTER_PORT` defines the TCP/IP port number on which the master process will listen for connections. This port needs to be available and unblocked, both by firewalls and by other processes already using the same port. If a port is already in use by another application, the DDP setup will encounter a `socket.error` as it will be unable to bind the address. Additionally, the chosen port needs to be consistent across all processes in the distributed training job. A mismatch in the port number will result in the worker processes targeting the incorrect listening port, again disrupting the communication and initialization procedures of the DDP. It's common practice to use a port that is greater than 1024 to avoid conflict with well-known system ports and services.

The effect of `MASTER_ADDR` and `MASTER_PORT` is most prominent in the initialization phase of PyTorch DDP using `torch.distributed.init_process_group()`. Internally, this function uses these environment variables to configure the communication backend and establish the process group.

Consider a hypothetical scenario of deploying DDP training across multiple machines. Here is a breakdown with examples and code.

**Example 1: Single Machine Multi-Process DDP**

In this case, all processes are running on the same machine, so the master can be addressed using the loopback address `127.0.0.1`.

```python
# Environment variables set prior to launching script (e.g., using `os.environ`)
# os.environ["MASTER_ADDR"] = "127.0.0.1"
# os.environ["MASTER_PORT"] = "12355"

import os
import torch
import torch.distributed as dist

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4 #Example with four processes
    import torch.multiprocessing as mp
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=run, args=(rank, world_size))
        processes.append(p)
        p.start()
    for p in processes:
        p.join()

def run(rank, world_size):
    setup(rank, world_size)
    print(f"Rank {rank}: Process group initialized")

    # Perform distributed operations...

    cleanup()
```

*Commentary:* This code initializes the distributed environment for a single-machine scenario. The `MASTER_ADDR` and `MASTER_PORT` are configured before initializing the process group with `dist.init_process_group()`. Each process has a distinct rank (0 through 3 in this example) and they are all part of the same communication group of size 4. The `"gloo"` backend, suitable for a single machine setup, is specified for simplicity.

**Example 2: Multi-Machine DDP**

Here, training is distributed across two machines, requiring precise address configuration. Let's assume machine 1 has IP `192.168.1.100` and machine 2 is `192.168.1.101`. We'll run the master process (rank 0) on machine 1.

*   **On machine 1:**
    ```python
    # Environment variables set prior to launching script
    # os.environ["MASTER_ADDR"] = "192.168.1.100"
    # os.environ["MASTER_PORT"] = "23456"
    # os.environ["RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "2"
    # Launch: python -m torch.distributed.launch --nproc_per_node=1 my_training_script.py
    ```
*   **On machine 2:**

    ```python
    # Environment variables set prior to launching script
    # os.environ["MASTER_ADDR"] = "192.168.1.100"  # same as the master!
    # os.environ["MASTER_PORT"] = "23456" # same as the master!
    # os.environ["RANK"] = "1"
    # os.environ["WORLD_SIZE"] = "2"
    # Launch: python -m torch.distributed.launch --nproc_per_node=1 my_training_script.py
    ```

*   **`my_training_script.py` (run on both machines):**

    ```python
    import os
    import torch
    import torch.distributed as dist

    def setup(rank, world_size):
        dist.init_process_group("nccl", rank=rank, world_size=world_size)

    def cleanup():
        dist.destroy_process_group()

    if __name__ == "__main__":
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        setup(rank, world_size)
        print(f"Rank {rank}: Process group initialized")

        # Perform distributed operations...

        cleanup()
    ```

*Commentary:* This example demonstrates how to distribute the training process across two different machines. The `MASTER_ADDR` is set to the IP address of the machine where the master process is running (192.168.1.100) and all worker processes are set to the same port. This example also leverages environment variables `RANK` and `WORLD_SIZE` as required by `torch.distributed.launch`. The `nccl` backend, preferable for multi-GPU situations, is specified here for efficient high-bandwidth communication between the nodes. Note that in a real scenario, launching the training script in a multi-node environment is typically done via a cluster manager which automatically handles the setup of these environment variables.

**Example 3: Error Scenario: Incorrect Port**

Let's consider an incorrect setup, where one worker process is using a different port number.

*   **On machine 1:**
    ```python
    # Environment variables set prior to launching script
    # os.environ["MASTER_ADDR"] = "192.168.1.100"
    # os.environ["MASTER_PORT"] = "23456"
    # os.environ["RANK"] = "0"
    # os.environ["WORLD_SIZE"] = "2"
    # Launch: python -m torch.distributed.launch --nproc_per_node=1 my_training_script.py
    ```
*   **On machine 2:**
    ```python
    # Environment variables set prior to launching script
    # os.environ["MASTER_ADDR"] = "192.168.1.100"  # same as the master!
    # os.environ["MASTER_PORT"] = "23457" #incorrect port number!
    # os.environ["RANK"] = "1"
    # os.environ["WORLD_SIZE"] = "2"
    # Launch: python -m torch.distributed.launch --nproc_per_node=1 my_training_script.py
    ```

The `my_training_script.py` code remains the same.

*Commentary:* In this case, when the processes on each machine attempt to initialize the process group, the second machine's process will fail to connect to the master process because of the mismatched port. This will cause a timeout error, or other error related to network communication, and prevent the DDP setup.

For further understanding of distributed training in PyTorch, I recommend consulting the PyTorch documentation on distributed training; specifically, the sections dealing with launching distributed jobs, and the initialization procedures using `init_process_group()`. The official PyTorch tutorials on DDP provide further examples and best practices, including considerations for fault tolerance and efficient communication strategies. Additionally, researching specific communication backends like NCCL and Gloo is helpful for understanding the underlying mechanics of the distributed processes and their configurations. Further study on the specifics of environment variable management, either within the code directly or using launching libraries, will assist in debugging misconfigurations. Also reviewing the documentation of the respective backend implementations can be valuable for detailed understanding.
