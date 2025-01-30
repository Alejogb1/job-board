---
title: "Why is dist.init_process_group() stalling when training an ML model across multiple GPUs?"
date: "2025-01-30"
id: "why-is-distinitprocessgroup-stalling-when-training-an-ml"
---
The `dist.init_process_group()` call in PyTorch's distributed training framework often stalls due to network communication issues, particularly when dealing with incorrect configurations of the underlying communication backend or resource allocation problems. My experience scaling large-language model training to multiple GPU nodes has shown me that the root cause typically lies within the inter-process communication (IPC) setup rather than in the model architecture itself.

**1. Explanation of the Stalling Issue**

The `dist.init_process_group()` function's purpose is to establish a communication channel between different processes (typically one process per GPU) so they can exchange gradient information during the distributed training. It relies on a selected backend, such as NCCL (NVIDIA Collective Communications Library) or Gloo, to facilitate this communication. The process group initiation requires all participating processes to discover and connect to each other before they can proceed to the training loop. When a process encounters a problem during this handshake phase, it essentially waits indefinitely for the other processes to complete their respective setup steps. This results in a stall, as at least one process is stuck and preventing the overall initialization from completing. The most common reasons for such stalls can be categorized as follows:

*   **Network Configuration Problems:** Incorrect network interface settings, mismatched hostnames or IP addresses, firewall restrictions, and insufficient network bandwidth can all disrupt the required communication between nodes. Misconfigured or absent DNS resolution can also cause failures in the handshake.
*   **Incorrect Backend Selection or Configuration:** The choice of the communication backend (NCCL, Gloo) must be appropriate for the hardware environment. NCCL is optimized for NVIDIA GPUs and typically provides the best performance. Gloo is a more general-purpose backend that can be used on CPUs and various hardware platforms, but it might not be as performant as NCCL on GPU systems. Further, certain backends may have specific environment variable requirements, like `NCCL_DEBUG` for debugging NCCL issues. Improperly set environment variables can lead to communication failures.
*   **Resource Allocation Issues:** Inadequate or incorrectly configured resources on individual nodes can cause processes to stall or crash during the initialization. This includes issues related to insufficient shared memory, too many processes competing for limited hardware, or even GPU driver incompatibilities. Sometimes, incorrect `rank` assignments across the processes can prevent successful setup.
*   **Firewall Blocking:** Firewalls present on systems may actively block the ports used by distributed training for communication. This can happen when firewalls are not disabled or when specific exceptions for these ports are not created, which will then block incoming connections from other nodes and cause a deadlock while the processes are awaiting the connection.
*   **Synchronization Problems:** While `init_process_group` is meant to provide a synchronized setup, subtle timing issues can still occur. If processes across nodes attempt to initialize before the underlying network setup is complete (or before each node has network interface addresses ready), the communication may fail. This could happen in orchestrated environments or job queues where process initialization order isn't explicitly controlled.
*   **Host Name Resolution Failure:** Incorrect or missing host name to IP address mappings, especially in situations where distributed processes are set up across different machines, may prevent processes from finding each other in the initial phase, and this can be due to improper hostname setup in the environment.

**2. Code Examples and Commentary**

Here, I will illustrate common problems and solutions through examples.

**Example 1: Incorrectly Configured Host Addresses**

```python
import os
import torch
import torch.distributed as dist

def init_distributed(rank, world_size, backend='nccl'):
  os.environ['MASTER_ADDR'] = 'wrong.address' # Issue
  os.environ['MASTER_PORT'] = '12345'
  dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


if __name__ == '__main__':
    world_size = int(os.environ['WORLD_SIZE']) # Correctly retrieved env vars
    rank = int(os.environ['RANK'])

    try:
      init_distributed(rank, world_size)
      print(f"Rank {rank}: Process group initialized successfully!")
    except RuntimeError as e:
      print(f"Rank {rank}: Initialization failed: {e}")
      dist.destroy_process_group()
```

*   **Commentary:** In this example, I have intentionally set the `MASTER_ADDR` environment variable to a non-existent or incorrect address (`wrong.address`). This results in a stall during initialization, as processes are unable to establish communication with the master process. The exception handler catches the runtime error and prints a message, along with correctly destroying the partially initialized process group in a good practice way. In real-world scenarios, the `MASTER_ADDR` should point to the IP address of the machine serving as the master, and all the nodes should be able to connect to this machine. When deploying across different machines, this will be the machine where a process of `rank` 0 is executing. The port number will need to be accessible to each host, which may require firewall considerations.

**Example 2: Incorrect Backend and Device Mismatch**

```python
import os
import torch
import torch.distributed as dist

def init_distributed(rank, world_size, backend='gloo'):
    os.environ['MASTER_ADDR'] = 'localhost' # Correct address for local multiple processes
    os.environ['MASTER_PORT'] = '12345'
    try:
      dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    except RuntimeError as e:
      print(f"Rank {rank}: Initialization failed: {e}")
      dist.destroy_process_group()

    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    print(f"Rank {rank} Device: {device}")


if __name__ == '__main__':
    world_size = int(os.environ['WORLD_SIZE']) # Correctly retrieved env vars
    rank = int(os.environ['RANK'])

    init_distributed(rank, world_size)

    if torch.cuda.is_available():
        try:
           local_rank = int(os.environ["LOCAL_RANK"]) # Correctly retrieved env vars
           torch.cuda.set_device(local_rank) # Sets device for the GPU on each process
           print(f"Rank {rank}: Successfully set device to {local_rank}")
        except KeyError as e:
          print(f"Rank {rank}: Failed to set device: {e}")
          dist.destroy_process_group() # Destroyed since it could not set local rank

```

*   **Commentary:** In this example, I've forced the backend to `gloo`, while I assume there are NVIDIA GPUs that could run on NCCL, which is optimal for GPU communication. When `gloo` is selected, it falls back to CPU for communication when cuda devices are initialized, which leads to inefficient communication when GPUs are intended to be used. Additionally, `local_rank` is required to allow each of the distributed processes to manage the GPUs correctly, when the training process intends to work on multiple GPUs on a single machine.  Further, when GPUs are being used by `gloo`, the code does not use any of them and may not be able to communicate efficiently over CPU RAM.

**Example 3: Firewall Restrictions**

```python
import os
import torch
import torch.distributed as dist
import time

def init_distributed(rank, world_size, backend='nccl'):
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12345'

  print(f"Rank {rank}: Attempting to init process group")
  try:
      dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
      print(f"Rank {rank}: Process group initialized successfully!")

  except RuntimeError as e:
      print(f"Rank {rank}: Initialization failed: {e}")
      dist.destroy_process_group()
      # Simulating firewall block
      time.sleep(10) # Simulate firewall blocking attempt for 10 seconds
      print(f"Rank {rank}: Exiting due to firewall sim...")
      return False

  return True

if __name__ == '__main__':
    world_size = int(os.environ['WORLD_SIZE']) # Correctly retrieved env vars
    rank = int(os.environ['RANK'])
    success = init_distributed(rank, world_size)

    if success:
      # Proceed with training, if any
      print(f"Rank {rank}: Distributed training can begin")
```

*   **Commentary:** In this example, I've introduced a simulated firewall block by adding a time delay in case of an exception during `init_process_group`. In real-world scenarios, a firewall might block communication on the specified ports, which will lead to a deadlock during the initial connection attempts. This will lead to similar behavior to a long timeout, making debugging more difficult. In the code, after timeout (simulated here), the process will exit gracefully after cleaning up the process group, a good practice.

**3. Resource Recommendations**

To further understand the intricacies of distributed training and avoid such stalls, I recommend the following resources:

*   **PyTorch Documentation:** The official PyTorch documentation on distributed training is comprehensive and provides a thorough explanation of all concepts and functions, which are vital in building solid distributed code. It describes best practices for using `dist.init_process_group()`.
*   **NVIDIA Collective Communications Library (NCCL) Documentation:** If your system employs NVIDIA GPUs, understanding NCCL and its environment variable configurations is crucial for achieving high performance. The document provides valuable information that can help in debugging.
*   **High-Performance Computing (HPC) Resources:** Textbooks and online resources that cover high-performance computing, especially those that focus on distributed memory programming and inter-process communication. The resources can help in better debugging complex networking issues when using distributed training.

These resources can be leveraged to grasp the fundamentals and the nuances related to distributed training, which can assist in debugging and reducing runtime errors. Understanding the intricacies of distributed training on GPUs is essential for ensuring that models are being trained efficiently and optimally.
