---
title: "Why does `dist.init_process_group()` hang when training an ML model across multiple GPUs?"
date: "2024-12-23"
id: "why-does-distinitprocessgroup-hang-when-training-an-ml-model-across-multiple-gpus"
---

Let's unpack the frustrating scenario of `dist.init_process_group()` hanging during multi-gpu model training. I've certainly been there, staring at a frozen console after launching a script, feeling like I've missed some crucial detail. It's usually not a single, easily identifiable problem, but rather a confluence of factors that can lead to this deadlock. The `dist.init_process_group()` function, as we know, is the gateway to distributed training in PyTorch (and similar libraries), and when it gets stuck, the entire distributed setup grinds to a halt. Let's dive into the common culprits based on my past experiences battling these issues.

First, the function’s core purpose is to establish communication channels between the various processes that constitute your distributed training setup. This involves a rendezvous point where each process waits to connect with all others. If any process fails to connect to this rendezvous point, or if one or more processes don't arrive at that point, the function hangs indefinitely, waiting for all participants.

One of the most frequent causes I've seen, particularly in early distributed projects, is incorrect network configuration. If the processes are running across different machines, it's crucial to verify that each machine is reachable by others, and the firewall rules aren't blocking the designated ports. Specifically, you'll need to pay very close attention to the `init_method` parameter you use. Typically, I've used TCP based initialization, like `tcp://<master_ip>:<master_port>`, where `<master_ip>` is the ip of the machine where rank=0 is running. It’s important that this IP is accessible by all the other training nodes. Sometimes the hostnames resolution is not setup correctly or the firewall is blocking that port. Always remember to double check network accessibility with simple tools like `ping` or `telnet`.

Another common source of problems is inconsistent environment variables. When multiple GPUs are on the same machine or multiple machines are used, you must ensure that each process has the same values for `WORLD_SIZE` and `RANK`. `WORLD_SIZE` indicates how many processes are participating in the training, while `RANK` is a unique id for a given process, usually from `0` to `WORLD_SIZE - 1`. Misconfigurations here, often caused by incorrectly launched scripts or environment variable overriding, will cause the hanging issue, since some processes might expect a different number of training nodes.

Incorrectly formatted host lists when using the `file://` initialization method can also cause this problem, especially in more complex cluster environments. If the shared file does not contain accurate and complete information about all the machines participating in the training, or is not properly set up with all processes having read and write permissions, `init_process_group()` will hang.

Finally, process synchronization issues can cause hangs. Even if the network and environment variables are correctly configured, certain operations that might cause unintended synchronization during the initialization process, such as excessive file system usage, can still cause one process to be delayed, leading to the hang at initialization. For example, during setup, if one process is waiting for a file to be written by another, but the second process is not making progress, this might result in a deadlock.

Now, let’s illustrate this with some code examples.

**Example 1: Network and Rendezvous Configuration (TCP Initialization)**

```python
import os
import torch
import torch.distributed as dist

def setup(rank, world_size, master_ip, master_port):
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_ip}:{master_port}",
        rank=rank,
        world_size=world_size
    )

    print(f"Process {rank} initialized. World size: {world_size}")


if __name__ == "__main__":
    world_size = 2  # Example number of processes
    master_ip = "192.168.1.100" # Replace with the actual master IP.
    master_port = 12345 # Replace with any free port.

    # For simplicity, I'll simulate multiple processes within the same script
    for rank in range(world_size):
      import multiprocessing
      p = multiprocessing.Process(target=setup, args=(rank, world_size, master_ip, master_port))
      p.start()
```

In this snippet, `setup` is the function called on each process, setting up the necessary environment variables, and initializing with a tcp backend. If the specified IP address of the master is incorrect, the firewall is blocking the communication or the port is used, the processes will hang within `dist.init_process_group()`. This directly illustrates the network accessibility issues I previously mentioned. When debugging this type of hang, a crucial step is to ensure the `master_ip` is reachable, and the `master_port` is open. The `nccl` backend is used because it's the recommended and most efficient way to do distributed training across gpus, but other backends such as `gloo` or `mpi` can be used.

**Example 2: Inconsistent Environment Variables**

```python
import os
import torch
import torch.distributed as dist

def setup_bad_ranks(rank, world_size, master_ip, master_port):
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = str(master_port)
    # Here, the ranks are incorrectly assigned.
    if rank == 0:
      os.environ['RANK'] = str(0)
    else:
      os.environ['RANK'] = str(2)
    os.environ['WORLD_SIZE'] = str(world_size)


    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_ip}:{master_port}",
        rank=int(os.environ['RANK']),
        world_size=world_size
    )

    print(f"Process {rank} initialized. World size: {world_size}")


if __name__ == "__main__":
    world_size = 2  # Example number of processes
    master_ip = "192.168.1.100"
    master_port = 12346

    # For simplicity, I'll simulate multiple processes within the same script
    for rank in range(world_size):
      import multiprocessing
      p = multiprocessing.Process(target=setup_bad_ranks, args=(rank, world_size, master_ip, master_port))
      p.start()
```
In this second example, I intentionally make a mistake during the `RANK` assignment. The `rank` variable within the function still reflects the correct rank, but the environment variable might hold different values which will make `dist.init_process_group()` hang because two processes share the same rank while the third process will never be reached during the group initialization.

**Example 3: Synchronization issues during initialization**

```python
import os
import torch
import torch.distributed as dist
import time

def setup_with_sync_issue(rank, world_size, master_ip, master_port):
    os.environ['MASTER_ADDR'] = master_ip
    os.environ['MASTER_PORT'] = str(master_port)
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)

    # Simulate a delay on one process that causes initialization to hang
    if rank == 1:
       time.sleep(10)


    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_ip}:{master_port}",
        rank=rank,
        world_size=world_size
    )

    print(f"Process {rank} initialized. World size: {world_size}")



if __name__ == "__main__":
    world_size = 2  # Example number of processes
    master_ip = "192.168.1.100"
    master_port = 12347

    # For simplicity, I'll simulate multiple processes within the same script
    for rank in range(world_size):
      import multiprocessing
      p = multiprocessing.Process(target=setup_with_sync_issue, args=(rank, world_size, master_ip, master_port))
      p.start()

```
In this example, a delay is introduced in process with rank 1 to illustrate synchronization issues where one process is delayed during initialization, causing a hang. This is a simplified illustration, but the same problem can occur with filesystem operations, or any other operation that is performed during setup and might delay certain processes more than others.

For further understanding of these distributed training problems, I’d recommend looking into specific resources. For the underlying distributed computing principles, "Parallel Programming: Techniques and Applications Using Networked Workstations and Parallel Computers" by Barry Wilkinson and Michael Allen is invaluable. For PyTorch specifically, the official documentation is obviously essential. Additionally, papers describing specific synchronization primitives such as those implemented in NCCL and libraries like OpenMPI can be enlightening. Finally, I have found that spending some time studying well known distributed computing paradigms, such as those covered in "Introduction to Parallel Computing" by Ananth Grama, Anshul Gupta, George Karypis, Vipin Kumar provides you with an excellent general understanding.

Debugging these hangs always requires careful examination of your specific environment, including network settings, environment variables, and any unusual operations you perform prior to calling `init_process_group()`. Often, it’s a combination of factors, and systematically eliminating each potential problem is the key to resolving the issue. I hope this detailed analysis and examples help you in your own distributed training efforts.
