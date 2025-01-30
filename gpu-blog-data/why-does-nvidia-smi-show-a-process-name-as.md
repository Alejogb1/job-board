---
title: "Why does nvidia-smi show a process name as '-' and prevent killing its PID with -9?"
date: "2025-01-30"
id: "why-does-nvidia-smi-show-a-process-name-as"
---
The appearance of a hyphen ("-") as a process name in `nvidia-smi` output, coupled with the inability to terminate its corresponding PID using `kill -9`, typically indicates the presence of a defunct or 'zombie' process that was previously utilizing the GPU. This situation does not signify an active, runaway GPU application; rather, it points to a process that has completed its execution but whose cleanup has not been properly handled by the operating system’s process management mechanism. This issue frequently arises within the context of machine learning workflows and multi-processing environments, particularly where GPU resources are heavily utilized.

The underlying cause lies in how operating systems manage process lifecycle and parent-child process relationships. When a child process terminates, it transitions to a zombie state, awaiting its parent process to acknowledge its termination via a `wait()` system call. During this zombie state, the process has ceased execution, relinquishing its resources except for its entry in the process table. In the context of GPU-accelerated applications, the situation is further complicated by CUDA driver interactions. The child process may have acquired a CUDA context, a structure managing GPU memory and execution contexts, which remains attached to the defunct process's entry. This CUDA context's lifecycle is not directly managed by the operating system's process termination signals like `SIGKILL`. Therefore, while the process itself is technically dead, its CUDA-related resources remain associated, potentially causing confusion and resource contention with subsequently launched GPU processes.

`nvidia-smi` displays the "-" placeholder as the process name precisely because the operating system no longer identifies the associated executable. It only retains the process ID and, crucially for `nvidia-smi`, the information that this process still holds a CUDA context. The fact that `kill -9`, which sends the `SIGKILL` signal, does not resolve the issue indicates the operating system has already done its part in terminating the process, and the issue is specifically related to the lingering CUDA context. Standard process termination mechanisms simply do not address or release the GPU-related resources held by a zombie process.

Resolving this issue generally requires either identifying and correcting the behavior of the parent process, which failed to properly wait on the terminated child, or, as a last resort, a more forceful approach to reclaim the associated CUDA context. This often involves restarting the CUDA driver or even rebooting the system. Let's consider several concrete scenarios and code examples from my experience as a machine learning engineer.

**Example 1: Unhandled Process Termination in a Python Multiprocessing Scenario**

```python
import torch
import torch.multiprocessing as mp
import time

def gpu_worker(rank, device_id):
  torch.cuda.set_device(device_id)
  print(f"Process {rank}: Device {torch.cuda.current_device()}")
  time.sleep(5) #Simulate some work
  print(f"Process {rank}: Exiting")

if __name__ == '__main__':
  mp.set_start_method('spawn')  # Necessary on some platforms for cuda
  processes = []
  num_devices = torch.cuda.device_count()
  for i in range(num_devices):
      p = mp.Process(target=gpu_worker, args=(i,i))
      processes.append(p)
      p.start()

  # Parent Process NOT waiting for children!
  #time.sleep(10)
  #for p in processes:
  #    p.join()

  print("Parent process exiting")
```

*   **Commentary:** This Python code utilizes the `torch.multiprocessing` module to launch multiple GPU workers. Each worker acquires a CUDA context upon initialization. The critical issue here is the parent process immediately exits without waiting for its child processes to complete their tasks using a `join()`. The commented-out section demonstrates the proper way to wait. Because the parent terminates prematurely, the child processes become zombies. Subsequent `nvidia-smi` calls will show the child process PID with a "-" name, and `kill -9` will be ineffective in removing it as explained above. In a real-world application, issues such as unhandled exceptions within subprocesses, or a premature termination of the parent process can lead to similar zombie process generation.

**Example 2: A Bash Script Simulating Resource Leak**

```bash
#!/bin/bash

# Launch a dummy CUDA program (replace with your actual CUDA program)
# For demonstration, a simple infinite loop utilizing GPU
python -c "import torch; torch.randn(1000, 1000).cuda()" &
echo $! > child.pid

sleep 3  #Allow time for it to start

# Parent process terminates without cleaning up
echo "Parent exiting"
```

*   **Commentary:** This bash script simulates a situation where a child process acquires a CUDA context (represented by the python command that imports torch and allocates a GPU tensor, running in the background) but is not properly managed by the parent script. The `&` at the end of the python line launches the program in background which implies that the parent script does not manage its lifecycle. The PID of the background process is captured and stored in a file. The parent process exits after a short sleep, leaving the child process detached. This mirrors the scenario in example 1 but implemented in a Bash script. After running this script, `nvidia-smi` will likely display the orphaned process with a "-" process name and `kill -9` on the captured pid will not release the GPU context. This case is simpler than example 1. It highlights how even non-pythonic code execution or orchestration of scripts might cause the same problem.

**Example 3: Using PyTorch's multiprocessing spawn with incorrect cleanup.**
```python
import torch
import torch.multiprocessing as mp
import time

def gpu_work():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tensor = torch.randn(1000, 1000).to(device)
    time.sleep(5)

if __name__ == '__main__':
    mp.set_start_method('spawn') #Crucial in some environments for GPU with multiprocessing
    p = mp.Process(target=gpu_work)
    p.start()

    #Intentionally skipping proper waiting on the child process
    #p.join()

    print("Parent exiting")
```

*   **Commentary:** This example again demonstrates the process using PyTorch multiprocessing. It is similar to the first example, but it is simpler, focusing on a single child process. This process utilizes CUDA if available and creates a tensor, thus getting a CUDA context. Crucially, the parent process, similarly to the first example, does not `join` the child before exiting. As a result, the child becomes a zombie after finishing its work and is displayed in `nvidia-smi` with the "-" placeholder and is resistant to `kill -9`. The essential point here is that the absence of proper process management using `join` leads to orphaned processes, regardless of the specific work done by the child.

**Recommended Resources:**

For deeper understanding of process management and operating system behavior, the following resources provide comprehensive information:

1.  **Operating System Concepts:** A thorough book on operating systems will provide fundamental knowledge of process life cycles, signals, and process management mechanisms. These foundational concepts are critical to understand the root cause of the zombie process issue.
2.  **CUDA Documentation:** The official CUDA documentation provides essential information on CUDA contexts, their lifecycle, and best practices for managing GPU resources when using CUDA APIs directly or indirectly. It also provides insight into CUDA drivers.
3.  **Python Multiprocessing Documentation:** The official Python documentation provides specific guidance on managing process lifecycles when using the `multiprocessing` library. This resource is crucial for resolving process management issues when using Python in GPU-accelerated contexts.

In conclusion, the `nvidia-smi` output showing a process as "-" and its associated PID being resistant to `kill -9` points towards a defunct process holding an orphaned CUDA context. This problem stems from inadequate process management, often in multiprocessing or parallel computation contexts where parent processes do not correctly clean up after child processes utilizing GPU resources. Properly handling process termination, especially with child processes utilizing GPUs, is critical for reliable GPU resource management and prevention of issues such as this. Reviewing operating system concepts, CUDA, and Python multiprocessing documentation is recommended for a deep understanding and to solve such cases.
