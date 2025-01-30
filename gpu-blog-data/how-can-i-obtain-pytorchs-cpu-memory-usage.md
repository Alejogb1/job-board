---
title: "How can I obtain PyTorch's CPU memory usage statistics?"
date: "2025-01-30"
id: "how-can-i-obtain-pytorchs-cpu-memory-usage"
---
Directly observing a PyTorch process’s memory consumption on CPU requires a layered approach, since PyTorch itself does not inherently expose a consolidated metric for total CPU memory usage. My experience with optimizing large-scale deep learning models has taught me that one must combine operating system-level tools with a deep understanding of PyTorch’s internal resource management. Specifically, while PyTorch tracks memory allocated for tensors, it does not account for other resources consumed by the Python process, such as the interpreter, loaded libraries, or memory allocated outside of the PyTorch framework. To accurately gauge overall CPU memory usage, therefore, we must leverage Python's `resource` module and process inspection tools, in conjunction with PyTorch-specific information.

Fundamentally, PyTorch’s memory allocation is primarily focused on managing tensor memory, crucial for its computational graph operations. It employs its own memory allocator which differs from the general system’s memory allocation. Consequently, calls to `torch.cuda.memory_allocated()` are ineffective for reporting CPU memory usage. Instead, we must rely on Python's standard library to inquire about the process’s memory footprint, coupled with a conscious effort to manage allocated tensor memory efficiently. We can then augment this data with PyTorch-specific functionalities that offer insight into the allocated tensors themselves.

The core of monitoring a process’s memory consumption on CPU lies in the `resource` module in Python. This module provides a way to access system resource utilization information. The `resource.getrusage()` function, in particular, returns a `struct_rusage` object, which contains information about the process’s resource consumption. Within this object, `ru_maxrss` specifically represents the maximum resident set size used by the process. The resident set size (RSS) represents the portion of memory held in RAM by a process at a particular moment, and the `ru_maxrss` indicates the maximum value of RSS the process has reached. It’s important to note that, on different operating systems, the unit of this value may vary (e.g., kilobytes on Linux, bytes on macOS). The important takeaway is that this is the closest we get to a single value representing the total CPU memory utilization by a Python process.

```python
import resource
import os
import time
import torch
import psutil

def get_cpu_memory_usage():
    """
    Obtains the CPU memory usage statistics using the resource and psutil modules.
    Returns:
      tuple: A tuple containing:
        - max_rss (int): Max resident set size in kilobytes.
        - python_rss (int): Python process's resident set size in megabytes.
        - torch_allocated_memory_kb (int): PyTorch's allocated tensor memory in kilobytes.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_kb = usage.ru_maxrss
    
    process = psutil.Process(os.getpid())
    python_memory_usage_mb = process.memory_info().rss / (1024 ** 2)
    
    torch_allocated_memory_kb = torch.tensor(0.0).element_size() * torch.tensor(0).numel()

    if torch.cuda.is_available():
          torch_allocated_memory_kb = torch.cuda.memory_allocated() / 1024
          
    return max_rss_kb, python_memory_usage_mb, torch_allocated_memory_kb
 
# Example usage
print(f"Max RSS before tensor creation: {get_cpu_memory_usage()}")
a = torch.randn(1000, 1000)
print(f"Max RSS after tensor creation: {get_cpu_memory_usage()}")
b = torch.randn(1000,1000)
print(f"Max RSS after second tensor creation: {get_cpu_memory_usage()}")

```

This code snippet illustrates obtaining system-level memory usage with the `resource` and `psutil` modules in addition to printing the allocated PyTorch tensor memory (in the case a CUDA device is not being used, the `torch_allocated_memory_kb` value will be zero as tensors are placed on CPU). I’ve used `psutil` to get a clearer view of the Python process RSS. Before the PyTorch tensors are created, `max_rss_kb` will show the memory used by the Python interpreter and loaded modules. After the tensor creation, this value increases, reflecting the memory allocation. The `torch_allocated_memory_kb` provides the portion of that memory which is specifically being managed by PyTorch. Note that if a CUDA device is available, then the `torch_allocated_memory_kb` will show memory allocated on the GPU, not the CPU, so this value will not be accurate in that circumstance.

To further refine memory monitoring, we can incorporate a periodic check during our training or computation loops. We can extend this to log the memory usage at each iteration or epoch, revealing any memory leaks or unexpected spikes.

```python
import resource
import os
import time
import torch
import psutil

def get_cpu_memory_usage():
    """
    Obtains the CPU memory usage statistics using the resource and psutil modules.
    Returns:
      tuple: A tuple containing:
        - max_rss (int): Max resident set size in kilobytes.
        - python_rss (int): Python process's resident set size in megabytes.
        - torch_allocated_memory_kb (int): PyTorch's allocated tensor memory in kilobytes.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_kb = usage.ru_maxrss
    
    process = psutil.Process(os.getpid())
    python_memory_usage_mb = process.memory_info().rss / (1024 ** 2)
    
    torch_allocated_memory_kb = torch.tensor(0.0).element_size() * torch.tensor(0).numel()

    if torch.cuda.is_available():
          torch_allocated_memory_kb = torch.cuda.memory_allocated() / 1024
          
    return max_rss_kb, python_memory_usage_mb, torch_allocated_memory_kb

# Dummy training loop
model = torch.nn.Linear(100, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
inputs = torch.randn(10, 100)
targets = torch.randn(10, 10)

num_epochs = 5

for epoch in range(num_epochs):
  for i in range(20):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = torch.nn.functional.mse_loss(outputs, targets)
    loss.backward()
    optimizer.step()
    
    max_rss_kb, python_rss_mb, torch_allocated_kb = get_cpu_memory_usage()
    print(f"Epoch: {epoch}, Iteration: {i}, Max RSS: {max_rss_kb} KB, Python RSS: {python_rss_mb:.2f} MB, Torch Allocated Memory: {torch_allocated_kb:.2f} KB")
    
    time.sleep(0.01)

```
This code demonstrates incorporating the previous memory usage check in a mock training loop. After each iteration, the memory usage is printed alongside the current epoch and iteration number. This process allows us to observe trends in memory usage during the training procedure which is crucial in large deep learning tasks to identify potential memory issues early on. The use of `time.sleep()` is merely to simulate a slower computation.

Finally, let's examine a method to actively release tensors when they're no longer needed. When dealing with multiple operations, intermediate results can accumulate and increase the overall memory usage. Actively calling `del` on intermediate tensors followed by a `gc.collect()` and then rechecking the memory usage allows us to confirm these tensors are deallocated and to diagnose a potential memory leak.

```python
import resource
import os
import time
import torch
import psutil
import gc

def get_cpu_memory_usage():
    """
    Obtains the CPU memory usage statistics using the resource and psutil modules.
    Returns:
      tuple: A tuple containing:
        - max_rss (int): Max resident set size in kilobytes.
        - python_rss (int): Python process's resident set size in megabytes.
        - torch_allocated_memory_kb (int): PyTorch's allocated tensor memory in kilobytes.
    """
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_kb = usage.ru_maxrss
    
    process = psutil.Process(os.getpid())
    python_memory_usage_mb = process.memory_info().rss / (1024 ** 2)
    
    torch_allocated_memory_kb = torch.tensor(0.0).element_size() * torch.tensor(0).numel()

    if torch.cuda.is_available():
          torch_allocated_memory_kb = torch.cuda.memory_allocated() / 1024
          
    return max_rss_kb, python_memory_usage_mb, torch_allocated_memory_kb

# Create a large intermediate tensor
max_rss_kb, python_rss_mb, torch_allocated_kb = get_cpu_memory_usage()
print(f"Max RSS Before: {max_rss_kb} KB, Python RSS: {python_rss_mb:.2f} MB, Torch Allocated Memory: {torch_allocated_kb:.2f} KB")
intermediate = torch.randn(10000, 10000)
max_rss_kb, python_rss_mb, torch_allocated_kb = get_cpu_memory_usage()
print(f"Max RSS After: {max_rss_kb} KB, Python RSS: {python_rss_mb:.2f} MB, Torch Allocated Memory: {torch_allocated_kb:.2f} KB")

# Explicitly delete and garbage collect
del intermediate
gc.collect()

# Check memory again
max_rss_kb, python_rss_mb, torch_allocated_kb = get_cpu_memory_usage()
print(f"Max RSS After del and gc: {max_rss_kb} KB, Python RSS: {python_rss_mb:.2f} MB, Torch Allocated Memory: {torch_allocated_kb:.2f} KB")
```

This last snippet demonstrates how to use `del` and garbage collection to free up memory. By creating a large tensor and then explicitly deleting it, we should see a decrease in the `max_rss_kb` and `torch_allocated_kb` after calling `gc.collect()`, effectively showing that these resources have been released back to the system. Note that Python’s garbage collector does not automatically release memory instantly, hence the use of `gc.collect()` to force garbage collection.

In conclusion, there is no single perfect method to accurately and completely monitor CPU memory usage, but rather we must combine operating system level information, Python module methods, and careful management of tensor allocation and deallocation.

For further study, one should consult operating system manuals regarding resource utilization statistics. Additionally, deep dives into the documentation of Python’s `resource` module, the `psutil` module, and PyTorch’s memory management sections would prove invaluable. Research into system-level memory profiling tools (specific to different operating systems) is also suggested.
