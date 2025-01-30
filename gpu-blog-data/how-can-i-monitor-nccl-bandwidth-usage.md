---
title: "How can I monitor NCCL bandwidth usage?"
date: "2025-01-30"
id: "how-can-i-monitor-nccl-bandwidth-usage"
---
NCCL bandwidth monitoring requires a multifaceted approach due to its distributed nature and the hardware-specific performance characteristics it exhibits. My experience over the last five years optimizing large-scale deep learning models on multi-GPU clusters has demonstrated that no single tool offers complete visibility. Therefore, a combination of system-level tools, NCCL-specific environment variables, and custom profiling scripts provides the most effective monitoring strategy.

The core challenge in monitoring NCCL bandwidth lies in understanding that NCCL operations are asynchronous and often intertwined with other communication and computation tasks. Bandwidth utilization isn't a constant; it varies based on data sizes, communication patterns (e.g., all-reduce, all-gather), and the specific interconnect hardware in use (e.g., NVLink, InfiniBand). Therefore, passive observation isn’t sufficient; active profiling and analysis are essential for identifying bottlenecks.

First, understanding the underlying communication patterns is paramount. NCCL’s performance critically depends on how the data is distributed and gathered across GPUs. An imbalanced workload can lead to underutilization of certain GPUs and significant delays in communication. For example, a poorly configured all-reduce operation can result in a few GPUs waiting idly while one or two are heavily utilized with data accumulation.

Secondly, monitoring system-level resource utilization alongside NCCL activity is crucial. Over-saturated PCIe lanes, for instance, can create a bottleneck irrespective of NCCL configuration. High CPU load involved in data preparation and transfer can also significantly impact bandwidth. Tools like `nvidia-smi` and `htop` are invaluable for observing overall system resource usage. `nvidia-smi`, specifically, allows observation of GPU utilization, memory usage, and power consumption, which indirectly reflects the effectiveness of NCCL communication. High GPU utilization with low data movement is an indicator of a communication bottleneck that needs further investigation.

Third, environment variables can be manipulated to control logging output of NCCL’s internal operations, though be prepared for a significant volume of console output. The `NCCL_DEBUG` variable, for example, when set to `INFO` or a higher level like `TRACE`, will produce detailed information about NCCL activities and can point to low-bandwidth communication. Additionally, the variable `NCCL_DEBUG_FILE` is useful for storing this output to a file for later analysis rather than flooding the terminal.

However, these system tools and logging utilities provide only a partial picture. To gain deeper insights into NCCL bandwidth usage, I have found custom profiling scripts using the NCCL API and PyTorch hooks to be most reliable. Here are several example implementations based on my workflow.

**Code Example 1: Profiling NCCL All-Reduce Operations in PyTorch**

This Python script utilizes PyTorch hooks to measure the duration of all-reduce operations and calculate the effective bandwidth based on the data volume.

```python
import torch
import torch.distributed as dist
import time

def measure_all_reduce_bandwidth(data_size, iterations=10):
    """Measures bandwidth of all-reduce operation."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    tensor = torch.randn(data_size, device=rank, dtype=torch.float32)

    start_times = []
    end_times = []
    for _ in range(iterations):
      torch.cuda.synchronize()
      start = time.perf_counter()
      dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
      torch.cuda.synchronize()
      end = time.perf_counter()
      start_times.append(start)
      end_times.append(end)
    total_time = sum(end-start for start, end in zip(start_times, end_times)) / iterations
    total_bytes = data_size * 4 * dist.get_world_size()  # 4 bytes per float, all-reduced across all processes
    bandwidth = total_bytes / total_time
    print(f"Rank: {rank}, All-reduce bandwidth: {bandwidth/10**9:.2f} GB/s")

if __name__ == "__main__":
  data_size = 1024 * 1024 * 100 # 100 MB
  measure_all_reduce_bandwidth(data_size)
```

This script initializes distributed processing with NCCL, creates a random tensor, and then runs an all-reduce operation multiple times, synchronizing the GPU to make timing accurate.  The bandwidth is then calculated based on total data transferred and average time consumed across the iterations. The bandwidth value is in Gigabytes per second, with floating-point number format. This provides a specific measurement of the all-reduce operations bandwidth performance, and it should be run on each rank of the distributed setup.

**Code Example 2: Capturing NCCL Debug Output from Environment Variable in Python**

This script demonstrates how to use the `subprocess` library to execute Python code while setting `NCCL_DEBUG` and `NCCL_DEBUG_FILE` environment variables, effectively capturing the detailed NCCL output for debugging purposes.

```python
import subprocess
import os

def run_with_nccl_debug(script_path, log_file):
    """Runs a script with NCCL debugging enabled."""
    env = os.environ.copy()
    env["NCCL_DEBUG"] = "INFO"
    env["NCCL_DEBUG_FILE"] = log_file
    subprocess.run(["python", script_path], env=env, check=True)

if __name__ == "__main__":
    example_script = "example_dist_script.py" # assuming this exists
    log_file = "nccl_debug.log"
    run_with_nccl_debug(example_script, log_file)
    print(f"NCCL debug output written to: {log_file}")
```

This script is a wrapper that launches another Python script. Before launching it, it adds the `NCCL_DEBUG` and `NCCL_DEBUG_FILE` environment variables to the environment of the child process. This forces all NCCL operations within the launched script to output very verbose information, which is redirected to the specified log file. The information within the log can then be parsed for debugging purposes, providing insights into the NCCL configuration used and operations. It should be noted that this can result in large log files.

**Code Example 3: NCCL Bandwidth Monitoring using a PyTorch Training Hook**

This script integrates with the PyTorch training loop via a hook, monitoring the all-reduce operations in the backward pass of a deep learning model. This provides a more in-context understanding of where bandwidth bottlenecks might arise during the actual training process.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import time

class BandwidthMonitorHook:
  def __init__(self):
      self.start_time = None
      self.data_size = 0

  def pre_forward(self, module, input):
      pass

  def post_forward(self, module, input, output):
      pass

  def pre_backward(self, module, grad_output):
      if dist.is_initialized() and module.__class__.__name__ != "DataParallel" and any(p.requires_grad for p in module.parameters()):
        # Assuming gradients are the tensors being all-reduced, calculate data size and capture time
        param = next(p for p in module.parameters() if p.requires_grad)
        self.data_size = param.numel() * 4 * dist.get_world_size()
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()

  def post_backward(self, module, input, output):
    if dist.is_initialized() and self.start_time is not None and module.__class__.__name__ != "DataParallel":
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        duration = end_time - self.start_time
        bandwidth = self.data_size / duration
        rank = dist.get_rank()
        print(f"Rank: {rank}, module {module.__class__.__name__}, All-reduce bandwidth: {bandwidth/10**9:.2f} GB/s")
        self.start_time = None

if __name__ == "__main__":
    if not dist.is_initialized():
      dist.init_process_group(backend="nccl")
    device = dist.get_rank()
    model = nn.Linear(100, 10, bias=False).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    hook = BandwidthMonitorHook()

    for layer in model.modules():
        layer.register_forward_pre_hook(hook.pre_forward)
        layer.register_forward_hook(hook.post_forward)
        layer.register_backward_pre_hook(hook.pre_backward)
        layer.register_backward_hook(hook.post_backward)
    for _ in range(10):
        inputs = torch.randn(10, 100).to(device)
        target = torch.randn(10, 10).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.MSELoss()(outputs, target)
        loss.backward()
        optimizer.step()
```

This script registers a custom hook that intercepts forward and backward passes through the PyTorch model. In the backward pass, it estimates the data volume being passed through the all-reduce operation for the gradient updates and calculates the bandwidth based on its duration. This allows pinpointing where within the neural network bandwidth saturation happens. The script provides an idea of how to integrate such monitoring in a full deep learning pipeline.

For further reading, I recommend exploring the official NVIDIA documentation for NCCL.  Additionally, performance tuning guides for multi-GPU deep learning from various institutions and libraries offer valuable techniques for optimized data-parallel processing. Familiarizing oneself with CUDA profiling tools like Nsight systems will also offer additional detail into system and GPU utilization. Performance evaluation papers that target distributed deep learning systems can also be useful in identifying common bottleneck scenarios and mitigation strategies.
