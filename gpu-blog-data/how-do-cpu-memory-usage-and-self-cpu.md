---
title: "How do CPU memory usage and self CPU memory usage differ in PyTorch profiling?"
date: "2025-01-30"
id: "how-do-cpu-memory-usage-and-self-cpu"
---
Within PyTorch’s profiling tools, specifically when examining memory consumption, the distinction between “CPU memory usage” and “self CPU memory usage” is critical for accurate performance analysis and optimization. My experience, having spent several years optimizing deep learning models on various hardware configurations, underscores that misunderstanding this difference frequently leads to misdirected debugging and inefficient code modifications. The “CPU memory usage” encompasses the total RAM allocated on the host machine during an operation or function's execution, whereas “self CPU memory usage” represents the memory solely attributed to the profiled operation or function itself, excluding memory allocated by any child calls.

Let’s delve into this with an example. Consider a complex convolutional neural network. During a forward pass, a convolutional layer (`nn.Conv2d`) might allocate memory for its weight matrices and intermediate feature maps. This memory allocation contributes to both the “CPU memory usage” and the “self CPU memory usage” of the convolutional layer operation. However, this convolutional layer also likely relies on other lower-level operations such as matrix multiplication or tensor manipulation. These subordinate operations may, in turn, allocate their own memory. The “CPU memory usage” will reflect the sum of all these memory allocations. The “self CPU memory usage”, in contrast, isolates the direct allocation performed within the `nn.Conv2d` forward method, not including allocations from the lower-level operations triggered by it. This separation allows for isolating the specific layers or operations that are most memory intensive and aids in focused optimization efforts.

To solidify this understanding, consider these scenarios in a more hands-on fashion. Below are Python code examples utilizing PyTorch’s `torch.profiler` and `torch.utils.checkpoint`, with commentaries on their respective memory usage outputs.

**Example 1: Basic Tensor Creation and Operation**

```python
import torch
import torch.profiler as profiler

with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    with profiler.record_function("tensor_creation_op"):
        a = torch.rand(1000, 1000)  # Direct allocation
        b = torch.rand(1000, 1000)
        c = a + b                 # Additional allocation for the result
    with profiler.record_function("tensor_op_2"):
        d = torch.matmul(a, b.T)

print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
```

In this first example, we create two random tensors `a` and `b`, then add them, assigning the result to `c`, and then perform a matrix multiplication. The profiler output, when sorted by “cpu_memory_usage,” will show the combined memory usage for the “tensor\_creation\_op” including the memory allocated for `a`, `b` and `c`. Sorting by “self\_cpu\_memory\_usage,” it will isolate the memory directly allocated in the `torch.rand` calls inside the context of “tensor\_creation\_op”. Similarly, the memory from inside matmul operation is included in the "cpu_memory_usage" of the “tensor\_op\_2” whereas the “self\_cpu\_memory\_usage” will isolate the allocation of the output tensor `d`. It’s important to note that the "+" operator also involves an allocation for the result `c`, which gets included in the total "cpu\_memory\_usage", but doesn't get counted under the "self\_cpu\_memory\_usage". This demonstrates that seemingly simple operations can have substantial underlying allocations, a reality that becomes critical when dealing with more complex computations.

**Example 2: Function Calls and Child Operations**

```python
import torch
import torch.profiler as profiler

def inner_function(tensor_in):
  with profiler.record_function("inner_op"):
      return torch.relu(tensor_in) # Child operation

def outer_function(size):
    with profiler.record_function("outer_op"):
        tensor = torch.randn(size, size)
        result = inner_function(tensor)
        return result


with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    size_param = 1000
    output = outer_function(size_param)

print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
```

In this second example, we define two functions: `outer_function`, which generates a random tensor and calls `inner_function`, which applies the ReLU activation. “cpu\_memory\_usage” will reflect the total memory consumed by both, including the allocation of the random tensor in "outer\_op" and the activation and return operation in "inner\_op" . The “self\_cpu\_memory\_usage” for "outer\_op" will primarily reflect the memory allocated for the tensor within it, and similarly for "inner\_op" will isolate the memory specifically within it. It will not reflect any memory allocations from its children functions. This hierarchical structure frequently encountered in deep learning models highlights why analyzing self memory is so essential when pinpointing specific bottlenecks. The `inner_function`'s contribution to the total memory usage can be directly examined by `self_cpu_memory_usage`, while the `cpu_memory_usage` of `outer_function` will include memory from its call to the `inner_function`

**Example 3: Memory Checkpointing with `torch.utils.checkpoint`**

```python
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.profiler as profiler


class ResidualBlock(nn.Module):
    def __init__(self, channels):
      super().__init__()
      self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
      self.relu = nn.ReLU()
      self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
      residual = x
      x = self.conv1(x)
      x = self.relu(x)
      x = self.conv2(x)
      return x + residual


class Model(nn.Module):
    def __init__(self, channels, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(channels) for _ in range(num_blocks)])

    def forward(self, x):
        for i, block in enumerate(self.blocks):
          x = checkpoint.checkpoint(block, x)
        return x

with profiler.profile(profile_memory=True, record_shapes=True) as prof:
    model = Model(32, 4)
    dummy_input = torch.randn(1, 32, 64, 64)
    output = model(dummy_input)

print(prof.key_averages().table(sort_by="cpu_memory_usage", row_limit=10))
print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
```

In this example, we employ memory checkpointing via `torch.utils.checkpoint`. The `checkpoint.checkpoint` function recomputes portions of the computational graph during the backward pass, reducing memory footprint at the expense of increased computation time. The "cpu\_memory\_usage" will show the total cumulative memory usage including the memory allocated inside the checkpointed blocks. When sorted by “self\_cpu\_memory\_usage”, the profiler reveals the memory allocated directly within the checkpointing function itself, not including the memory from inside the residual block's `forward` calls. This demonstrates how checkpointing functions, while beneficial for overall memory reduction, can also contribute their own memory footprint and is something that needs to be considered when tuning for performance. The impact of this difference in reporting can be significant. During forward passes of models with multiple levels of checkpointing, for example, the differences between the two measurements will reveal where the most benefit can be found using this technique.

From these examples and my experience, profiling CPU memory usage without distinguishing between total and self usage can be deceiving, leading to misguided optimization efforts. When troubleshooting memory problems, I primarily utilize these two metrics in conjunction, allowing a targeted approach for code changes. The “cpu\_memory\_usage” helps identify overall memory bottlenecks while the “self\_cpu\_memory\_usage” pinpoints the code segments responsible for direct allocations.

For further learning, I recommend examining PyTorch’s official documentation on the `torch.profiler` module, specifically sections on memory profiling and the interpretation of its various metrics. Additionally, reviewing research papers that utilize these profiler metrics and the underlying design rationale behind checkpointing can provide deeper insight. Textbooks that detail memory management techniques in deep learning frameworks can also be valuable. A good grasp of operating system level memory management will certainly benefit those trying to deep dive into performance bottlenecks.
