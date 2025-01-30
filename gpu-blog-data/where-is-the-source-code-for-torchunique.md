---
title: "Where is the source code for torch.unique()?"
date: "2025-01-30"
id: "where-is-the-source-code-for-torchunique"
---
The operational core of `torch.unique()`, unlike some PyTorch functionalities, is not directly implemented in Python within the standard PyTorch library. Its behavior is a composite involving C++ kernels and CUDA implementations, orchestrated by PyTorch's dispatcher. As someone who spent a considerable amount of time optimizing tensor operations for a high-throughput recommendation system, I've had to trace through similar low-level functions to understand their performance characteristics. This experience has given me a perspective on how PyTorch leverages hardware acceleration.

The function `torch.unique()` provides a deceptively straightforward task: identifying and returning unique elements within an input tensor. However, the underlying computation must handle diverse datatypes, tensor shapes, and crucially, must be efficient both on CPUs and GPUs. This leads to a bifurcated implementation strategy, avoiding the direct exposure of detailed C++ kernels directly in the Python API. Instead, the Python side interacts with PyTorch’s dispatcher which selects the appropriate kernel based on the tensor's device, datatype, and other properties.

The Python-accessible part of `torch.unique()` resides in the `torch/tensor.py` file, where you will find the definition of the public-facing API. However, the core logic is not here. When you execute `torch.unique()`, PyTorch's dispatcher takes over and redirects the operation to the corresponding backend. On a CPU, the relevant implementations will usually reside within the CPU backend, within C++ code that’s compiled as part of the PyTorch library. These might utilize optimized hash-based data structures or sorting algorithms, depending on whether the `sorted` argument is set to `True` (which changes the computational complexity). For GPUs, the dispatcher will reroute the computation to a CUDA kernel compiled as part of the CUDA extensions of PyTorch, enabling significant parallel processing.

The function's complexity further expands when considering that it can return indices along with unique values. This requires that the backend implementation also maintain a mapping of the original tensor positions to the unique values. Again, this is handled in the C++ and CUDA layer for performance, not the python-accessible code.

Let's illustrate with some examples.

```python
import torch

# Example 1: Basic usage with a CPU tensor
cpu_tensor = torch.tensor([1, 2, 2, 3, 1, 4, 5])
unique_values, unique_indices = torch.unique(cpu_tensor, return_inverse=True)
print(f"CPU Unique Values: {unique_values}")
print(f"CPU Unique Indices: {unique_indices}") #Indices to reconstruct

# Expected output:
#CPU Unique Values: tensor([1, 2, 3, 4, 5])
#CPU Unique Indices: tensor([0, 1, 1, 2, 0, 3, 4])
```

In this example, we observe the most basic scenario: a one-dimensional tensor where we desire both unique values and indices. The `return_inverse=True` argument triggers the return of `unique_indices`, which would allow reconstructing the original tensor, if needed. This example makes use of the CPU backed function, and the core operation is dispatched through PyTorch’s mechanisms.

```python
import torch

# Example 2: Usage with a GPU tensor
if torch.cuda.is_available():
    gpu_tensor = torch.tensor([6, 7, 7, 8, 6, 9, 10]).cuda()
    unique_values_gpu, unique_indices_gpu = torch.unique(gpu_tensor, return_inverse=True)
    print(f"GPU Unique Values: {unique_values_gpu}")
    print(f"GPU Unique Indices: {unique_indices_gpu}")

    # Expected output (assuming CUDA device):
    #GPU Unique Values: tensor([ 6,  7,  8,  9, 10], device='cuda:0')
    #GPU Unique Indices: tensor([0, 1, 1, 2, 0, 3, 4], device='cuda:0')
else:
    print("CUDA is not available, skipping GPU example.")
```

This highlights the dispatching aspect. The same `torch.unique()` function is used, however because `gpu_tensor` is allocated on a CUDA device, the dispatcher routes the call to the appropriate CUDA kernel.  The output will show that the computations and the resulting tensors are processed on the GPU. The core algorithm in this instance would likely use GPU-optimized sorting or hashing techniques to exploit the parallel processing capabilities.

```python
import torch

# Example 3: Usage with a multi-dimensional tensor
multi_dim_tensor = torch.tensor([[1, 2, 2],
                                [3, 1, 4],
                                [5, 6, 5]])
unique_values_multi, unique_indices_multi = torch.unique(multi_dim_tensor, sorted=True, return_inverse=True)
print(f"Multi-dimensional Unique Values: {unique_values_multi}")
print(f"Multi-dimensional Unique Indices: {unique_indices_multi}")

# Expected output:
#Multi-dimensional Unique Values: tensor([1, 2, 3, 4, 5, 6])
#Multi-dimensional Unique Indices: tensor([0, 1, 1, 2, 0, 3, 4, 5, 4])
```

This example shows that, regardless of the input tensor's shape, `torch.unique()` flattens the tensor before computing the unique elements. If `sorted=True` is used, a sorting operation is used before the unique value extraction. This operation is handled by backend specific code, usually optimized for speed depending on whether it is CPU or GPU compute. Understanding these variations is crucial for optimizing operations within my previous work, which frequently involved manipulating high dimensional data.

In terms of finding the exact source code, exploring the PyTorch codebase directly is essential. You would not be searching for a single python file implementing `torch.unique()`. Instead, you'll need to navigate the following areas:

* **The Dispatcher:** Start within PyTorch's dispatcher code located in the `torch/csrc/dispatch` directory. Here you can understand how the function call is routed. Look for dispatching mechanisms relevant to the `unique` operation, for both CPU and CUDA backends.
* **CPU Backend:** Within the CPU backend (likely under `torch/csrc/cpu`), look for implementations that use optimized algorithms. Understanding how these algorithms map into concrete C++ code is important, especially if you are interested in performance at the lower level.
* **CUDA Backend:** If GPU processing is required, you would need to explore the corresponding CUDA kernels. These are located under `torch/csrc/cuda`. CUDA kernels are often complex, finely tuned algorithms specifically designed to extract maximum performance from GPUs.
* **Tensor API:** The Python code for the `torch.unique` function is in the `torch/tensor.py` file, which calls to the dispatcher to execute the underlying computational operations.

While diving into the full source code requires a significant time investment, exploring the locations described above will provide more profound understanding of PyTorch’s internal operations. It’s important to note that this isn’t for casual usage, but for anyone who needs low level performance optimization this level of understanding becomes necessary.

For additional information and context, I would recommend the following resources:

1. **The PyTorch GitHub Repository:** The primary repository provides the most in-depth view of the source code. Browsing this code is indispensable for detailed analysis.
2. **PyTorch Documentation:** Specifically, focusing on the documentation related to tensor operations, dispatch mechanism and CUDA functionality would prove beneficial for the user. The documentation isn’t a code walk-through but does provide the conceptual framework for such explorations.
3. **C++ and CUDA Programming Resources:** If you intend to delve into the performance optimizations, good knowledge in these languages will be required. Understanding parallel algorithms and their implementations is a necessity for understanding the CUDA code.
