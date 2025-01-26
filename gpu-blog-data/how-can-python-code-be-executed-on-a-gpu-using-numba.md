---
title: "How can Python code be executed on a GPU using Numba?"
date: "2025-01-26"
id: "how-can-python-code-be-executed-on-a-gpu-using-numba"
---

Numba's `@jit` decorator, when combined with CUDA or ROCm support, offers a pathway for accelerating Python code execution via the parallel processing capabilities of a GPU. The key is annotating the correct functions with specific Numba decorators, enabling the compiler to generate GPU-compatible code. The process involves transitioning from a standard, CPU-bound function to a parallelized GPU kernel.

My experience over the last few years with numerical simulations, particularly fluid dynamics, has heavily relied on this capability for performance. Simple loops and array manipulations that are prohibitively slow on the CPU become viable when offloaded to the GPU using Numba. The speedup is not always automatic; careful consideration of data layout, memory access patterns, and threading model is crucial.

The core mechanism lies in Numba’s just-in-time (JIT) compilation. By applying the `@jit` decorator, Numba analyzes the decorated function and, if possible, produces optimized machine code targeted at the specific hardware architecture – in this case, a GPU. When CUDA or ROCm is available and configured correctly, Numba automatically detects this and attempts GPU compilation, if designated. This process involves translating a subset of Python code into the target language, such as CUDA C or HIP, compiling it with the associated compiler toolchain (nvcc for CUDA, hipcc for ROCm), and then managing the execution on the GPU device. The result is a significantly accelerated version of the Python function, executed directly on the GPU hardware. This approach avoids explicit C/C++ programming and complex GPU API management, retaining the productivity of Python.

The transition often requires some adjustment in thinking compared to purely CPU code. For instance, operations that are implicitly vectorized in NumPy require explicit management of thread indices in Numba kernels running on GPUs. Data must also be moved to the GPU’s memory before the kernel execution and retrieved afterwards. These transfers constitute overheads that are important to minimize. Numba assists by implicitly performing data movement for NumPy arrays, making it significantly easier to develop these GPU kernels.

Here are three concrete examples illustrating GPU execution with Numba:

**Example 1: Simple Array Addition**

```python
import numpy as np
from numba import cuda

@cuda.jit
def gpu_add(x, y, out):
    idx = cuda.grid(1)
    out[idx] = x[idx] + y[idx]

n = 1024
x = np.arange(n, dtype=np.float32)
y = np.arange(n, dtype=np.float32)
out = np.empty_like(x)

threadsperblock = 128
blockspergrid = (n + (threadsperblock - 1)) // threadsperblock

gpu_add[blockspergrid, threadsperblock](x,y,out)

print(f"First 5 elements of output: {out[:5]}")
```

In this example, `cuda.jit` designates `gpu_add` as a function to run on the GPU. The `cuda.grid(1)` function returns a unique index for each thread created during execution. We use this index to perform the element-wise addition of `x` and `y` and write the result into `out`. Before launching the kernel, we must determine the grid and block dimensions, ensuring we cover the entire range of the input arrays, accounting for the number of threads in each block. The call `gpu_add[blockspergrid, threadsperblock](x,y,out)` launches the GPU kernel with these specified parameters. This example showcases a basic parallelizable operation suitable for GPU processing. The output array contains the sums of the corresponding elements of the input arrays.

**Example 2: Matrix Multiplication**

```python
import numpy as np
from numba import cuda

@cuda.jit
def gpu_matmul(A, B, C):
  row = cuda.grid(0)
  col = cuda.grid(1)

  if row < C.shape[0] and col < C.shape[1]:
    tmp_sum = 0.0
    for k in range(A.shape[1]):
      tmp_sum += A[row,k] * B[k,col]
    C[row, col] = tmp_sum

n = 256
A = np.random.rand(n, n).astype(np.float32)
B = np.random.rand(n, n).astype(np.float32)
C = np.empty((n,n), dtype=np.float32)

threadsperblock = (16,16)
blockspergrid_x = (C.shape[0] + (threadsperblock[0] - 1)) // threadsperblock[0]
blockspergrid_y = (C.shape[1] + (threadsperblock[1] - 1)) // threadsperblock[1]

blockspergrid = (blockspergrid_x, blockspergrid_y)
gpu_matmul[blockspergrid,threadsperblock](A,B,C)

print(f"Top left 2x2 of result: \n{C[:2,:2]}")
```

This second example demonstrates matrix multiplication. This function operates on two-dimensional grids, using `cuda.grid(0)` and `cuda.grid(1)` to get the row and column indices, respectively. Each thread calculates one element of the resulting matrix, `C`. The inner `for` loop performs the dot product for the given element. Similar to the previous example, grid and block dimensions need careful setup to ensure all matrix elements are calculated. The output shows the top-left 2x2 portion of the resulting matrix.

**Example 3: Reduction Operation (Summation)**

```python
import numpy as np
from numba import cuda

@cuda.jit
def gpu_reduce(in_array, out_array):
    thread_id = cuda.grid(1)
    stride = cuda.gridsize(1)

    temp_sum = 0.0
    for i in range(thread_id, in_array.shape[0], stride):
        temp_sum += in_array[i]

    cuda.atomic.add(out_array, 0, temp_sum)

n = 1024
in_array = np.arange(n, dtype=np.float32)
out_array = np.zeros(1, dtype=np.float32)


threadsperblock = 256
blockspergrid = (n + (threadsperblock - 1)) // threadsperblock
gpu_reduce[blockspergrid, threadsperblock](in_array, out_array)


print(f"Sum of array elements: {out_array[0]}")
```

The final example showcases a reduction operation – in this instance, summation. Each thread computes the partial sum of a segment of the input array. The key here is the use of `cuda.atomic.add`. This function ensures that multiple threads can safely add their partial results to a shared output location (`out_array`) without race conditions. `cuda.gridsize(1)` provides the total number of threads in the grid, which is utilized to calculate the stride for each thread. The final output is a single number, representing the sum of all elements in the input array. This demonstrates a common pattern when applying GPUs to reduce operations.

When debugging, Numba's error messages can be particularly helpful. Errors during JIT compilation often stem from using Python features that are not yet supported within the CUDA or ROCm backend. A clear understanding of the underlying hardware architecture, the threading model, and memory access patterns is beneficial for optimizing performance. Furthermore, profiling tools can reveal bottlenecks and guide code optimization.

For further learning and practical implementation, consider exploring the documentation and tutorials offered by the Numba project directly. Consult resources specific to CUDA programming for deeper insight into the underlying CUDA model and its performance characteristics. While these examples are relatively basic, they form a foundation for more complex GPU-accelerated applications in numerical computation, scientific computing, and data analysis. Additionally, I recommend investigating best practices for data movement and memory access within the GPU architecture for improved performance in your specific domain. The documentation of the relevant GPU architectures (NVIDIA’s CUDA programming guide, or AMD’s ROCm documentation) provides crucial information for optimizing kernel performance. Books on parallel computing can also provide a theoretical framework for understanding the parallel concepts that are key to GPU programming.
