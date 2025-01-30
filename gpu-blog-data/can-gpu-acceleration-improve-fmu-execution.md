---
title: "Can GPU acceleration improve FMU execution?"
date: "2025-01-30"
id: "can-gpu-acceleration-improve-fmu-execution"
---
The significant computational demands of Functional Mock-up Unit (FMU) co-simulation, particularly for complex systems, often present a performance bottleneck. GPU acceleration offers a potential solution by leveraging massively parallel architectures to expedite these calculations. FMUs, inherently serial in their input-output processing at the core algorithm, typically involve repeated matrix operations and function evaluations where parallelization can be beneficial. I’ve encountered this optimization challenge frequently while working on real-time hardware-in-the-loop simulations, where FMU execution speed was the limiting factor in achieving required cycle times.

The core limitation in direct FMU execution on a GPU stems from the standard FMU interface definition, which assumes a step-by-step sequential execution model. The primary 'doStep' function, responsible for advancing the FMU state at each time instance, relies on results from the previous step and therefore limits the potential for massive parallelization at the level of the entire FMU. However, many of the *internal* computations within an FMU, especially those involving linear algebra, numerical solvers, and function evaluations, can be reformulated to exploit GPU parallelism.  It's crucial to identify which segments of the FMU's internal computations are most computationally intensive and parallelizable to determine where GPU acceleration will be most impactful.

The process isn’t simply a matter of executing the FMU "as is" on a GPU; instead, a hybrid approach is necessary.  We need to offload specific computationally expensive parts of the FMU calculation to the GPU while the rest of the FMU control flow remains on the CPU. I’ve consistently found that focusing on the numerical computations within the FMU provides the most significant performance increase.  Specifically, matrix-vector operations, especially those within implicit solvers, and computationally expensive function calls are excellent candidates for acceleration. The data transfer between CPU and GPU also plays a crucial role in overall performance. Frequent data transfers can negate the performance benefits of the GPU if not managed carefully.

To illustrate the concepts with code, consider a simple scenario where an FMU requires solving a linear system of equations during each simulation step. Assuming that the coefficient matrix A and the vector b are known, the solution x is found by solving Ax = b. The most commonly used implementation on the CPU is LU decomposition. This can be significantly sped up by using GPU based linear algebra libraries. The following pseudo-code outlines the standard CPU implementation. Note that the `doStep` function is part of the FMU’s core interface, which cannot itself be easily parallelized.

```python
# CPU Implementation
import numpy as np
import time

def solve_linear_system_cpu(A, b):
    # Simulates a linear system solver on the CPU
    x = np.linalg.solve(A, b)
    return x

def doStep_cpu(time, h, inputs, states, derivatives):
   # Simplified FMU-like step function
   A = np.random.rand(1000, 1000) # Example System matrix. FMU would manage this
   b = np.random.rand(1000)       # Example right-hand vector. FMU would manage this
   # ...  Update states/derivatives for FMU here using CPU solve result ...
   start_time = time.time()
   x = solve_linear_system_cpu(A, b)
   end_time = time.time()
   # ...
   return x, end_time - start_time # returns the result plus timing info

# Example run
num_steps = 10
total_time_cpu = 0
for i in range(num_steps):
  result, step_time = doStep_cpu(i, 0.1, None, None, None)
  total_time_cpu += step_time
print(f"Total time CPU: {total_time_cpu:.4f} seconds")
```

This example simulates a computationally expensive step within the FMU, the linear system solve.  The `doStep_cpu` function emulates a call to the FMU's primary step function and includes a `solve_linear_system_cpu` function, acting as a placeholder for potentially intensive computations within the FMU.

Now, the same computational kernel can be offloaded to a GPU.  Here, I'm utilizing a high-level framework for demonstration, but the underlying principles are consistent across most GPU libraries.

```python
# GPU Implementation (using CuPy)
import cupy as cp
import numpy as np
import time

def solve_linear_system_gpu(A, b):
    # Simulates a linear system solver on the GPU
    x = cp.linalg.solve(cp.asarray(A), cp.asarray(b))
    return x.get() # Copy results back to CPU

def doStep_gpu(time, h, inputs, states, derivatives):
    # Simplified FMU-like step function with GPU solver
    A = np.random.rand(1000, 1000) # Example system matrix
    b = np.random.rand(1000)       # Example right-hand vector
    # ...  Update states/derivatives for FMU here using GPU solve result ...
    start_time = time.time()
    x = solve_linear_system_gpu(A, b)
    end_time = time.time()
    #...
    return x, end_time - start_time # returns result plus timing

# Example run
num_steps = 10
total_time_gpu = 0
for i in range(num_steps):
   result, step_time = doStep_gpu(i, 0.1, None, None, None)
   total_time_gpu += step_time
print(f"Total time GPU: {total_time_gpu:.4f} seconds")
```

This GPU-accelerated version leverages the CuPy library.  The linear algebra computations are performed on the GPU using `cp.linalg.solve`. Crucially, we convert the numpy arrays to cupy arrays prior to the solver operation to move the data to the GPU. The final `.get()` command copies the result back to the CPU. This introduces data transfer overhead, which, while less in this example than the computation saving, can be a limitation in some FMU scenarios where transfer is a significant overhead. We need to take into consideration the overhead from the data transfer when implementing GPU acceleration.

Lastly, to further emphasize the potential for performance gains within the FMU implementation, consider the possibility of accelerating custom functions or function evaluations that may be called multiple times in each simulation step.  The code below provides a simple example of a computationally expensive custom function call using the Numba library for CUDA based acceleration:

```python
# GPU Implementation (using Numba)
from numba import cuda
import numpy as np
import time

@cuda.jit
def custom_function_gpu(input_array, output_array):
    idx = cuda.grid(1)
    if idx < input_array.size:
        output_array[idx] = np.sin(input_array[idx])**2  # Example compute-intensive function

def solve_custom_function_gpu(input_array):
    d_input = cuda.to_device(input_array) # Copy to GPU device memory
    d_output = cuda.device_array_like(d_input)  # allocate memory on the device.
    threadsperblock = 32
    blockspergrid = (input_array.size + (threadsperblock - 1)) // threadsperblock
    custom_function_gpu[blockspergrid, threadsperblock](d_input, d_output) # Launch kernel
    return d_output.copy_to_host() # copy back to the CPU.

def doStep_custom_gpu(time, h, inputs, states, derivatives):
    # Simplified FMU-like step function with GPU-accelerated custom function
    input_array = np.random.rand(1000000) # example input array
    start_time = time.time()
    x = solve_custom_function_gpu(input_array)
    end_time = time.time()
    #...
    return x, end_time-start_time


# Example run
num_steps = 10
total_time_custom_gpu = 0
for i in range(num_steps):
  result, step_time = doStep_custom_gpu(i, 0.1, None, None, None)
  total_time_custom_gpu += step_time
print(f"Total time custom GPU: {total_time_custom_gpu:.4f} seconds")
```

Here, we use the Numba library with CUDA capabilities, defining a kernel `custom_function_gpu` for parallel computation. A large input array is generated, and the execution time is measured for the GPU version. This example emphasizes that custom functions can also be a large computation bottleneck in the FMU computation cycle and are amenable to GPU acceleration.

In all the provided examples, the CPU continues to manage the overall FMU execution flow, while the computationally intensive parts are offloaded to the GPU. This hybrid approach, while not perfect, usually allows for significant performance gains over a pure CPU implementation.  The choice between GPU libraries such as CuPy, Numba, or a lower-level approach (e.g., CUDA C++) depends heavily on the specifics of the FMU, the complexity of computations, and the level of control needed over the GPU operations.

For further study, consider exploring resources that delve into GPU programming, specifically those pertaining to numerical methods and linear algebra.  Materials covering high-level Python libraries for GPU programming (such as CuPy, Numba) can prove invaluable. Additionally, focusing on literature related to sparse matrix operations and implicit solvers on GPUs will provide deeper insight, as these are frequent components in simulations. It’s important to also learn about CPU-GPU data management, to minimize performance bottlenecks during data transfers. Also consider literature on hybrid approaches for accelerating co-simulations. Examining open-source scientific libraries for linear algebra will offer a more in-depth view of optimization techniques. The key to successful GPU acceleration lies in understanding the FMU's internal operations and strategically selecting which computations to offload, balancing computation gains against the costs of data transfers.
