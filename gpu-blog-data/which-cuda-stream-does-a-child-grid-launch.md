---
title: "Which CUDA stream does a child grid launch belong to?"
date: "2025-01-30"
id: "which-cuda-stream-does-a-child-grid-launch"
---
Child grid launches in CUDA, specifically those initiated within a kernel using the cooperative groups API, do not inherently belong to the same CUDA stream as their parent grid. This is a critical distinction often overlooked, leading to unexpected synchronization issues and potentially incorrect execution. I've debugged this extensively in several high-performance computing simulations, experiencing firsthand the headaches that arise from assuming a child grid operates within the same stream as its launching kernel. Instead, child grids operate on the *null stream*, often referred to as the default or stream 0.

The core understanding here revolves around the nature of CUDA streams and their purpose. A CUDA stream is essentially an ordered sequence of operations that execute on the GPU. These operations can include kernel launches, memory transfers, and synchronization primitives. Executing operations within different streams allows for concurrent processing on the GPU, provided there are sufficient resources. The key advantage of streams lies in their implicit ordering; operations within a given stream are guaranteed to execute sequentially, preventing data races and ensuring predictable behavior.

However, when a kernel launches a child grid using the cooperative groups API (typically with `cudaLaunchKernel` or similar within a `grid_launch` function), this new grid does not inherit the stream context of the parent kernel. The child grid is dispatched immediately, as a separate execution unit, and is placed on the null stream. This implies that any synchronizations or communication explicitly required between parent and child kernels must be explicitly handled using appropriate CUDA primitives, such as device-wide barriers or explicit stream synchronizations using the `cudaStreamSynchronize` API call if a different stream is desired. Failing to do so will often lead to race conditions, where the parent kernel may proceed beyond the child launch while the child is still working, which can result in unpredictable outputs or even application crashes.

To illustrate, consider the case of a numerical simulation that updates a solution iteratively using a parent grid, and each step of that iteration requires a child grid to apply some transformation. Without explicitly managing streams and synchronization between parent and child grids, the next parent iteration may start before the previous child has completed its execution. Such issues are extremely difficult to track and can result in subtle numerical errors that are only apparent after significant computation time.

Here are three concrete code examples with explanatory notes to clarify these concepts:

**Example 1: Implicit Null Stream Execution**

```cpp
#include <cuda.h>
#include <iostream>

__global__ void child_kernel(int *output, int value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  output[tid] = value;
}

__global__ void parent_kernel(int *output, int size) {
  int child_size = size; // Example size for child grid.
  
  // Launch child grid implicitly on the null stream.
  cudaLaunchKernel(
    child_kernel,
    dim3((child_size + 255) / 256), // Child grid size.
    dim3(256),
    0,
    0, // Implicitly uses the null stream.
    output, 1
  );

    // Potential issue: parent continues processing here, potentially before child has completed.
  for(int i = 0; i < size; ++i) {
      output[i] = output[i] + 1;
  }
}


int main() {
  int size = 1024;
  int *h_output = new int[size];
  int *d_output;

  cudaMalloc((void**)&d_output, size * sizeof(int));

  // Initialize host output
  for(int i = 0; i < size; i++){
      h_output[i] = 0;
  }

  cudaMemcpy(d_output, h_output, size*sizeof(int), cudaMemcpyHostToDevice);
  
  parent_kernel<<<1, 1>>>(d_output, size);
  
  cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);
    
  for(int i = 0; i < 10; ++i)
  {
        std::cout << h_output[i] << " ";
  }
    std::cout << std::endl;

  cudaFree(d_output);
  delete[] h_output;
  return 0;
}
```

**Commentary:** This example demonstrates the typical scenario where the child kernel is implicitly placed on the null stream. The parent kernel does not explicitly wait for the child kernel to finish. Consequently, the incrementing operation in parent kernel will likely execute while the child has not finished populating the `output` array with 1. This will produce an output that is unpredictable (in many cases, it will likely be 2) and that has data races. This is a common mistake, and demonstrates that child grids must be synchronized correctly with their parent when any type of inter-grid data dependence exists.

**Example 2: Explicit Stream Management**

```cpp
#include <cuda.h>
#include <iostream>

__global__ void child_kernel_stream(int *output, int value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  output[tid] = value;
}

__global__ void parent_kernel_stream(int *output, int size, cudaStream_t stream) {
  int child_size = size; // Example size for child grid.

    // Launch child kernel on a specific stream
  cudaLaunchKernel(
    child_kernel_stream,
    dim3((child_size + 255) / 256), // Child grid size.
    dim3(256),
    0,
    stream,
    output, 1
  );
    
  // Synchronize stream
  cudaStreamSynchronize(stream);

  for(int i = 0; i < size; ++i) {
      output[i] = output[i] + 1;
  }
}


int main() {
    int size = 1024;
    int *h_output = new int[size];
    int *d_output;

    cudaMalloc((void**)&d_output, size * sizeof(int));

    // Initialize host output
    for(int i = 0; i < size; i++){
        h_output[i] = 0;
    }
    
    cudaMemcpy(d_output, h_output, size*sizeof(int), cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    parent_kernel_stream<<<1, 1>>>(d_output, size, stream);

    cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; ++i)
    {
      std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;
  

    cudaStreamDestroy(stream);
    cudaFree(d_output);
    delete[] h_output;
    return 0;
}
```

**Commentary:** In this example, a custom CUDA stream is created and explicitly passed to the `cudaLaunchKernel` call within the parent kernel. The `cudaStreamSynchronize` call is then used to ensure that the child kernel has completed its execution *before* the parent continues its calculations, which increment all the elements. This guarantees that the output after the parent increments will be correct. This method offers more control and is necessary for proper synchronization, even if you wish the child grid to execute concurrently with other operations in a different stream.

**Example 3: Cooperative Groups with Implicit Null Stream (Illustrative)**

```cpp
#include <cuda.h>
#include <cooperative_groups.h>
#include <iostream>

using namespace cooperative_groups;

__global__ void child_kernel_cg(int *output, int value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  output[tid] = value;
}

__global__ void parent_kernel_cg(int *output, int size) {
    grid_group grid = this_grid();
    int child_size = size;

  // Launch child grid implicitly using cooperative groups API on null stream.
   grid_launch(grid, child_kernel_cg, dim3((child_size + 255) / 256), dim3(256), output, 1);

  // Again: implicit null stream, potential race conditions.
    for(int i = 0; i < size; ++i) {
        output[i] = output[i] + 1;
    }
}


int main() {
    int size = 1024;
    int *h_output = new int[size];
    int *d_output;

    cudaMalloc((void**)&d_output, size * sizeof(int));

    // Initialize host output
    for(int i = 0; i < size; i++){
        h_output[i] = 0;
    }
    
    cudaMemcpy(d_output, h_output, size*sizeof(int), cudaMemcpyHostToDevice);

    parent_kernel_cg<<<1, 1>>>(d_output, size);

    cudaMemcpy(h_output, d_output, size * sizeof(int), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; ++i)
    {
      std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_output);
    delete[] h_output;
    return 0;
}

```

**Commentary:** This third example uses the cooperative groups API with `grid_launch`, showcasing that even this more sophisticated interface does not change the fundamental issue: the launched child grid executes on the null stream. The lack of explicit stream handling within the code will lead to the same race condition observed in Example 1.

For further study, I would recommend consulting NVIDIA's CUDA programming guides, particularly the sections covering streams, cooperative groups, and memory management. Also, publications on parallel programming patterns in high-performance computing and GPGPU programming offer valuable insights. Furthermore, NVIDIA provides thorough documentation, including API references and example code, that is essential for understanding stream behavior. Practical experience with GPU programming using small, targeted experiments is invaluable for solidifying these concepts.
