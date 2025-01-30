---
title: "How can NVTX markers be visualized in Nsight Systems on a single Windows machine?"
date: "2025-01-30"
id: "how-can-nvtx-markers-be-visualized-in-nsight"
---
The core challenge in visualizing NVTX markers within Nsight Systems on a single Windows machine lies in ensuring proper instrumentation of your application and the correct configuration of Nsight Systems to capture and display this data.  My experience profiling CUDA applications over the past five years has highlighted the importance of meticulous setup.  Incorrectly placed markers, misconfigured profiling sessions, or driver issues can readily lead to seemingly invisible markers, despite their presence in the application code.

**1. Clear Explanation:**

NVTX (NVIDIA Tools Extension) markers are lightweight annotations inserted into your code to demarcate specific regions of interest.  These regions, representing sections of computation or data transfer, can be subsequently visualized within performance analysis tools like Nsight Systems. The visualization allows for granular performance examination, enabling pinpointing bottlenecks in CUDA kernels, memory copies, or CPU-side operations.  However, the successful visualization hinges on several interconnected factors.

First, the NVTX library must be correctly integrated into your project.  This typically involves linking against the necessary libraries and including the appropriate header files.  Second, the markers themselves need to be placed strategically within the code to capture meaningful performance events.  Third, Nsight Systems needs to be configured to capture NVTX data during the profiling session, which necessitates selecting the appropriate profiling configuration and ensuring that the NVTX instrumentation is enabled.  Finally, careful attention must be paid to the scope of the profiling session, particularly if working with multi-threaded or multi-process applications. Incorrect scope can lead to the omission of critical marker data.


**2. Code Examples with Commentary:**

**Example 1: Basic Marker Placement in CUDA Kernel:**

```cpp
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.h>

__global__ void myKernel(int *data, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    nvtxRangePushA(0x0000FF00, "Kernel Computation"); // Green colored range
    // Perform kernel computation
    data[i] *= 2;
    nvtxRangePop();
  }
}

int main() {
  // ... CUDA memory allocation and data initialization ...

  nvtxRangePushA(0x0000FFFF, "Kernel Launch"); // Yellow colored range

  myKernel<<<blocks, threads>>>(dev_data, data_size);

  cudaDeviceSynchronize(); //Crucial for accurate timing

  nvtxRangePop();

  // ... CUDA memory deallocation and result processing ...
  return 0;
}
```

This example demonstrates the basic usage of `nvtxRangePushA` and `nvtxRangePop` within a CUDA kernel.  `nvtxRangePushA` takes a 32-bit color value (ARGB format) and a string identifier for the marker. The color allows for visual differentiation of various events.  `nvtxRangePop()` marks the end of the timed section. The use of `cudaDeviceSynchronize()` ensures that the kernel completes before the `nvtxRangePop()` call, preventing inaccurate timing data.  The main function also uses NVTX ranges to demarcate the kernel launch.


**Example 2:  Markers for CPU-side Operations:**

```cpp
#include <nvtx3/nvtx3.h>
#include <iostream>
#include <vector>

int main() {
  std::vector<int> host_data(1024*1024);
  nvtxRangePushA(0xFF000000, "CPU Data Initialization"); // Black colored range
  for (int i = 0; i < host_data.size(); ++i) {
    host_data[i] = i;
  }
  nvtxRangePop();

  // ... further CPU operations ...

  nvtxRangePushA(0x00FF0000, "CPU Data Processing"); // Red colored range
  // ... CPU intensive data processing ...
  nvtxRangePop();

  return 0;
}
```

This illustrates marking CPU-side operations.  This is crucial because bottlenecks may exist outside of the GPU code.  Similar to the CUDA example, the color-coded ranges allow for easy visual identification of different phases within the application.


**Example 3:  Using NVTX with Multiple Kernels:**

```cpp
#include <cuda_runtime.h>
#include <nvtx3/nvtx3.h>

__global__ void kernel1(int *data, int size) {
    nvtxRangePushA(0x00FF0000, "Kernel 1");
    // ...kernel 1 operations...
    nvtxRangePop();
}

__global__ void kernel2(int *data, int size) {
    nvtxRangePushA(0x0000FF00, "Kernel 2");
    // ...kernel 2 operations...
    nvtxRangePop();
}

int main() {
  // ... CUDA setup ...

  nvtxRangePushA(0xFFFF0000, "Kernel Launch Sequence");
  kernel1<<<blocks1, threads1>>>(dev_data, size);
  cudaDeviceSynchronize();
  kernel2<<<blocks2, threads2>>>(dev_data, size);
  cudaDeviceSynchronize();
  nvtxRangePop();

  // ... CUDA cleanup ...
  return 0;
}
```

This example showcases the marking of multiple kernels, essential for understanding the relative performance and dependencies between different stages of GPU computation. Each kernel has its own colored range, allowing Nsight Systems to separate their execution times for analysis. The outer range tracks the entire sequence.


**3. Resource Recommendations:**

The NVIDIA Nsight Systems documentation, including the user guide and tutorials, provides invaluable information on setup and interpretation. The CUDA C++ Programming Guide offers a comprehensive overview of CUDA programming concepts relevant to effective marker placement.  Finally, consulting NVIDIA's official CUDA samples repository can provide practical examples showcasing best practices in NVTX integration.  Careful examination of the provided error messages within Nsight Systems is also crucial for diagnosing potential issues.  Understanding how to read and interpret the profiling output, including the call stacks and timing data, will be beneficial.
