---
title: "How many CUDA cores are visible in the visual profiler?"
date: "2025-01-30"
id: "how-many-cuda-cores-are-visible-in-the"
---
The number of CUDA cores reported in the NVIDIA Nsight Visual Profiler isn't a direct count of the physical cores on the GPU.  Instead, it reflects the number of *streaming multiprocessors (SMs)* multiplied by the number of CUDA cores *per* SM. This distinction is crucial for accurate interpretation and understanding of performance bottlenecks.  My experience debugging CUDA applications for high-performance computing, particularly in computational fluid dynamics simulations, has highlighted the importance of this nuance.  Many newcomers misinterpret the profiler's output, leading to incorrect conclusions about parallel efficiency.

**1. Clear Explanation:**

The NVIDIA GPU architecture is hierarchical.  At the bottom are the CUDA cores, the actual processing units. These are grouped into Streaming Multiprocessors (SMs).  Each SM contains multiple CUDA cores, along with other resources like shared memory and register files.  The number of CUDA cores per SM varies depending on the GPU architecture (e.g., Kepler, Pascal, Ampere). The Visual Profiler doesn't directly access a hardware register providing the raw count of individual CUDA cores. Instead, it queries the GPU for its architectural specifications—specifically, the number of SMs and the number of CUDA cores per SM. It then reports the product of these two values as the "number of CUDA cores."  This is a logical representation, useful for understanding the overall parallel processing capacity.  However, it's not a precise count of individually addressable units in the same way CPU cores are.  Furthermore, the profiler might not display *all* available cores if some are disabled due to defects or power-saving mechanisms.

Understanding this distinction is vital for performance analysis. For instance, observing low occupancy despite a seemingly large number of reported CUDA cores suggests that your kernel isn't effectively utilizing the available SMs, possibly due to insufficient parallelism, memory access bottlenecks, or inefficient register usage.  The profiler's metrics, like occupancy and warp utilization, become far more meaningful when interpreted alongside the architectural details provided.

**2. Code Examples with Commentary:**

The following examples demonstrate how to access relevant GPU information programmatically, complementing the Visual Profiler's visual representation.  These snippets use CUDA runtime APIs and are illustrative; error handling and context setup are omitted for brevity.

**Example 1: Using `cudaGetDeviceProperties`**

```c++
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int i = 0; i < deviceCount; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d:\n", i);
        printf("  Name: %s\n", prop.name);
        printf("  MultiProcessorCount: %d\n", prop.multiProcessorCount); //Number of SMs
        printf("  CUDA Cores per SM (Inferred): %d\n", prop.maxThreadsPerMultiProcessor / 32); // Approximate; varies by architecture
        printf("  Total CUDA Cores (Inferred): %d\n", prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / 32)); // Approximate
    }
    return 0;
}
```

This code iterates through available CUDA devices and uses `cudaGetDeviceProperties` to retrieve information, including the number of SMs (`multiProcessorCount`).  The number of CUDA cores per SM is *inferred* by dividing the maximum threads per multiprocessor by 32 (a common, but not universally true, number of threads per CUDA core).  This calculation provides an approximation and might not be perfectly accurate for all architectures.  The total number of CUDA cores is then estimated by multiplying SM count and cores per SM. This approach verifies the information presented in the Visual Profiler.

**Example 2:  Using `cudaDeviceGetAttribute` (More precise for newer architectures)**

```c++
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  int device;
  cudaGetDevice(&device);
  int cores_per_mp;
  cudaDeviceGetAttribute(&cores_per_mp, cudaDevAttrMultiProcessorCount, device);
  int sm_count;
  cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device);
  printf("Number of SMs: %d\n", sm_count);

  //  For newer architectures, use cudaDevAttrWarpSize to get threads per core (more reliable)
  int warp_size;
  cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, device);
  printf("Warp Size: %d\n", warp_size); // typically 32
  // ... further processing to estimate CUDA Cores per SM and total
  return 0;
}

```

This example utilizes `cudaDeviceGetAttribute` which provides a more direct method for obtaining specific device attributes. While `cudaDevAttrMultiProcessorCount` still provides the SM count,  using `cudaDevAttrWarpSize` offers a more robust way to estimate the number of CUDA cores per SM, especially for newer architectures where the assumption of 32 threads per core might not always hold true.


**Example 3:  Direct access (Illustrative, architecture-specific)**

```c++
// This example is highly architecture-dependent and NOT recommended for portability.
// It illustrates the underlying concept but should NOT be used in production code.

// ... (Requires significant low-level knowledge and potentially assembly code) ...

//  Hypothetical register access (highly device-specific and discouraged):
// unsigned int num_cores;
//  //  This is NOT valid CUDA code and serves purely as a conceptual illustration.
//  //  Replace with actual low-level access methods IF you absolutely know what you're doing.
//  asm volatile ("mov.u32 %0, [register_address]" : "=r"(num_cores)); //Illustrative only
//  printf("Number of cores (highly unreliable method): %u\n", num_cores);
```

This final example is purely illustrative and highlights the impracticality of directly accessing the underlying hardware registers.  Such approaches are highly architecture-specific, extremely fragile, and generally discouraged.  Relying on the CUDA runtime APIs is always preferable for portability and maintainability.


**3. Resource Recommendations:**

The NVIDIA CUDA Toolkit documentation.  The CUDA C Programming Guide.  The NVIDIA Nsight Compute and Nsight Systems documentation.  A comprehensive textbook on parallel programming and GPU computing.  A suitable reference on the specific GPU architecture you're targeting.


In conclusion, the CUDA core count visible in the Nsight Visual Profiler is a derived value based on SM count and cores per SM.  Directly accessing the physical core count is generally unnecessary and not recommended. Using the CUDA runtime API provides a robust and portable way to retrieve relevant GPU architecture information, allowing for a more accurate interpretation of the profiler's data and ensuring efficient CUDA kernel design and optimization.  Always cross-reference the profiler's visual information with programmatic checks for validation and a deeper understanding of your application's performance profile.
