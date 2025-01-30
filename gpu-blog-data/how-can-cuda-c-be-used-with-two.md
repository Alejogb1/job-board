---
title: "How can CUDA C be used with two video cards?"
date: "2025-01-30"
id: "how-can-cuda-c-be-used-with-two"
---
Multi-GPU programming in CUDA C requires a nuanced understanding of data transfer mechanisms and kernel launch strategies.  My experience developing high-performance computing applications for geophysical modeling has underscored the critical role of efficient inter-GPU communication in achieving substantial speedups with multiple cards.  Simply assigning tasks to different GPUs is insufficient; careful consideration of memory access patterns and communication overhead is paramount.

**1.  Understanding CUDA's Multi-GPU Paradigm:**

CUDA's multi-GPU support relies on the concept of a *context*.  Each GPU has its own context, which manages its memory space, streams, and kernels.  To utilize two GPUs, you'll need to create separate contexts, one for each.  However, merely creating separate contexts doesn't automatically distribute work; you must explicitly manage data transfer between the GPUs and synchronize execution across them.  Naive approaches, such as independent kernel launches on each GPU without inter-GPU communication, may lead to minimal or no performance gains and, in certain cases, even performance degradation due to increased overhead.

There are several methods for achieving multi-GPU computation. The most common approaches involve using either peer-to-peer (P2P) memory access or CUDA's unified virtual addressing (UVA) along with explicit data transfer via CUDA streams and events.  P2P allows direct memory access between GPUs without going through the host CPU, leading to lower latency compared to CPU-mediated transfers.  However, P2P access requires driver-level support, and its availability depends on the GPUs and their interconnection. UVA simplifies memory management but might introduce more overhead than P2P for large datasets.

**2.  Code Examples with Commentary:**

**Example 1:  P2P Memory Access (assuming P2P is enabled):**

```c++
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  // Get GPU device IDs
  int devCount;
  cudaGetDeviceCount(&devCount);
  if (devCount < 2) {
    printf("Error: Requires at least two GPUs\n");
    return 1;
  }
  int devID1 = 0;
  int devID2 = 1;

  // Check for P2P access
  int canAccessPeer;
  cudaDeviceCanAccessPeer(&canAccessPeer, devID1, devID2);
  if (!canAccessPeer) {
    printf("Error: P2P access not available\n");
    return 1;
  }
  cudaDeviceEnablePeerAccess(devID2, 0); // Enable P2P access


  // Allocate memory on each GPU
  float *d_data1, *d_data2;
  cudaMalloc((void**)&d_data1, 1024 * sizeof(float));
  cudaSetDevice(devID2);
  cudaMalloc((void**)&d_data2, 1024 * sizeof(float));

  // ... (Kernel launches on GPU 0 and GPU 1, accessing d_data1 and d_data2 using appropriate device IDs)...


  // ... (Copy results back to host)...

  cudaFree(d_data1);
  cudaFree(d_data2);

  return 0;
}

__global__ void kernel1(float *data){
  // Access and process data residing on the device
}
```

This example demonstrates the crucial steps: retrieving GPU device IDs, verifying P2P access, enabling it, and allocating memory on each device separately.  Note that the kernels (not shown fully for brevity) launched on each GPU will explicitly use the appropriate `d_data` pointers.  The actual kernel operations depend entirely on the computation task.  Error checking is essential in all CUDA code, as shown here, to handle potential hardware limitations.

**Example 2:  CUDA Streams and Events for Asynchronous Data Transfer:**

```c++
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
  // ... (GPU initialization and memory allocation as in Example 1, but without P2P)...

  cudaStream_t stream1, stream2;
  cudaStreamCreate(&stream1);
  cudaStreamCreate(&stream2);

  cudaEvent_t event;
  cudaEventCreate(&event);


  //Launch kernel 1 on GPU 0 asynchronously
  kernel1<<<...>>>(d_data1, stream1);

  // Copy data from GPU 0 to GPU 1 asynchronously
  cudaMemcpyAsync(d_data2, d_data1, 1024 * sizeof(float), cudaMemcpyDeviceToDevice, stream1);

  // Record event after copy completes
  cudaEventRecord(event, stream1);

  // Wait for event, ensuring data is copied before kernel 2 starts
  cudaEventSynchronize(event);

  //Launch kernel 2 on GPU 1
  kernel2<<<...>>>(d_data2, stream2);


  // ... (Clean up streams and events)...

  return 0;
}
```

This illustrates asynchronous data transfer using CUDA streams and events.  `cudaMemcpyAsync` copies data between GPUs without blocking execution.  `cudaEventRecord` and `cudaEventSynchronize` are used for synchronization, ensuring kernel 2 on GPU 1 only begins after the data transfer from GPU 0 is complete.  This approach is essential for maximizing GPU utilization and avoiding idle time.


**Example 3:  Using CUDA streams and UVA (for simplicity, assuming memory allocation is managed efficiently by UVA):**

```c++
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    // ... (GPU initialization and memory allocation using UVA, potentially simpler than explicit device allocation)...

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    // Launch kernel 1 on GPU 0 asynchronously, accessing data via UVA
    kernel1<<<...,stream1>>>(d_data);

    // Launch kernel 2 on GPU 1 asynchronously, accessing the same data via UVA
    kernel2<<<...,stream2>>>(d_data);

    //synchronize both streams
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    // ... (Clean up streams and events)...

    return 0;
}
```

This example shows a simplified scenario where UVA simplifies memory management. Both kernels access the same data, potentially resulting in less code, but careful management of data dependencies is still critical to avoid race conditions.  The simplicity is deceptive; successful utilization of UVA relies heavily on effective memory management and preventing data conflicts.


**3. Resource Recommendations:**

The NVIDIA CUDA Toolkit documentation is the definitive resource.  Furthermore, consult books on high-performance computing and parallel programming focusing on GPU architectures.  A strong understanding of linear algebra and parallel algorithms is crucial for optimizing CUDA code.  Finally, profiling tools within the CUDA Toolkit are essential for identifying performance bottlenecks and refining your code for optimal efficiency.  Thorough testing across various hardware configurations is vital for ensuring robust and portable performance.
