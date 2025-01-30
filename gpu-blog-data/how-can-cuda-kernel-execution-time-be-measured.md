---
title: "How can CUDA kernel execution time be measured?"
date: "2025-01-30"
id: "how-can-cuda-kernel-execution-time-be-measured"
---
CUDA kernel performance analysis is critically dependent on accurate measurement of execution time. Achieving this requires understanding the nuances of CUDA's asynchronous model and choosing the appropriate tools. I've spent considerable time optimizing CUDA kernels in my work, and through that experience, I’ve found that the most reliable method involves leveraging CUDA events. Directly timing host-side code blocks calling kernel launches is often inadequate, as the CPU initiates the kernel launch asynchronously and doesn't typically wait for kernel completion. This approach will only measure launch overhead, not the kernel execution time itself.

The core challenge lies in the fact that kernel launches are added to a command queue, and the host proceeds without waiting for the device to complete the requested computation. To measure the actual time spent on the GPU, one must insert markers or signals that can be tracked on the host. CUDA events, specifically `cudaEvent_t`, serve precisely this purpose. These events act as timestamps, marking points in the GPU's execution stream.

Here's the fundamental process: You create two events, one to record the start of the kernel execution and another to record its end. Prior to launching the kernel, you record the start event. After launching the kernel, you record the end event. Importantly, the host needs to synchronize with the GPU to ensure that the end event is indeed after the kernel has finished. This synchronization is crucial for obtaining accurate timing data. Subsequently, you calculate the time difference between the two events. This difference then provides the time taken for the kernel execution, excluding the overhead of the kernel launch itself.

Synchronization is achieved by either forcing the host to wait for the device to finish all preceding commands with `cudaDeviceSynchronize()` or by waiting on a particular event with `cudaEventSynchronize()`. Utilizing `cudaDeviceSynchronize()` synchronizes the entire device, which can be less efficient if you only need to wait for a specific kernel. `cudaEventSynchronize()` allows you to wait only on the specific event that signals the completion of your target kernel.

Let's look at some code examples demonstrating this technique.

**Example 1: Basic Kernel Timing with Events**

This example shows the basic structure for measuring kernel execution time using CUDA events.

```cpp
#include <cuda.h>
#include <iostream>

__global__ void simpleKernel(float* data, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        data[i] *= 2.0f;
    }
}

int main() {
  int size = 1024;
  size_t byteSize = size * sizeof(float);
  float* h_data = new float[size];
  float* d_data;

  // Initialize host data
  for (int i=0; i<size; ++i) {
      h_data[i] = (float)i;
  }

  // Allocate device memory
  cudaMalloc(&d_data, byteSize);
  cudaMemcpy(d_data, h_data, byteSize, cudaMemcpyHostToDevice);

  // Create CUDA events
  cudaEvent_t startEvent, stopEvent;
  cudaEventCreate(&startEvent);
  cudaEventCreate(&stopEvent);


  // Record start event
  cudaEventRecord(startEvent, 0);

  // Launch kernel
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
  simpleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);

  // Record stop event
  cudaEventRecord(stopEvent, 0);


  // Synchronize on stop event
  cudaEventSynchronize(stopEvent);


  // Calculate elapsed time
  float elapsedTime;
  cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

  std::cout << "Kernel execution time: " << elapsedTime << " ms" << std::endl;

  // Cleanup
  cudaFree(d_data);
  cudaEventDestroy(startEvent);
  cudaEventDestroy(stopEvent);
  delete[] h_data;
  return 0;
}
```

In this example, we first create events using `cudaEventCreate`. Then, before and after the kernel launch, the events are recorded using `cudaEventRecord`. The key part is `cudaEventSynchronize(stopEvent)`, where the host thread waits until the stop event has been recorded on the GPU. Finally, the elapsed time is obtained using `cudaEventElapsedTime`. It's important to clean up allocated resources including the events, preventing memory leaks. Note that the last argument to `cudaEventRecord` is the stream where the events are recorded; a value of `0` indicates the default stream.

**Example 2: Kernel Timing with Separate Streams**

This example demonstrates recording events in separate CUDA streams. While the previous example works fine for basic cases, asynchrony in CUDA often benefits from using multiple streams.

```cpp
#include <cuda.h>
#include <iostream>

__global__ void complexKernel(float* a, float* b, int size) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    b[i] = sqrt(a[i] * a[i] + 1.0f);
  }
}
int main() {
    int size = 1024 * 1024;
    size_t byteSize = size * sizeof(float);
    float* h_a = new float[size];
    float* h_b = new float[size];
    float* d_a;
    float* d_b;

    for (int i=0; i<size; ++i) {
        h_a[i] = (float)(i + 1);
    }


    cudaMalloc(&d_a, byteSize);
    cudaMalloc(&d_b, byteSize);

    cudaMemcpy(d_a, h_a, byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, byteSize, cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);


    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);


    cudaEventRecord(startEvent, stream);

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    complexKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_a, d_b, size);


    cudaEventRecord(stopEvent, stream);


    cudaEventSynchronize(stopEvent);


    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    std::cout << "Kernel execution time with stream: " << elapsedTime << " ms" << std::endl;


    cudaFree(d_a);
    cudaFree(d_b);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    cudaStreamDestroy(stream);
    delete[] h_a;
    delete[] h_b;
    return 0;
}
```

In this version, a separate stream is created with `cudaStreamCreate`.  The kernel launch and the event recordings use the same specified stream.  This ensures the events and kernel are queued in the specific order in that stream, providing correct timing. Streams enable concurrent execution of different operations, so proper stream management is essential for accurate timing. Failing to specify the correct stream when recording or synchronizing will lead to incorrect time measurements.

**Example 3: Timing Multiple Kernel Launches with Single Start/Stop Events**

Here, we demonstrate timing multiple kernel executions in sequence. This is useful when you want to measure the combined time of a sequence of operations.

```cpp
#include <cuda.h>
#include <iostream>
#include <vector>

__global__ void smallKernel(float* data, int size) {
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < size) {
       data[i] += 1.0f;
   }
}

int main() {
    int size = 1024;
    size_t byteSize = size * sizeof(float);
    float* h_data = new float[size];
    float* d_data;

    for(int i = 0; i<size; i++){
        h_data[i] = (float)i;
    }

    cudaMalloc(&d_data, byteSize);
    cudaMemcpy(d_data, h_data, byteSize, cudaMemcpyHostToDevice);
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);


    cudaEventRecord(startEvent, 0);


    int numKernels = 5;
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    for(int i=0; i< numKernels; ++i){
         smallKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, size);
    }

    cudaEventRecord(stopEvent, 0);


    cudaEventSynchronize(stopEvent);


    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, startEvent, stopEvent);

    std::cout << "Total Execution Time of "<< numKernels << " kernels: " << elapsedTime << " ms" << std::endl;


    cudaFree(d_data);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
    delete[] h_data;
    return 0;
}
```

In this final example, we encapsulate the execution of five `smallKernel` calls within a single measurement using events. Notice that the start event is recorded before the loop begins, and the stop event is recorded after the last kernel launch within the loop. The measured time therefore includes the combined time for all five kernel executions. This approach is useful when examining the aggregate performance of multiple kernel launches in a sequence, where individual timing would be less informative.

For further study, the following resources are recommended: CUDA programming guides produced by NVIDIA, as well as books and online articles that cover performance analysis techniques for CUDA code. These often include detailed information on the various CUDA profiling tools available and their specific use cases. The CUDA Toolkit documentation is the definitive resource for API-specific details and best practices. Thoroughly familiarizing oneself with these materials is essential for developing and optimizing high-performance CUDA applications. Experimenting with different kernel configurations and timing them systematically is the most effective strategy for learning the practical nuances of CUDA performance measurement.
