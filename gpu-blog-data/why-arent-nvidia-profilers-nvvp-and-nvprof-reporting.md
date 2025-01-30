---
title: "Why aren't NVIDIA profilers (nvvp and nvprof) reporting page fault information?"
date: "2025-01-30"
id: "why-arent-nvidia-profilers-nvvp-and-nvprof-reporting"
---
NVIDIA profilers, specifically `nvprof` and its successor, `nvvp` (NVIDIA Visual Profiler), do not directly report page fault information because their primary function revolves around analyzing GPU activity and performance within the CUDA execution environment. Page faults, however, originate from the operating system's virtual memory management system when a program attempts to access a memory page that isn't currently loaded into physical RAM.  These are generally considered a system-level concern, outside the direct purview of the GPU and its driver.  My experience, over several years optimizing CUDA kernels for high-performance computing applications, has consistently underscored this distinction. I've encountered scenarios where seemingly inexplicable slowdowns in GPU execution traced back, not to inefficient kernel code, but to excessive page faulting impacting the data loading pipeline before the data even reached the device.

The core functionality of `nvprof` and `nvvp` is to trace and analyze CUDA API calls, kernel execution times, memory transfers to and from the GPU, and other aspects of GPU hardware utilization. This tracing process is implemented via callbacks within the CUDA driver and runtime environment. These profilers inject instrumentation into the CUDA calls and kernel launches, capturing timestamps and performance metrics directly related to the GPU. Page faults, while potentially impacting overall application performance by delaying data availability, occur outside the execution flow these profilers monitor. The operating system handles page fault resolution transparently to the GPU, and these resolution activities do not typically trigger CUDA API calls that the profilers would instrument. The profilers collect data from the CUDA API, which is insulated from the lower-level virtual memory management. Therefore, the profilers are effectively "blind" to page faults. It is a design choice that prioritizes detailed GPU execution analysis at the expense of this more broad system view.  It is also important to note that the GPU device memory is considered "pinned" and not virtual in that sense. Therefore, a device memory allocation will not cause a page fault and therefore it is not seen by the profiler.

Furthermore, the granularity of data collection contributes to this limitation. `nvprof` and `nvvp` focus on CUDA-specific events occurring at a relatively fine-grained timescale, such as individual kernel execution or memory copy operations. Page faults, conversely, are managed by the operating system kernel, occur asynchronously relative to the CUDA execution flow, and can involve complex disk I/O. Capturing this additional, lower-level data with these tools would require fundamentally altering their architecture and might significantly impact their ability to provide detailed, high-frequency GPU-centric profiling information, leading to potential overhead and obfuscation of the GPU performance metrics.  The profilers do monitor Host to Device transfers, and any slowdown there could be an indication of paging, but the profilers do not pinpoint that.

In practical terms, the absence of page fault information means that when I observe a CPU-to-GPU data transfer or a kernel launch taking unexpectedly long,  I cannot directly use `nvprof` or `nvvp` to determine if page faulting on the CPU side is the cause. This necessitates a multi-faceted diagnostic approach. This generally entails system-level monitoring tools to track page fault activity concurrently with GPU performance data, using techniques like system tracing to capture the lower-level context.

To illustrate, consider a simplified scenario where we are loading an array of floats from host memory to the GPU.

```cpp
#include <cuda_runtime.h>
#include <iostream>

#define N (1024 * 1024)
int main() {
  float *h_data = new float[N];
  for (int i = 0; i < N; ++i) {
    h_data[i] = static_cast<float>(i);
  }
  float *d_data;
  cudaMalloc((void **)&d_data, N * sizeof(float));
  cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaFree(d_data);
  delete[] h_data;
  return 0;
}
```

This code allocates and initializes a host array, then transfers it to the GPU. If this process, especially the `cudaMemcpy` call, were noticeably slow, `nvprof` or `nvvp` would indicate the time spent in the data transfer operation. However, if page faulting on `h_data` is the underlying issue, such information would not be provided directly. The profiler would show an extended time spent in the host to device copy but would not reveal if page faults on the host were the cause of that increase. The resolution would require separate system analysis tools.

Now let us look at an example of GPU work with no paging problem:

```cpp
#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        c[i] = a[i] + b[i];
}

int main() {
  int n = 1024*1024;
  float *a, *b, *c;

    cudaMallocManaged(&a, n * sizeof(float));
    cudaMallocManaged(&b, n * sizeof(float));
    cudaMallocManaged(&c, n * sizeof(float));
    
    for(int i = 0; i < n; ++i)
    {
        a[i] = (float)i;
        b[i] = (float)(i*2);
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);
    cudaDeviceSynchronize();


    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}
```

This code performs a vector addition on the GPU. It uses `cudaMallocManaged`, meaning that it is allocated in the unified memory model. It is possible that page faulting could be occurring with the use of unified memory, but again the profilers will not point that out specifically. The profilers will show the kernel execution times, memory reads, writes and device memory bandwidth utilization, but if there is a slowdown due to paging, that is not seen.

Finally let us look at a code that could be impacted by page faulting.

```cpp
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <numeric>


__global__ void kernel(float *output, int size){
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < size)
  {
    output[idx] = 1.0f;
  }
}


int main() {
  int size = 1024*1024*64;
    std::vector<float> host_data(size);
    std::iota(host_data.begin(), host_data.end(), 0.0f); // Initialize with sequential values

    float *device_data;
    cudaMalloc((void**)&device_data, size * sizeof(float));


    cudaMemcpy(device_data, host_data.data(), size * sizeof(float), cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    kernel<<<blocksPerGrid, threadsPerBlock>>>(device_data, size);
    cudaDeviceSynchronize();
    
    std::vector<float> output(size);
    cudaMemcpy(output.data(), device_data, size * sizeof(float), cudaMemcpyDeviceToHost);


    cudaFree(device_data);


  return 0;
}
```

In this example, there is an allocation of a large array on the host and then it is transferred to the device. If the host memory was not recently accessed, this allocation can cause many page faults. Profilers would not directly tell you that, but this can be observed with other tools.  It may show a long host-to-device transfer.

To address the issue of missing page fault information, I have found that OS level profiling tools are critical. For Linux systems, `perf` and `strace` are valuable. They enable observation of system-level calls, including the frequency of page fault events. For Windows, tools like Windows Performance Analyzer provide similar system-level tracing capabilities. Further, examining overall system performance using utilities like `top`, `htop` or task manager is helpful. These tools provide a system-wide view, showing not just GPU activity but also other processes, memory usage and CPU utilization. These tools don't provide specific context to the CUDA runtime so they should be used in conjunction with CUDA profiling. This often provides an explanation why a data transfer or device synchronization appears to be taking so long. Books focusing on operating system concepts and memory management have provided valuable insights, particularly those focusing on virtual memory and paging algorithms, helping to understand the root causes and mitigate these types of issues. A proper understanding of the interplay between CPU host memory and GPU device memory is crucial for optimizing CUDA code.
