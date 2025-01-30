---
title: "How does CPU memory copy speed compare to GPU memory copy speed?"
date: "2025-01-30"
id: "how-does-cpu-memory-copy-speed-compare-to"
---
Direct memory access (DMA) substantially influences the performance characteristics of memory copies on CPUs and GPUs. Specifically, the architecture of each processor and the mechanisms they employ for data transfer result in vastly different copy speeds. My experience optimizing compute pipelines for high-performance image processing has highlighted this contrast repeatedly; CPU-based copies often appear as a bottleneck when GPUs are actively processing the same data.

The key distinction lies in their fundamental design. CPUs are general-purpose processors optimized for latency; they excel at executing a variety of instructions sequentially. Memory access is handled via the system bus, a relatively high-latency communication channel shared with numerous peripherals and components. Data is typically copied word by word by the CPU core, moving data from one memory location to another through a load-store operation. This process involves multiple clock cycles and is inherently serial, which limits copy throughput, particularly when dealing with large datasets. The system bus’s bandwidth is also a limitation in these cases. When a CPU executes a memory copy, it becomes the primary task, leaving less capacity for other tasks. Moreover, cache mechanisms designed to improve performance with frequently accessed data don’t typically aid in large, single-use copy operations.

Conversely, GPUs are massively parallel processors designed for throughput-oriented tasks. They employ numerous processing cores capable of executing the same instruction simultaneously on different data. GPU memory copies often leverage dedicated hardware, like a DMA engine, that operates independently of the primary GPU cores. This DMA engine moves large blocks of data in parallel across high-bandwidth memory interfaces, like GDDR6, with limited CPU intervention after the initiation. As such, the GPU’s copy operation can be significantly faster for large blocks, approaching its theoretical memory bandwidth limit, as opposed to a serial operation constrained by CPU’s architecture and bus. This is because the GPU is designed to maximize data transfer throughput and utilizes specialized hardware and parallel techniques to achieve that.

The architectural differences in accessing main system memory vs. dedicated GPU memory further influence copy speeds. CPU cores usually access main system RAM through a memory controller integrated into the CPU or motherboard. This is a single pathway shared among different cores and components, subject to contention from various memory access requests. GPUs, however, often operate with dedicated memory, connected via a high-speed bus, often with a direct path to the processor. These dedicated memory buses have far greater bandwidth than main system memory connections. Consequently, data transfers between GPU memory and its processing units are often faster than the CPU accessing its RAM. This is a design consideration intended for performance as GPU workloads tend to be memory intensive. However, when a GPU needs to access system memory, this process is significantly slower as the transfer must transit through the system bus, which impacts transfer performance. This is often the case with `memcpy` operations when the source or destination is CPU-accessible system memory, rather than GPU memory. This leads to notable bottlenecks, making careful allocation management crucial for high-performance computing.

Let’s consider three examples to illustrate these principles. These are conceptual examples and may not fully reflect the intricacies of vendor-specific implementations, but are useful for general understanding.

**Example 1: CPU Copy (C++)**

```cpp
#include <iostream>
#include <chrono>
#include <vector>
#include <cstring>

void cpu_copy(size_t size_bytes) {
  std::vector<char> src(size_bytes, 'A');
  std::vector<char> dst(size_bytes);

  auto start = std::chrono::high_resolution_clock::now();
  std::memcpy(dst.data(), src.data(), size_bytes);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "CPU memcpy: " << duration.count() << " microseconds for " << size_bytes << " bytes." << std::endl;
}

int main() {
    cpu_copy(1024*1024);  // 1 MB copy
    cpu_copy(10 * 1024 * 1024); // 10 MB copy
    cpu_copy(100 * 1024 * 1024);  // 100 MB copy
    return 0;
}

```
This C++ example demonstrates a conventional CPU-based memory copy utilizing the `memcpy` function, which is typically optimized for general-purpose processing. Here, `memcpy` takes three parameters: destination buffer, source buffer, and the number of bytes to copy. The CPU core performs this operation word-by-word, sequentially moving the data. When we increase the size of the data, we observe a near linear increase in time taken, due to the sequential nature of the process.

**Example 2: GPU Copy within GPU Memory (CUDA)**

```cpp
#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

void gpu_copy(size_t size_bytes) {
  char *d_src, *d_dst;

  cudaMalloc((void**)&d_src, size_bytes);
  cudaMalloc((void**)&d_dst, size_bytes);

  cudaMemset(d_src, 'A', size_bytes);  // Initialize source memory on GPU

  auto start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_dst, d_src, size_bytes, cudaMemcpyDeviceToDevice);
  auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "GPU DeviceToDevice memcpy: " << duration.count() << " microseconds for " << size_bytes << " bytes." << std::endl;

  cudaFree(d_src);
  cudaFree(d_dst);
}


int main(){
    gpu_copy(1024*1024); //1 MB Copy
    gpu_copy(10 * 1024 * 1024); // 10 MB Copy
    gpu_copy(100 * 1024 * 1024); // 100 MB Copy
    return 0;
}
```
This CUDA example illustrates a memory copy within the GPU’s dedicated memory using `cudaMemcpy`. Notice that `cudaMemcpy` also takes three parameters: destination buffer, source buffer, and the number of bytes to copy. The additional fourth parameter is the copy type, set to `cudaMemcpyDeviceToDevice` in this example. This operation benefits from dedicated DMA hardware within the GPU. In my experience, when memory is allocated entirely within the GPU, copying between two locations there tends to be significantly faster than the equivalent CPU based copy. The bandwidth advantages are very apparent with large datasets.

**Example 3: GPU Copy from Host to Device (CUDA)**

```cpp
#include <iostream>
#include <chrono>
#include <vector>
#include <cuda_runtime.h>

void gpu_host_copy(size_t size_bytes) {
  std::vector<char> h_src(size_bytes, 'A');
  char *d_dst;

  cudaMalloc((void**)&d_dst, size_bytes);

  auto start = std::chrono::high_resolution_clock::now();
  cudaMemcpy(d_dst, h_src.data(), size_bytes, cudaMemcpyHostToDevice);
    auto end = std::chrono::high_resolution_clock::now();

  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  std::cout << "GPU HostToDevice memcpy: " << duration.count() << " microseconds for " << size_bytes << " bytes." << std::endl;

  cudaFree(d_dst);
}

int main(){
    gpu_host_copy(1024*1024); //1 MB Copy
    gpu_host_copy(10 * 1024 * 1024); // 10 MB Copy
    gpu_host_copy(100 * 1024 * 1024); // 100 MB Copy
    return 0;
}

```

Here we have a CUDA example demonstrating a memory copy from host (CPU) memory to GPU memory. The fourth parameter of `cudaMemcpy` is set to `cudaMemcpyHostToDevice`. We must use the host memory directly, rather than an intermediate variable on the CPU side. This example highlights the impact of the PCI-Express bus, which forms the path for data transfers between the CPU and GPU. Because of this bottleneck, copying from CPU memory to GPU memory is often slower compared to device-to-device copies, and generally still much faster than CPU-to-CPU copies for large datasets, but not as fast as intra-GPU transfers. The system bus becomes the primary limitation as the transfer must go through that route.

For further study, I recommend reviewing publications on computer architecture focusing on memory hierarchies and bus systems. Exploring vendor-specific documentation for CPUs and GPUs, such as Intel’s manuals and NVIDIA’s CUDA programming guides, can also provide deep insights. Textbooks on parallel computing and high-performance computing usually dedicate chapters to memory transfer mechanisms and performance optimization. Research papers detailing DMA engine implementation and usage are also beneficial in gaining a complete understanding. Furthermore, looking into the documentation for different memory technologies, like DDR5 and GDDR6 will help explain the different bandwidth capabilities. Experimentation with benchmarks measuring memory transfer speeds for different configurations and data sizes is an effective way to observe the practical implications of these differences.
