---
title: "What does NVIDIA-smi gpu-util represent?"
date: "2025-01-30"
id: "what-does-nvidia-smi-gpu-util-represent"
---
The percentage reported by `nvidia-smi` under the `GPU-Util` column represents the utilization of the GPU's compute engine—specifically, the proportion of time, measured over a recent interval, that one or more Streaming Multiprocessors (SMs) within the GPU are actively executing instructions. It's critical to understand that `GPU-Util` does *not* reflect the overall activity or load on the entire GPU. Instead, it primarily indicates how busy the core computational units are. Low GPU utilization can occur even when other parts of the GPU, like memory or copy engines, are heavily engaged. It’s a performance metric focused on the arithmetic logic units (ALUs) within the SMs, and is a critical figure for debugging performance issues.

The `GPU-Util` percentage is derived from sampling the activity of the SMs at regular intervals. NVIDIA's driver monitors whether an SM is actively processing data or waiting. This information is aggregated, and the percentage is then computed over the recent history. Typically, this monitoring interval is a fraction of a second, providing a near real-time view of SM activity. A higher percentage generally implies more intensive computational workload on the GPU, while a low percentage may indicate a bottleneck elsewhere in the processing pipeline, such as data transfer or preparation. It’s crucial to remember that a 100% `GPU-Util` figure doesn’t automatically equate to optimal performance or efficient use of all GPU resources, but indicates that the computational units are fully occupied.

Based on my experience optimizing CUDA code for high-performance computing (HPC) applications and analyzing GPU performance in various machine learning (ML) models, I’ve observed common scenarios where understanding `GPU-Util` is essential. A common misunderstanding is assuming that low `GPU-Util` implies that the GPU is idle. This can often be due to a lack of data feeding the GPU, such as inefficient CPU-to-GPU transfers or inadequately structured kernels. In such instances, the GPU has idle SMs because it is waiting on data, despite there being computational work to perform. Conversely, very high `GPU-Util`, approaching or hitting 100%, doesn't always mean that the GPU is operating optimally. For example, if there are resource contention issues, such as excessive bank conflicts when accessing memory or instruction dependencies, the GPU might be fully utilized, but it’s not performing calculations efficiently. These factors underscore why solely monitoring `GPU-Util` is insufficient for complete performance evaluation, and why other metrics, such as memory utilization and transfer rates, must be considered in parallel.

To illustrate these points further, consider a few practical scenarios with their corresponding code snippets:

**Example 1: CPU Bottleneck**

This first example demonstrates how a CPU-bound pre-processing step can lead to low GPU utilization, even though the core computations are straightforward. I've encountered this problem frequently in data-heavy AI workflows. Assume the following CUDA kernel, which is intentionally simple:

```c++
__global__ void add_scalar(float* data, float scalar, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    data[i] += scalar;
  }
}

// Host-side CPU code
int main() {
  int n = 1024 * 1024 * 128; // Large data size
  float* host_data = new float[n];
  float* device_data;
  cudaMalloc((void**)&device_data, n * sizeof(float));

    // (1) CPU-heavy preprocessing - artificially slowed down
    for (int i = 0; i < n; ++i) {
      host_data[i] = i; // Simulate complex data prep.
      std::this_thread::sleep_for(std::chrono::nanoseconds(1)); // Artifical CPU delay
    }

    // (2) GPU kernel execution
    cudaMemcpy(device_data, host_data, n * sizeof(float), cudaMemcpyHostToDevice);
    add_scalar<<<256, 256>>>(device_data, 1.0f, n);
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(device_data);
    delete[] host_data;
    return 0;
}
```

In this scenario, despite the GPU kernel being easily parallelizable and ready to process the entire array, the CPU preprocessing step (labelled (1)) which includes a simulated delay, takes significantly longer, causing the GPU to wait. Observing `nvidia-smi` during the run will show low `GPU-Util`, despite a large amount of data being processed by the kernel itself, because the data is not being fed to the GPU quickly enough. This is a classic example of a CPU bottleneck.

**Example 2: Inefficient Kernel Design**

This example demonstrates how a suboptimal kernel design can cause low `GPU-Util` due to memory access patterns and synchronization issues, an experience I’ve had while tuning custom neural network layers. Let's consider the following problematic kernel:

```c++
__global__ void inefficient_transpose(float* in, float* out, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        out[col * rows + row] = in[row * cols + col]; // Unaligned memory access
    }
    __syncthreads(); // Forcing synchronization on every write
}
// Host-side code (omitted for brevity, similar to Example 1 in principle)
```
Here, the kernel performs a matrix transposition. However, the memory access pattern (non-coalesced writes into `out`) is poor and `__syncthreads()` is excessive. This design leads to underutilization of the SMs due to poor data locality and forced synchronization. The `nvidia-smi` output will indicate a moderate to low `GPU-Util`, because despite the workload, SMs are frequently idle waiting for memory operations or sync points. Re-writing the kernel to ensure coalesced accesses and minimizing synchronization requirements would greatly enhance GPU utilization.

**Example 3: Computationally Intensive Kernel**

The final example represents a situation where the computational load is correctly mapped to the GPU, leading to a high `GPU-Util`. This is commonly seen in computationally demanding models, like training large language models. Consider this matrix multiplication kernel.

```c++
__global__ void matrix_multiply(float* A, float* B, float* C, int M, int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < M && col < N) {
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
  }
}
// Host-side code (omitted for brevity) - involves generating large matrices.
```
In this scenario, if the dimensions `M`, `N`, and `K` are substantial, the workload is highly parallelizable and requires intensive computation. When this kernel executes, `nvidia-smi` would typically report high `GPU-Util`, approaching 100%, provided the data is ready. This happens because the SMs are kept busy performing a large amount of calculations with minimal latency. This is an ideal scenario where the `GPU-Util` value reflects a large computation.

In summary, while the `GPU-Util` metric from `nvidia-smi` provides vital information about the active use of SMs on NVIDIA GPUs, it does not provide a comprehensive view of all GPU activity. A low percentage does not necessarily imply inactivity but can often point to bottlenecks elsewhere, while 100% `GPU-Util` does not guarantee peak performance. I’ve learned, through countless debugging sessions, to treat this value as an initial diagnostic; additional monitoring of memory bandwidth, bus usage, and CPU-to-GPU transfer speeds is crucial for identifying the root cause of underutilized GPU resources and for effective code optimization.

For further study, I would suggest consulting the NVIDIA CUDA programming guide which includes detailed sections about performance profiling and memory management as a starting point. The documentation associated with specific NVIDIA driver versions can also prove helpful as it contains crucial context on the reported metrics. Academic publications and conference proceedings on parallel computing often present advanced methods for performance analysis. Accessing these primary sources of knowledge has proven essential during my time as a developer, and I highly recommend engaging with such resources for a deeper understanding.
