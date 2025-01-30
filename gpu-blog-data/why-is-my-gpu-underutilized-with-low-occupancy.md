---
title: "Why is my GPU underutilized with low occupancy?"
date: "2025-01-30"
id: "why-is-my-gpu-underutilized-with-low-occupancy"
---
GPU underutilization with low occupancy often stems from a mismatch between the parallel processing capabilities of the GPU and the workload presented by the software. Fundamentally, a GPU achieves peak performance through massive parallelism; if the tasks it’s receiving don't fully exploit this parallelism, occupancy—the percentage of active processing units—drops, resulting in underutilization. I've personally encountered this across numerous projects, from high-throughput simulations to complex rendering pipelines. The issue rarely lies with the GPU itself, but rather how code is structured and how data is managed for GPU processing.

Occupancy is a key metric that directly relates to a GPU's ability to hide latency. Latency on a GPU arises primarily from accessing memory, particularly global memory. When a processing unit (a thread or warp/wavefront) needs to read or write to global memory, it typically has to wait before the operation is completed. To avoid stalling, the GPU needs a sufficient number of other active units to switch to while it waits for the memory operation of the previous unit. The more active units, the higher the occupancy and the better the latency hiding. Low occupancy results in idle units waiting for memory operations to complete, rather than switching to other tasks.

Several factors contribute to low occupancy. First, **insufficient work per thread/warp** is a common culprit. If a kernel launch specifies a number of thread blocks (or compute units) that's too low, the GPU's multiprocessors won't be fully populated with active warps or wavefronts. This under-utilization limits the amount of work the GPU can actively schedule, forcing it to wait for completion of each operation, without being able to shift to other parallel tasks. This also includes scenarios where thread blocks are very small. For instance, a single thread block with only 32 threads is generally much less than the capacity of a single GPU multiprocessor, resulting in less occupancy.

Second, **memory access patterns** can drastically impact performance. If threads within a warp access memory in an uncoalesced manner (non-contiguous, scattered across the memory space), the GPU ends up performing multiple independent memory transactions instead of a single wide transaction. This greatly impacts memory throughput, causing warps to stall, increasing the latency of memory access, and thus reduces occupancy. Similarly, excessive reliance on global memory operations can limit occupancy, as global memory access typically involves significant latency. Constant re-reading data or writing partially-used results to global memory will also slow down processing.

Third, **algorithmic limitations and divergent control flow** can cause occupancy issues. If the kernel code contains if-else conditions where different threads in the same warp execute different code paths (divergence), the warp will stall since all threads within a warp need to execute the same instruction. GPUs are optimized to execute the same instruction in parallel across threads in the same warp and only handle divergences efficiently in newer architectures, but even then, excessive divergence can still cause serialization and lower occupancy. Additionally, suboptimal algorithm design—such as processing data in a way that is not inherently parallelizable— can limit GPU utilization.

Here are some code examples to illustrate these points, based on my experience with CUDA. Although the syntax is CUDA, the principle applies broadly to other parallel computing APIs like OpenCL.

**Example 1: Insufficient Work Per Thread Block**

```c++
// Inefficient kernel launch with too few thread blocks
__global__ void inefficient_kernel(float* output, float* input, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = input[i] * 2.0f;
    }
}

int main() {
  int size = 1024 * 1024;
  float* input = allocate_memory_cpu(size);
  float* output = allocate_memory_cpu(size);
  initialize_array(input, size);
  float* d_input;
  float* d_output;
  allocate_memory_gpu(&d_input, size);
  allocate_memory_gpu(&d_output, size);
  copy_cpu_to_gpu(d_input, input, size);

  // Launch using a limited number of blocks with large block size
  dim3 threadsPerBlock(256);
  dim3 blocksPerGrid(2); // Very few blocks which wont populate the GPU
  inefficient_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_input, size);
  cudaDeviceSynchronize();
  copy_gpu_to_cpu(output, d_output, size);

  //... Cleanup code omitted for brevity.
}
```

In this example, `blocksPerGrid` is set to 2, meaning only a small number of the GPU's multiprocessors would be active. Even though each thread block has 256 threads, the limited number of blocks prevents the GPU from fully utilizing its parallel processing capabilities. The GPU would be idle for a large portion of its potential throughput, resulting in low occupancy and underutilization. The solution lies in launching more thread blocks – specifically, launch the number of blocks that fully occupy the multiprocessors. The exact number of multiprocessors per GPU can be queried from the runtime.

**Example 2: Uncoalesced Memory Access**

```c++
// Inefficient memory access pattern with uncoalesced access
__global__ void uncoalesced_access_kernel(float* output, float* input, int width, int height) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < height && col < width) {
    output[col * height + row] = input[col * height + row] * 2.0f; //Uncoalesced access for column major indexing
  }
}


int main() {
  int width = 1024;
  int height = 1024;
  float* input = allocate_memory_cpu(width * height);
  float* output = allocate_memory_cpu(width* height);
  initialize_array(input, width * height);
  float* d_input;
  float* d_output;
  allocate_memory_gpu(&d_input, width * height);
  allocate_memory_gpu(&d_output, width * height);
  copy_cpu_to_gpu(d_input, input, width * height);

  dim3 threadsPerBlock(32, 8);
  dim3 blocksPerGrid((width + 31) / 32, (height + 7) / 8);
  uncoalesced_access_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_input, width, height);
  cudaDeviceSynchronize();
  copy_gpu_to_cpu(output, d_output, width* height);

  //... Cleanup code omitted for brevity.
}
```

Here, I assume a 2D data matrix. The `output[col * height + row]` access pattern is uncoalesced for row-major layout, as threads within a warp end up accessing memory locations that are far apart. This forces the memory controller to serve each request separately. The fix is to ensure contiguous memory access by having threads in the warp access adjacent data elements. A simple change to iterate using `row * width + col` which maps to the contiguous elements of the 2d array in a row-major layout and would yield coalesced memory access.

**Example 3: Divergent Control Flow**

```c++
// Inefficient use of divergence in control flow
__global__ void divergent_control_flow_kernel(float* output, float* input, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        if (input[i] > 0.5f) {
            output[i] = input[i] * 2.0f;
        } else {
            output[i] = input[i] / 2.0f;
        }
    }
}

int main() {
    int size = 1024 * 1024;
    float* input = allocate_memory_cpu(size);
    float* output = allocate_memory_cpu(size);
    initialize_array(input, size);
    float* d_input;
    float* d_output;
    allocate_memory_gpu(&d_input, size);
    allocate_memory_gpu(&d_output, size);
    copy_cpu_to_gpu(d_input, input, size);
    dim3 threadsPerBlock(256);
    dim3 blocksPerGrid((size + 255) / 256);
    divergent_control_flow_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_input, size);
    cudaDeviceSynchronize();
    copy_gpu_to_cpu(output, d_output, size);

    //... Cleanup code omitted for brevity.
}
```

The `if (input[i] > 0.5f)` condition causes warp divergence. The threads within a warp will take different paths which forces the warp to perform the conditional code serially, rather than in parallel, thus reducing the occupancy and throughput. For some simple computations, using `select` operation as supported in most GPU programming languages may alleviate such divergent paths by computing both paths and selecting results later. However, for complex conditional branches, divergence becomes unavoidable, but must be understood to achieve maximum performance.

To address low occupancy and GPU underutilization, I recommend studying resources that describe CUDA architecture in detail, focusing particularly on warp scheduling, memory access coalescing, and strategies to minimize divergence. Publications and guides that delve into optimization techniques for parallel computing environments would also be highly beneficial. Focusing on the theoretical underpinnings of GPU architecture, rather than specific implementations, will provide a deeper understanding of the factors that influence occupancy. Additionally, using profiling tools to identify specific areas where occupancy is low can provide targeted insights for optimization. Detailed performance analysis of your GPU kernels is crucial to finding bottlenecks and improving efficiency.
