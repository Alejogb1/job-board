---
title: "Why is CUDA shared memory slower than global memory for tiled matrix multiplication?"
date: "2025-01-30"
id: "why-is-cuda-shared-memory-slower-than-global"
---
Direct memory access patterns critically influence performance in CUDA, and the seemingly counter-intuitive speed difference between shared and global memory when performing tiled matrix multiplication underscores this point. I’ve encountered this issue repeatedly during my years optimizing numerical kernels, and the root cause lies in the interplay of memory access coalescing and the effective use of shared memory’s low latency. The presumption that shared memory, being on-chip and thus faster, will automatically yield superior performance is simply incorrect in scenarios where access patterns impede its strengths.

At its core, tiled matrix multiplication breaks down the problem into smaller blocks of data that can be processed within the compute capability of a single Streaming Multiprocessor (SM). The typical implementation loads a submatrix from global memory into shared memory, performs the multiplication, and writes the result back to global memory. The intent is to leverage shared memory’s low latency to speed up access during the multiplication step. However, global memory access, when done correctly (i.e., coalesced), can significantly outperform non-coalesced shared memory access, thus negating the benefits of shared memory in some implementations.

Here's why this performance anomaly can occur:

1.  **Coalesced Access:** Global memory accesses, if arranged appropriately, can be coalesced. This means that consecutive threads within a warp access consecutive memory locations. This coalescing effectively maximizes bandwidth to the DRAM, resulting in a single high-bandwidth memory transaction instead of multiple smaller ones. CUDA hardware is optimized for coalesced accesses, significantly accelerating data transfers.

2.  **Bank Conflicts:** Shared memory is physically organized into banks to enable parallel access by multiple threads. If multiple threads in a warp request data from the *same* shared memory bank simultaneously, a bank conflict occurs, serializing access and degrading performance drastically. A common misstep in tiled matrix multiplication is to not carefully plan the layout of the submatrices in shared memory, causing widespread bank conflicts when threads access elements during the multiplication phase.

3.  **Limited Bandwidth with Bank Conflicts:** Even though shared memory has low latency, its effective bandwidth suffers when bank conflicts are pervasive. When a program is experiencing bank conflicts in shared memory, the total bandwidth is lower than the available DRAM bandwidth when accessed in a coalesced manner. Effectively, the lower latency is not able to compensate for reduced bandwidth.

4.  **Read-Only Data** Frequently in matrix multiplication the same submatrix is loaded into shared memory multiple times. Often, an intermediate result that will only be read from can be located in global memory and accessed with coalesced reads. This bypasses the overhead of loading the intermediate result into shared memory.

To solidify these points, let's consider a series of simplified, illustrative CUDA kernels, where I demonstrate how these factors contribute to the observed performance discrepancies. I will focus on the core issue of loading the tiles and avoid the actual multiplication to illustrate the loading problem:

**Example 1: Naive Shared Memory Loading (Poor Performance)**

```cuda
__global__ void naive_shared_load(float* input, float* output, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float tile[TILE_SIZE][TILE_SIZE];

    if (row < width && col < width) {
      tile[threadIdx.y][threadIdx.x] = input[row * width + col];
    }
     __syncthreads();

     if (row < width && col < width) {
       output[row * width + col] = tile[threadIdx.y][threadIdx.x];
     }
}
```

*   **Commentary:** This kernel attempts to load a tile into shared memory using a straightforward, but problematic, indexing approach where threads map directly to shared memory array locations. This design is highly susceptible to bank conflicts. Threads in a warp are likely to access the same bank, particularly in the common case where `TILE_SIZE` is a power of two.

**Example 2: Coalesced Global Memory Loading (Good Performance)**

```cuda
__global__ void coalesced_global_load(float* input, float* output, int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if(row < width && col < width) {
    output[row*width + col] = input[row*width + col];
  }
}
```

*   **Commentary:** This kernel loads the output directly from the global memory. No shared memory is used. Note that consecutive threads are loading consecutive memory locations within the global `input` array. This satisfies the coalescing requirement of global memory and results in an efficient load. Assuming `width` is aligned to the warp size, this is a very fast operation.

**Example 3: Shared Memory Loading with Bank Conflict Mitigation (Better, but not always ideal)**

```cuda
__global__ void mitigated_shared_load(float* input, float* output, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];

    if (row < width && col < width) {
        tile[threadIdx.y][threadIdx.x] = input[row * width + col];
    }
    __syncthreads();

    if (row < width && col < width) {
        output[row * width + col] = tile[threadIdx.y][threadIdx.x];
    }
}
```

*   **Commentary:** This kernel addresses bank conflicts by adding padding to the shared memory array. By increasing the column dimension of the shared memory array to `TILE_SIZE + 1`, we are likely to distribute the data across shared memory banks, reducing conflicts. This approach can yield better performance than the first example, but requires padding. This is only applicable to specific cases of the TILE_SIZE.

In summary, while shared memory is low latency, its potential benefits are easily negated by poor access patterns that cause bank conflicts. Coalesced global memory access, on the other hand, capitalizes on the high-bandwidth DRAM interface and results in a faster data transfer when implemented correctly.

It's crucial to remember the performance equation is always context-dependent. Factors such as the specific GPU architecture, the size of the matrices, and how efficiently data is arranged in global memory all contribute to the optimal approach. While adding padding to shared memory arrays can help, the overhead of writing to shared memory, especially when compared to a simple global memory access, is not negligible. This means that, in many instances, the overhead of writing to shared memory and reading from shared memory is less beneficial than simply performing coalesced global memory reads.

For a deeper dive into these concepts, I recommend exploring the following resources:

*   CUDA C Programming Guide: This Nvidia-provided document offers an exhaustive overview of CUDA architecture and programming techniques. Particular attention should be given to sections covering memory management and optimization.
*   Various online courses concerning parallel programming. Several major educational platforms offer courses specifically focused on CUDA.
*   Books focusing on GPU programming: A number of books go into detail on how to use GPUs in scientific applications. Seek ones that deal with memory usage.

My practical experience leads me to emphasize that blind application of shared memory is not an automatic performance win. Understanding coalesced access, bank conflicts, and the memory hierarchy is crucial for writing efficient CUDA kernels. Carefully measure and analyze performance to truly understand bottlenecks, and never assume that shared memory is always the correct solution.
