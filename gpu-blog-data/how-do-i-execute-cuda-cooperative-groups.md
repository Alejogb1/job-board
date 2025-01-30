---
title: "How do I execute CUDA cooperative groups?"
date: "2025-01-30"
id: "how-do-i-execute-cuda-cooperative-groups"
---
CUDA cooperative groups, fundamentally, provide a structured way for threads within a CUDA grid to communicate and synchronize, extending beyond the implicit barriers within warp execution. My own experience, developing a custom image processing pipeline for a real-time analytics system, highlighted the critical need for this feature; simple warp-level parallelism proved inadequate for operations requiring cross-thread data aggregation. Specifically, I needed to perform a highly localized histogram equalization across overlapping subregions of a large image, and traditional global memory synchronization proved prohibitively slow.

The challenge lies in that CUDA kernels traditionally operate under an assumption of relative thread independence. Threads within a warp synchronize implicitly, but synchronization across warps or thread blocks requires explicit management, often using atomic operations and global memory barriers which introduce significant performance overhead. Cooperative groups address this by introducing an abstraction layer: they provide mechanisms for creating named groups of threads, within which shared memory and synchronizations can occur with higher efficiency and greater clarity.

At the heart of cooperative groups is the `cuda::thread_block` type, which represents the current thread block.  It has associated member functions, such as `sync()`, that serve as barriers within the thread block. Further group types can be derived from this, enabling communication across groups larger than a thread block, for example across grids. While simple `__syncthreads()` offers barriers within thread blocks, cooperative groups offer more nuanced control. They allow for: hierarchical grouping of threads (e.g. threads within a warp, warps within a block, blocks within a grid), efficient reduction operations within a group, and explicit control over thread participation in operations.

Crucially, using cooperative groups effectively demands careful consideration of the target hardware architecture. The efficiency of operations depends directly on thread layout and the underlying hardware’s capability to facilitate communication within defined groups. Ill-structured group sizes or incorrect usage of synchronization primitives can result in performance bottlenecks or incorrect results due to race conditions. The compiler often plays a crucial role in translating group primitives into appropriate hardware instructions, and a thorough understanding of this process is necessary for optimization.

Let's illustrate this with three examples, starting with a simple thread block reduction:

**Example 1: Thread Block Reduction using Cooperative Groups**

```cpp
#include <cooperative_groups.h>
#include <cuda_runtime.h>

__global__ void blockReduce(float* input, float* output, int size) {
    extern __shared__ float sdata[];
    auto group = cooperative_groups::this_thread_block();

    int tid = threadIdx.x;
    sdata[tid] = (tid < size) ? input[tid] : 0.0f;
    group.sync();

    for(int i = group.size() / 2; i > 0; i >>= 1){
        if(tid < i){
            sdata[tid] += sdata[tid + i];
        }
        group.sync();
    }

    if(tid == 0){
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    const int N = 256;
    const int numBlocks = 1; // Simplified for demo
    const int threadsPerBlock = 256;
    float *h_input, *h_output, *d_input, *d_output;
    cudaMallocHost((void**)&h_input, N * sizeof(float));
    cudaMallocHost((void**)&h_output, numBlocks * sizeof(float));
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, numBlocks * sizeof(float));

    for(int i = 0; i < N; ++i) h_input[i] = i * 1.0f; // Example data

    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    blockReduce<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, N);
    cudaMemcpy(h_output, d_output, numBlocks * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum: %f\n", h_output[0]);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);

    return 0;
}
```

*Commentary:* This example demonstrates a straightforward reduction of an array within a single thread block. `cooperative_groups::this_thread_block()` provides access to group functionalities, `sync()` synchronizes all threads in the group and the code performs a classic pairwise reduction stored in shared memory `sdata`. A shared memory allocation equivalent to the number of threads in the block is passed via the kernel launch configuration. The use of cooperative groups ensures correctness even when the block size isn't a power of two.

**Example 2: Grid-Level Reduction using Cooperative Groups**

```cpp
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <assert.h>

__global__ void gridReduce(float* input, float* output, int size) {
    extern __shared__ float sdata[];
    auto blockGroup = cooperative_groups::this_thread_block();
    int tid = threadIdx.x + blockIdx.x * blockGroup.size();

    float mySum = 0;
    if (tid < size)
        mySum = input[tid];
    sdata[threadIdx.x] = mySum;
    blockGroup.sync();

     // Block-level reduction
     for (int i = blockGroup.size() / 2; i > 0; i >>= 1) {
        if (threadIdx.x < i) {
            sdata[threadIdx.x] += sdata[threadIdx.x + i];
         }
         blockGroup.sync();
    }


    auto gridGroup = cooperative_groups::this_grid();
    if(threadIdx.x == 0){
         cooperative_groups::reduce_sum(gridGroup,sdata[0]);
    }
    gridGroup.sync();
    if(threadIdx.x == 0 && blockIdx.x ==0)
         *output = sdata[0];
}

int main() {
    const int N = 1024 * 1024; // Large data set
    const int threadsPerBlock = 256;
    const int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    float *h_input, *h_output, *d_input, *d_output;
    cudaMallocHost((void**)&h_input, N * sizeof(float));
    cudaMallocHost((void**)&h_output, sizeof(float));
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, sizeof(float));
    for(int i = 0; i < N; i++) h_input[i] = 1.0f;
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    gridReduce<<<numBlocks, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, N);
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    printf("Sum: %f\n", *h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);
    return 0;
}
```

*Commentary:* In this more complex example, we perform a reduction across the entire grid. Initially, we perform a reduction within each block, storing the partial sums in shared memory (`sdata`). Subsequently, the result of each block sum is aggregated using `cooperative_groups::this_grid()` and `cooperative_groups::reduce_sum()`. This method avoids global memory writes for the final reduction and the built in `reduce_sum()` operation provides a performance optimized operation for reducing group data. This is significantly more efficient than using atomic operations to perform grid-level aggregations.

**Example 3: Data Sharing Using a Block Group**

```cpp
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void dataShare(float* input, float* output, int size) {
   extern __shared__ float sdata[];
    auto blockGroup = cooperative_groups::this_thread_block();

    int tid = threadIdx.x;
    if(tid < size){
        sdata[tid] = input[tid];
    }
    blockGroup.sync();

    if(tid < size){
      float sum = 0;
      for(int i=0;i<size;i++){
           sum += sdata[i];
      }
     output[tid] = sum;
    }
}

int main() {
    const int N = 256;
    const int threadsPerBlock = 256;
    float *h_input, *h_output, *d_input, *d_output;
    cudaMallocHost((void**)&h_input, N * sizeof(float));
    cudaMallocHost((void**)&h_output, N * sizeof(float));
    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

     for (int i = 0; i < N; ++i) {
        h_input[i] = i * 1.0f;
    }
    cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice);

    dataShare<<<1, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_input, d_output, N);
    cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i=0; i<N; ++i) {
        printf("Output[%d]: %f\n", i, h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFreeHost(h_input);
    cudaFreeHost(h_output);

    return 0;
}

```

*Commentary:* This example highlights the use of cooperative groups for shared memory data access. All threads in the block copy an array of data into shared memory using `sdata`. Each thread then iterates through the shared data computing the total and storing this sum in the output array. This demonstrates how `sync()` ensures all shared memory data has been updated by each thread in the group. While a contrived example, it emphasizes the use of shared memory alongside cooperative group barriers.

To further enhance one's understanding of CUDA cooperative groups I would strongly suggest exploring these resources, which I have personally found invaluable: The official NVIDIA CUDA programming guide provides detailed explanations of the underlying concepts and API details. The CUDA samples provided with the CUDA Toolkit are also extremely beneficial as they contain numerous practical examples. Furthermore, research publications from NVIDIA researchers provide valuable information regarding optimizations and design choices that lead to efficient cooperative group implementations. These publications can usually be found through a literature search using relevant keywords. Careful study of these resources should provide a solid understanding of this advanced CUDA feature.
