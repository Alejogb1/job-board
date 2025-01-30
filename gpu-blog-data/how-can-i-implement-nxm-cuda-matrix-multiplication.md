---
title: "How can I implement NxM CUDA matrix multiplication?"
date: "2025-01-30"
id: "how-can-i-implement-nxm-cuda-matrix-multiplication"
---
A foundational aspect of high-performance computing, NxM matrix multiplication on CUDA presents a practical challenge: optimizing memory access patterns and thread utilization to maximize throughput on massively parallel GPUs. Having spent considerable time optimizing numerical algorithms for scientific simulations, I've found that a strategic approach to kernel design and data layout is paramount. Efficient CUDA matrix multiplication requires careful consideration of block and thread dimensions, shared memory usage, and avoiding memory access bottlenecks.

The core idea involves dividing the matrices into manageable tiles that can be processed by CUDA thread blocks. Rather than having each thread compute an entire output matrix element, we assign each thread to compute a single element or a small portion of a matrix tile. This approach allows us to leverage the parallel processing capabilities of the GPU, where thousands of threads can concurrently perform calculations. A simple but often effective implementation involves dividing both the input matrices and the output matrix into blocks, processing those blocks in parallel by GPU blocks and dividing the work on blocks to threads, while employing shared memory to minimize global memory accesses.

Here’s the general algorithm, broken down into steps:

1.  **Data Allocation and Transfer:** Allocate memory on both the host (CPU) and device (GPU) for the input matrices (A and B) and the output matrix (C). Transfer the input matrix data from host to device.
2.  **Kernel Configuration:** Define block dimensions (number of threads per block) and grid dimensions (number of blocks) that adequately cover the output matrix, adjusting to the number of Streaming Multiprocessors and shared memory resources.
3.  **Tiled Matrix Access:** In the kernel, each thread computes a small portion of the output matrix (a tile). To perform this multiplication efficiently, each block needs to load a tile of the A and B matrices into shared memory. Shared memory is fast, on-chip memory accessible by all threads in the block. The tiles are loaded into shared memory, and all threads within the block perform the tile multiplication step, contributing to the tile of the output matrix.
4.  **Synchronization:** Synchronization is essential after loading a tile to ensure that all threads within the block have access to the data in shared memory. The `__syncthreads()` command is used for this operation.
5.  **Accumulation and Output:** Threads calculate elements of the output matrix within the tile by multiplying a row of tile A with column of tile B and accumulating in the corresponding element of the output matrix C. The results are then written back to global memory.
6.  **Data Transfer (Optional):** Transfer the results (matrix C) back to the host after kernel completion, if required.

Let's examine a series of CUDA C++ code examples to illustrate this concept. The examples will use the classic matrix multiplication algorithm (C = A x B), and for clarity, I'll omit host-side setup and cleanup details, focusing instead on the CUDA kernel implementation. Error checking and performance optimization techniques (such as loop unrolling or compiler directives) are not included to make the code easier to read.

**Example 1: Basic Shared Memory Multiplication**

This first example demonstrates a straightforward approach to using shared memory. It introduces basic concepts, but lacks optimal performance due to the hard-coded tile size.

```cpp
__global__ void matrixMulBasic(float* A, float* B, float* C, int N, int M, int K) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int TILE_SIZE = 16;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    for(int k=0; k < (K + TILE_SIZE - 1) / TILE_SIZE; ++k)
    {
        if (row < N && (k*TILE_SIZE + tx) < K)
            As[ty][tx] = A[row * K + k * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if ((k*TILE_SIZE + ty) < K && col < M)
            Bs[ty][tx] = B[(k * TILE_SIZE + ty) * M + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

         __syncthreads();
    }
    if (row < N && col < M) {
       C[row * M + col] = sum;
    }

}
```

*   **Commentary:** The kernel calculates the row and column for each thread within the output matrix, based on block and thread indices. Each thread loads a tile of matrices A and B into shared memory `As` and `Bs`, using a loop to account for matrix K dimension being larger than tile size. Synchronization using `__syncthreads()` ensures that all threads finish loading shared memory before commencing matrix multiplication. It is important to note that padding is done so that we do not write out of bounds on global memory. This basic example illustrates the core concept; however, using a fixed tile size limits flexibility.

**Example 2: Flexible Tile Size and Handling Uneven Matrix Sizes**

This example improves upon the first by allowing for a non-fixed tile size and handling uneven matrix sizes.

```cpp
__global__ void matrixMulFlexible(float* A, float* B, float* C, int N, int M, int K, int TILE_SIZE) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
     float sum = 0.0f;

    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; ++k) {
       if (row < N && (k*TILE_SIZE + tx) < K)
            As[ty][tx] = A[row * K + k * TILE_SIZE + tx];
        else
            As[ty][tx] = 0.0f;

        if ((k*TILE_SIZE + ty) < K && col < M)
            Bs[ty][tx] = B[(k * TILE_SIZE + ty) * M + col];
        else
           Bs[ty][tx] = 0.0f;

        __syncthreads();

        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
        }

        __syncthreads();
    }
     if (row < N && col < M) {
        C[row * M + col] = sum;
    }

}

```

*   **Commentary:**  The significant difference here is that the `TILE_SIZE` is now a kernel parameter. The loop over K dimension now takes into consideration the general case where the K dimension is not a multiple of the tile size by calculating number of steps based on how many tiles are required. Padding with zeros prevents memory access errors. This improved approach offers more flexibility for different matrix sizes and allows for experimental optimization of tile size.

**Example 3: Optimized Memory Access for Non-Square Matrices**

This example uses tiling and shared memory, but this time, instead of having the threads simply load their corresponding matrix element, it leverages the fact that shared memory access is much faster when performed contiguously. Each thread loads consecutive elements from global memory into shared memory, resulting in memory access coalescing and improving performance. We do this by assigning multiple values per thread, up to the tile size.

```cpp
__global__ void matrixMulOptimized(float* A, float* B, float* C, int N, int M, int K, int TILE_SIZE) {
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

     int row = by * TILE_SIZE + ty;
     int col = bx * TILE_SIZE + tx;

    float sum = 0.0f;
    for (int k = 0; k < (K + TILE_SIZE - 1) / TILE_SIZE; ++k) {
         for(int i=0; i < TILE_SIZE; ++i){
           if (row < N && (k*TILE_SIZE + i) < K)
               As[ty][i] = A[row * K + k * TILE_SIZE + i];
           else
               As[ty][i] = 0.0f;

           if((k*TILE_SIZE + i) < K && col < M)
               Bs[i][tx] = B[(k * TILE_SIZE + i) * M + col];
           else
              Bs[i][tx] = 0.0f;
         }
        __syncthreads();
    
        for (int i = 0; i < TILE_SIZE; i++) {
            sum += As[ty][i] * Bs[i][tx];
         }
    
       __syncthreads();
    }

     if (row < N && col < M) {
        C[row * M + col] = sum;
    }
}
```

*   **Commentary:** In this example, threads load entire rows of matrix A, and columns of matrix B into shared memory. By using loops to load the `TILE_SIZE` elements, the algorithm accesses sequential memory locations on global memory, resulting in coalesced reads and writes for faster global memory transactions. Additionally, a better usage of the shared memory space can be obtained if the number of values loaded by each thread is optimized. This version provides the best performance in practice, demonstrating the value of coalesced memory access.

For further study and advanced techniques, the following resources are suggested. These are books and online resources that will help in understanding the details of CUDA programming for scientific computing.

*   **Programming Massively Parallel Processors** by David B. Kirk and Wen-mei W. Hwu: A comprehensive introduction to CUDA programming covering various aspects including memory management and performance optimization.
*   **CUDA by Example** by Jason Sanders and Edward Kandrot: Practical examples and case studies, a great guide for hands-on learning.
*   **NVIDIA CUDA documentation**: The official NVIDIA documentation provides in-depth explanations of all CUDA features and capabilities. The programming guide and API references are particularly valuable.
*   **Online Courses on Parallel Programming**: Platforms such as Coursera and edX offer courses on parallel computing that often include material on GPU programming. These courses frequently cover advanced optimization techniques.

Implementing efficient NxM CUDA matrix multiplication requires an understanding of the underlying hardware architecture and careful tuning of the kernel parameters and memory access patterns. The examples provided here offer a starting point for developing customized and optimized matrix multiplication kernels.
