---
title: "How can CUDA optimize matrix multiplication?"
date: "2025-01-30"
id: "how-can-cuda-optimize-matrix-multiplication"
---
Matrix multiplication, when performed naively on a CPU, often becomes a bottleneck in computationally intensive applications. Parallelizing this operation via CUDA leverages the massive parallelism offered by GPUs, leading to substantial performance gains. I have personally observed this during my time optimizing physics simulations where matrix operations are ubiquitous, and I’ve found that a solid understanding of CUDA’s memory hierarchy and execution model is critical for achieving optimal speedups. The core optimization revolves around strategically mapping the inherently parallel structure of matrix multiplication to the parallel processing capabilities of the GPU.

The basic operation, C = A * B, where A is an *m x k* matrix, B is a *k x n* matrix, and C is an *m x n* matrix, involves calculating each element of C as the dot product of a row of A and a column of B. This seemingly simple operation results in a computational complexity of *O(mkn)*. To exploit parallelism, we assign each thread on the GPU to calculate a single element of the C matrix, or potentially a small block of elements in C to minimize memory access. This requires carefully crafting a kernel, which is the function executed on the GPU, and considering how memory is handled. The GPU memory, while significantly faster than CPU memory in many cases, is not uniformly accessed, therefore we need to be aware of global memory accesses, which are expensive and can become a limiting factor to performance. Therefore, techniques for maximizing data locality and coalesced memory access are key.

Furthermore, CUDA offers shared memory, a fast, on-chip memory shared by threads within a single thread block. By loading portions of the matrices A and B into shared memory before computation, each thread in the block can access it rapidly, minimizing global memory accesses. The size of thread blocks is critical, as it impacts the amount of shared memory available and how effectively we utilize the GPU’s execution units. Proper grid dimension configuration, coupled with effective use of the block and thread index within the kernel itself, is also critical for correctly mapping the problem to the GPU’s parallel architecture.

Let's look at an initial approach using CUDA. This example assumes square matrices of size N x N and shows a naive element-wise multiplication where each thread computes a single element of the C matrix.

```c++
__global__ void matrixMulNaive(float* A, float* B, float* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

In this `matrixMulNaive` kernel, we use `blockIdx`, `blockDim`, and `threadIdx` to determine the row and column index for each thread. The nested loop performs the dot product calculation. This implementation will work functionally, but it is not optimized for performance. Each thread reads data directly from global memory repeatedly, leading to a large number of uncoalesced global memory accesses. Also, the global memory accesses are not reused, meaning the same data may be requested many times. This approach serves as a baseline for comparison.

To improve upon this, we need to use shared memory and tiling to perform the matrix multiplication in blocks. We load tiles of A and B into shared memory and perform dot products using only the fast on-chip memory. This approach uses thread blocks to compute smaller sub-matrices.

```c++
__global__ void matrixMulTiled(float* A, float* B, float* C, int N, int TILE_SIZE) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;
    int numTiles = N / TILE_SIZE;

    for (int tile = 0; tile < numTiles; tile++) {
        int a_row = blockIdx.y * TILE_SIZE + threadIdx.y;
        int a_col = tile * TILE_SIZE + threadIdx.x;
        int b_row = tile * TILE_SIZE + threadIdx.y;
        int b_col = blockIdx.x * TILE_SIZE + threadIdx.x;

        if (a_row < N && a_col < N) {
          As[threadIdx.y][threadIdx.x] = A[a_row * N + a_col];
        } else {
          As[threadIdx.y][threadIdx.x] = 0.0;
        }
        if (b_row < N && b_col < N) {
          Bs[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
        } else {
          Bs[threadIdx.y][threadIdx.x] = 0.0;
        }
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++) {
          sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }

    if(row < N && col < N) {
        C[row * N + col] = sum;
    }
}
```

In this `matrixMulTiled` kernel, we use `__shared__` memory to create `As` and `Bs` shared arrays within each thread block. Each thread loads a small section from the `A` and `B` matrices into shared memory. The `__syncthreads()` calls ensure that all threads within a block have completed the loading before computation begins.  The outer loop iterates through tiles of the matrices, allowing the shared memory to be reused effectively. Importantly, the use of shared memory in this manner significantly reduces the number of costly global memory access. The tile size, `TILE_SIZE`, should be chosen carefully to maximize performance, generally a power of two value. If your input matrix size is not a multiple of your `TILE_SIZE`, then you will need to handle the boundary conditions using proper checks, such as I have done here using `if` statements.

While the previous approach drastically improves performance, it can be further enhanced.  The use of unrolling, multiple accumulators, and vectorization within the kernel can improve instruction-level parallelism. While a full implementation of this would exceed the length constraint here, I can illustrate a method of using multiple accumulators:

```c++
__global__ void matrixMulMultipleAccum(float* A, float* B, float* C, int N, int TILE_SIZE) {
  __shared__ float As[TILE_SIZE][TILE_SIZE];
  __shared__ float Bs[TILE_SIZE][TILE_SIZE];

  int row = blockIdx.y * TILE_SIZE + threadIdx.y;
  int col = blockIdx.x * TILE_SIZE + threadIdx.x;

  float sum0 = 0.0f;
  float sum1 = 0.0f;
  float sum2 = 0.0f;
  float sum3 = 0.0f;

  int numTiles = N / TILE_SIZE;
  for (int tile = 0; tile < numTiles; ++tile) {
    int a_row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int a_col = tile * TILE_SIZE + threadIdx.x;
    int b_row = tile * TILE_SIZE + threadIdx.y;
    int b_col = blockIdx.x * TILE_SIZE + threadIdx.x;

    if (a_row < N && a_col < N) {
      As[threadIdx.y][threadIdx.x] = A[a_row * N + a_col];
    } else {
      As[threadIdx.y][threadIdx.x] = 0.0;
    }
    if (b_row < N && b_col < N) {
       Bs[threadIdx.y][threadIdx.x] = B[b_row * N + b_col];
    } else {
      Bs[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();

    for (int k = 0; k < TILE_SIZE; k+=4) {
        sum0 += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        if (k+1 < TILE_SIZE) {
            sum1 += As[threadIdx.y][k+1] * Bs[k+1][threadIdx.x];
        }
        if (k+2 < TILE_SIZE) {
           sum2 += As[threadIdx.y][k+2] * Bs[k+2][threadIdx.x];
        }
        if (k+3 < TILE_SIZE) {
            sum3 += As[threadIdx.y][k+3] * Bs[k+3][threadIdx.x];
        }
    }

    __syncthreads();
  }

  if(row < N && col < N) {
        C[row * N + col] = sum0 + sum1 + sum2 + sum3;
  }
}
```

This `matrixMulMultipleAccum` kernel is similar to the tiled version, but uses four accumulators, `sum0`, `sum1`, `sum2`, and `sum3` to compute partial sums which are then combined before writing to memory.  This exposes more independent operations and increases the instruction-level parallelism that the compiler can take advantage of.

Achieving peak performance often requires careful tuning and experimentation with various tile sizes, block sizes, and memory access patterns. The optimal values are specific to the GPU architecture in use. Additionally, one must also consider whether to use other features, such as CUDA streams, which allow overlapping of computation and data transfers.

For those looking to expand their understanding of CUDA optimization, I would suggest reviewing resources that discuss the CUDA memory model in detail, particularly the differences between global memory, shared memory, and constant memory. Textbooks covering parallel programming with CUDA provide strong theoretical foundations and practical implementations. It is also worthwhile to investigate the NVIDIA CUDA programming guide which provides clear explanations of the API, along with recommendations for optimization and best practices. Finally, exploring open-source libraries that perform matrix operations via CUDA can provide real-world examples of advanced optimization techniques.
