---
title: "How can CUDA be used to parallelize array operations?"
date: "2025-01-30"
id: "how-can-cuda-be-used-to-parallelize-array"
---
The core strength of CUDA lies in its ability to leverage the massively parallel architecture of NVIDIA GPUs for computationally intensive tasks, particularly those involving array operations. I've spent several years optimizing scientific simulations, and the performance gain achieved by transitioning array processing from CPU to GPU using CUDA is often dramatic, sometimes exceeding an order of magnitude. This stems from the GPU’s architecture, featuring thousands of cores capable of executing the same instruction on different data, which aligns perfectly with the typical operations we perform on arrays—element-wise addition, multiplication, or more complex functions.

Fundamentally, CUDA allows us to define *kernels*, which are functions executed by these individual GPU cores. When you launch a CUDA kernel, you specify the grid and block dimensions. The *grid* represents the overall arrangement of blocks, and each *block* contains multiple threads that execute the kernel. A single *thread* is the fundamental unit of execution. Thus, we map our array operation to this structure, letting each thread process a small section of our array. This parallel execution is how we achieve the performance gains.

To illustrate this, consider a simple element-wise addition of two arrays. In a traditional CPU setting, this would typically involve a loop. In CUDA, we would instead launch a kernel where each thread adds corresponding elements from two arrays and stores the result in a third array.

Here's a basic example in CUDA C++ demonstrating this concept.

```cpp
#include <iostream>
#include <cuda.h>

// CUDA kernel for element-wise addition
__global__ void addArrays(float *a, float *b, float *c, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
      c[i] = a[i] + b[i];
  }
}

int main() {
  int size = 1024; // Example array size
  size_t memSize = size * sizeof(float);

  // Host memory allocation
  float *h_a = new float[size];
  float *h_b = new float[size];
  float *h_c = new float[size];

  // Initialize host arrays
  for (int i=0; i < size; ++i) {
      h_a[i] = static_cast<float>(i);
      h_b[i] = static_cast<float>(i * 2);
  }

  // Device memory allocation
  float *d_a, *d_b, *d_c;
  cudaMalloc((void**)&d_a, memSize);
  cudaMalloc((void**)&d_b, memSize);
  cudaMalloc((void**)&d_c, memSize);

  // Copy host arrays to device arrays
  cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, h_b, memSize, cudaMemcpyHostToDevice);


  // Define grid and block dimensions
  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;


  // Launch the kernel
  addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

  // Copy result from device to host
  cudaMemcpy(h_c, d_c, memSize, cudaMemcpyDeviceToHost);

  // Verify the result (optional)
  // for (int i=0; i<size; ++i) {
  //   std::cout << h_c[i] << " ";
  // }
  // std::cout << std::endl;

  // Cleanup
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  delete[] h_a;
  delete[] h_b;
  delete[] h_c;

  return 0;
}
```

This code first allocates memory on both the host (CPU) and device (GPU). We initialize the host arrays `h_a` and `h_b` with sample data, copy this data to the device arrays `d_a` and `d_b`, then launch the `addArrays` kernel, where the index `i` is calculated by `blockIdx.x * blockDim.x + threadIdx.x`.  This allows us to correctly map a thread to an element of the input arrays. The `<<<blocksPerGrid, threadsPerBlock>>>` syntax specifies the number of thread blocks and threads per block. The result is copied back to host array `h_c`. Key operations here include `cudaMalloc` for device memory allocation, `cudaMemcpy` for data transfer, and the kernel launch syntax, `<<<...>>>`. This example shows the basic structure of a typical CUDA implementation for array operations.

Beyond simple element-wise operations, CUDA’s power becomes even more apparent with more complex array manipulations. Consider the reduction operation, which combines all elements of an array into a single value using some operation, such as summation. This operation is not inherently parallelizable, but carefully designed CUDA kernels can achieve significant speedup by applying a hierarchical approach where each block computes a partial sum and then these partial sums are summed. Here is an example of how a reduction operation can be parallelized using CUDA.

```cpp
#include <iostream>
#include <cuda.h>
#include <limits>

__global__ void reduceArray(float *d_in, float *d_out, int size) {

  __shared__ float sharedMem[256];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + tid;

  sharedMem[tid] = (i < size) ? d_in[i] : 0;

  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sharedMem[tid] += sharedMem[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    d_out[blockIdx.x] = sharedMem[0];
  }
}


int main() {
  int size = 1024;
  size_t memSize = size * sizeof(float);

  float *h_in = new float[size];
  float h_out_cpu = 0;

  for (int i=0; i<size; ++i) {
      h_in[i] = 1.0f;
      h_out_cpu += h_in[i];
  }

  float *d_in, *d_out;
  cudaMalloc((void**)&d_in, memSize);
  cudaMalloc((void**)&d_out, (size / 256 + 1)* sizeof(float));


  cudaMemcpy(d_in, h_in, memSize, cudaMemcpyHostToDevice);


  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  reduceArray<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, size);


  float* h_out_partial = new float[blocksPerGrid];
  cudaMemcpy(h_out_partial, d_out, (size / 256 + 1) * sizeof(float), cudaMemcpyDeviceToHost);

  float h_out_gpu = 0.0f;
  for(int i =0; i < blocksPerGrid; ++i)
      h_out_gpu += h_out_partial[i];

  std::cout << "CPU sum: " << h_out_cpu << std::endl;
  std::cout << "GPU sum: " << h_out_gpu << std::endl;


  cudaFree(d_in);
  cudaFree(d_out);
  delete[] h_in;
  delete[] h_out_partial;

  return 0;
}
```

This reduction kernel utilizes shared memory, which is a fast on-chip memory accessible by all threads within a block. The algorithm combines elements within a block using a tree-like reduction pattern. The partial sums from each block are written to the `d_out` array.  A final reduction would be needed for larger array sizes. The `__syncthreads()` ensures all threads within the block have completed their operations before the next phase. While this example only reduces sums, the same pattern can be extended for other reduction operations like max, min or logical operations. This provides a powerful general pattern for efficient parallel computations.

For applications requiring data movement between arrays, such as transposition or matrix multiplication, CUDA can also perform these efficiently. In my work with image processing, matrix multiplication is a common operation. While naive implementations can be inefficient on GPUs due to memory access patterns, tiled algorithms or using highly-optimized CUDA libraries (like cuBLAS) can lead to significantly improved performance. Here's a simplified CUDA implementation of matrix multiplication:

```cpp
#include <iostream>
#include <cuda.h>
#include <vector>

// Kernel for matrix multiplication
__global__ void matrixMultiply(float *a, float *b, float *c, int widthA, int widthB, int widthC) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float sum = 0.0f;
    if (row < widthA && col < widthB) {
        for (int i = 0; i < widthC; i++) {
            sum += a[row * widthC + i] * b[i * widthB + col];
        }
        c[row * widthB + col] = sum;
    }
}

int main() {
    int widthA = 100;
    int widthB = 100;
    int widthC = 100;

    size_t sizeA = widthA * widthC * sizeof(float);
    size_t sizeB = widthC * widthB * sizeof(float);
    size_t sizeC = widthA * widthB * sizeof(float);

     // Initialize host matrices with sample data (fill with simple values for demonstration)
    std::vector<float> h_a(widthA * widthC);
    std::vector<float> h_b(widthC * widthB);
    std::vector<float> h_c(widthA * widthB, 0.0f);

    for (int i = 0; i < widthA * widthC; i++) h_a[i] = 1.0f;
    for (int i = 0; i < widthC * widthB; i++) h_b[i] = 1.0f;

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, sizeA);
    cudaMalloc((void**)&d_b, sizeB);
    cudaMalloc((void**)&d_c, sizeC);

    cudaMemcpy(d_a, h_a.data(), sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((widthB + threadsPerBlock.x - 1) / threadsPerBlock.x,
                      (widthA + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, widthA, widthB, widthC);

    cudaMemcpy(h_c.data(), d_c, sizeC, cudaMemcpyDeviceToHost);

    // Verification is not shown

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

This example uses a block and thread arrangement that allows multiple threads to participate in the calculation. Each thread is responsible for calculating one element of the resulting matrix. Optimized matrix multiplications use more advanced tiling techniques, which reduce memory access bottlenecks.  In practical scenarios,  I consistently favor using specialized libraries like cuBLAS for such complex operations as they are much more optimized.

When starting with CUDA for array operations, it is beneficial to familiarize oneself with the core CUDA programming guide, focusing on memory management, kernel launch configuration, and the various memory models available within the GPU architecture. Understanding thread indexing is crucial and the various optimization strategies available should be examined.  For more complex operations, I would recommend studying how to use libraries such as cuBLAS, cuFFT, and Thrust. These tools will significantly ease the process of developing optimized CUDA applications.
