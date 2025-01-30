---
title: "How can CUDA C++ be used to implement an odd-even sort?"
date: "2025-01-30"
id: "how-can-cuda-c-be-used-to-implement"
---
Odd-even sort, a relatively simple parallel sorting algorithm, benefits significantly from the computational power of CUDA. Its inherent structure, involving independent comparisons and swaps in alternating phases, aligns well with the Single Instruction, Multiple Data (SIMD) nature of GPUs. Leveraging CUDA, I’ve optimized odd-even sort implementations that regularly outperform their CPU counterparts, particularly for larger datasets, providing concrete experience in harnessing CUDA for parallel sorting operations.

At its core, odd-even sort operates in a series of phases. Each phase involves comparing and potentially swapping adjacent elements in the input array. Specifically, 'odd' phases compare and swap elements at odd indices (e.g., elements at positions 1 and 2, 3 and 4, etc.), while 'even' phases handle the elements at even indices (0 and 1, 2 and 3, etc.).  Importantly, these comparisons within a phase are independent of one another and thus suitable for parallel execution.  The algorithm continues through alternating odd and even phases until the entire array is sorted.

Implementing odd-even sort in CUDA C++ requires defining a kernel function that executes on the GPU, as well as memory management functions for transferring data between the host (CPU) and device (GPU). Crucially, the CUDA kernel executes for a subset of the array at a time, managed by threads, blocks, and grids. Each thread within a block compares and potentially swaps a specific pair of elements as dictated by the current phase, which is decided on the host. This contrasts with a typical serial implementation where the entire array is worked on via a single thread or process.

To illustrate, consider the first code example, which sets up a basic CUDA kernel for a single odd or even phase.

```cpp
__global__ void oddEvenKernel(int* data, int phase, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = 2;

    if (phase % 2 == 0) { // Even phase
        if (index * stride + 1 < n && index * stride % 2 == 0 )
        {
          if (data[index * stride] > data[index * stride + 1]) {
            int temp = data[index * stride];
            data[index * stride] = data[index * stride + 1];
            data[index * stride + 1] = temp;
          }
        }
    } else {  // Odd phase
        if (index * stride + 1 < n && index * stride % 2 != 0 )
        {
          if (data[index * stride] > data[index * stride + 1]) {
            int temp = data[index * stride];
            data[index * stride] = data[index * stride + 1];
            data[index * stride + 1] = temp;
          }
        }
    }
}
```

This `oddEvenKernel` function is executed by each thread on the GPU. The `index` calculates the thread's position in the array.  The `phase` input determines whether this is an odd or even phase comparison.  Each thread checks if it is within the array bounds and then, if necessary, compares and swaps adjacent array elements, `data[index * stride]` and `data[index * stride + 1]`.  The modular calculation in the condition ensures that odd or even index pairs are targeted based on the phase of execution. The use of strides ensures that only adjacent elements are compared in each phase. This kernel will be launched repeatedly from the host, alternating between even and odd values for the 'phase' parameter.  It's crucial to note that threads within each phase work on independent parts of the array.

The second example demonstrates how the host-side code might set up and call the kernel. This code will prepare data, copy it to the GPU, launch the kernel repeatedly, and return the sorted data back.

```cpp
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

void oddEvenSortGPU(int* data, int n) {
    int* d_data;
    size_t size = n * sizeof(int);

    cudaMalloc((void**)&d_data, size);
    cudaMemcpy(d_data, data, size, cudaMemcpyHostToDevice);

    int numBlocks = (n / 2 + 255) / 256 ;
    int numThreadsPerBlock = 256;

    for (int phase = 0; phase < n; ++phase) {
       oddEvenKernel<<<numBlocks, numThreadsPerBlock>>>(d_data, phase, n);
       cudaDeviceSynchronize();
    }

    cudaMemcpy(data, d_data, size, cudaMemcpyDeviceToHost);
    cudaFree(d_data);
}

int main() {
    int data[] = {5, 2, 8, 1, 9, 4, 7, 3, 6};
    int n = sizeof(data) / sizeof(data[0]);

    oddEvenSortGPU(data, n);

    std::cout << "Sorted Array: ";
    for (int i = 0; i < n; ++i) {
        std::cout << data[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

The `oddEvenSortGPU` function first allocates device memory using `cudaMalloc` and copies the host data to the device using `cudaMemcpy`. The number of blocks is calculated to efficiently utilize the GPU resources. Inside the loop, the `oddEvenKernel` is launched with the appropriate `phase`. The call to `cudaDeviceSynchronize` ensures that each kernel launch completes before the next iteration. This is crucial to guarantee that updates made by the kernel are available before the next phase begins. After the sorting is completed, data is copied back to the host. The `main` function then calls the host function, and outputs the result. The use of error checking for memory allocation and CUDA operations is implied but left out for brevity.

The final example shows how one might optimize the kernel by leveraging shared memory within blocks for faster data access and reduce global memory access latency. This implementation only targets elements within a block.

```cpp
__global__ void optimizedOddEvenKernel(int* data, int phase, int n) {
  extern __shared__ int sharedData[];
  int tid = threadIdx.x;
  int index = blockIdx.x * blockDim.x + tid;
  int stride = 2;

  if(index < n) {
      sharedData[tid] = data[index];
  }
  __syncthreads();

  if (phase % 2 == 0) {
    if(tid * stride + 1 < blockDim.x && tid * stride % 2 == 0) {
        if(sharedData[tid*stride] > sharedData[tid*stride+1]){
          int temp = sharedData[tid*stride];
          sharedData[tid*stride] = sharedData[tid*stride+1];
          sharedData[tid*stride+1] = temp;
        }
    }
  }
   else {
    if(tid * stride + 1 < blockDim.x && tid * stride % 2 != 0) {
        if(sharedData[tid*stride] > sharedData[tid*stride+1]){
          int temp = sharedData[tid*stride];
          sharedData[tid*stride] = sharedData[tid*stride+1];
          sharedData[tid*stride+1] = temp;
        }
    }
  }
  __syncthreads();

  if(index < n) {
      data[index] = sharedData[tid];
  }
}
```

In this optimized kernel, we allocate shared memory, which is faster than global memory. Initially, all threads copy their respective data elements from global memory to shared memory, using `__shared__ int sharedData[]`. The call to `__syncthreads()` guarantees all data transfer to shared memory has completed before proceeding with comparison and swapping.  The logic of comparison and swapping is essentially the same as in previous example except using shared memory, and only for pairs within each block. Lastly, we write data back to global memory. This reduces the overall latency by utilizing the faster shared memory. The launch of this kernel requires specifying the amount of shared memory to be used via kernel launch syntax.

Further optimizations can be achieved by incorporating different block sizes to ensure optimal GPU occupancy, exploring memory access patterns to maximize memory bandwidth, and reducing unnecessary synchronization barriers. Additionally, for extremely large datasets, considering data decomposition techniques can further enhance performance.

For deeper understanding of CUDA programming, refer to the NVIDIA CUDA Programming Guide and the CUDA C++ Best Practices Guide. Also, books focused on parallel algorithm design and GPU architecture provide detailed theoretical background.  Understanding how specific GPU architectures handle memory and thread scheduling is fundamental for writing efficient CUDA code, and these guides provide that crucial information.
