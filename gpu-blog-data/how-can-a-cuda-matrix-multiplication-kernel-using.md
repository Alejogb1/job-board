---
title: "How can a CUDA matrix multiplication kernel using tiling be rewritten for OpenCL?"
date: "2025-01-30"
id: "how-can-a-cuda-matrix-multiplication-kernel-using"
---
My experience over the past several years in high-performance computing, particularly within GPU accelerated workflows, has demonstrated the performance benefits of memory access optimization when dealing with large matrices. When porting from CUDA to OpenCL, one of the critical aspects is adapting the core logic of memory access patterns, like those used in tiling, for the OpenCL programming model. Fundamentally, the process revolves around translating CUDA's thread hierarchy and shared memory mechanisms to OpenCL’s work-item and local memory concepts.

The CUDA kernel, optimized for tiling, breaks down matrix multiplication into smaller blocks, loading these blocks into shared memory, and then performing computations within those blocks. OpenCL achieves similar behavior using work-groups and local memory. The essential difference lies in the syntax and some abstraction differences between the two APIs. In essence, we're transitioning from a language tied to NVIDIA's hardware architecture to a more portable, hardware-agnostic language.

First, let's explore the core steps to port a CUDA tiled matrix multiplication kernel to OpenCL:

1. **Global and Local Index Translation:** In CUDA, `threadIdx.x`, `threadIdx.y`, `blockIdx.x`, and `blockIdx.y` represent thread and block IDs within a grid. OpenCL uses `get_global_id(0)`, `get_global_id(1)` for global indices and `get_local_id(0)`, `get_local_id(1)` for local indices, where the arguments indicate the dimension. You must carefully map these to the corresponding global and local work dimensions when configuring your kernel execution.

2. **Shared Memory to Local Memory:** CUDA’s shared memory is emulated by OpenCL using local memory, which is specified with the `__local` keyword. Allocating the size and type of this local memory is identical, but we need to make sure that we are mapping the global work-item’s access to a relevant location in the local memory.

3. **Synchronization Barriers:** CUDA's `__syncthreads()` must be replaced by OpenCL's `barrier(CLK_LOCAL_MEM_FENCE)` which provides barrier synchronization within a work-group. It is crucial to ensure that all work-items in a work-group reach this point before proceeding to computations, specifically after loading the local memory and before reading data from local memory.

4. **Global Memory Access:**  Both CUDA and OpenCL use a similar concept of global memory access, but OpenCL utilizes buffers as the main way to interact with global memory. Input matrices and the result matrix are stored as memory buffers and passed as arguments to the kernel. Indexing into these buffers must also be adjusted based on OpenCL’s global ID and the global size, matching the original access pattern in the CUDA kernel.

Let's illustrate this with code examples. We will assume a basic setup where a square matrix of dimension N is multiplied by itself, with tiling into blocks of dimension BLOCK_SIZE.

**Example 1:  CUDA Kernel (Conceptual)**

```c++
__global__ void matrixMulCUDA(float* A, float* B, float* C, int N) {
    const int BLOCK_SIZE = 16;
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    for(int k = 0; k < N; k += BLOCK_SIZE) {
        if (row < N && (k + threadIdx.y) < N) {
            As[threadIdx.y][threadIdx.x] = A[row * N + (k + threadIdx.x)];
        } else {
             As[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && (k + threadIdx.y) < N) {
            Bs[threadIdx.y][threadIdx.x] = B[(k+threadIdx.y) * N + col];
        } else {
             Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }


        __syncthreads();

        for(int i = 0; i < BLOCK_SIZE; i++) {
          sum += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        }
        __syncthreads();
    }
    if (row < N && col < N){
        C[row * N + col] = sum;
    }
}
```

*Commentary:* This CUDA kernel demonstrates the typical structure: it retrieves row and column indices, allocates shared memory, loads matrix blocks into this shared memory, and then performs the dot product within the block. Crucially, the access pattern relies on `threadIdx` and `blockIdx`.

**Example 2: Equivalent OpenCL Kernel**

```c
__kernel void matrixMulOpenCL(__global float* A, __global float* B, __global float* C, int N) {
    const int BLOCK_SIZE = 16;
    int row = get_global_id(1);
    int col = get_global_id(0);
    int localRow = get_local_id(1);
    int localCol = get_local_id(0);

    __local float As[BLOCK_SIZE][BLOCK_SIZE];
    __local float Bs[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    for(int k = 0; k < N; k += BLOCK_SIZE) {
        if(row < N && (k + localRow) < N) {
            As[localRow][localCol] = A[row * N + (k + localCol)];
        } else {
           As[localRow][localCol] = 0.0f;
        }

         if(col < N && (k + localRow) < N) {
           Bs[localRow][localCol] = B[(k+localRow) * N + col];
         } else {
           Bs[localRow][localCol] = 0.0f;
         }


        barrier(CLK_LOCAL_MEM_FENCE);

       for(int i = 0; i < BLOCK_SIZE; i++) {
            sum += As[localRow][i] * Bs[i][localCol];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
     if(row < N && col < N){
        C[row * N + col] = sum;
    }
}
```

*Commentary:* In this OpenCL kernel, `get_global_id` is used to determine global row and column values, and `get_local_id` is used to map each work item within the work group. Local memory `As` and `Bs` have the same role as the shared memory in CUDA, and the `barrier(CLK_LOCAL_MEM_FENCE)` replaces `__syncthreads()`. Note that the access patterns of both the input and output are mapped from the global indexes in both cases.

**Example 3: OpenCL Host Code (Partial Illustrative Example)**

```c++
#include <CL/cl.h>
#include <iostream>
#include <vector>

// ... Error check helper functions and OpenCL setup

int main() {
    // ... Platform and device selection, context and command queue setup

    const int N = 1024;
    const int BLOCK_SIZE = 16;
    size_t globalSize[2] = { N, N };
    size_t localSize[2] = { BLOCK_SIZE, BLOCK_SIZE };


    // Allocate host matrices A, B and C, fill A,B.
    std::vector<float> A(N*N), B(N*N), C(N*N, 0.0);
    //... fill A, B with data
    for (int i = 0; i<N*N; ++i){
      A[i] = 1.0f;
      B[i] = 1.0f;
    }

    // Create OpenCL buffers
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N, A.data(), NULL);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float) * N * N, B.data(), NULL);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * N * N, NULL, NULL);


    // Build kernel (Assume kernelSource is available, e.g. read from a text file)
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "matrixMulOpenCL", NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);
    clSetKernelArg(kernel, 3, sizeof(int), &N);

    // Launch kernel
    clEnqueueNDRangeKernel(commandQueue, kernel, 2, NULL, globalSize, localSize, 0, NULL, NULL);

    // Read back results
    clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0, sizeof(float) * N * N, C.data(), 0, NULL, NULL);

    // Clean up
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
   // ... Release context and queue.
    std::cout<<"result = "<< C[0] << std::endl;
    return 0;
}

```

*Commentary:*  This partial host code illustrates how to allocate and populate the input arrays, transfer them to the device using OpenCL buffers, execute the `matrixMulOpenCL` kernel, and read the results back to the host. The key is defining the `globalSize` and `localSize` variables, corresponding to global work items and work group size respectively. The clEnqueueNDRangeKernel launches the kernel. It showcases the necessary steps for preparing data and executing the OpenCL kernel on the chosen hardware.

In conclusion, migrating a tiled matrix multiplication from CUDA to OpenCL requires a detailed understanding of how thread hierarchies map to work-items and work-groups. The syntax and API differ, but the underlying principles of shared/local memory use, barrier synchronization, and global memory access patterns remain consistent. Resources such as vendor-specific OpenCL documentation, online tutorials, and academic papers on parallel computing provide more detailed insight on how to optimize this type of code. Understanding OpenCL’s explicit memory management, error handling and profiling is critical for developing robust and efficient GPU kernels.
