---
title: "What are CUDA compiler macros?"
date: "2025-01-30"
id: "what-are-cuda-compiler-macros"
---
CUDA compiler macros, in essence, are preprocessor directives recognized and processed by the NVIDIA CUDA compiler, `nvcc`, before the actual compilation stage into machine code. They’re not inherent to the C++ language itself, but rather an extension provided by the CUDA toolchain to facilitate conditional compilation, control code behavior based on the target GPU architecture, and manage memory layout details. My experience working on high-performance computing applications using CUDA for the last five years has consistently shown the indispensable nature of these macros in crafting portable and optimized code.

The primary purpose of CUDA compiler macros is to enable selective compilation of code blocks. This stems from the variability in NVIDIA GPU architectures, where features and capabilities can differ drastically between different generations (e.g., Kepler, Maxwell, Pascal, Volta, Turing, Ampere, Hopper). Different architectures might have different memory access patterns, register usage, instruction sets, and warp sizes. Hardcoding for a single architecture is not only brittle; it also drastically reduces performance on other devices. Using macros allows us to write a single codebase that adapts at compile time to different architectural features, without runtime overhead associated with conditionals. This process involves checking a macro defined by `nvcc` against the current compilation target and only compiling the blocks under conditional statements where the criteria are met.

CUDA compiler macros can be categorized into several groups based on their function. Some relate to device architecture (e.g., identifying the compute capability or the GPU architecture's name), while others concern themselves with hardware features (e.g., the presence of atomic operations or shared memory sizes). There are macros that help manage thread indexing within a kernel, like `threadIdx`, `blockIdx`, `blockDim`, and `gridDim`, which, although not technically preprocessor macros, are essential to the kernel execution model and work very much in the same way. Finally, others are utilized for debugging or diagnostics.

My first project where these macros became indispensable was a molecular dynamics simulation code. The initial version was architected for a Pascal GPU; however, I needed it to work on the older Maxwell architecture as well, without sacrificing performance. This is where the macro `__CUDA_ARCH__` came into focus. The following example will illustrate its use.

```c++
#include <cuda.h>

__global__ void myKernel(float* out, const float* in, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        #if __CUDA_ARCH__ >= 600
           // For Pascal and later architectures (compute capability 6.0+)
           // Pascal has better support for fp16 arithmetic so we can use them
           half h = __float2half(in[i]);
           out[i] = __half2float(h) * 2.0f; 
        #else
           // For Maxwell and earlier architectures (compute capability < 6.0)
           // No native fp16 support, use regular fp32 math
           out[i] = in[i] * 2.0f;
        #endif
     }
}
```

In this example, the `#if __CUDA_ARCH__ >= 600` directive checks if the target architecture's compute capability is greater than or equal to 6.0, which corresponds to Pascal and later GPUs. If true, the code uses the CUDA intrinsic functions `__float2half` and `__half2float` for reduced-precision floating-point arithmetic (fp16) which improves the throughput for memory-bound calculations on Pascal and later GPUs, where such support was improved. Otherwise, the code will use traditional fp32.  The `nvcc` compiler will strip away the code branch that does not apply to the architecture targeted during compilation, avoiding runtime conditional checks. This example allowed me to target a wider range of GPUs, by making use of advanced instructions available only on more modern hardware, while retaining performance on legacy hardware.

Another significant macro that I have found very useful is `__CUDA_API_PER_THREAD_DEFAULT_STREAM`. Consider a scenario where we want to launch kernels on a per-thread basis, perhaps when doing asynchronous operations using CUDA streams. Let’s assume that you are working with multiple streams and wish to launch a different kernel on each of those streams. In a previous version of the software, I had to explicitly manage the stream assignment for each thread, which was a substantial pain. This is when I discovered that `__CUDA_API_PER_THREAD_DEFAULT_STREAM` had just been released.

```c++
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void myStreamKernel(float* data) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    data[i] = 123.0f;
    // Simulate processing by occupying the device for a short time.
    // In a real use case, replace with meaningful calculations
    for (int j = 0; j < 100000; j++)
    {
        float some_var = 1.0f;
        some_var = some_var * 1.0f;
    }
}


int main() {
    const int N = 1024;
    float* deviceData;
    cudaMalloc((void**)&deviceData, N * sizeof(float));

    #if __CUDA_API_PER_THREAD_DEFAULT_STREAM
        printf("Per-thread default stream API is active.\n");
        myStreamKernel<<<N/256, 256>>>(deviceData);
        cudaDeviceSynchronize(); // Necessary to wait for all kernels to finish
    #else
        printf("Per-thread default stream API is not active.\n");
        cudaStream_t streams[N / 256];
        for (int i = 0; i < N / 256; i++) {
            cudaStreamCreate(&streams[i]);
            myStreamKernel<<<1, 256, 0, streams[i]>>>(deviceData + i * 256);

        }
        for (int i = 0; i < N/256; i++)
            cudaStreamSynchronize(streams[i]);
        cudaDeviceSynchronize();
    #endif

    float hostData[N];
    cudaMemcpy(hostData, deviceData, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Validate the result
    for (int i = 0; i < N; i++) {
      if (hostData[i] != 123.0f){
        printf("Error! Element %d has value %.2f\n", i, hostData[i]);
        return 1;
        }
    }
    printf("Kernel completed successfully with each element being 123.0f\n");

    cudaFree(deviceData);
    #ifndef __CUDA_API_PER_THREAD_DEFAULT_STREAM
        for (int i = 0; i < N/256; i++) {
             cudaStreamDestroy(streams[i]);
        }
    #endif
    return 0;
}
```

Here, the code checks the macro `__CUDA_API_PER_THREAD_DEFAULT_STREAM`. When the macro is defined, it implies the CUDA runtime API will automatically create and assign a unique default stream for each thread, so that the kernel launches on different streams automatically. If not available, we must explicitly create and manage a number of streams, then manually assign kernel executions to different streams. This macro enabled simplification of complex asynchronous operations, reducing boilerplate and potential bugs. It also enabled each thread to execute without interfering with others threads of the same kernel instance, which increases occupancy as the program avoids synchronization bottlenecks.

Lastly, `__CUDA_ARCH__` can be further used to conditionally select different kernel implementations depending on the architecture, often for optimization purposes. Here's an example where we choose different shared memory usage. Note, the amount of shared memory available per block is also fixed and varies between generations.

```c++
#include <cuda.h>

__global__ void mySharedMemKernel(float* out, const float* in, int size) {
    extern __shared__ float sdata[];

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        #if __CUDA_ARCH__ >= 700
            //Volta or later, can potentially use larger shared memory
            // Using more shared memory can improve access performance.
            sdata[threadIdx.x] = in[i];
            __syncthreads();
            out[i] = sdata[threadIdx.x];
        #else
            // Older architectures, can use a more minimal amount
            sdata[threadIdx.x] = in[i];
            __syncthreads();
            out[i] = sdata[threadIdx.x] * 2.0f;

        #endif
    }
}

// Example function that calls the kernel.
void kernelLaunch(float* d_out, float* d_in, int size, int blockSize)
{
    #if __CUDA_ARCH__ >= 700
        mySharedMemKernel<<< (size + blockSize - 1) / blockSize, blockSize, blockSize * sizeof(float)>>>(d_out, d_in, size);
    #else
         mySharedMemKernel<<< (size + blockSize - 1) / blockSize, blockSize, blockSize * sizeof(float)>>>(d_out, d_in, size);
    #endif
}
```

Here, based on whether the compute capability is greater than or equal to 7.0 (Volta architecture and later), the kernel can use a variable amount of shared memory to hold the input data. This highlights that different CUDA architectures may have different capabilities or limits for shared memory, as well as different optimal uses. The kernel launch uses the `blockSize*sizeof(float)` to pass the size of shared memory in bytes as a third argument. This ability is particularly helpful in tuning kernels for specific performance characteristics.

For anyone seeking to deepen their understanding of these macros and their implications, I would recommend the official NVIDIA CUDA Toolkit documentation. In particular, the sections on "CUDA Programming Guide," "CUDA Runtime API," and "nvcc" will contain all relevant information about the available macros, intrinsics, and compiler behavior. Also, reviewing the example CUDA projects that NVIDIA often releases as part of their toolkit installation will give tangible examples of these macros being used in real applications. Open-source projects written in CUDA are also invaluable educational tools. Exploring their implementations helps to discern not just what macros are available but *when* to use them and *how*.
