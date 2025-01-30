---
title: "Can nvprof (or alternative methods) determine if kernel execution utilized Tensor Cores?"
date: "2025-01-30"
id: "can-nvprof-or-alternative-methods-determine-if-kernel"
---
Tensor Cores, specialized processing units on modern NVIDIA GPUs, accelerate matrix multiplication and convolution operations pivotal in deep learning and other high-performance computing domains. Detecting their utilization during kernel execution is crucial for performance optimization. While `nvprof` was historically the primary tool for such analysis, its deprecation necessitates a shift towards alternatives, primarily NVIDIA Nsight Systems and Nsight Compute.

From my experience analyzing numerous CUDA kernels for machine learning applications, precisely identifying Tensor Core usage requires more than a simple runtime profile. It demands examining hardware metrics exposed by the profiling tools. A high percentage of floating-point operations, especially fused multiply-add (FMA) operations, is not sufficient evidence. Tensor Core activity needs specific flags from the hardware counters. The presence of `sm__sass_thread_inst_executed_op_tensor_core_fma_i` or similar counters in the profiling output is a strong indication that Tensor Cores were engaged during kernel execution. Without these specific counters, merely observing higher throughput could imply other performance optimizations rather than active Tensor Core usage. The effectiveness of Tensor Core utilization also depends heavily on data layout and data types.

Let's consider a basic matrix multiplication kernel. A naive implementation may not use Tensor Cores automatically. Proper layout and type considerations are critical.

**Example 1: Naive Matrix Multiplication (Likely No Tensor Core Usage)**

```cpp
__global__ void matrixMultiplyNaive(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; ++k) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}
```

In this kernel, the direct memory access patterns and data type make it unlikely for Tensor Cores to be utilized by the compiler. The float data type will normally use the standard CUDA cores. When profiling this kernel with Nsight Systems or Nsight Compute, I would not expect to see the specialized Tensor Core instruction counters active. The performance bottleneck will lie in the memory access, not necessarily arithmetic, because this is a naive matrix multiplication.

**Example 2: Optimized Matrix Multiplication (Tensor Core Candidates)**

```cpp
#include <mma.h>
using namespace nvcuda;

__global__ void matrixMultiplyOptimized(float* A, float* B, float* C, int width) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < width && col < width) {
    float sum = 0.0f;
    for (int k = 0; k < width; k+=16) {

        mma_sync<float, 16, 16, 16, true> mma;
        float a[16];
        float b[16];
        float c[16];


        for (int i = 0; i < 16; ++i) {
            a[i] = A[row * width + k + i];
            b[i] = B[(k+i) * width + col];
        }


        mma.load_a(a);
        mma.load_b(b);
        mma.load_c(c);
        mma.mma();
        mma.store_c(c);

         for(int i = 0; i < 16; i++){
            sum+=c[i];
        }


    }

       C[row * width + col] = sum;
  }
}
```

Here, I've introduced `mma_sync`, directly invoking the matrix multiply-accumulate functionality offered by the `nvcuda` namespace. This example explicitly attempts to leverage Tensor Cores by loading input fragments of 16x16 matrices.  If this code successfully compiles targeting an architecture with Tensor Cores and the correct compiler flags are enabled (e.g., using flags like `-arch=sm_XX`, where XX is the target GPU architecture), the performance profile will likely exhibit an increased number of `sm__sass_thread_inst_executed_op_tensor_core_fma_i` instructions. The data alignment and the type of operations being performed are suitable for Tensor Core acceleration.  Note that this is just one example and the implementation of the matrix multiplication using MMA will change from one generation of the GPU to another.

**Example 3: Data Type Manipulation (Tensor Core Optimization)**

```cpp
#include <mma.h>
using namespace nvcuda;

__global__ void matrixMultiplyMixedPrecision(half* A, half* B, float* C, int width) {
     int row = blockIdx.y * blockDim.y + threadIdx.y;
     int col = blockIdx.x * blockDim.x + threadIdx.x;


    if (row < width && col < width) {
    float sum = 0.0f;
    for (int k = 0; k < width; k+=16) {

        mma_sync<half, 16, 16, 16, true> mma;
        half a[16];
        half b[16];
        float c[16];


        for (int i = 0; i < 16; ++i) {
            a[i] = A[row * width + k + i];
            b[i] = B[(k+i) * width + col];
        }


        mma.load_a(a);
        mma.load_b(b);
        mma.load_c(c);
        mma.mma();
        mma.store_c(c);

         for(int i = 0; i < 16; i++){
            sum+=c[i];
        }

    }

     C[row * width + col] = sum;

   }
}
```

This kernel uses half-precision floating-point (half) for input matrices `A` and `B`, and single-precision float for the accumulator `C`. Many Tensor Core implementations perform particularly well with half-precision, and if the architecture supports it, the use of this data type will substantially improve Tensor Core usage. The specific combination of input data types influences what operations Tensor Cores can perform. I've seen many cases where changing the input data type results in dramatic performance gains due to the effective utilization of Tensor Cores, and I have often used this technique during the optimization process. Again, a profile will reveal the presence of `sm__sass_thread_inst_executed_op_tensor_core_fma_i` (or related counters) should Tensor Cores be active.  The data movement overhead may also change when using half data types due to the change in the number of bytes moved.

**Profiling Methods**

To verify if Tensor Cores are active, use Nsight Systems or Nsight Compute, available as part of the NVIDIA Nsight development suite. Nsight Systems provides a high-level overview of the application, showing kernel execution times, API calls, and other system-wide performance characteristics. It helps identify hotspots in your application. After profiling with Nsight Systems, I usually switch to Nsight Compute when a bottleneck in a CUDA kernel is identified.

Nsight Compute performs detailed kernel analysis, providing a vast array of hardware performance counters, including specific metrics related to Tensor Cores, such as those counting FMA operations performed by the units. In the reports generated by Nsight Compute, these counters will either be present, indicating Tensor Core usage, or absent, indicating otherwise.  The reported instruction counters can be searched for directly, in the output of the command line profiler `nv-nsight-cu-cli`.

**Resource Recommendations**

For deep dives into CUDA programming, the official NVIDIA CUDA documentation is indispensable. The specific sections detailing the warp-level functions (`__wmma` intrinsics) as well as the matrix instruction (`mma_sync` function) are essential for understanding Tensor Core acceleration. Additionally, NVIDIA publishes programming guides and blog posts detailing the use of Tensor Cores across different GPU architectures which are a great place to start reading about new features in newer GPU generations. When working on a specific area of the code, it's always worth looking for examples from NVIDIA's code repositories, such as the CUDA samples. Lastly, NVIDIA webinars and presentations, often available on their developer website, offer valuable insights and advanced techniques for optimizing GPU kernels for deep learning and scientific computing. These are more targeted than the general documentation. Through consistent study of these resources, the process of debugging and optimization can be carried out with much greater ease.
