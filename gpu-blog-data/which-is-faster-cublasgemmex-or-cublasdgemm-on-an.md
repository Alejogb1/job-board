---
title: "Which is faster, cublasGemmEx or cublasDgemm on an A100?"
date: "2025-01-30"
id: "which-is-faster-cublasgemmex-or-cublasdgemm-on-an"
---
On an NVIDIA A100 GPU, `cublasGemmEx` will, in many scenarios, exhibit performance advantages over `cublasDgemm`, primarily due to its greater flexibility in data type handling and the potential for using optimized algorithms not available to `cublasDgemm`. My observations from numerous high-performance computing projects indicate that choosing between these routines depends heavily on the specific numerical precision requirements and the overall system architecture.

`cublasDgemm` operates exclusively on double-precision floating-point numbers (FP64). It performs the general matrix multiplication operation, C = αAB + βC, where A, B, and C are matrices, and α and β are scalar values. The limitation to FP64 arithmetic, while ensuring maximum precision, can be detrimental to performance when lower precisions suffice, as the underlying computational resources of the GPU are not being utilized optimally. `cublasDgemm` implementations in cuBLAS are highly optimized for its specific purpose. Nevertheless, the inherent overhead of double-precision arithmetic, in terms of both memory bandwidth and floating-point operations per second, often results in it being slower compared to what can be achieved with lower precision data.

`cublasGemmEx`, on the other hand, introduces a more versatile approach. This function permits the specification of different input and output data types through its `dataTypeA`, `dataTypeB`, `dataTypeC`, and `computeType` parameters. Crucially, this allows for the exploitation of tensor cores present in newer NVIDIA GPUs such as the A100, which are specialized units that deliver significantly higher performance for lower precision computations, particularly mixed-precision operations such as single-precision accumulation of half-precision (FP16) matrix products. By leveraging these tensor cores with `cublasGemmEx`, a substantial increase in throughput can often be achieved. Further, `cublasGemmEx` exposes the ability to choose a specific algorithm via `cublasGemmAlgo_t`, allowing further fine-tuning and optimization for certain matrix dimensions or layouts, which is not exposed by `cublasDgemm`. This parameter lets one select from the available gemm algorithm implementations of cuBLAS, offering potentially substantial gains if the application’s numerical requirements allow for it and the problem structure is suitable. In practice, I've found that profiling various algorithms offered by `cublasGemmEx` often reveals an optimal choice not obvious beforehand.

For a clear demonstration, I will present three code examples utilizing C++ and the cuBLAS library. The first will show a straightforward implementation of `cublasDgemm`, the second will illustrate the use of `cublasGemmEx` with FP32 input and output, and the third will demonstrate `cublasGemmEx` with FP16 input and FP32 output (mixed precision). These examples will be concise and focus on the core function calls. I'll omit memory allocation and CUDA context setup, as that is application dependent and not central to the performance comparison here. I'll use a small matrix size so the examples are runnable by others even on resource constrained machines.

**Code Example 1: `cublasDgemm` Implementation**

```cpp
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int m = 128;
    const int n = 128;
    const int k = 128;
    const double alpha = 1.0;
    const double beta = 0.0;
    double* dA, *dB, *dC; // Device memory pointers.
    
    cudaMalloc(&dA, m * k * sizeof(double));
    cudaMalloc(&dB, k * n * sizeof(double));
    cudaMalloc(&dC, m * n * sizeof(double));

    // Initialize dA, dB... Not shown for brevity.

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
               m, n, k,
               &alpha, dA, m,
               dB, k, &beta,
               dC, m);

    // ... Cleanup (cudaFree, cublasDestroy)... Not shown.

    return 0;
}
```

Here, `cublasDgemm` is invoked directly using double-precision matrix data.  The parameters `CUBLAS_OP_N` indicate no transposition on A and B. The matrix dimensions are given by `m, n, k`. The scalar multipliers are `alpha` and `beta`. Finally, the device pointers for the input matrices and result are included. While being straight forward, this example highlights the rigidness of data type specification: it only works on doubles.

**Code Example 2: `cublasGemmEx` with Single Precision**

```cpp
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int m = 128;
    const int n = 128;
    const int k = 128;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    float* dA, *dB, *dC; // Device memory pointers.
    
    cudaMalloc(&dA, m * k * sizeof(float));
    cudaMalloc(&dB, k * n * sizeof(float));
    cudaMalloc(&dC, m * n * sizeof(float));

    // Initialize dA, dB... Not shown for brevity.

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              m, n, k,
              &alpha, dA, CUDA_R_32F, m,
              dB, CUDA_R_32F, k, &beta,
              dC, CUDA_R_32F, m, CUDA_R_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP);

    // ... Cleanup (cudaFree, cublasDestroy)... Not shown.

    return 0;
}
```
This example demonstrates `cublasGemmEx` with all single-precision inputs and outputs, explicitly setting the data type to `CUDA_R_32F`. I have specified `CUBLAS_GEMM_DEFAULT_TENSOR_OP` to allow cuBLAS to select the appropriate algorithm for this. While this implementation has similar data type specification to `cublasDgemm`, its advantage lies in the fact that we now have the possibility to choose other data types or gemm algorithms. This example will often show a small performance boost over the `cublasDgemm` case, even though no tensor cores are being explicitly used here.

**Code Example 3: `cublasGemmEx` with Mixed Precision (FP16 Input, FP32 Output)**

```cpp
#include <iostream>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cuda.h>

int main() {
    cublasHandle_t handle;
    cublasCreate(&handle);

    const int m = 128;
    const int n = 128;
    const int k = 128;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    half* dA, *dB; // Device memory pointers.
    float* dC;
        
    cudaMalloc(&dA, m * k * sizeof(half));
    cudaMalloc(&dB, k * n * sizeof(half));
    cudaMalloc(&dC, m * n * sizeof(float));

    // Initialize dA, dB... Not shown for brevity.

    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
              m, n, k,
              &alpha, dA, CUDA_R_16F, m,
              dB, CUDA_R_16F, k, &beta,
              dC, CUDA_R_32F, m, CUDA_R_32F,
              CUBLAS_GEMM_DEFAULT_TENSOR_OP);


    // ... Cleanup (cudaFree, cublasDestroy)... Not shown.

    return 0;
}
```

This example uses FP16 for the input matrices A and B while accumulating the result into FP32, which is a very common mixed-precision pattern. The data types `CUDA_R_16F` and `CUDA_R_32F` are used to make this explicit. Here, the A100's tensor cores are engaged, leading to significantly faster computation compared to either of the previous two examples, assuming the matrix sizes are large enough. This example highlights how `cublasGemmEx` provides the flexibility necessary to exploit GPU hardware specific features and drastically improve execution times.

In summary, while `cublasDgemm` provides a robust and straightforward solution for double-precision matrix multiplications, `cublasGemmEx` offers a substantial advantage in performance due to its data type flexibility, access to mixed-precision operations, and the capacity to choose from various algorithms, especially when utilizing tensor cores available on an A100. The speed difference between them is often substantial and should be measured empirically on a problem by problem basis. I have found that the appropriate data types for an application is not always obvious, and is often arrived at by careful profiling.

For further information, I recommend consulting the NVIDIA cuBLAS documentation. Additionally, publications on NVIDIA's tensor core technology provide in depth knowledge. Finally, benchmarking suites such as those released by NVIDIA are invaluable for gaining a deeper understanding of the actual performance. Thorough understanding of the limitations and advantages of each of these functions can only come from a deep knowledge of both the API and the underlying hardware architecture, as well as careful empirical testing.
