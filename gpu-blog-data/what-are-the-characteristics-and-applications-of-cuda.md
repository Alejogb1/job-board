---
title: "What are the characteristics and applications of CUDA half-precision floating point?"
date: "2025-01-30"
id: "what-are-the-characteristics-and-applications-of-cuda"
---
The primary driver for half-precision floating-point (FP16) in CUDA is the reduction in memory bandwidth and computational load compared to single-precision (FP32) arithmetic, enabling faster processing and higher throughput, particularly within deep learning workloads. I’ve frequently observed speedups of up to 2x simply by migrating key tensor operations to FP16 on suitable hardware. These gains, however, require careful management of potential precision losses and the nuances of FP16 representation.

FP16, adhering to the IEEE 754 standard for binary floating-point arithmetic, uses 16 bits: one sign bit, five bits for the exponent, and ten bits for the mantissa (also known as significand). In contrast, FP32 uses 32 bits: one sign bit, eight exponent bits, and 23 mantissa bits. This difference directly translates into a smaller dynamic range and reduced precision for FP16. The limited exponent range means FP16 is more prone to overflow and underflow, and the fewer mantissa bits result in fewer representable numbers and larger gaps between those numbers.

A key characteristic is that FP16's limited precision can manifest in various forms during computation. The most prominent is a reduction in the number of representable digits after the decimal point, which leads to larger rounding errors during arithmetic operations. This becomes especially noticeable when dealing with very small or very large numbers, or when a series of arithmetic operations are involved. The consequence can be that FP16 outputs may deviate from FP32 equivalents, a divergence that can become severe if unchecked. This limitation means algorithms that rely on precise, iterative calculations (such as certain physics simulations, numerical solvers) may be unsuitable for pure FP16 execution without adjustment.

Another characteristic is the presence of subnormal numbers. These numbers, also referred to as denormalized numbers, are representable in both FP32 and FP16 but are not typically directly supported in older CUDA cores. They fill the gap between zero and the smallest representable normal number, allowing underflow to be gradual rather than abrupt. While this can help maintain numerical stability in some cases, it comes at a performance penalty since subnormal arithmetic requires more operations and generally slower handling by the hardware. Current CUDA architectures (Volta and later) can handle subnormal numbers with less overhead.

The reduced memory footprint of FP16 is a considerable advantage. Storing and moving FP16 data requires half the memory bandwidth compared to FP32. This reduction is invaluable in large-scale AI training models, where memory bandwidth is often the limiting factor. The impact of this reduced bandwidth requirement scales with the size of the datasets and network structures. Another advantage stems from the architecture of modern CUDA cores, which are designed for efficient FP16 computation. The Tensor Cores, first introduced in Volta architecture, have revolutionized the use of FP16 in AI. These specialized processing units can perform mixed-precision calculations (often using FP16 for matrix multiplication or convolution and FP32 for accumulation) at very high speeds, often significantly outperforming standard FP32 operations.

Applications of FP16 in CUDA are widespread, particularly within the domain of deep learning. In training, using FP16 for activations and weights while maintaining FP32 for gradients and loss calculations, a strategy known as mixed-precision training, allows for larger batch sizes, reduced training time, and improved memory utilization. Additionally, FP16 is heavily used for neural network inference. Post-training, converting models to use FP16 for the computations, provided the model is robust to the precision shift, can substantially accelerate inference time while requiring less memory, especially when deploying onto resource-constrained devices. Beyond AI, certain image processing algorithms and signal processing pipelines that are not critically sensitive to small loss in precision can leverage FP16's efficiency.

Here are three code examples with explanations of common usage patterns:

**Example 1: Simple Matrix Multiplication with FP16**

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMul(half* A, half* B, half* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        half sum = __float2half(0.0f);
        for (int k = 0; k < K; k++) {
            sum = __hadd(sum, __hmul(A[row * K + k], B[k * N + col]));
        }
        C[row * N + col] = sum;
    }
}

int main() {
    int M = 128; int N = 128; int K = 128;
    size_t sizeA = M * K * sizeof(half);
    size_t sizeB = K * N * sizeof(half);
    size_t sizeC = M * N * sizeof(half);

    half* h_A = (half*)malloc(sizeA);
    half* h_B = (half*)malloc(sizeB);
    half* h_C = (half*)malloc(sizeC);

    // Initialize h_A and h_B with data (omitted for brevity)

    half* d_A; cudaMalloc((void**)&d_A, sizeA); cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    half* d_B; cudaMalloc((void**)&d_B, sizeB); cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    half* d_C; cudaMalloc((void**)&d_C, sizeC);

    dim3 dimBlock(16, 16);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);

    matrixMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Use results in h_C

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}

```

This example showcases a basic matrix multiplication on the GPU utilizing `half` data types. Note the usage of `cuda_fp16.h` for half-precision specific functions, and that explicit casting using `__float2half` is used to convert from float to half. The `__hmul` and `__hadd` intrinsics perform the half-precision multiplication and addition, respectively.

**Example 2: Mixed Precision Operation using Tensor Cores (Simplified)**

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

// Example assumes a small matrix size and that input matrices and output are aligned for Tensor Core usage.
__global__ void tensorCoreMul(half* A, half* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for(int k = 0; k < K; k++) {
             float a_val = __half2float(A[row*K + k]);
             float b_val = __half2float(B[k*N + col]);
            sum += a_val * b_val;  //FP32 accumulation, but using FP16 input.
         }

        C[row * N + col] = sum;
    }
}

int main() {
    int M = 16; int N = 16; int K = 16;
    size_t sizeA = M * K * sizeof(half);
    size_t sizeB = K * N * sizeof(half);
    size_t sizeC = M * N * sizeof(float);

    half* h_A = (half*)malloc(sizeA);
    half* h_B = (half*)malloc(sizeB);
    float* h_C = (float*)malloc(sizeC);

    // Initialize h_A and h_B (omitted for brevity)

    half* d_A; cudaMalloc((void**)&d_A, sizeA); cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice);
    half* d_B; cudaMalloc((void**)&d_B, sizeB); cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    float* d_C; cudaMalloc((void**)&d_C, sizeC);

    dim3 dimBlock(4, 4);
    dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x, (M + dimBlock.y - 1) / dimBlock.y);

    tensorCoreMul<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, M, N, K);
    cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Use results in h_C

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);

    return 0;
}
```

This example simulates the behavior of Tensor Cores. While not directly utilizing the Tensor Core API, it illustrates how FP16 inputs can be multiplied and the results accumulated in FP32. This approach helps mitigate precision loss during accumulation. In practice, the CUDA API provides mechanisms to directly access Tensor Cores, for example using `wmma` namespace.

**Example 3: Mixed-Precision Data Storage**

```cpp
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

int main() {
    int size = 1024;
    float* h_data_fp32 = (float*)malloc(size * sizeof(float));
    half* h_data_fp16 = (half*)malloc(size * sizeof(half));

    // Initialize h_data_fp32 with some floating point data (omitted for brevity)

    // Convert and store in FP16
    for (int i = 0; i < size; i++) {
        h_data_fp16[i] = __float2half(h_data_fp32[i]);
    }

    float* d_data_fp32; cudaMalloc((void**)&d_data_fp32, size * sizeof(float));
    half* d_data_fp16; cudaMalloc((void**)&d_data_fp16, size * sizeof(half));

    cudaMemcpy(d_data_fp16, h_data_fp16, size * sizeof(half), cudaMemcpyHostToDevice);

    // Some processing using d_data_fp16. (omitted for brevity)

    // Convert d_data_fp16 to FP32 and store
    cudaMemcpy(d_data_fp32, d_data_fp16, size * sizeof(half), cudaMemcpyDeviceToHost);
    for(int i = 0; i< size; i++){
      h_data_fp32[i] = __half2float(h_data_fp16[i]);
    }


    // Use data in h_data_fp32. (omitted for brevity)

    free(h_data_fp32);
    free(h_data_fp16);
    cudaFree(d_data_fp32);
    cudaFree(d_data_fp16);

    return 0;
}
```

This example demonstrates the conversion and storage of data between FP32 and FP16.  The code illustrates a common workflow where FP32 input data is converted to FP16 for storage and manipulation on the GPU, showcasing the memory savings and subsequently converted back to FP32 for further processing if necessary, and the usage of `__half2float` to convert back.

For further exploration and best practices concerning CUDA, I would recommend the official CUDA programming guide provided by NVIDIA. Their documentation on CUDA libraries, especially cuBLAS and cuDNN, which are heavily used in deep learning, provides detailed guidance on using FP16 and mixed-precision techniques effectively. In addition, NVIDIA provides numerous examples and tutorials.  Books on parallel programming using CUDA offer a broad theoretical background in GPU computing, complementing specific information about FP16 handling. Finally, reading papers on mixed-precision training from research communities provide invaluable insights and up-to-date strategies.
