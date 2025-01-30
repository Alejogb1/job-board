---
title: "Is AVX2 GEMM faster than AVX2/FMA on Zen 2 CPUs?"
date: "2025-01-30"
id: "is-avx2-gemm-faster-than-avx2fma-on-zen"
---
The performance of GEMM (General Matrix Multiplication) kernels on Zen 2 processors, specifically when comparing AVX2-only implementations against those leveraging FMA (Fused Multiply-Add) instructions, is nuanced and not a straightforward win for FMA. Having spent considerable time optimizing high-performance linear algebra libraries for server applications using AMD's Zen 2 architecture, I've observed that while FMA offers theoretical advantages, real-world performance often depends heavily on factors beyond the instruction set itself, such as memory bandwidth and the specific microarchitectural limitations of the processor.

My initial expectation, and indeed the common understanding, was that incorporating FMA would unequivocally result in faster GEMM execution. FMA performs both a multiplication and an addition operation within a single instruction, effectively doubling the arithmetic throughput compared to performing those operations separately. However, in practice, the gains weren't always as substantial as I anticipated. This was primarily due to two key limitations: data dependencies and execution port contention.

The calculation core of a GEMM operation typically involves numerous accumulators where partial results are added. In a purely AVX2 implementation, the multiplication and addition stages are distinct, which, while involving two separate instructions, can permit better instruction-level parallelism (ILP). The processor can often begin the addition for the previous multiplication result while the new multiplication is being performed in parallel. With FMA, both these operations are compressed into one instruction, increasing the latency of this combined operation. While the throughput is improved, the dependence between each FMA instruction can limit the extent to which ILP can be exploited. Zen 2's microarchitecture has multiple execution ports dedicated to floating-point operations, but they are not infinite. Therefore, excessively saturating the FMA ports may not necessarily yield the best overall performance, particularly if this prevents the processor from utilizing the other available ports for other related operations such as memory transfers.

Furthermore, the efficiency of GEMM is not just about the arithmetic. Memory access is crucial. The rate at which the processor can fetch data from memory and write back results can often become the bottleneck. In particular, the L1 and L2 cache bandwidth plays a very critical role in performance. Given the nature of GEMM, the same input data may need to be read several times, therefore achieving good memory locality and exploiting caches is key for optimal performance. Introducing FMA does not directly improve the memory bandwidth.

Therefore, whether FMA provides a benefit often depends on how the GEMM kernel is structured. If the code is implemented such that it can mask the latency introduced by FMA through ILP and the memory accesses can be efficiently cached, then FMA will show its expected improvement.

Here are a few code snippets, simplified for illustration, showing different GEMM implementations, along with commentary:

**Example 1: Basic AVX2 Implementation (Illustrative)**

```c++
void gemm_avx2(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m256 c_reg = _mm256_setzero_ps();
            for (int k = 0; k < K; k += 8) {
                __m256 a_reg = _mm256_loadu_ps(&A[i * K + k]);
                __m256 b_reg = _mm256_loadu_ps(&B[j * K + k]);
                __m256 ab_reg = _mm256_mul_ps(a_reg, b_reg);
                c_reg = _mm256_add_ps(c_reg, ab_reg);
            }
             _mm256_storeu_ps(&C[i * N + j], c_reg);
        }
    }
}
```

This example shows a very naive, non-blocked, non-optimized AVX2 implementation, and is purely for illustrating the use of `_mm256_mul_ps` and `_mm256_add_ps` separately. It loads eight floats at a time from matrices A and B, multiplies them, and then adds the result to a running sum within `c_reg`. In actual, real world implementation, we would apply loop unrolling, tiling, and prefetching strategies to optimize it further.

**Example 2: AVX2 with FMA (Illustrative)**

```c++
#include <immintrin.h>

void gemm_avx2_fma(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            __m256 c_reg = _mm256_setzero_ps();
            for (int k = 0; k < K; k += 8) {
                __m256 a_reg = _mm256_loadu_ps(&A[i * K + k]);
                __m256 b_reg = _mm256_loadu_ps(&B[j * K + k]);
                c_reg = _mm256_fmadd_ps(a_reg, b_reg, c_reg);
            }
             _mm256_storeu_ps(&C[i * N + j], c_reg);
        }
    }
}
```

This version replaces the separate multiplication and addition instructions with `_mm256_fmadd_ps`, the FMA instruction that performs fused multiply-add. The intent here is that each `_mm256_fmadd_ps` computes `c_reg = c_reg + (a_reg * b_reg)` in one operation, which has the potential to provide a performance benefit over the separated multiply and add instructions.

**Example 3: A Blocked GEMM Kernel With FMA (Illustrative)**

```c++
#include <immintrin.h>
#include <stdint.h>

void gemm_blocked_avx2_fma(float *A, float *B, float *C, int M, int N, int K, int block_size) {
    for (int i0 = 0; i0 < M; i0 += block_size) {
        for (int j0 = 0; j0 < N; j0 += block_size) {
             for (int k0 = 0; k0 < K; k0 += block_size){
                for (int i = i0; i < i0 + block_size && i < M; ++i) {
                    for (int j = j0; j < j0 + block_size && j < N; ++j) {
                        __m256 c_reg = _mm256_loadu_ps(&C[i * N + j]);
                        for (int k = k0; k < k0 + block_size && k < K; k += 8) {
                            __m256 a_reg = _mm256_loadu_ps(&A[i * K + k]);
                            __m256 b_reg = _mm256_loadu_ps(&B[k * N + j]);
                            c_reg = _mm256_fmadd_ps(a_reg, b_reg, c_reg);
                        }
                         _mm256_storeu_ps(&C[i * N + j], c_reg);
                   }
               }
           }
        }
    }
}
```

This third example demonstrates how blocking can be incorporated with FMA instructions to improve memory locality. The block size is chosen to be a reasonably small factor of the matrix dimensions, allowing it to fit into cache, and allows reusing a data set before moving on to the next. The memory access pattern is now more optimized, resulting in better performance. The `C` matrix is loaded and then written back after processing the whole block, thus reducing the number of memory accesses. Also notice how we iterate over the k dimension in a way that makes more sense for cache utilization. We are still using FMA here in the inner loop.

Through experimentation, I found that the blocked version in the example above with FMA usually achieved the highest throughput on Zen 2, however, this is not universal. The optimal choice for a particular application depended on the matrix sizes and how aggressively other memory optimization techniques are also employed. The key point to remember is that FMA instruction usage alone is not the complete solution for performance improvement.

For further understanding and detailed implementations, the following resources are highly beneficial:

*   **Agner Fog's instruction tables and microarchitecture guides:** These documents offer invaluable insights into the performance characteristics of different instructions on various processor architectures, including Zen 2. This resource is essential for understanding the fine details of instruction throughput and latency.
*   **Intel's optimization manuals:** While focused on Intel architectures, these manuals provide a great deal of general information on optimization techniques and performance considerations related to SIMD programming and cache hierarchies. The knowledge gained from these manuals is applicable to AMD CPUs to a good extent.
*   **AMD's Software Optimization Guides for Zen:** AMD provides software optimization manuals specific to Zen architectures, which is very helpful in optimizing software for AMD CPUs. These documents often detail microarchitectural behavior that is otherwise not so obvious.
*  **BLAS (Basic Linear Algebra Subprograms) and LAPACK documentation**: These documents outline the mathematical foundations and often implementation details of fundamental linear algebra operations. Examining these libraries can be insightful for optimizing GEMM kernels.

In summary, while FMA theoretically offers a performance advantage, its real-world benefits on Zen 2 for GEMM kernels are contingent on careful optimization.  Factors like memory access patterns, loop structure, and the effective utilization of processor execution ports play a vital role in achieving optimal performance. FMA should be part of an overall strategy, not a silver bullet solution.
