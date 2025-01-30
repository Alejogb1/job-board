---
title: "How to ensure CUDA computations are complete?"
date: "2025-01-30"
id: "how-to-ensure-cuda-computations-are-complete"
---
The asynchronous nature of CUDA kernels necessitates explicit mechanisms to guarantee completion, a point I've frequently encountered when debugging performance-critical applications in high-performance computing. The GPU executes kernels concurrently with the host CPU, and without proper synchronization, data races and incorrect results are inevitable. Completion of CUDA computations isn’t inherent; it requires deliberate steps to enforce order.

The core challenge arises because kernel launches are *non-blocking*. When `cudaLaunchKernel` is called, it returns immediately to the CPU, without waiting for the GPU to finish. The execution occurs in a separate stream of operations. Unless the programmer takes action, the CPU could begin modifying data that the GPU is concurrently processing, or attempt to read data before the GPU has finished writing it. This asynchronicity, while enabling performance, requires explicit synchronization to preserve program correctness.

The primary means of ensuring CUDA computations are complete are: explicit synchronization using `cudaDeviceSynchronize`, event-based synchronization, and implicit synchronization via stream semantics. Each approach has specific use cases and performance implications.

**1. Explicit Synchronization with `cudaDeviceSynchronize`**

The most direct, though potentially performance-limiting, method is to invoke `cudaDeviceSynchronize`. This function blocks the host CPU thread until *all* previously launched CUDA kernels and operations on the specified device have completed. It essentially creates a global barrier for the entire device.

*Code Example 1:*
```c++
#include <cuda_runtime.h>
#include <iostream>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
    const int n = 1024;
    size_t size = n * sizeof(float);
    float *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;

    // Allocate host memory
    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c = (float*)malloc(size);

    // Initialize host data
    for(int i = 0; i < n; ++i){
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2*i);
    }


    // Allocate device memory
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);


    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    // Synchronize the device
    cudaDeviceSynchronize();

    // Copy results from device to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Verify results (example)
    for (int i=0; i < n; i++){
        if (abs(h_c[i] - (h_a[i] + h_b[i])) > 0.00001){
            std::cout << "Error at index " << i << ": Expected " << (h_a[i] + h_b[i]) << ", got " << h_c[i] << std::endl;
        }
    }
     std::cout << "Vector Addition Complete" << std::endl;

    // Free allocated memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
```

In this example, `cudaDeviceSynchronize` ensures that the `vectorAdd` kernel has completed *before* the results are copied back to the host via `cudaMemcpy`, guaranteeing that the data in `d_c` is accurate. Without the `cudaDeviceSynchronize`, the copy to `h_c` could start while the kernel is still executing, leading to incorrect values. This approach is straightforward but introduces a complete halt to host execution, which may be undesirable in scenarios where other processing could be performed concurrently.

**2. Event-based Synchronization**

CUDA events offer more granular control over synchronization. Events record points in time within a CUDA stream and enable synchronization relative to those points. This approach allows for overlapping host and device work more effectively than `cudaDeviceSynchronize`.

*Code Example 2:*
```c++
#include <cuda_runtime.h>
#include <iostream>

__global__ void matrixMultiply(float *A, float *B, float *C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}


int main() {
    const int M = 256;
    const int N = 256;
    const int K = 256;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    // Allocate Host Memory
    h_A = (float*)malloc(size_A);
    h_B = (float*)malloc(size_B);
    h_C = (float*)malloc(size_C);

    // Initialize host data
    for(int i = 0; i < M * K; i++){
        h_A[i] = static_cast<float>(i);
    }

    for(int i=0; i < K * N; i++){
        h_B[i] = static_cast<float>(2*i);
    }



    // Allocate device memory
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    // Copy data from host to device
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (M + threadsPerBlock.y - 1) / threadsPerBlock.y);
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, M, N, K);

    // Create and record an event after kernel launch
    cudaEvent_t event;
    cudaEventCreate(&event);
    cudaEventRecord(event, 0);  // Record event on stream 0

    // Asynchronous copy
    cudaMemcpyAsync(h_C, d_C, size_C, cudaMemcpyDeviceToHost, 0);

    // Wait for event to complete
    cudaEventSynchronize(event);
    cudaEventDestroy(event);

    // Verify the computation
    bool error = false;

    for(int row = 0; row < M; ++row){
        for(int col = 0; col < N; ++col){
                float expectedSum = 0.0f;
                for (int k=0; k<K; ++k){
                     expectedSum += h_A[row * K + k] * h_B[k*N + col];
                }
                if (abs(h_C[row*N + col] - expectedSum) > 0.00001) {
                    std::cout << "Error at position [" << row << "," << col << "]: Expected " << expectedSum << ", got " << h_C[row * N + col] << std::endl;
                    error = true;
                }
            }
        }
    if(!error) std::cout << "Matrix Multiplication Completed" << std::endl;

   // Free allocated memory
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

Here, we use `cudaEventCreate`, `cudaEventRecord`, and `cudaEventSynchronize` to ensure that the `matrixMultiply` kernel finishes before the asynchronous memory copy begins, preventing data corruption and enabling potential host-device overlap. We are also able to destroy the event once synchronization is completed to free up device resources. This provides more control compared to synchronizing the entire device.

**3. Stream Semantics (Implicit Synchronization)**

CUDA streams provide a mechanism to group GPU operations. Operations within the same stream execute sequentially, implying an ordering that automatically enforces data dependencies. This eliminates the need for explicit synchronization calls when operations within the same stream depend on each other.

*Code Example 3:*
```c++
#include <cuda_runtime.h>
#include <iostream>


__global__ void increment(int *data, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i < n){
        data[i]++;
    }
}

__global__ void doubleData(int *data, int n){
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if( i < n){
        data[i] *= 2;
    }
}

int main() {
    const int n = 1024;
    size_t size = n * sizeof(int);

    int *h_data, *d_data;

    h_data = (int*)malloc(size);

    for (int i=0; i < n; ++i)
        h_data[i] = i;



    cudaMalloc((void**)&d_data, size);


    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // Use stream 0 (default stream)
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;


    increment<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, n);
    doubleData<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_data, n);

    cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream); // Wait until all tasks within stream are done
    cudaStreamDestroy(stream); // Clean up resources

    // Check Results
    bool error = false;
    for (int i=0; i < n; i++){
        if (h_data[i] != (i +1 )* 2){
            std::cout << "Error at index " << i << ": Expected " << (i+1)*2 << ", got " << h_data[i] << std::endl;
            error = true;
        }
    }
    if(!error) std::cout << "Stream Operations Complete" << std::endl;

    free(h_data);
    cudaFree(d_data);

    return 0;
}
```

In this case, two kernels, `increment` and `doubleData`, along with the memory copy, are launched into the same stream. Because operations in a stream execute sequentially, the `doubleData` kernel will not begin until the `increment` kernel has completed, and the copy back to the host will not commence until both kernel operations are finished. This behavior is automatically enforced by the stream, reducing explicit synchronization calls. `cudaStreamSynchronize` is still necessary before continuing host execution which relies on the completed stream operations.

**Resource Recommendations**

For a deeper understanding of CUDA synchronization, explore the NVIDIA CUDA Programming Guide, which details stream semantics, event usage, and memory management. A thorough understanding of these principles is crucial for constructing high-performance CUDA applications. Textbooks on parallel programming with CUDA are also beneficial. Experimenting with diverse synchronization techniques using simple, custom-designed kernel operations is paramount for a solid understanding. Always profile code changes to quantify performance impacts.
