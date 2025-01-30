---
title: "How can multiple GPUs be used with CUDA MPS?"
date: "2025-01-30"
id: "how-can-multiple-gpus-be-used-with-cuda"
---
CUDA Multi-Process Service (MPS) significantly improves GPU utilization in scenarios where multiple independent processes need to share GPU resources concurrently. Without MPS, each process vying for GPU access operates in exclusive mode, leading to underutilization as processes wait for their turn. This is particularly apparent in inference workloads or simulations where numerous small tasks are launched rapidly. The core challenge MPS addresses is enabling these disparate processes to efficiently share a single GPU, thereby reducing idle time and increasing throughput.

I've frequently encountered this bottleneck when simulating large-scale physical systems where multiple agents require individual, yet parallel, computation on the GPU. Before employing MPS, I was constrained by the single-process-per-GPU model, which severely hampered resource utilization and increased the overall simulation runtime. The shift to MPS, while initially requiring some configuration, provided a considerable performance boost by allowing multiple simulation instances to access the GPU concurrently. This wasn't about faster single process execution; it was about the aggregated throughput of multiple processes running at the same time.

Fundamentally, MPS operates as a client-server system. A single MPS server manages GPU resources and schedules computation requests from multiple client processes. The server ensures that each client receives its fair share of GPU time without direct access to the hardware. Communication between clients and the server is handled through inter-process communication mechanisms, typically utilizing shared memory. When a client process needs to execute a CUDA kernel, it doesn't directly interact with the GPU. Instead, it sends a request to the MPS server which then orchestrates the kernel launch on the GPU. This approach abstracts the complexity of managing GPU access and allows several processes to share the same GPU without interfering with each other. This isolation reduces the likelihood of program crashes or incorrect results due to concurrent memory access.

Setting up MPS involves several steps. Firstly, the MPS server application, `nvidia-cuda-mps-control`, must be running. This server is responsible for listening for client connections and scheduling GPU resources. The server can be started with various options including user restrictions and persistence flags. It is also possible to control the level of parallelism using environment variables such as `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`, which dictates the percentage of GPU threads allocated to MPS, and `CUDA_MPS_PIPE_DIRECTORY`, which specifies the location of the communication pipes.

On the client side, code changes are minimal. The key requirement is ensuring that the process utilizes a CUDA context within the scope that has been registered with the MPS server. Specifically, client processes need to use an environment variable, `CUDA_VISIBLE_DEVICES`, which can be set to a virtual device identifier. This virtual identifier is provided by the MPS server when the client connects. Without setting this variable, the client will attempt to create a new context and fail, as the actual GPU hardware is now managed by the server. The client applications interact with the GPU via the standard CUDA API calls; no special libraries or functions specific to MPS are required. The server handles the allocation and deallocation of GPU resources on behalf of the clients.

Let's examine a basic C++ code snippet illustrating this:

```cpp
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    exit(EXIT_FAILURE);
  }
}

__global__ void vectorAdd(float *a, float *b, float *c, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  int size = 1024;
  size_t memSize = size * sizeof(float);

  float *h_a = (float*) malloc(memSize);
  float *h_b = (float*) malloc(memSize);
  float *h_c = (float*) malloc(memSize);

  for (int i=0; i<size; i++) {
      h_a[i] = 1.0f;
      h_b[i] = 2.0f;
  }

  float *d_a, *d_b, *d_c;
  checkCudaError(cudaMalloc((void**)&d_a, memSize));
  checkCudaError(cudaMalloc((void**)&d_b, memSize));
  checkCudaError(cudaMalloc((void**)&d_c, memSize));

  checkCudaError(cudaMemcpy(d_a, h_a, memSize, cudaMemcpyHostToDevice));
  checkCudaError(cudaMemcpy(d_b, h_b, memSize, cudaMemcpyHostToDevice));


  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);
  checkCudaError(cudaDeviceSynchronize());

  checkCudaError(cudaMemcpy(h_c, d_c, memSize, cudaMemcpyDeviceToHost));

  for (int i = 0; i < 10; ++i) {
    std::cout << "Result at index " << i << ": " << h_c[i] << std::endl;
  }

  free(h_a);
  free(h_b);
  free(h_c);
  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);

  return 0;
}
```

This code performs a simple vector addition. To make this MPS-enabled, the same code can be executed, but the client processes need to have the `CUDA_VISIBLE_DEVICES` environment variable set. If MPS is correctly running with server process id 1000, and is allocating virtual GPU id '0' to this process, the client must be launched as:

`CUDA_VISIBLE_DEVICES=0 ./vectorAdd`

The core change for using MPS is outside of the application code. No special CUDA calls are required within the vectorAdd application to use MPS. MPS does not require any special API functions. The program interacts with the CUDA API as it would under normal operation. The crucial part is the environment variable setting that guides the CUDA runtime to connect with the MPS server instead of directly to the GPU driver. I tested this scenario with several processes launching the same `vectorAdd` application concurrently. Under MPS, these processes shared the same GPU which improved the aggregated performance. When running each process without MPS concurrently, these processes would compete for GPU resources, and often result in slower execution.

Another area where MPS becomes useful is when individual processes require independent CUDA contexts, but are executing relatively short kernels. In such cases, using a single CUDA context with its own thread pool can introduce significant overhead due to context switching. MPS can mitigate this by providing each process with its virtual context and executing these kernels as efficiently as possible.

Consider an application where multiple instances each perform a matrix multiplication:

```cpp
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t error) {
  if (error != cudaSuccess) {
    std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
    exit(EXIT_FAILURE);
  }
}


__global__ void matrixMultiply(float *A, float *B, float *C, int N) {
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


int main() {
    int N = 512;
    size_t memSize = N * N * sizeof(float);


    float *h_A = (float*) malloc(memSize);
    float *h_B = (float*) malloc(memSize);
    float *h_C = (float*) malloc(memSize);

    for(int i=0; i< N * N; ++i) {
        h_A[i] = (float)rand()/(float)RAND_MAX;
        h_B[i] = (float)rand()/(float)RAND_MAX;
    }


    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc((void**)&d_A, memSize));
    checkCudaError(cudaMalloc((void**)&d_B, memSize));
    checkCudaError(cudaMalloc((void**)&d_C, memSize));


    checkCudaError(cudaMemcpy(d_A, h_A, memSize, cudaMemcpyHostToDevice));
    checkCudaError(cudaMemcpy(d_B, h_B, memSize, cudaMemcpyHostToDevice));



    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);


    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    checkCudaError(cudaDeviceSynchronize());

    checkCudaError(cudaMemcpy(h_C, d_C, memSize, cudaMemcpyDeviceToHost));

  std::cout << "First element of result: " << h_C[0] << std::endl;
    
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
```

This `matrixMultiply` example again operates within a standard CUDA context. No explicit calls are made to interact with MPS. Similar to the previous example, MPS is enabled via the `CUDA_VISIBLE_DEVICES` environment variable. Multiple instances of this program can then be launched concurrently, sharing the GPU through the MPS server.

Lastly, configuration of the MPS server is a detail that can be easily missed. The MPS server has several configuration options such as `CUDA_MPS_PINNED_MEM_LIMIT`, which controls the amount of pinned memory allocated for MPS, and `CUDA_MPS_LOG_DIRECTORY`, which specifies the location where logs will be saved. Tuning these values can be crucial depending on workload size.

```bash
nvidia-cuda-mps-control -d # Start MPS in daemon mode
nvidia-cuda-mps-control query # Check MPS status
nvidia-cuda-mps-control quit # Stop MPS
```

This bash script shows how to start, query and quit the MPS server. This management is done independently of the client programs.

To fully leverage MPS, it is essential to understand its limitations. It is not intended to be used with CUDA applications that have dependencies or are tightly coupled. If the processes have complex interaction or require global synchronization across processes on the GPU, MPS may not offer significant advantages. It is suitable for scenarios where each process performs independent computation. In my experience, MPS has been most effective when dealing with large batches of independent inference requests and individual simulations. For resources, consult NVIDIA's CUDA Toolkit documentation, particularly sections detailing MPS, and the relevant programming guides which offer detailed instruction and configuration advice. Further guidance can be found in papers detailing shared memory GPU systems and job scheduling on many-core devices. Additionally, practical usage tutorials available through NVIDIA's developer website can be of assistance.
