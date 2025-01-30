---
title: "When does CUDA code compile?"
date: "2025-01-30"
id: "when-does-cuda-code-compile"
---
CUDA code compilation is not a monolithic event; it occurs in multiple stages, each with distinct outputs and requirements, spanning from source code to machine-executable instructions. I’ve observed this complexity firsthand over years of developing parallel algorithms, including intricate simulations and deep learning models. The understanding of this compilation process is crucial for optimizing performance and debugging issues effectively. The core distinction rests between host code (typically C/C++) and device code (CUDA C/C++), which necessitate different compilation flows and tools.

Firstly, the host code, which executes on the CPU, is compiled using a standard C/C++ compiler like `gcc` or `clang`. This process is largely conventional, yielding an executable binary or shared library. In a typical CUDA application, this host code is responsible for tasks such as memory management, data transfer between host and device memory, launching kernel functions on the GPU, and processing results. The host compiler transforms C/C++ into machine code specific to the CPU architecture.

Secondly, device code, containing kernel functions designed to execute in parallel on the GPU, undergoes a more intricate compilation journey. This begins with the `nvcc` compiler, which is part of the CUDA toolkit. `nvcc` isn’t a conventional compiler; it acts as a driver, performing a complex orchestration of several steps to convert CUDA source code into executable GPU instructions. Initially, `nvcc` identifies CUDA kernels and separates host and device code. The device code, typically identified using the `__global__` keyword, is then preprocessed by `nvcc`, using the system's preprocessor. The preprocessed CUDA source file is then parsed and a Parallel Thread Execution (PTX) assembly code intermediate representation is generated. This PTX code is a virtual assembly language, intended to be relatively portable across different generations of NVIDIA GPUs, though not across differing architectures from other manufacturers.

The intermediate PTX code is not directly executable on the GPU hardware. This is where the third and final phase comes into play: Just-In-Time (JIT) compilation, performed by the NVIDIA driver at runtime. When a CUDA kernel is launched for the first time within a specific application, the driver compiles the PTX code into machine code specifically for the target GPU architecture, referred to as the SASS (Shader Assembly) code. This SASS code is the actual instruction set that the GPU executes. The driver caches the generated SASS code for future use, so subsequent invocations of the same kernel on the same GPU will not trigger recompilation. This JIT compilation approach enables CUDA code to be more portable across different generations of GPUs without requiring the end user to recompile their application for every new hardware iteration.

This three-stage process ensures that developers can write code in a relatively stable environment (CUDA C/C++) and have that code adapted at runtime to the specific capabilities of the GPU it's executing on. However, a misunderstanding of when compilation occurs and what each stage entails can lead to performance issues. For example, relying on JIT compilation every time a program is executed introduces an unnecessary startup delay.

The first code example below illustrates a basic CUDA kernel:

```cpp
// simple_kernel.cu
#include <stdio.h>
#include <cuda.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1024;
    float *a, *b, *c, *d_a, *d_b, *d_c;
    cudaMallocManaged(&a, n * sizeof(float));
    cudaMallocManaged(&b, n * sizeof(float));
    cudaMallocManaged(&c, n * sizeof(float));

    // Initialize arrays a and b
    for (int i = 0; i < n; i++) {
        a[i] = (float)i;
        b[i] = (float)(n - i);
    }

    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));


    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    vectorAdd<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n);

    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < 10; i++){
        printf("c[%d] = %f\n", i, c[i]);
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
```

This code will be compiled using `nvcc simple_kernel.cu -o simple_kernel`.  The `nvcc` command here produces the host executable, along with an embedded PTX representation of the `vectorAdd` kernel function. During the first invocation of `vectorAdd<<<numBlocks, blockSize>>>`, the NVIDIA driver will perform the JIT compilation into SASS code, and this will be cached for the remainder of execution.

A second example illustrates how different compilation parameters can affect which SASS code is generated:

```cpp
// variant_kernel.cu
#include <stdio.h>
#include <cuda.h>

__global__ void complexKernel(float *a, float *b, float *c, int n, float scalar) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = (a[i] * b[i]) + (scalar * a[i]);
    }
}

int main() {
     int n = 1024;
    float *a, *b, *c, *d_a, *d_b, *d_c;
    float scalar = 2.5f;

    cudaMallocManaged(&a, n * sizeof(float));
    cudaMallocManaged(&b, n * sizeof(float));
    cudaMallocManaged(&c, n * sizeof(float));

    for (int i = 0; i < n; i++) {
        a[i] = (float)i;
        b[i] = (float)(n - i);
    }


    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n * sizeof(float));


    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 512;
    int numBlocks = (n + blockSize - 1) / blockSize;

    complexKernel<<<numBlocks, blockSize>>>(d_a, d_b, d_c, n, scalar);

    cudaMemcpy(c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < 10; i++){
        printf("c[%d] = %f\n", i, c[i]);
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);


    return 0;
}
```

In this scenario, compiling using the command `nvcc variant_kernel.cu -arch=sm_70 -o variant_kernel_sm70` or  `nvcc variant_kernel.cu -arch=sm_80 -o variant_kernel_sm80` demonstrates how different target architectures affect generated output. Specifying the architecture with the `-arch` flag instructs `nvcc` to optimize the PTX specifically for the specified NVIDIA GPU architecture. Subsequent JIT compilation during runtime will be faster because the PTX code is more specific, leading to faster SASS code generation. If the `-arch` flag is not specified, the PTX will be optimized for the general capability of the machine, and the JIT will determine the best optimization based on the device.

Finally, considering CUDA Dynamic Parallelism, where one kernel can launch other kernels, the picture becomes even more nuanced. The code below will compile without issue, but only run on a GPU with appropriate capabilities.

```cpp
// dynamic_kernel.cu
#include <stdio.h>
#include <cuda.h>

__global__ void innerKernel(int id) {
    printf("Inner kernel thread id: %d\n", id);
}

__global__ void outerKernel() {
   int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    innerKernel<<<1,1>>>(threadId);
}

int main() {

    outerKernel<<<10,1>>>();
    cudaDeviceSynchronize();
     return 0;
}
```

The command `nvcc dynamic_kernel.cu -rdc=true -o dynamic_kernel` compiles this code. The `-rdc=true` flag enables relocatable device code which is necessary for device code to launch other device code. The initial JIT compilation of `outerKernel` will occur as usual. However, when `outerKernel` launches `innerKernel`, there will be a second JIT compilation performed by the driver. This occurs not at the time of host execution, but later, on-device, as the launched inner kernel executes for the first time. The PTX representation of the inner kernel is included in the device code generated from the compilation of the outer kernel. This illustrates how compilation can be a multi-tiered process in advanced CUDA features.

For further understanding of CUDA compilation, the official NVIDIA CUDA documentation is indispensable; the programming guide and the `nvcc` documentation are invaluable resources. Additionally, exploring the CUDA Toolkit samples provides practical insights into how compilation options are used in real-world applications. The NVIDIA developer blog often provides deeper explanations of particular advanced CUDA features and compilation methods. Understanding that CUDA code compilation is a phased process involving `nvcc`, PTX generation, and JIT SASS compilation allows for more targeted optimization and debugging of GPU applications.
