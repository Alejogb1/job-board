---
title: "How can functions defined in an .h file be declared in a .cu file?"
date: "2025-01-30"
id: "how-can-functions-defined-in-an-h-file"
---
Direct inclusion of function definitions within a header file, coupled with subsequent inclusion in a CUDA (.cu) file, presents a unique compilation challenge. CUDA's compiler, `nvcc`, manages the separation of host and device code. This process necessitates a specific strategy for integrating functions, especially those intended for device execution, into a .cu file after initial declaration in a .h file.

The core issue stems from how `nvcc` handles compilation. When a `.h` file containing function definitions is included in multiple `.cu` files, the compiler will generate multiple definitions of the same function. This violates the one-definition rule (ODR) leading to linker errors during the build process. Specifically, `nvcc` compiles `.cu` files to target both host and device architectures and, without proper management, will attempt to create multiple compiled instances of the functions.

The correct approach lies in a combination of function declarations in the header and definitions in a single source (.cu or .cpp) file. Header files should only contain *declarations*. Function prototypes, using the `extern` keyword, serve this purpose. The definitions, the actual implementation of the function, should be located in a single `.cu` or `.cpp` file and compiled only once, then linked into the final executable. Crucially, for device functions, these definitions must utilize the `__device__` or `__global__` qualifiers.

Here’s how I’ve structured similar setups in the past, including strategies to avoid the pitfalls of the ODR and ensuring proper execution on both host and device:

**1. Declaration in Header (.h) File**

The header file (`my_functions.h`), should only include the function prototype with the `extern` specifier, and for device functions, the correct device qualifier. This separates interface from implementation. I generally also guard my headers against multiple inclusions using include guards.

```c++
#ifndef MY_FUNCTIONS_H
#define MY_FUNCTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

// Host Function Declaration
extern int host_function(int a, int b);

// Device Function Declarations
extern __device__ int device_function(int a, int b);
extern __global__ void kernel_function(float *output, float *input, int size);

#ifdef __cplusplus
}
#endif

#endif
```

**Commentary:**

*   `#ifndef MY_FUNCTIONS_H` and `#define MY_FUNCTIONS_H`: These are include guards, preventing multiple inclusion of the header.
*   `extern "C"`: Ensures that the function names are not mangled by C++'s name mangling process when compiling. This is particularly important when interfacing with C code or libraries. While C++ code could use `extern "C++"`, I've found it's simpler to declare everything as extern "C" when mixing.
*   `extern int host_function(int a, int b);`:  This declares a regular host function. The `extern` keyword signals that the actual implementation is located elsewhere.
*   `extern __device__ int device_function(int a, int b);`: This declares a device function which can be called by other device functions or from host functions executing on the device (via `<<<>>>` launch configuration). The `__device__` qualifier is critical for indicating its execution environment.
*   `extern __global__ void kernel_function(float *output, float *input, int size);`: This declares a kernel function, which is designed to run on the GPU. The `__global__` qualifier denotes a kernel function that is launched by the host.

**2. Definition in a CUDA (.cu) File**

The implementation of the function is then placed within a single `.cu` file (`my_functions.cu`). This prevents duplicate definitions during compilation. This file is compiled using `nvcc`.

```c++
#include "my_functions.h"

// Host Function Implementation
int host_function(int a, int b) {
  return a + b;
}

// Device Function Implementation
__device__ int device_function(int a, int b) {
    return a * b;
}

// Kernel Function Implementation
__global__ void kernel_function(float *output, float *input, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    output[i] = input[i] * 2.0f;
  }
}

```

**Commentary:**

*   `#include "my_functions.h"`: Includes the header file to access function declarations. This ensures consistency between declarations and definitions.
*   `int host_function(int a, int b) { ... }`: This defines the actual implementation of the `host_function`.
*   `__device__ int device_function(int a, int b) { ... }`: This provides the actual implementation of the device function `device_function`, with the `__device__` qualifier, signaling it can execute on the GPU.
*   `__global__ void kernel_function(float *output, float *input, int size) { ... }`: Defines the implementation of the CUDA kernel function, with the `__global__` qualifier. Inside it, `blockIdx`, `blockDim`, and `threadIdx` are CUDA built-in variables used to determine the unique index of each thread within the grid. The implementation within this kernel is simply scaling each input by 2.

**3. Usage in a Main CUDA (.cu) File**

The final step is to include the header within a primary `.cu` file (`main.cu`) and utilize the defined functions. This .cu file is also compiled by `nvcc`

```c++
#include <iostream>
#include "my_functions.h"
#include <cuda.h>

int main() {
    int a = 5;
    int b = 10;
    int host_result = host_function(a, b);
    std::cout << "Host Function Result: " << host_result << std::endl;

    // Device Function Execution (example within kernel)
    int size = 1024;
    float *h_input, *h_output, *d_input, *d_output;
    cudaMallocHost((void**)&h_input, size * sizeof(float));
    cudaMallocHost((void**)&h_output, size * sizeof(float));
    cudaMalloc((void**)&d_input, size * sizeof(float));
    cudaMalloc((void**)&d_output, size * sizeof(float));

    for(int i = 0; i < size; i++) { h_input[i] = static_cast<float>(i); }
    cudaMemcpy(d_input, h_input, size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockDim(256);
    dim3 gridDim((size + blockDim.x - 1) / blockDim.x);
    kernel_function<<<gridDim, blockDim>>>(d_output, d_input, size);

    cudaMemcpy(h_output, d_output, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    int device_result = 0;

    for(int i = 0; i < size; i++)
    {
       device_result += static_cast<int>(h_output[i]);
    }
      std::cout << "Partial Result from Kernel : " << device_result << std::endl;
      cudaFree(d_input);
      cudaFree(d_output);
      cudaFreeHost(h_input);
      cudaFreeHost(h_output);

    return 0;
}
```

**Commentary:**

*   Includes the necessary headers, including the custom “my_functions.h” header.
*   Demonstrates usage of the host function, obtaining the result and printing to standard out.
*   Shows a basic example of memory allocation for the host and device, as well as the setup and launch of a CUDA kernel. The kernel’s result, copied back to host memory, undergoes partial processing on the host to demonstrate proper kernel execution, followed by a print to standard output.
*   Proper device and host memory deallocation is important to prevent memory leaks.

**Resources**

When seeking further information on this topic, I’ve found the following resources generally useful:

*   **CUDA Programming Guide:** The official NVIDIA documentation is the primary source for understanding CUDA concepts, including device function qualifiers, memory management, and kernel launch configurations.

*   **Textbooks on Parallel Programming:** Several academic textbooks on parallel programming concepts, particularly those focusing on GPU computing, can be invaluable. These typically cover memory management, thread organization, and performance optimization for CUDA.

*   **Online Forums:** Various online forums dedicated to CUDA programming (such as NVIDIA’s developer forums), provide community support, practical examples, and help to address specific issues. These forums can be a valuable resource for specific troubleshooting and detailed information.
By following the described approach, I have reliably incorporated functions from header files into CUDA projects and avoided issues with the one definition rule. Using external declarations in headers and defining functions in a single source file, specifically those with `__device__` and `__global__` qualifiers, is paramount for proper compilation and execution within CUDA's architecture. The provided examples illustrate the structure I’ve consistently utilized to manage both host and device code in my projects.
