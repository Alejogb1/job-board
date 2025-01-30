---
title: "What causes undefined CUDA references in a CMAKE project?"
date: "2025-01-30"
id: "what-causes-undefined-cuda-references-in-a-cmake"
---
Undefined CUDA references within a CMake project typically stem from a misconfiguration of the build process, specifically how the CUDA compiler (nvcc) and linker interact with the rest of your C++ project. Having spent years debugging large-scale simulations involving CUDA, I've consistently encountered this issue in various forms, often boiling down to discrepancies in include paths, library dependencies, or compilation settings. The core problem is that the linker, during its final stage of combining object files into an executable or library, cannot locate the CUDA-specific functions and variables referenced in your source code. This failure occurs because the necessary object code containing these definitions was not correctly generated or presented to the linker.

A common cause is the absence of proper CUDA compilation for `.cu` files. When a CMake project encounters a `.cu` file, it's not inherently obvious to the compiler, usually `g++`, that this needs to be processed by nvcc. Simply having a `.cu` extension is not enough; the project needs specific instructions. CMake offers functionality to add custom commands, but it also provides higher-level abstractions for CUDA projects that encapsulate this detail. If you are manually trying to invoke nvcc as an add_custom_command, it’s very easy to introduce path errors or forget dependencies. Consequently, these `.cu` files might get compiled as if they are C++, or worse, ignored entirely, leading to undefined references at the linking stage when the program attempts to utilize functions declared as `__global__`, `__device__`, or within the CUDA Runtime.

Furthermore, linking against the appropriate CUDA libraries is essential. While nvcc handles the compilation of CUDA source code into object files, these object files still need to be linked against the CUDA runtime libraries, such as `cudart`. If these libraries are not specified as linker dependencies, the linker won't be able to resolve function calls like `cudaMalloc`, `cudaMemcpy`, or kernel launches. Additionally, different CUDA toolkit versions often require different library paths and names, introducing another potential source of misconfiguration. For example, using a newer toolkit while building against an older set of library paths results in failure to locate essential symbols.

Improper include path settings are another culprit. CUDA headers such as `cuda_runtime.h` define necessary functions and structures. If your project's include directories don't include the appropriate CUDA SDK include paths, your source files won't correctly recognize CUDA declarations, leading to compile-time errors, or more subtly, issues during linking if compiled with incorrect definitions. This commonly occurs when the CUDA toolkit is installed in a non-standard location or when multiple toolkit versions are present on the system.

Let's examine some specific code examples to illustrate these points:

**Example 1: Incorrect Compilation of `.cu` Files**

This example demonstrates a case where `.cu` files are not recognized, resulting in a linking error. Suppose we have `kernel.cu` containing the following:

```cpp
#include <cuda_runtime.h>

__global__ void addArrays(float *a, float *b, float *c, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        c[i] = a[i] + b[i];
    }
}
```

And `main.cpp`, which uses the kernel:

```cpp
#include <iostream>
#include <cuda_runtime.h>

__global__ void addArrays(float *a, float *b, float *c, int N);

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);
    float *h_a, *h_b, *h_c;
    cudaMallocHost((void**)&h_a, size);
    cudaMallocHost((void**)&h_b, size);
    cudaMallocHost((void**)&h_c, size);
    for(int i = 0; i < N; i++) {
      h_a[i] = (float)i;
      h_b[i] = (float)i * 2.0f;
    }

    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, size);
    cudaMalloc((void**)&d_b, size);
    cudaMalloc((void**)&d_c, size);
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addArrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    for(int i = 0; i < 10; i++) {
      std::cout << h_c[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(h_a);
    cudaFreeHost(h_b);
    cudaFreeHost(h_c);
    return 0;
}
```

A naive CMakeLists.txt might look like:

```cmake
cmake_minimum_required(VERSION 3.15)
project(cuda_example)

add_executable(my_app main.cpp kernel.cu)
```

In this setup, CMake treats `kernel.cu` like a regular C++ file and will likely use `g++`, resulting in a link time error, as the necessary `nvcc` compilation steps were bypassed. This manifests as undefined references to functions like `addArrays` within the `main.cpp` executable.

**Example 2: Missing CUDA Library Linking**

Suppose we correct Example 1 with a CMake command using `CUDA` language support but forget the required library:

```cmake
cmake_minimum_required(VERSION 3.15)
project(cuda_example)

find_package(CUDA REQUIRED)
cuda_add_executable(my_app main.cpp kernel.cu)
```

While this correctly compiles the `.cu` file, it's still missing vital linkage. This example, though properly compiling CUDA code with `nvcc`, will result in undefined references to functions like `cudaMalloc` or `cudaMemcpy`. These functions reside within the `cudart` library which has to be explicitly linked against. The error messages would specifically complain about these CUDA runtime calls. The solution would involve using the `CUDA::cudart` target.

**Example 3: Incorrect Include Paths**

Let’s assume we have a correct compilation and linking but the include paths for CUDA are incorrect:

```cmake
cmake_minimum_required(VERSION 3.15)
project(cuda_example)

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})  # Add CUDA include dirs
cuda_add_executable(my_app main.cpp kernel.cu)
target_link_libraries(my_app CUDA::cudart) # Link the library
```

This code may be fine initially. However, if `CUDA_INCLUDE_DIRS` points to the wrong location, or is not set, compilation would fail during preprocess stages when encountering statements like `#include <cuda_runtime.h>`. The compiler might either not find the header or find a version that's incompatible with the CUDA toolkit targeted for compilation. This would manifest during compilation, with error messages indicating the absence of the CUDA header file or syntax errors within those files.

To diagnose and rectify such issues, start by verifying that the CUDA toolkit is correctly installed and accessible on your system. CMake must be able to locate the CUDA installation through the `find_package(CUDA)` command. Then, use `cuda_add_executable`, or `cuda_add_library`, to process CUDA source code. Ensure that your build process correctly incorporates `CUDA_INCLUDE_DIRS` into include paths, and links against the proper CUDA libraries, usually specified using targets such as `CUDA::cudart`. Examine the output of `cmake --debug-find` for details about the location of `CUDA` on your system to spot any potential path misconfigurations.

To further refine your workflow and enhance debugging capabilities consider these resources:

*   **CMake documentation**: The official CMake documentation provides comprehensive guidance on CUDA language support, including details on `find_package(CUDA)`, `cuda_add_executable` and other commands.
*   **CUDA toolkit documentation**:  NVIDIA's documentation for the CUDA toolkit provides crucial details about library dependencies and requirements for different CUDA versions.
*   **Advanced CMake books**: Some books dedicated to modern CMake offer invaluable information for setting up complex build systems involving CUDA. Specifically, pay attention to sections on custom commands and properties.

By carefully managing these aspects of the build process, you can avoid the frustration of undefined CUDA references and ensure a smooth compilation and execution of your projects. Thoroughly reading error messages from the compiler and linker provides important clues about the source of such issues.
