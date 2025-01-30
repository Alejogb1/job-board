---
title: "How do I obtain CUDA driver module handles for program functions and global variables?"
date: "2025-01-30"
id: "how-do-i-obtain-cuda-driver-module-handles"
---
CUDA's interoperability with host systems often necessitates direct interaction with the underlying driver. Accessing CUDA driver modules, specifically for obtaining handles to program functions and global variables, requires navigating a complex but well-defined interface provided by the CUDA Driver API. Direct interaction at this level, while powerful, sacrifices a degree of abstraction offered by the higher-level CUDA Runtime API. However, for tasks like low-level resource management, custom runtime implementations, or advanced debugging, understanding this mechanism is indispensable. I've spent considerable time in embedded systems development and GPU-accelerated scientific computing where precisely this interaction is often critical for performance tuning and resource optimization.

The core concept involves using the `cuModuleGetFunction()` and `cuModuleGetGlobal()` functions, which are part of the CUDA Driver API. These functions are not directly tied to source code like function names in your C/C++ code; instead, they reference symbols in the compiled device code represented by the `.cubin` or `.ptx` files. This requires careful compilation and symbol naming considerations.

First, the code destined for the GPU must be compiled into a module, typically an executable (.cubin) or intermediate (.ptx) file. This process involves the `nvcc` compiler and careful specification of compilation options. Then, the host application needs to load the compiled module into the CUDA context by using `cuModuleLoad()` or `cuModuleLoadData()`, resulting in a `CUmodule` handle. It's this `CUmodule` handle that becomes the starting point for obtaining function and global variable handles. Finally, using either `cuModuleGetFunction()` or `cuModuleGetGlobal()` with this module handle and a textual symbol name, you can acquire the `CUfunction` and `CUdeviceptr` handles respectively. These handles, in turn, are used for launching kernels and accessing global variables on the device.

Obtaining function handles requires the correct mangled name of the function. If the CUDA C++ code is compiled with C linkage, then the raw function name is directly used. However, if standard C++ is used, function names are mangled based on namespace, class and argument types. Therefore, using `extern "C"` to force C linkage is advisable for simple cases, as it eliminates the need to interpret the complex name mangling rules. For global variables, their symbol name usually matches the variable name in the CUDA C++ code, unless otherwise specified.

Below are three code examples demonstrating this process, along with explanations and commentary.

**Example 1: Retrieving a Function Handle using C Linkage**

This example demonstrates obtaining a function handle for a simple kernel compiled with C linkage, avoiding name mangling complexities.

```c++
#include <cuda.h>
#include <stdio.h>

// CUDA kernel code (kernel.cu)
extern "C" __global__ void myKernel(float* output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = (float)i;
    }
}


int main() {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    CUdeviceptr d_output;
    float *h_output;
    int size = 1024;

    // Initialization and context setup
    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // Load the CUDA module
    cuModuleLoad(&module, "kernel.cubin"); // Assumes kernel.cubin is in the current dir

    // Get function handle (Notice "myKernel" is used directly)
    CUresult result = cuModuleGetFunction(&function, module, "myKernel");
    if (result != CUDA_SUCCESS) {
        printf("Error getting function handle: %d\n", result);
        return 1;
    }

    // Memory allocation and kernel launch (Simplified)
    cuMemAlloc(&d_output, size * sizeof(float));
    h_output = (float*) malloc(size * sizeof(float));

    // Kernel launch dimensions
    dim3 block_dim(256);
    dim3 grid_dim((size + block_dim.x - 1) / block_dim.x);
    void* args[] = {&d_output, &size};

    cuLaunchKernel(function, grid_dim.x, grid_dim.y, grid_dim.z,
        block_dim.x, block_dim.y, block_dim.z, 0, NULL, args, NULL);
    cuCtxSynchronize();

    // Memory copy back to host
    cuMemcpyDtoH(h_output, d_output, size * sizeof(float));

    // Basic validation
    for (int i = 0; i < size; i++) {
      if (h_output[i] != (float)i) {
        printf("Validation Failed\n");
        break;
      }
    }
    printf("Validation Successful\n");


    // Cleanup
    cuMemFree(d_output);
    free(h_output);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}
```

In this example, the CUDA kernel is declared as `extern "C"`. This causes the compiler to not perform any name mangling, making the function name "myKernel" directly available for lookup via `cuModuleGetFunction()`. The rest of the code sets up the CUDA environment, loads the module, retrieves the function handle, and then launches the kernel. Note that the `.cubin` file named `kernel.cubin` should have been precompiled from the `kernel.cu` file using `nvcc`.

**Example 2: Retrieving a Global Variable Handle**

This example demonstrates obtaining a handle to a global variable residing on the device.

```c++
#include <cuda.h>
#include <stdio.h>

// CUDA kernel code (globals.cu)
__device__ int global_var = 100;

extern "C" __global__ void readGlobal(int* output) {
    *output = global_var;
}

int main() {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUdeviceptr d_global_var;
    CUfunction function;
    int h_output;
    CUdeviceptr d_output;


    // Initialization and context setup
    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // Load the CUDA module
    cuModuleLoad(&module, "globals.cubin"); // Assumes globals.cubin is in the current directory

    // Get global variable handle
    CUresult result = cuModuleGetGlobal(&d_global_var, NULL, module, "global_var");
    if(result != CUDA_SUCCESS) {
        printf("Error getting global variable handle: %d\n", result);
        return 1;
    }

    // Get function handle
     result = cuModuleGetFunction(&function, module, "readGlobal");
    if (result != CUDA_SUCCESS) {
        printf("Error getting function handle: %d\n", result);
        return 1;
    }

    // Allocate output variable
    cuMemAlloc(&d_output, sizeof(int));

    // Launch kernel to read global variable
    void * args[] = { &d_output};
     cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, NULL, args, NULL);
    cuCtxSynchronize();

    // Copy the output back to host
    cuMemcpyDtoH(&h_output, d_output, sizeof(int));

    printf("Value of global variable: %d\n", h_output);

    //Cleanup
    cuMemFree(d_output);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}
```

Here, we use `cuModuleGetGlobal()` to acquire a pointer to `global_var` defined in device memory. Unlike the previous example with a kernel function, the variable's symbol name is straightforward. The kernel then reads this global variable into a memory location allocated on the device, and finally it copies the value to the host memory.

**Example 3: Handling Name Mangling with C++**

This example shows how one might acquire a function handle using name mangling, illustrating the difficulties of this approach, and reinforcing the benefits of C linkage.

```c++
#include <cuda.h>
#include <stdio.h>
#include <string>
// CUDA kernel code (mangling.cu)
namespace myns {
    __global__ void myKernelCPlus(float* output, int size) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < size) {
          output[i] = (float)i;
        }
    }
}

int main() {
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction function;
    CUdeviceptr d_output;
    float *h_output;
    int size = 1024;

    // Initialization and context setup
    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);

    // Load the CUDA module
    cuModuleLoad(&module, "mangling.cubin"); // Assumes mangling.cubin is in the current directory

    // Get mangled function name using nvcc -Xptxas -dlcm=ptx <cuda_file>.cu
    std::string mangled_name = "_ZN4myns10myKernelCPlusEPfi"; // example

    // Get function handle
    CUresult result = cuModuleGetFunction(&function, module, mangled_name.c_str());
    if (result != CUDA_SUCCESS) {
        printf("Error getting function handle: %d\n", result);
        return 1;
    }

     // Memory allocation and kernel launch (Simplified)
    cuMemAlloc(&d_output, size * sizeof(float));
    h_output = (float*) malloc(size * sizeof(float));

    // Kernel launch dimensions
    dim3 block_dim(256);
    dim3 grid_dim((size + block_dim.x - 1) / block_dim.x);
    void* args[] = {&d_output, &size};

    cuLaunchKernel(function, grid_dim.x, grid_dim.y, grid_dim.z,
        block_dim.x, block_dim.y, block_dim.z, 0, NULL, args, NULL);
    cuCtxSynchronize();

    // Memory copy back to host
    cuMemcpyDtoH(h_output, d_output, size * sizeof(float));

    // Basic validation
    for (int i = 0; i < size; i++) {
      if (h_output[i] != (float)i) {
        printf("Validation Failed\n");
        break;
      }
    }
    printf("Validation Successful\n");

    // Cleanup
    cuMemFree(d_output);
    free(h_output);
    cuModuleUnload(module);
    cuCtxDestroy(context);
    return 0;
}
```
Here, the kernel `myKernelCPlus` resides inside the namespace `myns`. As a result, the name is mangled by the compiler. To determine the correct mangled name one approach is to compile with `-Xptxas -dlcm=ptx`, which will output the mangled symbol in the `.ptx` file. However, the `cuModuleGetFunction()` function needs the mangled string, which can be very complex and subject to compiler versions. This approach highlights why C linkage simplifies the process significantly.

For further study, I recommend consulting the official CUDA toolkit documentation, which contains the definitive API specifications for the functions mentioned above and provides detailed explanations regarding compilation options. Additionally, the book “CUDA by Example” by Sanders and Kandrot, provides good practical examples and the book "Programming Massively Parallel Processors" by Kirk and Hwu contains useful background.
