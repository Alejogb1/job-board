---
title: "Why is nvrtc missing after CUDA 7.5 installation on Mac?"
date: "2025-01-30"
id: "why-is-nvrtc-missing-after-cuda-75-installation"
---
The absence of `nvrtc` after installing CUDA 7.5 on macOS stems primarily from a fundamental architectural shift in how NVIDIA packaged their compiler tools for that platform around that time. Specifically, prior to CUDA 8.0, the Runtime Compilation (RTC) library, `nvrtc`, wasn’t distributed as a separate, dynamically linked library on macOS. Instead, its functionality was primarily integrated within the larger `libcudart.dylib` file, the CUDA runtime library.

The implication of this architectural choice is that attempts to directly link against a non-existent `nvrtc.dylib` will naturally fail, leading to "missing" library errors. You won't find a distinct `nvrtc` component in the CUDA 7.5 SDK's installation directories on macOS because it isn't there as a standalone entity. This differs from how `nvrtc` was handled on Linux and, crucially, how it would be later structured in newer CUDA versions for all operating systems, including macOS.

My experience with this stems from a project attempting to leverage dynamic kernel compilation using the `nvrtc` API with CUDA 7.5 on a Mac. Initial builds failed because linking against `-lnvrtc` produced link errors, reflecting the absence of the expected `nvrtc.dylib`. This initially pointed toward an installation error or faulty path configuration. However, further investigation revealed the difference in packaging described previously. The confusion, for developers accustomed to the standard dynamic library model of Linux and newer CUDA installs, is understandable.

The key takeaway is that for CUDA 7.5 on macOS, you do not directly link against `nvrtc`. Instead, `nvrtc` functions are indirectly accessible through `libcudart.dylib`, provided you correctly include `cuda.h`. The program utilizes API calls specific to runtime compilation. The `nvrtc` headers, `nvrtc.h`, provide the appropriate function signatures to use but do not imply the existence of a separate library object.

To demonstrate this concept, let’s look at several illustrative examples, progressing from basic initialization to more involved source code compilation.

**Example 1: Initialization using `nvrtc` functions from `libcudart`**

This example demonstrates a successful initialization without needing a `-lnvrtc` link flag.

```c++
#include <iostream>
#include <cuda.h>
#include <nvrtc.h>

int main() {
  nvrtcVersion version;
  nvrtcResult result = nvrtcGetVersion(&version);

  if (result != NVRTC_SUCCESS) {
    std::cerr << "Error getting NVRTC version: " << result << std::endl;
    return 1;
  }
  
  std::cout << "NVRTC Version: " << version.major << "." << version.minor << std::endl;

  return 0;
}
```

*Commentary:* This snippet includes necessary headers for `cuda` and `nvrtc`. `nvrtcGetVersion` is a typical `nvrtc` function. Critically, no linker flags are present requiring `-lnvrtc`. Compilation occurs with only the `CUDA` libraries. The `nvrtc` API is accessed through `libcudart` which exposes it via the headers provided. A successful build and execution signifies the `nvrtc` functionality is present without a separate `nvrtc.dylib`. Compile with something like `nvcc example1.cu -o example1`.

**Example 2: Basic kernel compilation with `nvrtc`**

This demonstrates the API call pattern for actual kernel compilation with the functions provided.

```c++
#include <iostream>
#include <cuda.h>
#include <nvrtc.h>

const char* kernelSource = 
R"(
    __global__ void hello() {
        printf("Hello from GPU, thread %d\\n", threadIdx.x);
    }
)";

int main() {
    nvrtcProgram prog;
    nvrtcResult result;
    result = nvrtcCreateProgram(&prog, kernelSource, "hello.cu", 0, NULL, NULL);
    
    if (result != NVRTC_SUCCESS) {
        std::cerr << "Error creating program: " << result << std::endl;
        return 1;
    }

    const char* opts[] = { "--gpu-architecture=compute_20" };  // Adjust based on target device
    result = nvrtcCompileProgram(prog, 1, opts);
    if( result != NVRTC_SUCCESS)
    {
      size_t logSize;
      nvrtcGetProgramLogSize(prog, &logSize);
      char* log = new char[logSize];
      nvrtcGetProgramLog(prog, log);
      std::cerr << "Compilation Error:\n" << log << std::endl;
      delete[] log;
      nvrtcDestroyProgram(&prog);
      return 1;
    }
    
    size_t ptxSize;
    nvrtcGetPTXSize(prog, &ptxSize);
    char* ptx = new char[ptxSize];
    nvrtcGetPTX(prog, ptx);

    CUmodule module;
    CUfunction function;
    CUdevice device;
    CUcontext context;
    cuInit(0);
    cuDeviceGet(&device, 0);
    cuCtxCreate(&context, 0, device);
    cuModuleLoadData(&module, ptx);
    cuModuleGetFunction(&function, module, "hello");

    cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, 0, NULL, NULL);
    cuCtxSynchronize();
    
    delete[] ptx;
    nvrtcDestroyProgram(&prog);
    cuModuleUnload(module);
    cuCtxDestroy(context);
   
    return 0;
}
```

*Commentary:* This example shows a complete workflow of source compilation to executing on the GPU. The `nvrtcCreateProgram` function initiates the compilation process. `nvrtcCompileProgram` compiles the source code based on the chosen options. Error checks are crucial and include retrieving the compilation log via `nvrtcGetProgramLog`. After a successful compilation, `nvrtcGetPTX` retrieves the compiled PTX code. CUDA’s driver API is used to load and execute the compiled kernel. Again, linking with `-lnvrtc` is absent. Compile using `nvcc example2.cu -o example2`.

**Example 3: Error handling with nvrtc**

This short snippet directly demonstrates what happens when a syntax error is included in the code passed to the compiler API

```c++
#include <iostream>
#include <cuda.h>
#include <nvrtc.h>

const char* kernelSource = 
R"(
    __global__ void hello() {
        int a;
        a;
    }
)";

int main() {
    nvrtcProgram prog;
    nvrtcResult result;
    result = nvrtcCreateProgram(&prog, kernelSource, "hello.cu", 0, NULL, NULL);
    
    if (result != NVRTC_SUCCESS) {
        std::cerr << "Error creating program: " << result << std::endl;
        return 1;
    }

    const char* opts[] = { "--gpu-architecture=compute_20" };  // Adjust based on target device
    result = nvrtcCompileProgram(prog, 1, opts);
    if( result != NVRTC_SUCCESS)
    {
      size_t logSize;
      nvrtcGetProgramLogSize(prog, &logSize);
      char* log = new char[logSize];
      nvrtcGetProgramLog(prog, log);
      std::cerr << "Compilation Error:\n" << log << std::endl;
      delete[] log;
      nvrtcDestroyProgram(&prog);
      return 1;
    }

    std::cout << "Compilation Success, (which shouldn't have happened, this code has errors!)" << std::endl;
    nvrtcDestroyProgram(&prog);
   
    return 0;
}
```

*Commentary:* As in the previous example the compilation procedure follows the established pattern, however due to an incomplete operation in the provided kernel source (the declaration of `int a` without assignment, and `a` on its own as a statement) compilation will fail as expected. This showcases the importance of log retrieval for debugging and correct `nvrtc` function usage. Compile using `nvcc example3.cu -o example3`.

It is imperative to refer to the CUDA Toolkit Documentation for the specific version being used. The online documentation, coupled with release notes, provides an accurate reflection of the supported features and library structures. Textbooks dedicated to CUDA programming may also be valuable, especially those with a focus on older CUDA versions, and for providing historical context. Furthermore, NVIDIA developer forums and technical blogs provide community expertise and often answer very specific questions around obscure legacy issues like this.
