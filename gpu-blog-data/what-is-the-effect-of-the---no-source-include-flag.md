---
title: "What is the effect of the --no-source-include flag in NVRTC?"
date: "2025-01-30"
id: "what-is-the-effect-of-the---no-source-include-flag"
---
The `--no-source-include` flag in NVIDIA's NVRTC (NVIDIA Virtual CUDA Runtime Compiler) significantly alters the compiler's behavior regarding header file inclusion.  My experience optimizing CUDA kernels for high-performance computing has underscored the importance of understanding this flag's impact on compilation speed and, critically, the resulting code's correctness.  It fundamentally changes how NVRTC handles `#include` directives within CUDA source code.  Specifically, it disables the compiler's ability to process directives that include external files, effectively limiting compilation to the contents of the provided source file itself.

**1. Explanation:**

Standard compilation processes involving header files involve several steps. First, the preprocessor expands `#include` directives, replacing them with the contents of the specified header file.  This process can recursively involve other included headers, creating a complex dependency tree.  The preprocessor then performs macro substitution and other pre-processing directives. Finally, the actual compilation process takes place on the preprocessed code.

NVRTC, as a just-in-time compiler, streamlines this process. However, the inclusion of external header files introduces complexities:  it necessitates locating and parsing potentially numerous files, increasing compilation time.  The `--no-source-include` flag directly addresses this overhead. By specifying this flag, one explicitly instructs NVRTC to *ignore* any `#include` directives.  The compiler operates solely on the provided source code, treating any `#include` statements as if they were simply comments.

This has profound consequences.  The absence of header files eliminates the need for the preprocessor to resolve dependencies, resulting in a faster compilation process, particularly beneficial when compiling many small kernels or in resource-constrained environments. However, this speed improvement comes at a cost:  the compiled code will lack the declarations and definitions present in the excluded headers. This directly impacts the functionality of the code if the included headers contain crucial declarations needed by the compiled kernel.

Therefore, the judicious use of `--no-source-include` depends heavily on the architecture of the CUDA code.  It’s suitable for self-contained kernels with all necessary declarations and definitions explicitly present within the main source file.  For code that relies on standard CUDA libraries or custom headers, using this flag will invariably result in compilation errors due to undefined symbols or missing types.

**2. Code Examples and Commentary:**

**Example 1: Successful Compilation with `--no-source-include` (Self-Contained Kernel)**

```cuda
__global__ void myKernel(int *data, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    data[i] *= 2;
  }
}
```

This kernel is self-contained.  It doesn’t rely on any external headers or libraries.  Compiling this with `--no-source-include` will succeed, because all necessary types and functions are defined within the kernel itself. The compilation process will be faster than if the header files were included.

**Example 2: Compilation Failure with `--no-source-include` (External Library Dependency)**

```cuda
#include <cuda_runtime.h>

__global__ void myKernel(int *data, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    cudaMemcpy(data, data, N * sizeof(int), cudaMemcpyDeviceToDevice); //Error
  }
}
```

This example attempts to use functions from the `cuda_runtime.h` header.  Compilation with `--no-source-include` will fail because the compiler will not find the necessary declarations for `cudaMemcpy`, `cudaMemcpyDeviceToDevice`, etc. The compiler will report undefined symbols.

**Example 3: Conditional Compilation to Manage Header Inclusion**

```cuda
#ifdef USE_HEADER
#include "myheader.h"
#endif

__global__ void myKernel(int *data, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        #ifdef USE_HEADER
        processWithHeader(data[i]); //Requires declaration from myheader.h
        #else
        data[i] *= 2;
        #endif
    }
}
```

This example demonstrates a conditional compilation approach.  By defining `USE_HEADER` during compilation (e.g., using a compiler flag or preprocessor directive), one can control whether `myheader.h` is included.  This allows for flexibility: building without `myheader.h` might be possible using `--no-source-include` if `myKernel` is modified to avoid header-dependent functionality. This provides a more refined control over compilation and allows adaptation to different compilation scenarios.


**3. Resource Recommendations:**

I would recommend consulting the official NVIDIA CUDA Toolkit documentation for the most up-to-date and detailed information on NVRTC and its compiler flags.  Pay close attention to the sections on pre-processing and compilation.  The CUDA C Programming Guide will provide a thorough understanding of the CUDA programming model and header file usage.  Finally, a comprehensive guide on optimizing CUDA code can greatly enhance your understanding of performance tuning techniques, where the judicious use of the `--no-source-include` flag plays a small but crucial part.  These resources will provide the foundational knowledge necessary to navigate the complexities of CUDA kernel compilation and optimization.
