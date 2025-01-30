---
title: "What is the cause of the undefined reference linking error in CUDA/C++ code?"
date: "2025-01-30"
id: "what-is-the-cause-of-the-undefined-reference"
---
The core issue behind undefined reference linking errors in CUDA/C++ projects stems from the compiler’s inability to locate the compiled definitions (function bodies, global variables) associated with declarations made in the source code during the linking phase. This inability can manifest in a variety of ways, often causing significant frustration, particularly when dealing with the complexities of CUDA and its specific compilation requirements.

Having spent considerable time debugging CUDA applications, I’ve frequently encountered this error, and it's almost always reducible to a mismatch between what the compiler knows about during compilation and what the linker can actually resolve. The process involves, first, compiling individual source files into object files. These object files contain the machine code and symbols. The linker then combines these object files into an executable or shared library. During the linking, the linker attempts to resolve symbol references (like function calls, global variables access) with their respective definitions present in other object files. When a definition cannot be found, an 'undefined reference' error is thrown, stopping the linking process. This is not an error in the *code* itself, but often an error in *how* the code is being compiled, or missing libraries.

The typical reasons this error arises in CUDA projects are multifaceted:

1. **Missing Compilation of CUDA Kernels:**  CUDA kernels, marked with `__global__`, need to be compiled using the NVIDIA `nvcc` compiler. Standard C++ compilers like `g++` will only process C++ host code, meaning it will only process the declarations of CUDA kernel calls, not the underlying CUDA implementation. If `nvcc` does not process the `.cu` file containing the kernel definitions, the generated object file will be missing the symbol definition, which results in the undefined reference.

2. **Incorrect Linking Order:** The order of object files and libraries passed to the linker matters. For instance, a host function calling a device function needs access to the device code through linking in the corresponding object file containing that device code. When the device object file isn't present *or* linked at the correct stage, an undefined reference error occurs. This extends to libraries as well; dependency libraries (e.g., CUDA libraries) must be linked *after* object files relying on them.

3. **Name Mangling Mismatches:** C++ applies name mangling (also called decoration) to function names, which involves encoding type information within the function's symbol. If a declaration differs in type or namespace from its definition, the linker will treat them as distinct symbols, resulting in an undefined reference during linkage. CUDA kernels are subject to specific name mangling rules which can lead to issues when combined with other code written in different conventions.

4. **Missing CUDA Runtime Library:** Linking against libraries such as the CUDA Runtime API (cudart) is necessary for CUDA functionality. If these libraries are absent in the linker command, undefined reference errors will surface for symbols related to CUDA, such as `cudaMalloc`, `cudaMemcpy`, etc.

5. **Improper Inclusion of Header Files:** While a source file may include a header containing a function declaration, the compiler may not be able to associate that declaration with the actual compiled code if the definition is not compiled into an object file being linked into the final binary. A common mistake is forgetting to compile a file that has function definitions related to declarations from the included header files.

Now, let's consider some concrete examples.

**Example 1: Missing Kernel Compilation**

Consider `kernel.cu`:

```cpp
// kernel.cu
#include <cuda.h>

__global__ void addArrays(int *a, int *b, int *c, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        c[i] = a[i] + b[i];
    }
}
```

And a host file `host.cpp`:

```cpp
// host.cpp
#include <iostream>
#include <cuda.h>
extern void addArrays(int *a, int *b, int *c, int size);

int main() {
    int size = 1024;
    int *a, *b, *c;
    cudaMallocManaged(&a, size * sizeof(int));
    cudaMallocManaged(&b, size * sizeof(int));
    cudaMallocManaged(&c, size * sizeof(int));

    for(int i = 0; i < size; ++i){
        a[i] = i;
        b[i] = i * 2;
    }

    addArrays<<<256, 256>>>(a, b, c, size);
    cudaDeviceSynchronize();

    for(int i = 0; i < 10; ++i){
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;


    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    return 0;
}
```
If you compile `host.cpp` with `g++ host.cpp -o host`, and do not include `kernel.cu` in the compilation with `nvcc`, the linker will fail with an undefined reference to `addArrays`. The `g++` compiler is aware that there exists an `addArrays` function, but never finds its compiled form, thus resulting in an undefined reference linking error.

**Example 2: Incorrect Linking Order**

Now consider a utility library in `util.cu`:

```cpp
// util.cu
#include <cuda.h>

__device__ int deviceAdd(int a, int b) {
    return a + b;
}
```

And the modified host file `host2.cpp` that calls it:

```cpp
// host2.cpp
#include <iostream>
#include <cuda.h>

__global__ void kernelCallDevice(int *a, int *b, int size){
   int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < size){
        a[i] = deviceAdd(a[i], b[i]);
   }
}
extern __device__ int deviceAdd(int a, int b);

int main() {
    int size = 1024;
    int *a, *b;
    cudaMallocManaged(&a, size * sizeof(int));
    cudaMallocManaged(&b, size * sizeof(int));

    for(int i = 0; i < size; ++i){
        a[i] = i;
        b[i] = i * 2;
    }

    kernelCallDevice<<<256, 256>>>(a, b, size);
    cudaDeviceSynchronize();

    for(int i = 0; i < 10; ++i){
        std::cout << a[i] << " ";
    }
    std::cout << std::endl;


    cudaFree(a);
    cudaFree(b);
    return 0;
}
```
If the `util.cu` object file (compiled with `nvcc -c util.cu`) is not included *or* is included before the `host2.cpp` object in the linking step (e.g., `nvcc host2.o util.o -o host2`, instead of `nvcc util.o host2.o -o host2`), this will result in an undefined reference to `deviceAdd` during linkage. The compiler needs `util.o` in the link step to actually find the definition of `deviceAdd`, and it has to be available before the object that needs it, `host2.o`.

**Example 3: Missing CUDA Runtime Library**

Continuing with `host2.cpp` example, suppose we have the correct linkage order but forget the CUDA Runtime Library. Using a command like:

```bash
nvcc host2.o util.o -o host2
```

will fail because `cudaMallocManaged`, `cudaDeviceSynchronize` are defined in the CUDA runtime library `libcudart.so` on Linux and `cudart.lib` on Windows. Without linking against it, we will see undefined reference to these functions. The correct command would be (Linux):

```bash
nvcc host2.o util.o -o host2 -lcudart
```
or (Windows):
```bash
nvcc host2.o util.o -o host2 cudart.lib
```

To resolve these issues, I usually adopt the following best practices:

* **Use `nvcc` for all `.cu` files:** Ensure that all files containing CUDA kernels or device functions are compiled using `nvcc`.
* **Explicitly include the CUDA runtime library:**  Link against `libcudart.so` or `cudart.lib` as needed.
* **Careful Link Order:** When linking, start with the object files that define the dependencies followed by object files that rely on them.
* **Use build systems (CMake, Make):**  Automated build systems allow for better organization and consistency when dealing with CUDA code, especially when projects grow in complexity.
* **Clean build when errors arise:** It is recommended to start with a clean environment when a linking problem occurs. Sometimes object files are corrupted or partially compiled and could create linking problems that did not previously exist.

**Resource Recommendations:**

1.  **NVIDIA CUDA Programming Guide:** This is the primary resource for learning CUDA development, including compilation and linking concepts. The documentation goes into more depth on the different compilation stages and the linking process, specifically for CUDA applications.
2.  **NVIDIA CUDA Toolkit Documentation:** Comprehensive documentation on the various aspects of the CUDA Toolkit, including APIs, libraries, and compiler specifics. This is a more practical guide to implementing CUDA code that complements the programming guide.
3.  **Your compiler’s manual:**  Familiarizing yourself with the documentation of `g++`, `nvcc`, and other compilers being used is invaluable. Understanding compiler options and linking nuances greatly assists in debugging these errors.

In summary, undefined reference linking errors in CUDA/C++ projects arise because the linker is unable to locate compiled definitions necessary to build the final executable. By ensuring correct compilation with `nvcc`, proper linking order, inclusion of CUDA runtime libraries, and careful examination of build processes, these errors can be effectively mitigated. Building experience with debugging these errors has led me to believe the process described above remains effective and applicable to all instances I have encountered in my practice.
