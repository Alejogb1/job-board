---
title: "How to resolve CUDA 9.1 compilation errors in Visual Studio 2017 Preview?"
date: "2025-01-30"
id: "how-to-resolve-cuda-91-compilation-errors-in"
---
CUDA 9.1's integration with Visual Studio 2017 Preview presented significant challenges, particularly around compiler compatibility and library linking. Specifically, the preview nature of both software versions led to subtle incompatibilities not always explicitly documented in their initial releases. These issues often manifested as opaque error messages during the build process, forcing a somewhat iterative debugging approach. In my experience, resolving these involved carefully aligning include paths, CUDA toolkit configurations, and compiler flags.

The root causes generally fell into a few categories. First, the Visual Studio 2017 Preview's C++ compiler and toolchain were in a state of flux, and while CUDA 9.1 was *theoretically* supported, the specific compiler versions it was truly optimized for weren't always precisely what the preview version provided. Secondly, include directories and library paths weren't always correctly configured by the installation process, requiring manual intervention. Finally, inconsistencies in the architecture targets and linking flags within the project properties could also contribute to build failures.

Let's break down the common error types and their resolutions. A prevalent issue revolved around header file access. The error message would frequently indicate the compiler failing to locate CUDA header files, despite the toolkit being ostensibly installed. This arose from the project’s include search paths not being properly set to include the CUDA installation’s “include” directory, typically found in `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include`. The solution is to navigate to the project's properties, C/C++ -> General -> Additional Include Directories, and manually add this path. Relative paths might be tempting, but the stability of using the full, absolute path proved more robust during my testing. This step ensures the compiler can resolve references like `#include <cuda.h>`.

Another source of grief was the linker stage. Error messages often manifested as "unresolved external symbol" related to CUDA runtime functions (e.g., `cudaMalloc`, `cudaMemcpy`). The cause was often the project not correctly linking against the necessary CUDA libraries. The fix involves navigating to Linker -> Input -> Additional Dependencies and adding the required library files, usually `cudart.lib`, `cuda.lib`, and potentially others depending on the specific features being used, such as `cublas.lib` for linear algebra operations or `curand.lib` for random number generation. Furthermore, the library directories, found in Linker -> General -> Additional Library Directories, had to be set to point to the library folder within the CUDA install folder (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\lib\x64` for a 64-bit build). Neglecting to set the correct library directory was a recurring cause of link-time errors. Incorrect specification of x86 versus x64 libraries also causes problems that require careful attention. The key is meticulous attention to pathing and architecture targets to ensure the linker can locate the CUDA libraries.

Finally, issues related to the correct code generation targets were also present. By default, Visual Studio might attempt to compile code using a target architecture that isn't optimal for the specific CUDA-enabled GPU. This would lead to runtime errors or unexpected behavior when trying to execute the CUDA kernel. This is resolved within the CUDA C/C++ properties, under Device -> Code Generation. Here, it is necessary to provide the appropriate GPU architecture specifications. These specifications use a format like `compute_xy,sm_xy` where `xy` is the compute capability of your GPU (e.g., `compute_35,sm_35` for Kepler, `compute_50,sm_50` for Maxwell, `compute_61,sm_61` for Pascal, and so on). One can query your specific GPU’s compute capability using the `deviceQuery` utility, included in the CUDA toolkit, and set these values accordingly. Specifying multiple targets provides a broader hardware compatibility at the cost of slightly larger executable sizes.

Let's look at some illustrative code and examples:

**Example 1: Basic CUDA Kernel**

```cpp
#include <cuda.h>
#include <iostream>

__global__ void vectorAdd(int *a, int *b, int *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    int n = 1024;
    int *h_a, *h_b, *h_c, *d_a, *d_b, *d_c;
    h_a = (int*)malloc(n * sizeof(int));
    h_b = (int*)malloc(n * sizeof(int));
    h_c = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    cudaMalloc((void**)&d_a, n * sizeof(int));
    cudaMalloc((void**)&d_b, n * sizeof(int));
    cudaMalloc((void**)&d_c, n * sizeof(int));

    cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
       std::cout << h_a[i] << " + " << h_b[i] << " = " << h_c[i] << std::endl;
    }
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
```

This demonstrates a simple vector addition. If the include paths are incorrect, the compiler will fail to resolve the `<cuda.h>` include. If the linker paths are incorrect, the calls to `cudaMalloc`, `cudaMemcpy`, and `cudaFree` will be unresolved. The absence of the correct compute architecture specification can lead to runtime failures or incorrect calculations.

**Example 2: Using CUDA Error Handling**

```cpp
#include <cuda.h>
#include <iostream>

void checkCudaError(cudaError_t error)
{
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        exit(EXIT_FAILURE);
    }
}

__global__ void kernel(int* d_output)
{
	d_output[threadIdx.x] = threadIdx.x;
}

int main() {
    int *d_output;
    int size = 256;
	cudaError_t err;

    err = cudaMalloc((void**)&d_output, size * sizeof(int));
	checkCudaError(err);

	kernel<<<1, size>>>(d_output);

    err = cudaGetLastError();
    checkCudaError(err);
  
    cudaFree(d_output);

    return 0;
}
```

Here, proper CUDA error checking is performed. Without the correctly configured linker, functions like `cudaGetErrorString` would also cause "unresolved external symbol" errors. This example highlights the importance of robust error handling in CUDA development.

**Example 3: Using CUDA Runtime API**
```cpp
#include <cuda_runtime.h>
#include <iostream>

int main()
{
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop,0);
  std::cout << "Device name: " << prop.name << std::endl;
  std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;

  return 0;
}
```
This program demonstrates using CUDA runtime calls to query device properties. Absence of `cudart.lib` when linking would result in link errors related to `cudaGetDeviceProperties`. The proper include path for `cuda_runtime.h` is essential for the program to compile, further illustrating the importance of meticulously specifying these settings.

For further understanding, I recommend thoroughly reading the CUDA Programming Guide, which NVIDIA provides in PDF format. Additionally, consulting the Visual Studio documentation about compiler and linker flags is essential. The CUDA toolkit release notes often include specifics about supported Visual Studio versions and known issues, which can be invaluable. Also, the documentation surrounding the properties system in Visual Studio can be useful in debugging build errors.

In closing, achieving seamless CUDA 9.1 integration with Visual Studio 2017 Preview required a precise and methodical approach to project configuration. While the provided examples are simple, the same principles of proper include paths, library linking, and architecture specification apply to larger projects. A deep dive into the documentation, coupled with careful diagnostics, formed the basis of a stable development environment during that time.
