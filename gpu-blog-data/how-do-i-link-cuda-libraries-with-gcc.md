---
title: "How do I link CUDA libraries with GCC on Windows?"
date: "2025-01-30"
id: "how-do-i-link-cuda-libraries-with-gcc"
---
The successful compilation and linking of CUDA applications using GCC on Windows hinges primarily on configuring the compiler and linker to correctly identify the CUDA toolkit installation and its associated libraries. This differs significantly from the more common Linux workflow, where environment variables often simplify the process. My experience managing cross-platform builds has shown that careful specification of include paths and library directories is paramount for Windows using GCC.

Fundamentally, the challenge arises from the fact that GCC, by default, is unaware of the CUDA toolkit’s installation locations. The CUDA toolkit, installed by Nvidia, places header files (.h) required for CUDA programming in one directory and the precompiled libraries (.lib for Windows) in another. Without explicit instructions, GCC’s compiler and linker won’t be able to resolve the CUDA function declarations and their corresponding implementations, resulting in compile-time and link-time errors, respectively.

The solution involves two main steps: First, informing the compiler about the location of the CUDA header files so that it can successfully parse CUDA code using include statements. Second, providing the linker with the paths to the necessary CUDA libraries. This allows it to resolve function calls to CUDA’s runtime and driver APIs during the linking phase. Achieving this requires the explicit use of GCC's compiler and linker flags.

Here’s how one would typically achieve this, along with specific examples:

**Compiler Configuration:**

We need to provide the `-I` flag to the GCC compiler to include the directory where CUDA header files are located. This path is usually within the `include` subdirectory of the CUDA toolkit installation directory. A common path on a standard Windows installation might be `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v<version>\include`, where `<version>` is the version number of the installed CUDA toolkit.  For example, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include`.  I've found that directly specifying the full path is more robust than relying on potentially unreliable environment variables, particularly within cross-platform builds.

**Linker Configuration:**

The linker requires two types of information. First, we need the path to the directory containing the CUDA libraries, specified using the `-L` flag. This directory is commonly located in the `lib/x64` (or sometimes `lib/Win32` for 32-bit builds, although these are rare now) subdirectory of the CUDA toolkit installation directory. For example, `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64`.  Second, we need to specify the individual CUDA library files the linker must include using the `-l` flag. On Windows, these typically end in `.lib` and we need to omit the extension. Necessary libraries typically include `cudart`, `cuda`, `nvrtc`, and potentially others depending on the specific features of the CUDA application. My experience has been that `cudart` (the CUDA Runtime library) is almost always required.

**Code Example 1: Basic CUDA Kernel**

Let’s consider a simple example involving a basic vector addition CUDA kernel within a single file `vector_add.cu`.

```c++
#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(int *a, int *b, int *c, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  int size = 1024;
  int *a, *b, *c;
  int *d_a, *d_b, *d_c;

  cudaMallocManaged((void**)&a, size * sizeof(int));
  cudaMallocManaged((void**)&b, size * sizeof(int));
  cudaMallocManaged((void**)&c, size * sizeof(int));
  cudaMalloc((void**)&d_a, size * sizeof(int));
  cudaMalloc((void**)&d_b, size * sizeof(int));
  cudaMalloc((void**)&d_c, size * sizeof(int));


  for (int i = 0; i < size; ++i) {
    a[i] = i;
    b[i] = i * 2;
  }


  cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

  vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, size);

  cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

  for(int i = 0; i < 10; ++i){
        std::cout << c[i] << " ";
    }
    std::cout << std::endl;


    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

  return 0;
}
```

To compile and link this using GCC, a suitable command might be:

```bash
g++ vector_add.cu -o vector_add.exe -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64" -lcudart -lcuda -lnvrtc
```

Here, I am using `g++` to compile and link the `vector_add.cu` file. The `-o vector_add.exe` flag specifies the output executable name.  The `-I` flag adds the CUDA include directory. The `-L` flag adds the CUDA library directory. Finally, the `-l` flags specify the CUDA libraries: `cudart`, `cuda`, and `nvrtc`. Omitting any of these will typically result in compilation or linker errors related to undefined references or missing header files.

**Code Example 2: Separate Compilation**

In larger projects, I’ve found that separate compilation and linking can be much more manageable. Consider a scenario where our kernel implementation is in `kernel.cu` and our main application is in `main.cpp`.

`kernel.cu`:

```c++
#include <cuda_runtime.h>

__global__ void scalarMultiply(int *data, int scalar, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i < size)
      data[i] *= scalar;
}
```
`main.cpp`:

```c++
#include <iostream>
#include <cuda_runtime.h>

extern "C" void scalarMultiply(int *data, int scalar, int size);

int main() {
    int size = 512;
    int *hostData, *deviceData;

    cudaMallocManaged((void**)&hostData, size * sizeof(int));
    cudaMalloc((void**)&deviceData, size * sizeof(int));


    for(int i = 0; i < size; ++i){
        hostData[i] = i+1;
    }

    cudaMemcpy(deviceData, hostData, size * sizeof(int), cudaMemcpyHostToDevice);

    int scalar = 5;

    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;

    scalarMultiply<<<blocksPerGrid, threadsPerBlock>>>(deviceData, scalar, size);

    cudaMemcpy(hostData, deviceData, size * sizeof(int), cudaMemcpyDeviceToHost);

   for(int i = 0; i < 10; ++i){
        std::cout << hostData[i] << " ";
    }
    std::cout << std::endl;

    cudaFree(hostData);
    cudaFree(deviceData);

    return 0;
}
```

The compilation and linking process would then look like this:

```bash
nvcc -c -I"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\include" kernel.cu -o kernel.o
g++ main.cpp kernel.o -o main.exe -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.2\lib\x64" -lcudart -lcuda -lnvrtc
```

Here, `nvcc`, the NVIDIA CUDA compiler driver, is used to compile `kernel.cu` into an object file `kernel.o`. Then, `g++` compiles `main.cpp` and links it with `kernel.o`, along with the CUDA libraries, as in the first example. The key is using `nvcc` to compile the CUDA code and then using `g++` to link the compiled code and the remaining components of your project.

**Code Example 3: Using a Makefile**

For more complex projects, using a `Makefile` simplifies the build process and ensures consistency. A basic `Makefile` for the above example might look like this (assuming the files `kernel.cu` and `main.cpp` exist in the current directory):

```makefile
CUDA_PATH = C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2
INCLUDE_DIR = $(CUDA_PATH)/include
LIB_DIR = $(CUDA_PATH)/lib/x64

CC = g++
NVCC = nvcc
LIBS = -lcudart -lcuda -lnvrtc

all: main.exe

main.exe: main.o kernel.o
	$(CC) main.o kernel.o -o $@ -L$(LIB_DIR) $(LIBS)

main.o: main.cpp
	$(CC) -c $< -o $@

kernel.o: kernel.cu
	$(NVCC) -c $< -o $@ -I$(INCLUDE_DIR)

clean:
	rm -f *.o *.exe
```
This `Makefile` defines variables for the CUDA path, include and library directories.  It then provides recipes to compile `main.cpp`, `kernel.cu`, and link the resulting object files into the executable `main.exe`.  A simple `make` command in the same directory as this `Makefile` would execute the build process.

In summary, linking CUDA libraries with GCC on Windows requires explicit configuration of compiler and linker flags to properly reference the CUDA toolkit’s installed files. It is not as automatic as Linux setups, therefore specifying paths using `-I` and `-L` alongside using specific library flags like `-lcudart`, `-lcuda`, and `-lnvrtc` is absolutely essential. Furthermore, employing separate compilation and makefiles becomes increasingly important for managing project complexity.

For further study, I would recommend reviewing the documentation for GCC on Windows and focusing particularly on compiler and linker flag specifications. Additionally, the official CUDA documentation provided by NVIDIA is an excellent resource for understanding the runtime environment, libraries, and associated APIs. Exploring examples provided within the CUDA toolkit installation directory itself can also be beneficial. Additionally, consider searching for examples and documentation provided by the users of the MinGW toolchain which frequently integrates GCC for Windows builds. Finally, a practical understanding of Makefiles is necessary for more complex projects and will streamline builds significantly.
