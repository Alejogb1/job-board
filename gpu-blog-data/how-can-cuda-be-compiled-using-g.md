---
title: "How can CUDA be compiled using g++?"
date: "2025-01-30"
id: "how-can-cuda-be-compiled-using-g"
---
The core challenge when compiling CUDA code with `g++` arises from the fact that `g++`, by itself, understands only standard C and C++. CUDA, developed by NVIDIA, requires specific extensions to these languages along with a separate compiler, `nvcc`, which handles the necessary device code compilation. Successfully integrating CUDA into a `g++` build process necessitates a workflow that combines the strengths of both compilers. I’ve personally encountered this hurdle numerous times, initially struggling with build system configurations and linker issues until settling on a reliable method involving explicit compilation steps and careful linking.

The fundamental strategy is to decompose the compilation process into two distinct phases. First, the CUDA code, typically residing in `.cu` files, undergoes compilation using `nvcc`. This produces object files or device-specific intermediary representation, which are then fed into the subsequent linking phase. Second, the host-side code, commonly residing in `.cpp` or `.c` files, is compiled with `g++` in the standard way. Finally, both sets of object files are linked together along with the necessary CUDA runtime libraries. This linking stage is crucial, as it resolves dependencies and creates the executable.

Let's delve into the specifics with some practical code examples, illustrating the process and clarifying the steps involved.

**Example 1: A Simple CUDA Kernel and Host Code**

Assume you have two files: `kernel.cu` containing the CUDA code and `main.cpp` containing the host code.

`kernel.cu`:

```cpp
#include <cuda.h>
#include <stdio.h>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}
```

`main.cpp`:

```cpp
#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

void checkCudaError(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s:%d: %s\n", file, line, cudaGetErrorString(error));
        exit(1);
    }
}


int main() {
    int n = 10;
    std::vector<float> a(n, 1.0f);
    std::vector<float> b(n, 2.0f);
    std::vector<float> c(n, 0.0f);

    float *d_a, *d_b, *d_c;

    checkCudaError(cudaMalloc((void**)&d_a, n * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMalloc((void**)&d_b, n * sizeof(float)), __FILE__, __LINE__);
    checkCudaError(cudaMalloc((void**)&d_c, n * sizeof(float)), __FILE__, __LINE__);

    checkCudaError(cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);
    checkCudaError(cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice), __FILE__, __LINE__);

    int threadsPerBlock = 2;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, n);

    checkCudaError(cudaMemcpy(c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    for (float val : c) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    checkCudaError(cudaFree(d_a), __FILE__, __LINE__);
    checkCudaError(cudaFree(d_b), __FILE__, __LINE__);
    checkCudaError(cudaFree(d_c), __FILE__, __LINE__);

    return 0;
}
```

The compilation process would involve the following steps:

1.  **Compile `kernel.cu` with `nvcc`:**

    ```bash
    nvcc -c kernel.cu -o kernel.o
    ```

    This command utilizes the NVIDIA CUDA Compiler (`nvcc`) to compile `kernel.cu` and generate an object file named `kernel.o`. The `-c` flag indicates that compilation should stop after the object file is created, without proceeding to linking. This object file encapsulates the device code ready for integration in the next phase.

2.  **Compile `main.cpp` with `g++`:**

    ```bash
    g++ -c main.cpp -o main.o
    ```

    Here, `g++` compiles `main.cpp` to create `main.o`. The `-c` flag operates similarly, generating an object file suitable for the linker.

3. **Link the object files with `g++` including the CUDA runtime:**

   ```bash
    g++ main.o kernel.o -o my_program -lcudart
   ```

   This final command directs `g++` to link the object files (`main.o` and `kernel.o`) into an executable named `my_program`. Crucially, the `-lcudart` flag specifies that the CUDA runtime library should be linked as well, ensuring the availability of CUDA APIs within the program's execution context.

**Example 2: Using a Makefile for Simplified Builds**

Manual compilation is tedious, especially for larger projects. A `Makefile` automates this process. Here's a basic `Makefile`:

```makefile
CC = g++
NVCC = nvcc

CUDA_FLAGS = -arch=sm_75 # Change to match your GPU architecture
CUDA_LIBS = -lcudart

all: my_program

my_program: main.o kernel.o
	$(CC) main.o kernel.o -o my_program $(CUDA_LIBS)

main.o: main.cpp
	$(CC) -c main.cpp -o main.o

kernel.o: kernel.cu
	$(NVCC) -c kernel.cu -o kernel.o $(CUDA_FLAGS)

clean:
	rm -f *.o my_program
```
This `Makefile` defines a set of rules. Typing `make` in the directory containing this file will compile the source code using the commands I detailed earlier and produce an executable `my_program`. It includes a `clean` rule to remove generated files. The CUDA architecture should be specified in `CUDA_FLAGS`, which requires knowledge of the specific GPU being used. I found this approach particularly beneficial when experimenting with various CUDA configurations because it greatly accelerates iteration times.

**Example 3: Handling Include Paths and External Libraries**

Real-world CUDA projects often rely on external headers and libraries. The compilation process must accommodate these. Let's imagine a situation where you have a third-party header file, located in a directory called `include`. The modified `Makefile` would be:

```makefile
CC = g++
NVCC = nvcc
INCLUDE_DIR = include
CUDA_FLAGS = -arch=sm_75 
CUDA_LIBS = -lcudart
CXX_FLAGS = -I$(INCLUDE_DIR)

all: my_program

my_program: main.o kernel.o
	$(CC) main.o kernel.o -o my_program $(CUDA_LIBS)

main.o: main.cpp
	$(CC) -c main.cpp -o main.o $(CXX_FLAGS)

kernel.o: kernel.cu
	$(NVCC) -c kernel.cu -o kernel.o $(CUDA_FLAGS) -I$(INCLUDE_DIR)

clean:
	rm -f *.o my_program
```

The modifications include:
1. `INCLUDE_DIR = include`: This defines a variable for the directory containing include files.
2. `CXX_FLAGS = -I$(INCLUDE_DIR)`: The `-I` flag tells the C++ compiler to include the specified path when searching for include files.
3. Added `-I$(INCLUDE_DIR)` to the nvcc step, as well. This ensures both the C++ and CUDA compiler know where to find the necessary headers.

This modified makefile now incorporates the include directory for both `g++` and `nvcc`, showcasing the necessary setup for including external headers in your project. This particular setup saved me countless hours in one project where complex header files had to be included for specialized mathematical operations.

**Resource Recommendations**

For further information, I recommend consulting NVIDIA’s official documentation on CUDA programming, which thoroughly covers API usage and compilation workflows. Additionally, various online guides and tutorials delve into the nuances of CUDA compilation with `nvcc`, which are beneficial for solidifying understanding. Finally, reading material covering general makefile usage, can help manage larger and more complex builds. These sources, taken together, provide a very robust foundation for CUDA development.
