---
title: "What causes the multiple definition error when compiling CUDA code with C++?"
date: "2025-01-30"
id: "what-causes-the-multiple-definition-error-when-compiling"
---
The crux of multiple definition errors in CUDA C++ compilation stems from the way the CUDA compiler, `nvcc`, and the host C++ compiler, often `g++`, interact during the compilation process. These errors generally arise when the same symbol – a function, variable, or class – is defined in multiple compilation units (source files). This contravenes the one-definition rule (ODR), a fundamental principle in C++ ensuring each symbol has a unique definition across the entire program.

Understanding this requires examining the distinct roles of `nvcc` and `g++`. `nvcc` acts as a hybrid compiler, processing CUDA-specific code (kernels and device functions) and generating PTX (Parallel Thread Execution) assembly or device code for the GPU. It also preprocesses standard C++ code and delegates its compilation to the host compiler, typically `g++`. This split processing is where multiple definition issues often manifest.

When a C++ header file containing definitions, not just declarations, is included in multiple source files, and those source files are separately compiled, the definitions become duplicated. During the linking phase, which combines the separately compiled object files into the final executable, the linker encounters these duplicate definitions and issues an error. The one-definition rule is violated because the same symbol exists in multiple object files.

Furthermore, CUDA introduces additional subtleties. Device code and host code have different compilation paths. Definitions that are intended only for the device should be declared with `__device__` or `__global__`, avoiding inclusion in host code, where they should instead be declared `__host__`. Without this distinction, a function defined without a qualifier may be compiled for both the host and device, leading to a double definition if the same header is included in both contexts.

Another source of confusion arises with template functions and classes. Templates are not compiled directly into object code, instead, they're instantiated upon use with specific template arguments. If a template function, for instance, is defined in a header file and used in multiple `.cu` files with identical template parameters, `nvcc` may attempt to compile and link those multiple instantiations, which leads to a duplicate definition error.

To clarify these concepts, consider the following three examples, and how these issues played out in past projects, and ultimately how I addressed them.

**Example 1: Header File Inclusion**

Consider a scenario from a past project involving image processing. I had a header file, `utility.h`, which defined a simple utility function intended for both host and device operations.

```c++
// utility.h
#ifndef UTILITY_H
#define UTILITY_H

#include <cmath>

__device__ __host__ float square(float x) {
    return x * x;
}

#endif
```
And then, a pair of source files:
```c++
// kernel.cu
#include "utility.h"
#include <cuda.h>

__global__ void myKernel(float *in, float *out, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
       out[i] = square(in[i]);
    }
}
```
```c++
// main.cpp
#include "utility.h"
#include <iostream>

int main(){
    float x = 5.0f;
    std::cout << "Square: " << square(x) << std::endl;
    return 0;
}
```

Initially, this seemed straightforward. However, when compiled, I encountered the dreaded multiple definition error. The `square` function was defined within `utility.h` which was included in both `kernel.cu` and `main.cpp`. This led to two definitions in the object files created from `kernel.cu` and `main.cpp` respectively. During the linking stage, the linker discovered the duplicate definition of `square` and flagged it as an error. The solution was to declare only the function, rather than define it, within `utility.h`. The definition of square was then relocated into a `utility.cu` file. Here's the fix:
```c++
// utility.h
#ifndef UTILITY_H
#define UTILITY_H

__device__ __host__ float square(float x);

#endif
```
```c++
// utility.cu
#include "utility.h"

__device__ __host__ float square(float x) {
    return x * x;
}
```
And then recompile. The `utility.cu` file is separately compiled and the definition of `square` then only exists once across all files.

**Example 2: Device Only Functions**

In another project, I made an error relating to device only functions, resulting in the following:

```c++
// device_math.h
#ifndef DEVICE_MATH_H
#define DEVICE_MATH_H
#include <cmath>

float deviceSin(float x){
    return sin(x);
}
#endif
```
And a basic `.cu` file:

```c++
// my_kernel.cu
#include "device_math.h"

__global__ void myKernel(float *input, float *output, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        output[i] = deviceSin(input[i]);
    }
}
```

I initially omitted the `__device__` qualifier within `device_math.h`. As a result, the `deviceSin` function was not recognized as device code and was implicitly treated as a host function, even though it was exclusively used within the device kernel. Subsequently, both `my_kernel.cu` and the host code were including a `deviceSin` definition, thus producing a duplicate error when compiled. The solution, in this case, was to ensure that the `deviceSin` is explicitly marked as a device function using `__device__`. The updated header file is:

```c++
// device_math.h
#ifndef DEVICE_MATH_H
#define DEVICE_MATH_H
#include <cmath>

__device__ float deviceSin(float x){
    return sin(x);
}
#endif
```

**Example 3: Template Instantiation**

My final example is related to template functions. In a data processing task, I developed a generic `maximum` function using templates, to avoid needing multiple functions for different types. Here is the initial faulty implementation:

```c++
// max.h
#ifndef MAX_H
#define MAX_H

template <typename T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

#endif
```

And then a simple kernel and main file using it:

```c++
// kernel_template.cu
#include "max.h"
#include <cuda.h>

__global__ void myKernel(float *in, float *out, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) {
        out[i] = maximum<float>(in[i], 10.0f);
    }
}
```
```c++
// main_template.cpp
#include "max.h"
#include <iostream>

int main(){
   int a = 5;
   int b = 10;
   std::cout << maximum<int>(a,b) << std::endl;
   return 0;
}
```

Although it seemed convenient to have a single header file defining the template function, this led to the template being instantiated in both `kernel_template.cu` and `main_template.cpp` when compiled. The resulting duplicate instantiations were then flagged as multiple definitions at link time. The fix was to isolate the template definition and only declare the function within the header. The template definition itself was placed in a separate `max.cu` file and compiled separately:

```c++
// max.h
#ifndef MAX_H
#define MAX_H

template <typename T>
T maximum(T a, T b);

#endif
```
```c++
// max.cu
#include "max.h"

template <typename T>
T maximum(T a, T b) {
    return (a > b) ? a : b;
}

template int maximum<int>(int a, int b);
template float maximum<float>(float a, float b);
```

By providing explicit instantiations for each type needed within the `max.cu` file, the compiler generates object code containing these instances, and avoids the multiple instantiation problem.

In each of these three scenarios, the core issue was the violation of the ODR due to duplicated definitions. Separating declarations and definitions, utilizing device qualifiers appropriately, and carefully controlling template instantiation were the effective strategies I have used to mitigate multiple definition errors.

For further study, I would recommend exploring the C++ standard documentation related to the One Definition Rule, and also referring to the NVIDIA CUDA Programming Guide. Additionally, comprehensive C++ books often include sections explaining compilation and linking, which are invaluable for understanding the error I described. Lastly, research papers related to CUDA and heterogeneous programming also offer valuable insight into GPU specific compilation processes.
