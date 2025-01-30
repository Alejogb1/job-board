---
title: "How do I create a PyTorch extension using CMake?"
date: "2025-01-30"
id: "how-do-i-create-a-pytorch-extension-using"
---
Crafting a custom PyTorch extension, especially one built with CMake, offers a significant boost in performance and flexibility when you need to optimize specific numerical routines or interface with existing C/C++ libraries. I've navigated this process several times while developing high-throughput signal processing pipelines, and the following explanation outlines the necessary steps and considerations.

The primary motivation for using CMake stems from its ability to manage complex build processes, cross-platform compilation, and dependency handling – aspects that the traditional Python setup scripts often struggle with when dealing with C++ source code. PyTorch leverages CMake internally, thus ensuring a consistent build environment and smoother integration for extensions. The core idea involves writing your custom C++ code, using CMake to compile this code into a shared library, and then loading it into Python using PyTorch’s C++ extension API.

**Explanation of the Process**

The workflow breaks down into three main phases: code authoring, build configuration using CMake, and Python integration.

1.  **C++ Code Authoring:** This entails crafting the core logic that you intend to expose to Python. This often consists of numerical algorithms, hardware-accelerated operations, or interfaces to external libraries. Your C++ code needs to adhere to PyTorch's C++ extension API, specifically by including `torch/extension.h`. This header provides functions and macros needed to bridge the gap between C++ data structures (tensors) and Python. Functions to be made accessible from Python should be defined and annotated with a macro to expose them to the Python binding. For instance, a simple addition function could be implemented like:
```cpp
#include <torch/extension.h>
#include <vector>

torch::Tensor add_cpu(torch::Tensor a, torch::Tensor b) {
    auto a_accessor = a.accessor<float, 1>();
    auto b_accessor = b.accessor<float, 1>();
    
    long size = a.size(0);
    torch::Tensor out = torch::empty_like(a);
    auto out_accessor = out.accessor<float, 1>();
    
    for (long i = 0; i < size; ++i) {
        out_accessor[i] = a_accessor[i] + b_accessor[i];
    }
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_cpu", &add_cpu, "Add two 1D tensors on CPU");
}
```

2.  **CMake Build Configuration:**  This phase centers around creating a `CMakeLists.txt` file that describes the compilation process to CMake. This file specifies the source files, include directories (particularly PyTorch's headers), compiler settings, and the target library name. The core directives in the file include `cmake_minimum_required`, `project`, `find_package`, and `add_library`. The `find_package` directive is crucial for finding the PyTorch library, whose information is provided by the `TORCH_INSTALL_PREFIX` environment variable. The `add_library` specifies the C++ library to build and how to build it. The CMakeLists.txt would be structured like:

```cmake
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(my_extension LANGUAGES CXX)

find_package(Torch REQUIRED)

add_library(my_extension SHARED src/extension.cpp) # Replace src/extension.cpp with your actual source file.
target_link_libraries(my_extension PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET my_extension PROPERTY CXX_STANDARD 17)
```

3.  **Python Integration:** The Python side requires a small initialization script to load the shared library.  PyTorch's extension API provides the `torch.utils.cpp_extension.load` function for this purpose. This function takes the name of the extension, paths to the source files, and other build settings. After successful compilation, the custom C++ functions become callable as ordinary functions from Python. The Python side script would appear as follows:

```python
import torch
from torch.utils.cpp_extension import load

source_files = ['src/extension.cpp'] # Replace with the path to your source file
extra_cflags=['-O3'] # Optional: add optimization flags

module = load(name='my_extension', sources=source_files, extra_cflags=extra_cflags, verbose=True)

a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
b = torch.tensor([4.0, 5.0, 6.0], dtype=torch.float32)

result = module.add_cpu(a, b)
print(result)
```

**Code Examples with Commentary**

Here are three examples, progressively adding features:

1.  **Example 1: Basic CPU Addition:** The first example, already outlined above, involves creating a shared library that performs simple element-wise addition of two 1D tensors.  The corresponding CMake and Python setup scripts are also provided above.  The key part here is that a tensor is read using accessors, and the operation is performed using the accessor's indexing. Using the accessor avoids direct pointer access.

2. **Example 2: Kernel Launch (CPU Vectorization):** This example demonstrates how you could use SIMD (Single Instruction, Multiple Data) intrinsics for improved performance, if the C++ compiler and processor support it. Here's an example using the x86 instruction set. Note that compiler directives are used to ensure the code only compiles when the necessary instruction set is supported.  It is important to perform such checks in your code, or unexpected errors may occur at runtime if the user compiles the extension on different hardware.

```cpp
#include <torch/extension.h>
#include <vector>
#ifdef __x86_64__
#include <immintrin.h>
#endif

torch::Tensor add_vec(torch::Tensor a, torch::Tensor b) {
    auto a_accessor = a.accessor<float, 1>();
    auto b_accessor = b.accessor<float, 1>();
    long size = a.size(0);
    torch::Tensor out = torch::empty_like(a);
    auto out_accessor = out.accessor<float, 1>();
    
    #ifdef __x86_64__
    long i = 0;
    for(; i + 7 < size; i += 8){
        __m256 vec_a = _mm256_loadu_ps(&a_accessor[i]);
        __m256 vec_b = _mm256_loadu_ps(&b_accessor[i]);
        __m256 vec_out = _mm256_add_ps(vec_a, vec_b);
        _mm256_storeu_ps(&out_accessor[i], vec_out);
    }
    for (; i < size; i++){
        out_accessor[i] = a_accessor[i] + b_accessor[i];
    }
    #else
    for (long i = 0; i < size; ++i) {
        out_accessor[i] = a_accessor[i] + b_accessor[i];
    }
    #endif
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_vec", &add_vec, "Add two 1D tensors with SIMD");
}
```
The corresponding `CMakeLists.txt` remains the same as above. On the Python side, the call to `add_vec` is similar:
```python
result = module.add_vec(a, b)
```
This showcases the ability to integrate hardware-specific optimizations. The conditional compilation via macros ensures the vectorized instructions are only used when applicable.  If you compile on an architecture without SIMD, the non-vectorized loop will be used.

3.  **Example 3:  Multiple Functions, Different Source Files, Custom Headers**  This example illustrates a more realistic setup, featuring a separate header file and multiple functions that do not use SIMD instructions.   Suppose that `my_operations.h` contains the following code:

```cpp
#ifndef MY_OPERATIONS_H
#define MY_OPERATIONS_H

#include <torch/extension.h>

torch::Tensor multiply_cpu(torch::Tensor a, torch::Tensor b);
torch::Tensor subtract_cpu(torch::Tensor a, torch::Tensor b);

#endif
```

And `my_operations.cpp` contains the implementation:

```cpp
#include "my_operations.h"

torch::Tensor multiply_cpu(torch::Tensor a, torch::Tensor b) {
    auto a_accessor = a.accessor<float, 1>();
    auto b_accessor = b.accessor<float, 1>();
    
    long size = a.size(0);
    torch::Tensor out = torch::empty_like(a);
    auto out_accessor = out.accessor<float, 1>();
    
    for (long i = 0; i < size; ++i) {
        out_accessor[i] = a_accessor[i] * b_accessor[i];
    }
    
    return out;
}

torch::Tensor subtract_cpu(torch::Tensor a, torch::Tensor b) {
    auto a_accessor = a.accessor<float, 1>();
    auto b_accessor = b.accessor<float, 1>();
    
    long size = a.size(0);
    torch::Tensor out = torch::empty_like(a);
    auto out_accessor = out.accessor<float, 1>();
    
    for (long i = 0; i < size; ++i) {
        out_accessor[i] = a_accessor[i] - b_accessor[i];
    }
    
    return out;
}

```

Then `extension.cpp` may look like:

```cpp
#include <torch/extension.h>
#include "my_operations.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("multiply_cpu", &multiply_cpu, "Multiply two 1D tensors");
    m.def("subtract_cpu", &subtract_cpu, "Subtract two 1D tensors");
}
```
The  `CMakeLists.txt` must then also be updated:

```cmake
cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(my_extension LANGUAGES CXX)

find_package(Torch REQUIRED)

add_library(my_extension SHARED src/extension.cpp src/my_operations.cpp) # Added my_operations.cpp here.
target_include_directories(my_extension PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src) # Added include directory
target_link_libraries(my_extension PRIVATE ${TORCH_LIBRARIES})
set_property(TARGET my_extension PROPERTY CXX_STANDARD 17)

```

The corresponding Python call would then be:

```python
result_mult = module.multiply_cpu(a, b)
result_sub = module.subtract_cpu(a,b)
print("Multiplication result:", result_mult)
print("Subtraction result:", result_sub)
```

This more realistic scenario demonstrates how a modular code base can be structured and compiled. The usage of a custom header file improves organization and code reuse.  Also shown is how multiple source files can be added to the `add_library` call, with the `include_directory` added to tell CMake where the `my_operations.h` header file can be located.

**Resource Recommendations**

For deeper dives, explore the following:

*   **PyTorch Documentation:** The official PyTorch documentation provides extensive explanations on extending PyTorch with C++, particularly concerning the use of the `torch.utils.cpp_extension` module. It also explains the requirements and restrictions of C++ extensions and is your primary point of information.
*   **CMake Documentation:** CMake's official documentation serves as an authoritative reference on build configurations and target management. A comprehensive understanding of CMake's directives can significantly ease your build process.
*   **C++ Standards Documentation:** Understanding the different C++ standards and the best practices for writing performant numerical algorithms is crucial. The cppreference website offers a wealth of information, and numerous books on C++ are also available that explain C++ standard details.

By methodically working through the code, build setup, and integration, custom PyTorch extensions built with CMake unlock powerful possibilities for optimization and customization within deep learning projects. Each of the above example progressively shows the increase in complexity one might add to a PyTorch extension.
