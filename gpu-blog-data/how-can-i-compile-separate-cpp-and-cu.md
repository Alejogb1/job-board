---
title: "How can I compile separate .cpp and .cu files using CMake?"
date: "2025-01-30"
id: "how-can-i-compile-separate-cpp-and-cu"
---
Managing the compilation of separate `.cpp` and `.cu` (CUDA) files within a CMake project requires a nuanced understanding of CMake's capabilities and the integration of the NVIDIA CUDA compiler (nvcc).  My experience developing high-performance computing applications heavily reliant on GPU acceleration has underscored the importance of carefully structuring the build process to avoid common pitfalls.  The core challenge lies in correctly specifying the compilation steps for each file type, managing dependencies, and linking the resulting object files effectively.  Ignoring these details can lead to linker errors or incorrect code generation.


The fundamental approach involves leveraging CMake's target-based build system.  We define separate targets for `.cpp` files (compiled with a standard C++ compiler like g++) and `.cu` files (compiled with nvcc).  Then, we link these targets together into an executable.  The complexity stems from managing the dependencies and ensuring that the compiler flags are appropriately set for each target.  Specifically, the `.cu` files require additional compiler flags for CUDA architecture specification, optimization levels, and potentially other CUDA-specific features.

**1.  Clear Explanation:**

The CMakeLists.txt file should contain explicit definitions for each compilation target.  For `.cpp` files, we utilize the `add_executable` command, specifying the source files.  For `.cu` files, we need to use `add_custom_target` or a more sophisticated approach involving `add_library` and `target_link_libraries`, leveraging nvcc’s compilation capabilities.  This strategy allows for the precise control necessary to handle CUDA-specific compilation options.  The key lies in using CMake variables to store compiler flags and include paths, ensuring clean, maintainable, and easily adaptable code.


**2. Code Examples:**

**Example 1: Simple Project (add_executable with nvcc)**

This example demonstrates a straightforward approach, suitable for projects with a small number of `.cu` files. It leverages the `NVCC_FLAGS` variable to manage CUDA compilation flags.  Note that this approach relies on nvcc being in the system's PATH.

```cmake
cmake_minimum_required(VERSION 3.10)
project(SimpleCUDA)

set(CMAKE_CXX_STANDARD 17) # Set C++ standard

set(NVCC_FLAGS "-arch=compute_75 -code=sm_75") # Adjust architecture as needed

add_executable(SimpleCUDA main.cpp kernel.cu)
target_compile_options(SimpleCUDA PRIVATE ${NVCC_FLAGS})
```

`main.cpp` would contain the host code, and `kernel.cu` would contain the CUDA kernel.  This method is concise but less flexible for larger projects with intricate dependencies.


**Example 2: Modular Project (add_library with separate targets)**

This approach promotes modularity, enabling better organization for larger projects. We create separate libraries for CUDA code and CPU code, then link them. This offers greater control and facilitates code reuse.


```cmake
cmake_minimum_required(VERSION 3.10)
project(ModularCUDA)

set(CMAKE_CXX_STANDARD 17)

set(CUDA_FLAGS "-arch=compute_75 -code=sm_75 -O3") # More robust flag management
set(CMAKE_CUDA_ARCHITECTURES 75) # Explicit architecture specification

add_library(cuda_lib cuda_functions.cu)
target_compile_options(cuda_lib PRIVATE ${CUDA_FLAGS})
target_include_directories(cuda_lib PUBLIC include_dir) # Include directories as needed

add_library(cpu_lib cpu_functions.cpp)
target_link_libraries(cpu_lib cuda_lib) # Link the CPU library to the CUDA library

add_executable(ModularCUDA main.cpp)
target_link_libraries(ModularCUDA cpu_lib)
```

This example showcases superior organization. The `cuda_lib` target handles CUDA compilation, while `cpu_lib` links against it. This structure clarifies dependencies and improves maintainability.


**Example 3:  Advanced Project with Separate Compilation Steps (add_custom_command and add_custom_target)**

For complex scenarios involving pre-compilation steps or custom build processes, this approach provides maximum control.  It exemplifies advanced CMake capabilities.  This is beneficial for handling external dependencies or custom build tools.

```cmake
cmake_minimum_required(VERSION 3.10)
project(AdvancedCUDA)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_FLAGS "-arch=compute_75 -code=sm_75 -O3")
set(CMAKE_CUDA_ARCHITECTURES 75)

add_custom_command(
    OUTPUT cuda_obj.o
    COMMAND nvcc -c ${CUDA_FLAGS} kernel.cu -o cuda_obj.o
    DEPENDS kernel.cu
)

add_library(cpu_lib cpu_functions.cpp)
add_executable(AdvancedCUDA main.cpp)
target_link_libraries(AdvancedCUDA cpu_lib cuda_obj.o)

```

This example meticulously handles the compilation of the CUDA code as a separate step, offering fine-grained control over the build process.


**3. Resource Recommendations:**

*   The official CMake documentation.  Pay close attention to the sections on targets, custom commands, and variables.
*   The NVIDIA CUDA Toolkit documentation, focusing on compiler directives and architecture specifications.
*   A good CMake tutorial focused on advanced features and external library integration.


In summary, effectively compiling separate `.cpp` and `.cu` files using CMake demands a thorough understanding of CMake’s target-based build system and the NVIDIA CUDA compiler’s requirements.  The choice between `add_executable`, `add_library`, and `add_custom_command` depends on project complexity and the need for control over the compilation process.  Using CMake variables to manage compiler flags promotes maintainability and facilitates adaptation to different hardware architectures.  Through careful planning and a structured approach, you can build robust and efficient CUDA applications using CMake.
