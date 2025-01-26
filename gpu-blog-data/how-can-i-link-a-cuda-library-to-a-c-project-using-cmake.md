---
title: "How can I link a CUDA library to a C++ project using CMake?"
date: "2025-01-26"
id: "how-can-i-link-a-cuda-library-to-a-c-project-using-cmake"
---

Successfully integrating CUDA libraries into a C++ project managed by CMake requires precise configuration of include directories, library locations, and compile-time flags. This process often presents challenges if the CMake project doesn't correctly locate the CUDA toolkit or its components. I've frequently encountered issues during the initial setup, especially when dealing with different CUDA versions or custom installation paths. I'll outline a robust approach that has proven reliable across multiple projects.

The foundational element is ensuring CMake can discover the CUDA toolkit. This is typically achieved using the `find_package(CUDA)` command within the `CMakeLists.txt` file. This command relies on CMake's module system to locate the necessary CUDA header files, libraries, and compiler. Upon successful discovery, CMake populates variables such as `CUDA_INCLUDE_DIRS`, `CUDA_LIBRARIES`, and `CUDA_NVCC_EXECUTABLE`, among others. Failure to find the CUDA toolkit can lead to build errors due to missing headers or libraries.

Following discovery, you'll need to configure your target to use the CUDA compiler (nvcc). This involves adding CUDA source files as a target and informing CMake to use nvcc to compile them.  Furthermore, the target must link with the necessary CUDA libraries. This is typically done through using the variables discovered during `find_package(CUDA)`. Crucially, C++ projects often include both host (CPU) code and device (GPU) code, necessitating a separation in the build process. You must designate which source files should be compiled with nvcc and which should be compiled with the standard C++ compiler (e.g., g++, clang++).

Finally, runtime considerations are critical. CUDA libraries rely on specific drivers installed on the system. If there is a mismatch between the CUDA toolkit version used to compile the project and the driver version installed on the deployment machine, runtime errors are highly probable. It's advisable to perform build and runtime checks to proactively identify these issues.

Here are three code examples, demonstrating the key principles, that I've used in my workflow.

**Example 1: Basic CUDA Detection and Inclusion**

This example demonstrates the foundational setup, focusing on the `find_package(CUDA)` command and setting the include directory.

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyCudaProject LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

if(CUDA_FOUND)
  message(STATUS "CUDA Toolkit found at: ${CUDA_TOOLKIT_ROOT_DIR}")
  message(STATUS "CUDA Include directories: ${CUDA_INCLUDE_DIRS}")

  add_executable(my_cuda_executable main.cpp cuda_kernel.cu)
  
  target_include_directories(my_cuda_executable PUBLIC ${CUDA_INCLUDE_DIRS})

else()
  message(FATAL_ERROR "CUDA Toolkit not found")
endif()
```

*   **Commentary:** This example starts by declaring the project and its supported languages (C++ and CUDA). The `find_package(CUDA REQUIRED)` line is crucial; it instructs CMake to search for a CUDA installation. The `REQUIRED` keyword ensures the process will fail if the toolkit isn't found. Conditional checks verify `CUDA_FOUND`. If found, diagnostic messages are outputting the location and include directories. The executable, `my_cuda_executable`, is defined, containing `main.cpp` and `cuda_kernel.cu`. Finally, `target_include_directories` adds the CUDA include paths to the `my_cuda_executable` target, ensuring that the compiler can find the appropriate header files.  If `CUDA_FOUND` evaluates to false, the CMake process terminates with a fatal error message. This example assumes that `cuda_kernel.cu` is a file containing CUDA code, and `main.cpp` is a regular C++ file.

**Example 2: Linking CUDA Libraries**

Building upon the previous example, this one demonstrates how to correctly link the CUDA runtime and other required libraries.

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyCudaProject LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

if(CUDA_FOUND)
  message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")

  add_executable(my_cuda_executable main.cpp cuda_kernel.cu)

  target_include_directories(my_cuda_executable PUBLIC ${CUDA_INCLUDE_DIRS})

  target_link_libraries(my_cuda_executable PUBLIC ${CUDA_LIBRARIES})

  set_property(TARGET my_cuda_executable PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  
else()
  message(FATAL_ERROR "CUDA Toolkit not found")
endif()
```
*   **Commentary:**  This example introduces the use of `target_link_libraries`. This adds CUDA libraries to the linking process when building the `my_cuda_executable` target. The variable `CUDA_LIBRARIES` is populated by the `find_package(CUDA)` command and generally contains paths to the necessary CUDA runtime libraries. This ensures that during linking, the application is provided with all the required CUDA symbols. `set_property(TARGET my_cuda_executable PROPERTY CUDA_SEPARABLE_COMPILATION ON)` is an important directive for CUDA projects that often improves compilation times by enabling a finer-grained compilation of the .cu files.  This directive lets CMake handle splitting the compilation of device code from host code. If this directive is omitted, the user would be required to explicitly invoke nvcc on the device code and use the resulting object files in the link step.

**Example 3: Custom CUDA Compilation Flags and Configuration**

This example shows how to add specific compilation flags and manage CUDA architecture settings.

```cmake
cmake_minimum_required(VERSION 3.15)
project(MyCudaProject LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)

if(CUDA_FOUND)

    set(CUDA_NVCC_FLAGS "-arch=sm_75 -O3 -Xcompiler=-fPIC")
    
    message(STATUS "CUDA NVCC Flags: ${CUDA_NVCC_FLAGS}")

    add_executable(my_cuda_executable main.cpp cuda_kernel.cu)

    target_include_directories(my_cuda_executable PUBLIC ${CUDA_INCLUDE_DIRS})

    target_link_libraries(my_cuda_executable PUBLIC ${CUDA_LIBRARIES})

    set_property(TARGET my_cuda_executable PROPERTY CUDA_SEPARABLE_COMPILATION ON)
    
    set_property(TARGET my_cuda_executable PROPERTY CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})


else()
  message(FATAL_ERROR "CUDA Toolkit not found")
endif()
```

*   **Commentary:** Here, I've introduced  `set(CUDA_NVCC_FLAGS)`, to define custom compiler flags for the CUDA code. The example sets  `-arch=sm_75`, specifying the compute capability, optimizing the code for a particular CUDA architecture (in this case, one in the 7.5 family). The flags `-O3` enables level 3 compiler optimizations. `-Xcompiler=-fPIC` injects the position-independent code flag for the C++ compiler (useful for creating libraries).  The `set_property(TARGET my_cuda_executable PROPERTY CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})` line then passes these defined flags to the `nvcc` compiler when compiling CUDA kernels. This provides fine-grained control over the compilation process. This configuration is particularly important if your project has to target specific architectures or needs specific optimization levels.

**Resource Recommendations**

For further study, I would recommend exploring the following documentation. Consult the CMake documentation specifically concerning the `find_package` command and the `CUDA` module.  The official NVIDIA documentation for the CUDA Toolkit is invaluable for understanding the different compilation flags and CUDA architectures. Additionally, the CMake documentation for target properties like `CUDA_NVCC_FLAGS` and `CUDA_SEPARABLE_COMPILATION` provide key insights. Exploring tutorials and example projects on platforms such as GitHub can also solidify understanding, showing real-world use cases. The NVIDIA Developer website has detailed resources as well. These resources, used in combination with the demonstrated principles, will facilitate successful integration of CUDA into a CMake-managed C++ project.
