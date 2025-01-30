---
title: "How can CMake be used to specify the CUDA runtime library?"
date: "2025-01-30"
id: "how-can-cmake-be-used-to-specify-the"
---
The critical challenge when integrating CUDA development into a CMake project lies in accurately identifying and linking against the correct CUDA runtime libraries. Specifically, misconfiguration can lead to compile-time or, more insidiously, run-time errors if the version or location of the CUDA runtime is not what your CUDA code expects. I've spent a good portion of my career debugging such issues, and a solid understanding of how CMake handles this is vital for reproducible builds.

Fundamentally, CMake does not automatically handle CUDA library location. Instead, it relies on a combination of find modules and explicit library linking commands. The CUDA SDK provides its own configuration files which, ideally, `find_package(CUDA)` should leverage. However, explicit settings often become necessary when multiple CUDA installations exist on a system or if non-standard installation paths are used. Therefore, controlling the CUDA runtime library becomes a multi-step process involving finding the CUDA Toolkit, extracting the necessary library paths, and then correctly linking against the required libraries using CMake's linking functionalities. This process is further complicated by the potential for static versus dynamic linking choices, as well as the different CUDA runtime libraries available, like the `cudart` library and the cuBLAS library if BLAS support is needed. We need to be precise about which CUDA runtime libraries our application depends on.

Let's break this down into specific code examples.

**Example 1: Basic CUDA Toolkit Detection and Linking with cudart**

This example demonstrates the most basic scenario: finding the CUDA toolkit and linking against the `cudart` (CUDA runtime) library. Here, I will focus on dynamic linking of the library to make the compiled executable smaller and portable.

```cmake
cmake_minimum_required(VERSION 3.15)
project(CUDAExample LANGUAGES CUDA CXX)

find_package(CUDA REQUIRED)

if(CUDA_FOUND)
  message(STATUS "CUDA Toolkit found at: ${CUDA_TOOLKIT_ROOT_DIR}")

  # Create a source list for compilation.
  set(CMAKE_CUDA_STANDARD 17)
  set(CUDA_SOURCES main.cu)
  
  add_executable(my_cuda_executable ${CUDA_SOURCES})
  
  # Link to the CUDA runtime library
  target_link_libraries(my_cuda_executable PRIVATE ${CUDA_CUDART_LIBRARY})

else()
  message(FATAL_ERROR "CUDA Toolkit not found. Ensure CUDA is installed and PATH is configured correctly.")
endif()
```

**Commentary:**

*   `cmake_minimum_required(VERSION 3.15)`: Specifies the minimum CMake version required. I always prefer using a recent version to get modern functionality.
*   `project(CUDAExample LANGUAGES CUDA CXX)`: This line declares the project and specifies CUDA and C++ as the supported languages, enabling the CUDA compiler.
*   `find_package(CUDA REQUIRED)`: This command attempts to locate the CUDA toolkit using its find module. The `REQUIRED` keyword makes the CMake process fail if CUDA is not found.
*   `if(CUDA_FOUND)`: This conditional ensures that subsequent commands are executed only if the CUDA toolkit is found successfully by `find_package`.
*   `message(STATUS "CUDA Toolkit found at: ${CUDA_TOOLKIT_ROOT_DIR}")`: This displays the root path of the CUDA installation which is useful for debugging.
*    `set(CMAKE_CUDA_STANDARD 17)`: This sets the CUDA standard for compilation.
*   `add_executable(my_cuda_executable ${CUDA_SOURCES})`: This defines the target executable and the CUDA source files.
*   `target_link_libraries(my_cuda_executable PRIVATE ${CUDA_CUDART_LIBRARY})`: This critical command links our executable against the CUDA runtime library which was found by `find_package` as `CUDA_CUDART_LIBRARY`. The use of `PRIVATE` indicates that this link is not propagated to other targets which is preferable in most cases.
*   The `else()` block handles the case when CUDA isn't found, exiting with an error.
*  This basic example shows how to link with the runtime library but omits other essential CUDA libraries like the math library, for simplicity.

**Example 2: Explicitly Specifying CUDA Runtime Library Path**

Sometimes, `find_package` might not locate the exact CUDA installation you intend to use (e.g., different CUDA versions). In these instances, explicit specification of the library path is essential. This example focuses on explicitly specifying the location of the `libcudart.so` (Linux) or `cudart.lib` (Windows) file.

```cmake
cmake_minimum_required(VERSION 3.15)
project(CUDAExample LANGUAGES CUDA CXX)

# Explicitly specify the CUDA installation path.
set(CUDA_ROOT "/usr/local/cuda-12.0" CACHE PATH "CUDA Toolkit Root directory") # Example: Adjust this path.

# Locate the CUDA library directory based on the OS.
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(CUDA_LIBRARY_PATH "${CUDA_ROOT}/lib64")
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(CUDA_LIBRARY_PATH "${CUDA_ROOT}/lib/x64")
else()
  message(FATAL_ERROR "Unsupported operating system.")
endif()


# Explicitly set the cudart library path.
find_library(CUDA_CUDART_LIBRARY
  NAMES cudart
  PATHS ${CUDA_LIBRARY_PATH}
  REQUIRED
)


if(CUDA_CUDART_LIBRARY)
   message(STATUS "CUDA Runtime Library found at: ${CUDA_CUDART_LIBRARY}")

   set(CMAKE_CUDA_STANDARD 17)
   set(CUDA_SOURCES main.cu)
  
   add_executable(my_cuda_executable ${CUDA_SOURCES})
  
   target_link_libraries(my_cuda_executable PRIVATE ${CUDA_CUDART_LIBRARY})
else()
  message(FATAL_ERROR "CUDA runtime library not found at specified path.")
endif()

```

**Commentary:**

*   `set(CUDA_ROOT "/usr/local/cuda-12.0" CACHE PATH "CUDA Toolkit Root directory")`: This sets the CUDA toolkit root directory explicitly. The `CACHE` keyword makes this value persistent across CMake runs, which is useful for customization.
*   The OS-specific logic (`if(CMAKE_SYSTEM_NAME MATCHES "Linux")`, etc.) determines the correct library directory based on the operating system. This is important because the library path changes.
*   `find_library(CUDA_CUDART_LIBRARY NAMES cudart PATHS ${CUDA_LIBRARY_PATH} REQUIRED)`: This command uses `find_library` to locate the specific `cudart` library, relying on the explicit path defined before. `REQUIRED` ensures that CMake fails if the library isn't found.
*   Subsequent logic mirrors Example 1. The critical difference is that the library location is directly controlled via a user-specified path.

**Example 3: Static linking of CUDA Runtime**

Sometimes you need to use the static version of the library to produce a standalone executable that does not rely on a dynamic library. This example focuses on static linking with the library (`libcudart_static.a` on Linux or `cudart_static.lib` on Windows).

```cmake
cmake_minimum_required(VERSION 3.15)
project(CUDAExample LANGUAGES CUDA CXX)

#Explicitly specify the CUDA installation path.
set(CUDA_ROOT "/usr/local/cuda-12.0" CACHE PATH "CUDA Toolkit Root directory") # Example: Adjust this path.

# Locate the CUDA library directory based on the OS.
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(CUDA_LIBRARY_PATH "${CUDA_ROOT}/lib64")
elseif(CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(CUDA_LIBRARY_PATH "${CUDA_ROOT}/lib/x64")
else()
  message(FATAL_ERROR "Unsupported operating system.")
endif()

# Explicitly set the cudart static library path
find_library(CUDA_CUDART_STATIC_LIBRARY
  NAMES cudart_static
  PATHS ${CUDA_LIBRARY_PATH}
  REQUIRED
)


if(CUDA_CUDART_STATIC_LIBRARY)
    message(STATUS "CUDA Static Runtime Library found at: ${CUDA_CUDART_STATIC_LIBRARY}")

   set(CMAKE_CUDA_STANDARD 17)
   set(CUDA_SOURCES main.cu)
   add_executable(my_cuda_executable ${CUDA_SOURCES})
    
   # Set the CMAKE_CUDA_LINKER_FLAGS to use static runtime
   set(CMAKE_CUDA_LINKER_FLAGS "${CMAKE_CUDA_LINKER_FLAGS} -cudart=static")
   
   target_link_libraries(my_cuda_executable PRIVATE ${CUDA_CUDART_STATIC_LIBRARY})
else()
 message(FATAL_ERROR "CUDA static runtime library not found at the specified path.")
endif()
```

**Commentary:**

*   The initial sections regarding CUDA root path specification remain identical to Example 2.
*   `find_library(CUDA_CUDART_STATIC_LIBRARY NAMES cudart_static PATHS ${CUDA_LIBRARY_PATH} REQUIRED)`: This searches for the static version of the CUDA runtime library (`cudart_static`).
*  `set(CMAKE_CUDA_LINKER_FLAGS "${CMAKE_CUDA_LINKER_FLAGS} -cudart=static")`: This setting forces CUDA to use the static version of the runtime libraries by passing the `-cudart=static` flag to the CUDA linker. This flag should be included when statically linking.
*   `target_link_libraries(my_cuda_executable PRIVATE ${CUDA_CUDART_STATIC_LIBRARY})`: This step links the target executable against the statically linked CUDA runtime library.

**Resource Recommendations:**

For deeper understanding and advanced usage, I would recommend exploring the CMake documentation itself, particularly sections related to `find_package`, `find_library`, `target_link_libraries`, and the `CUDA` module. Additionally, the official NVIDIA CUDA documentation provides in-depth details about the CUDA runtime, libraries and supported features. Specifically reviewing the `nvcc` documentation will be helpful for understanding the specific flags and link-time parameters.
Finally, reviewing the CMake documentation regarding platform detection will be useful for generalizing the code for different operating systems or specific build setups.
