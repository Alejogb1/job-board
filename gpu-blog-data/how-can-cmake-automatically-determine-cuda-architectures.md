---
title: "How can CMake automatically determine CUDA architectures?"
date: "2025-01-30"
id: "how-can-cmake-automatically-determine-cuda-architectures"
---
Detecting CUDA architectures automatically within CMake is critical for ensuring a project compiles and executes effectively across diverse hardware configurations. I've wrestled with this complexity in a number of high-performance computing projects, and a robust solution often involves a combination of CMake's built-in capabilities and careful use of the `nvcc` compiler itself. Fundamentally, the challenge stems from needing to tailor the compiled CUDA code to match the compute capabilities of the target GPUs at build time. This prevents issues like code that is compiled for an architecture that does not exist on the deployment machine.

The primary mechanism for determining CUDA architecture within CMake relies on querying the `nvcc` compiler directly. `nvcc` provides the `--gpu-architecture` or `-arch` flag (depending on version) to specify the target compute capability. Instead of hardcoding specific architectures, we interrogate `nvcc` to output a list of supported architectures and then leverage CMake's language facilities to intelligently select the appropriate options.

My approach involves using CMake's `execute_process` command to run `nvcc` and capture its output regarding supported architecture. Subsequently, we parse this output to create a list of viable architectures. Specifically, I've found that using a script or a custom CMake function to handle the `nvcc` invocation makes the process more maintainable and easier to reuse across various targets in larger projects. Once we've determined the target architecture, we then use CMake’s target properties to define the compiler flags.

Here's how the process typically unfolds: first, a CMake function is created to execute `nvcc -arch=list`.  The output of this command will contain a list of the available architecture flags. The architecture strings are separated by line breaks.  The parsing step extracts the compute capability strings (e.g., `compute_60`) and stores them in a CMake variable.  A separate function, or a segment of code in the original function, selects the architecture based on availability, user-supplied options or desired defaults. Once selected, the appropriate architecture string is then used to populate the `CUDA_ARCHITECTURES` target property, which then determines how `nvcc` will compile the CUDA sources.

**Code Example 1: Retrieving Architectures with execute_process**

```cmake
function(get_cuda_architectures out_variable)
  execute_process(
    COMMAND nvcc -arch=list
    OUTPUT_VARIABLE architectures
    ERROR_QUIET
    RESULT_VARIABLE nvcc_result
  )
  if(NOT ${nvcc_result} EQUAL 0)
    message(FATAL_ERROR "nvcc failed to list architectures")
  endif()
  string(REPLACE "\n" ";" architectures_list "${architectures}")
  
  set(${out_variable} "" PARENT_SCOPE)
  foreach(arch IN LISTS architectures_list)
    if(arch MATCHES "^compute_\\d+$")
        list(APPEND valid_architectures "${arch}")
    endif()
  endforeach()

  set(${out_variable} "${valid_architectures}" PARENT_SCOPE)
endfunction()

get_cuda_architectures(AVAILABLE_CUDA_ARCHITECTURES)

message(STATUS "Available CUDA Architectures: ${AVAILABLE_CUDA_ARCHITECTURES}")
```

This code example defines a CMake function named `get_cuda_architectures`. This function utilizes `execute_process` to run `nvcc -arch=list`. The output is stored in the `architectures` variable, which is then parsed. The function then iterates over each line in the output, and, if a line matches the pattern `^compute_\d+$` using a regular expression, it adds the architecture to a list.  Finally, the list is saved in the `AVAILABLE_CUDA_ARCHITECTURES` variable in the calling scope. The output is then printed to the console. Crucially, the function also checks for errors from the `nvcc` call.

**Code Example 2: Selecting Target Architecture with Fallback**

```cmake
function(select_cuda_architecture out_variable available_architectures)
    set(target_arch "")
    
    # Check for user-provided architecture preference
    if(DEFINED ENV{CUDA_ARCHITECTURE})
      set(requested_arch $ENV{CUDA_ARCHITECTURE})
       message(STATUS "User requested CUDA architecture: ${requested_arch}")
      if(requested_arch IN_LIST available_architectures)
        set(target_arch "${requested_arch}")
         message(STATUS "Found matching architecture")
      else()
        message(WARNING "Requested architecture ${requested_arch} not found. Using default architecture instead")
      endif()
    endif()


    if(NOT target_arch)
      # Select a default architecture (e.g., highest available)
      list(LENGTH available_architectures available_count)
      if(${available_count} GREATER 0)
        list(REVERSE available_architectures reversed_available_archs)
        list(GET reversed_available_archs 0 default_arch)
        set(target_arch ${default_arch})
         message(STATUS "Using Default CUDA architecture: ${target_arch}")
       else()
          message(FATAL_ERROR "No CUDA architecture detected")
       endif()
    endif()
  
    set(${out_variable} "${target_arch}" PARENT_SCOPE)
endfunction()

select_cuda_architecture(SELECTED_CUDA_ARCHITECTURE ${AVAILABLE_CUDA_ARCHITECTURES})

message(STATUS "Selected CUDA Architecture: ${SELECTED_CUDA_ARCHITECTURE}")
```

The second example implements a function named `select_cuda_architecture`, which selects a suitable CUDA architecture based on user preference, availability or defaults.  First, the environment variable `CUDA_ARCHITECTURE` is checked. If set and matching the architectures returned in the previous step, that value is selected. If not, or the environment variable is unset, the code reverses the list of available architectures and selects the last element, which based on the convention of how `nvcc` prints the output, represents the highest architecture.   This approach ensures that a compatible architecture is used and fallbacks are in place when none is explicitly provided. A message indicating the architecture being used is output to the console.

**Code Example 3: Applying Architecture to Target**

```cmake
add_executable(my_cuda_app src/main.cu)
find_package(CUDA REQUIRED)

set_target_properties(my_cuda_app PROPERTIES
  CUDA_ARCHITECTURES "${SELECTED_CUDA_ARCHITECTURE}"
  CUDA_TOOLKIT_ROOT_DIR "${CUDA_TOOLKIT_ROOT_DIR}"
)

# Example of additional flags
target_compile_options(my_cuda_app PRIVATE -rdc=true)

target_link_libraries(my_cuda_app PRIVATE CUDA::cudart_static)
```

This third code snippet applies the selected architecture from the previous step to a CUDA executable. The target property `CUDA_ARCHITECTURES` is set to the variable `SELECTED_CUDA_ARCHITECTURE`, which was determined by previous example. This tells `nvcc` to compile code only for the specified architecture, and prevents problems on target machines that don't support higher architectures.  This snippet also demonstrates how other necessary details, like the root CUDA install directory and linking with the CUDA runtime library, are established and applied to the target, demonstrating other necessary setup for a CUDA project. I’ve included an example of how to specify compilation flags like `-rdc=true` (for relocatable device code), which is frequently necessary in larger CUDA projects.

**Resource Recommendations:**

For further study, I recommend exploring the official CMake documentation, especially the sections pertaining to the `execute_process` command, target properties, and the `find_package` command when working with CUDA. Thoroughly examining the NVIDIA documentation pertaining to the `nvcc` compiler, especially its architecture flags (`--gpu-architecture`, `-arch`) and the conventions for specifying compute capabilities is extremely beneficial. Furthermore, reviewing example CMake projects using CUDA which are available on platforms like GitHub, can provide more context and usage patterns for a practical example. Finally, understanding the specific needs of the projects under development will aid in tailoring CMake configuration to the desired build behaviour.
