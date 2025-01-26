---
title: "Why are TensorFlow 2.8 static libraries built with CMake showing undefined references to ruy and flatbuffers?"
date: "2025-01-26"
id: "why-are-tensorflow-28-static-libraries-built-with-cmake-showing-undefined-references-to-ruy-and-flatbuffers"
---

TensorFlow 2.8's reliance on Ruy and FlatBuffers, particularly when building static libraries with CMake, often results in undefined reference errors. This stems from how these dependencies are managed within TensorFlow's build system and how CMake handles static library linking. In my experience, successfully building TensorFlow as a static library requires a thorough understanding of the dependency graph and proper configuration of CMake's linking procedures.

The core issue arises from the difference between static and dynamic linking. When building a dynamic library (.so or .dll), the linker primarily ensures that the symbol definitions are present somewhere in the loaded libraries at runtime. Conversely, when creating a static library (.a or .lib), the linker needs to pull all required code from the dependent libraries and embed it directly into the resultant static library at build time. If the linking process is not correctly configured, especially with transitive dependencies like Ruy and FlatBuffers, the necessary code sections will not be incorporated, leading to undefined reference errors. Specifically, when TensorFlow is configured using CMake, the default configurations may not automatically include these transitive dependencies for a static library target. This can happen due to several factors: missing target dependencies, improper linking flag propagation, or inconsistent definition of compile and link flags across various dependency levels.

Consider the build process conceptually. TensorFlow's CMake configuration scripts define different targets, such as the core TensorFlow library and the associated kernels. These kernels often utilize operations that internally depend on Ruy, a library for performing efficient matrix multiplications, and FlatBuffers, a serialization library. When building a shared library, the system can rely on dynamic linker resolving these dependencies during runtime. However, if creating a static library, the linker needs to be explicitly instructed to include Ruy and FlatBuffers implementations during the build. Without these specific linking directives, the TensorFlow static library will be missing those implementations.

Let’s break down how this might materialize in a practical context:

**Example 1: Missing Target Dependencies**

Assume our project depends on a TensorFlow static library, let's call it 'libtensorflow_static.a', and we are building it using CMake. A common error arises when our top-level CMakeLists.txt lacks a direct dependency declaration on Ruy or FlatBuffers. Although the 'libtensorflow_static.a' might use those libraries, if the CMake configuration doesn't list them as explicit requirements, the linker will not include the requisite code within the TensorFlow static archive. In the top-level `CMakeLists.txt` of our project, this would look something like:

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyTensorFlowProject)

find_package(TensorFlow REQUIRED) # Assume TensorFlow CMake config is available.

add_executable(my_app main.cpp)

# Incorrect: Missing explicit dependencies
target_link_libraries(my_app PRIVATE libtensorflow_static)

# Correct: Added dependencies
# target_link_libraries(my_app PRIVATE libtensorflow_static ruy flatbuffers)
```
In this scenario, the first approach leads to undefined references because the linker fails to recognize that `my_app` needs symbols from both `ruy` and `flatbuffers` which are used indirectly by `libtensorflow_static`. The second, corrected approach explicitly lists the direct dependencies. While a user might think `libtensorflow_static` would implicitly contain these, the static link process does not operate in this fashion. Therefore, if our application relies upon functions within `libtensorflow_static` that subsequently utilize symbols within `ruy` or `flatbuffers`, the linker would be unable to locate them unless explicitly provided via `target_link_libraries`.

**Example 2: Improper Link Flag Propagation within TensorFlow's Build**

Another common reason arises from incomplete or incorrect propagation of linking flags within TensorFlow's internal build system. Even if TensorFlow's CMake configurations define target dependencies, these can fail to properly propagate to the static library target if linking flags are not consistently applied. Suppose our user is trying to integrate their library with TensorFlow, they must add their flags to the `tensorflow_cc` target which contains all tensorflow c++ components. When building a static library, the default `cc_library` rules might inadvertently skip the required flags. Let’s simulate this behavior:
```cmake
# Inside TensorFlow's internal CMake configuration:

# Incorrect - Flags are not propagated to a separate static target properly
add_library(tensorflow_cc STATIC
    # Some source files
    )

# This does not fully propagate the transitive dependencies when linking a static library.
# target_link_libraries(tensorflow_cc PUBLIC ruy flatbuffers)

# Correct
add_library(tensorflow_cc SHARED
     # Some source files
     )
 target_link_libraries(tensorflow_cc PUBLIC ruy flatbuffers)

 # Separate definition for static tensorflow library

 add_library(tensorflow_static STATIC
   # copy sources from tensorflow_cc
 )
 # explicit transitive dependency for static target.
 target_link_libraries(tensorflow_static PUBLIC tensorflow_cc)
```

In the erroneous case, even though  `tensorflow_cc` is built correctly with `ruy` and `flatbuffers`, the propagation to `tensorflow_static` is omitted. The `tensorflow_static` would not be able to locate the relevant functions during compilation. To solve this, we can explicitly create a static target and link the `tensorflow_cc` output to that library and add the dependency to that newly created `tensorflow_static` library.

**Example 3: Conflicting Definitions of Compile and Link Flags**

A more subtle issue can occur with inconsistent compilation or linking flags. For instance, one part of TensorFlow might compile Ruy with position-independent code (PIC), while another does not. This can lead to linking errors when compiling a static library as all object files must be compiled consistently with PIC or without, when combining them into static archive.
```cmake
# Within TensorFlow's internal build
# Incorrect - Inconsistent compilation flag

# For Ruy
add_library(ruy_target STATIC
    # Ruy source files
    )
target_compile_options(ruy_target PRIVATE  -fno-PIC)

# For TensorFlow Core
add_library(tensorflow_core STATIC
    # Tensorflow source files
    )
target_compile_options(tensorflow_core PRIVATE -fPIC)

# This approach is likely to fail as object file PIC/non-PIC mix
target_link_libraries(tensorflow_static PUBLIC tensorflow_core ruy_target)

# Correct - Enforce consistent PIC flags across all relevant targets
add_library(ruy_target STATIC
    # Ruy source files
    )

add_library(tensorflow_core STATIC
    # Tensorflow source files
    )
target_compile_options(ruy_target tensorflow_core PRIVATE -fPIC) # consistent flag

target_link_libraries(tensorflow_static PUBLIC tensorflow_core ruy_target)

```

This scenario involves an inconsistency in PIC flag configuration. In the first example, object files from `ruy_target` will be compiled without PIC, and the object files from `tensorflow_core` will be compiled with PIC. When linking these together into a static archive using `target_link_libraries`, a link error will happen, as the object files have different configurations. The corrected solution enforces consistent flags for both libraries via `target_compile_options`, ensuring that the static linkage is successful.

To resolve these types of issues and create a functioning TensorFlow static library, one must ensure that all transitive dependencies of the library (e.g., Ruy and FlatBuffers) are explicitly declared and properly linked when configuring the project with CMake. This involves careful examination of the TensorFlow project's CMakeLists.txt files, potentially needing adjustments to propagate linking flags consistently across all targets. When working with a static library, all the necessary code should be incorporated into the archive during the link phase. It may be beneficial to create a separate target for a static library to explicitly define the necessary linking flags for the correct dependency resolution. Additionally, it is crucial to maintain consistency in compile and link flags across all parts of the project including third party dependencies.

For further reading, I would recommend exploring the CMake documentation, particularly the sections on target dependencies, linking behavior of static libraries, and compile options. Additionally, a thorough investigation into the TensorFlow source code, particularly the CMake configuration, is beneficial. Exploring build system documentation from other build systems like Bazel can also reveal how dependency management can be better structured when building large projects. Finally, consulting articles or documentation related to the specific dependencies (Ruy and FlatBuffers) can clarify their usage and provide additional context. These resources can assist in understanding the intricate process of building static libraries and resolving issues related to transitive dependencies within complex projects like TensorFlow.
