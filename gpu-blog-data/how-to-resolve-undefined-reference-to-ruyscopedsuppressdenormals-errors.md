---
title: "How to resolve 'undefined reference to `ruy::ScopedSuppressDenormals'' errors when linking TensorFlow Lite libraries in a C++ CMake project?"
date: "2025-01-30"
id: "how-to-resolve-undefined-reference-to-ruyscopedsuppressdenormals-errors"
---
The "undefined reference to `ruy::ScopedSuppressDenormals'" error during linking of TensorFlow Lite C++ projects, particularly within CMake environments, typically indicates a mismatch or omission in the required build dependencies for Ruy, TensorFlow Lite's core matrix multiplication library. I've encountered this issue numerous times, specifically when transitioning between different TensorFlow Lite versions or when manually managing dependency libraries instead of relying on pre-built packages. The root cause almost always stems from an incomplete or incorrect inclusion of the Ruy library and its associated compilation flags. The linker simply cannot find the symbols provided by Ruy's implementation.

The `ruy::ScopedSuppressDenormals` class is designed to handle floating-point denormal values, which can impact performance on certain processor architectures. When TensorFlow Lite attempts to use Ruy’s matrix multiplication functions, it often invokes this scope guard, hence the linker error when it's absent. Resolving this requires precise understanding of how Ruy is integrated into the build process. Incorrectly specified include directories, missing library links, or a conflict in library versions can easily lead to this specific undefined reference.

My strategy to resolve this consistently begins with ensuring the Ruy library is properly built and available for linking. TensorFlow Lite, being primarily C++, leverages CMake extensively for its build process, meaning Ruy’s inclusion is largely dictated by the CMake configuration. The most common errors revolve around either overlooking the need to build Ruy separately, or incorrectly specifying its location in the CMakeLists.txt file used in your project. It's also vital to remember that Ruy can sometimes be bundled inside the TensorFlow Lite library itself, depending on the build procedure used. When this happens, the errors usually point to issues with the version compatibility of TensorFlow Lite and its embedded Ruy.

To illustrate, consider a scenario where I am building a custom TensorFlow Lite application. I'll walk through common issues with three code examples that demonstrate possible pitfalls and their respective solutions within a CMake context.

**Example 1: Missing Ruy Library Link**

In this case, the CMakeLists.txt might have neglected to link the Ruy library directly, assuming it’s part of another linked library. This is a frequent mistake when relying on a generic, non-specific approach to linking TensorFlow Lite.

```cmake
cmake_minimum_required(VERSION 3.10)
project(TFLiteApp)

find_package(TensorFlowLite REQUIRED)

add_executable(TFLiteApp main.cpp)

# Incorrect: Assumes Ruy is implicitly included in TensorFlowLite
target_link_libraries(TFLiteApp TensorFlowLite::tensorflowlite)

# Additional target_include_directories to include other headers.
target_include_directories(TFLiteApp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
```

This will almost certainly result in the undefined reference error. The fix involves explicitly finding and linking Ruy, which may require building it separately. If using a pre-built TFLite package, the dependency handling should be part of the provided configuration. However, in many cases, especially when building TFLite from source, an explicit inclusion of the Ruy library is necessary. When attempting a custom build the code must look something like this.

```cmake
cmake_minimum_required(VERSION 3.10)
project(TFLiteApp)

find_package(TensorFlowLite REQUIRED)
# Assume Ruy is pre-built and Ruy_DIR variable contains the path to
# Ruy's install directory
find_package(Ruy REQUIRED)

add_executable(TFLiteApp main.cpp)

target_link_libraries(TFLiteApp TensorFlowLite::tensorflowlite Ruy::ruy)
target_include_directories(TFLiteApp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
```

The `find_package(Ruy REQUIRED)` line searches for the Ruy library package, as if it were installed as a separate project. The variable `Ruy_DIR` needs to be set either as a CMake variable on the command line or via `set(Ruy_DIR /path/to/ruy/install CACHE PATH "Ruy install path")` within the `CMakeLists.txt`. The key is the addition of `Ruy::ruy` to the `target_link_libraries` statement; this explicitly instructs the linker to search and pull in the Ruy library's code when linking the executable. If you are using a version of TensorFlow Lite that has Ruy baked in to the TensorFlow lite library, then you would not add `find_package(Ruy REQUIRED)` or include `Ruy::ruy` on the link line.

**Example 2: Incorrect Include Paths**

Sometimes, while the Ruy library is linked, the compiler might not be able to locate the Ruy header files, leading to build-time errors which manifest at link time. The compiler fails to include the files needed by the libraries which then results in a link time error later on.

```cmake
cmake_minimum_required(VERSION 3.10)
project(TFLiteApp)

find_package(TensorFlowLite REQUIRED)
find_package(Ruy REQUIRED)

add_executable(TFLiteApp main.cpp)

target_link_libraries(TFLiteApp TensorFlowLite::tensorflowlite Ruy::ruy)

# Missing target_include_directories for Ruy
# target_include_directories(TFLiteApp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Incorrectly specifying include paths of other libraries.
```

This configuration will successfully link but is incomplete, the code will fail due to missing include files. The correction involves using the Ruy package's include directories with the `target_include_directories` command.

```cmake
cmake_minimum_required(VERSION 3.10)
project(TFLiteApp)

find_package(TensorFlowLite REQUIRED)
find_package(Ruy REQUIRED)

add_executable(TFLiteApp main.cpp)

target_link_libraries(TFLiteApp TensorFlowLite::tensorflowlite Ruy::ruy)

# Correctly include the Ruy header files.
target_include_directories(TFLiteApp PRIVATE ${Ruy_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include )
```

By adding `Ruy_INCLUDE_DIRS` this makes sure the Ruy header files are available to the compiler. This is usually set by the find_package module. It’s critical to understand the correct way that include paths are specified via the `find_package` module. Incorrectly including paths can still cause build-time errors that resolve to undefined references at link time.

**Example 3: Library Version Conflicts**

In certain situations, I have encountered inconsistencies between the version of Ruy expected by TensorFlow Lite and the one included in the system or build. This is often the cause of an undefined reference to a specific class or function in Ruy when different versions of libraries are intermingled. The below example assumes the Ruy library to have been built correctly, but that it is not the version the tflite library is expecting.

```cmake
cmake_minimum_required(VERSION 3.10)
project(TFLiteApp)

find_package(TensorFlowLite REQUIRED)
find_package(Ruy REQUIRED)

add_executable(TFLiteApp main.cpp)

target_link_libraries(TFLiteApp TensorFlowLite::tensorflowlite Ruy::ruy)

#Include directores for project and dependencies.
target_include_directories(TFLiteApp PRIVATE ${Ruy_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
```

In this case, the `find_package` will find the Ruy library. It will successfully link and compile the code, but the link will error with an undefined reference to `ruy::ScopedSuppressDenormals`. This is because the compiled code is using the function from a different version of the library than the one that is being linked. To remedy this, either the version of the tflite library or the ruy library will need to be changed. Ideally the version of tflite will be changed to be compatible with the version of ruy as the tflite library should be developed to not break compatibility. Unfortunately this is not always the case and it becomes necessary to change the version of Ruy. To find a compatible version of Ruy you would need to trace back through the tflite library's source control to the ruy submodule. You would then need to checkout the compatible version of ruy.

These examples highlight the main pitfalls encountered. The resolution consistently involves ensuring that: 1) Ruy is properly included in the CMakeLists.txt. 2) The include paths are correctly configured so that the compiler can find the ruy header files. 3) There are no version mismatches between Ruy and the TensorFlow Lite library being linked.

To further assist troubleshooting, I recommend several resources which have consistently provided useful insights: The official CMake documentation offers comprehensive information about all CMake commands and configuration techniques. Additionally, reviewing the TensorFlow Lite source code, specifically the build scripts related to Ruy, can provide more precise instructions for handling dependency management. Lastly, the official TensorFlow Lite documentation provides a build from source section which can provide guidance. Examining the build process that TensorFlow Lite employs can often highlight build steps and nuances not easily found elsewhere. These, combined with systematic debugging, are usually sufficient to resolve the `ruy::ScopedSuppressDenormals` undefined reference.
