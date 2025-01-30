---
title: "How can I integrate TensorFlow Lite with CMake?"
date: "2025-01-30"
id: "how-can-i-integrate-tensorflow-lite-with-cmake"
---
Integrating TensorFlow Lite with CMake presents a specific set of challenges rooted in the diverse compilation environments and target platforms TensorFlow Lite aims to support. Direct inclusion of TensorFlow Lite libraries as simple header-only dependencies is often insufficient; rather, a more robust build system integration is required to ensure compatibility across different architectures and operating systems. The primary hurdle lies in the need to locate the pre-built TensorFlow Lite libraries, link them appropriately, and manage their associated dependencies.

My experience across several embedded projects has shown that the most reliable approach involves using CMake's `find_package` mechanism in conjunction with a thoughtfully structured `CMakeLists.txt` file. This allows for a platform-agnostic configuration that adapts to various TensorFlow Lite distributions, whether installed through package managers, pre-built binaries, or custom builds. The goal is to make the build process as smooth as possible, abstracting away the complexity of TensorFlow Lite's underlying build system.

The initial step is to locate the TensorFlow Lite libraries. Typically, these are provided as a static library (`.a` or `.lib`) and the corresponding header files. TensorFlow Lite distributions frequently package these within a specific directory structure. My approach involves creating a CMake module that intelligently searches for this structure, setting appropriate variables for linking later in the build process.

Here's an illustrative example of a `FindTensorFlowLite.cmake` file. Place this file in a `cmake/modules` directory within your project.

```cmake
# cmake/modules/FindTensorFlowLite.cmake

find_path(TENSORFLOWLITE_INCLUDE_DIR
  NAMES tensorflow/lite.h
  PATHS /usr/include /usr/local/include /opt/tensorflow/include
  PATH_SUFFIXES tensorflow-lite
)

find_library(TENSORFLOWLITE_LIBRARY
  NAMES tensorflowlite
  PATHS /usr/lib /usr/local/lib /opt/tensorflow/lib
  PATH_SUFFIXES lib
)

if(TENSORFLOWLITE_INCLUDE_DIR AND TENSORFLOWLITE_LIBRARY)
  set(TENSORFLOWLITE_FOUND TRUE)
  set(TENSORFLOWLITE_INCLUDE_DIRS ${TENSORFLOWLITE_INCLUDE_DIR})
  set(TENSORFLOWLITE_LIBRARIES ${TENSORFLOWLITE_LIBRARY})
else()
  set(TENSORFLOWLITE_FOUND FALSE)
  message(WARNING "TensorFlow Lite not found. Please ensure it is installed and the paths are correct.")
endif()

if(TENSORFLOWLITE_FOUND)
  if(NOT TARGET TensorFlowLite::TensorFlowLite)
      add_library(TensorFlowLite::TensorFlowLite UNKNOWN IMPORTED)
      set_target_properties(TensorFlowLite::TensorFlowLite PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${TENSORFLOWLITE_INCLUDE_DIRS}"
          IMPORTED_LOCATION "${TENSORFLOWLITE_LIBRARIES}"
      )
  endif()
endif()
```

This script uses `find_path` and `find_library` commands to locate the include directory containing `tensorflow/lite.h` and the library file itself. I have included common system paths like `/usr/include`, `/usr/local/include`, and `/opt/tensorflow/include`, which generally cover various installation methods. `PATH_SUFFIXES` is used to accommodate common directory structures used by TensorFlow Lite packages. If both are found, it sets `TENSORFLOWLITE_FOUND` to `TRUE` and creates an imported target `TensorFlowLite::TensorFlowLite`, making it easy to link against later. If the library is not found, the script displays a warning message. This targeted approach provides a more controlled and transparent process than relying on system-wide environment variables.

Now, in your main project's `CMakeLists.txt`, you integrate the module like this:

```cmake
# CMakeLists.txt

cmake_minimum_required(VERSION 3.10)
project(MyTFLiteProject)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

find_package(TensorFlowLite REQUIRED)

add_executable(my_tflite_app main.cpp)

if(TENSORFLOWLITE_FOUND)
    target_link_libraries(my_tflite_app PRIVATE TensorFlowLite::TensorFlowLite)
endif()
```

Here, we set the `CMAKE_MODULE_PATH` to include the directory containing our `FindTensorFlowLite.cmake` module. Then `find_package(TensorFlowLite REQUIRED)` invokes the search logic from the module. The `REQUIRED` keyword ensures that the CMake configuration will fail if the TensorFlow Lite library is not found, preventing compilation issues later. Lastly, if the package is found, the executable `my_tflite_app` is linked against the imported target `TensorFlowLite::TensorFlowLite`, enabling the application to use TensorFlow Lite functionalities. This structure allows separation between searching for dependencies and linking them, improving maintainability.

For projects that require more complex configurations, such as using specific TensorFlow Lite delegates (e.g., GPU or NNAPI), additional steps are required. You might need to link against other libraries and set relevant compiler flags. This can be accomplished by expanding the `FindTensorFlowLite.cmake` module and refining the target linking in the `CMakeLists.txt`.

Consider a situation where you need to compile with the GPU delegate. Here's an extended example of the `FindTensorFlowLite.cmake` module:

```cmake
# cmake/modules/FindTensorFlowLite.cmake (Extended)

find_path(TENSORFLOWLITE_INCLUDE_DIR
  NAMES tensorflow/lite.h
  PATHS /usr/include /usr/local/include /opt/tensorflow/include
  PATH_SUFFIXES tensorflow-lite
)

find_library(TENSORFLOWLITE_LIBRARY
  NAMES tensorflowlite
  PATHS /usr/lib /usr/local/lib /opt/tensorflow/lib
  PATH_SUFFIXES lib
)

find_library(TENSORFLOWLITE_GPU_LIBRARY
    NAMES tensorflowlite_gpu
    PATHS /usr/lib /usr/local/lib /opt/tensorflow/lib
    PATH_SUFFIXES lib
)

if(TENSORFLOWLITE_INCLUDE_DIR AND TENSORFLOWLITE_LIBRARY)
  set(TENSORFLOWLITE_FOUND TRUE)
  set(TENSORFLOWLITE_INCLUDE_DIRS ${TENSORFLOWLITE_INCLUDE_DIR})
  set(TENSORFLOWLITE_LIBRARIES ${TENSORFLOWLITE_LIBRARY})

  if(TENSORFLOWLITE_GPU_LIBRARY)
    set(TENSORFLOWLITE_GPU_FOUND TRUE)
    set(TENSORFLOWLITE_GPU_LIBRARIES ${TENSORFLOWLITE_GPU_LIBRARY})
  else()
    set(TENSORFLOWLITE_GPU_FOUND FALSE)
    message(WARNING "TensorFlow Lite GPU delegate library not found. GPU support will be disabled.")
  endif()

else()
  set(TENSORFLOWLITE_FOUND FALSE)
  message(WARNING "TensorFlow Lite not found. Please ensure it is installed and the paths are correct.")
endif()

if(TENSORFLOWLITE_FOUND)
  if(NOT TARGET TensorFlowLite::TensorFlowLite)
      add_library(TensorFlowLite::TensorFlowLite UNKNOWN IMPORTED)
      set_target_properties(TensorFlowLite::TensorFlowLite PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES "${TENSORFLOWLITE_INCLUDE_DIRS}"
          IMPORTED_LOCATION "${TENSORFLOWLITE_LIBRARIES}"
      )
  endif()

  if(TENSORFLOWLITE_GPU_FOUND)
    if(NOT TARGET TensorFlowLite::TensorFlowLiteGPU)
        add_library(TensorFlowLite::TensorFlowLiteGPU UNKNOWN IMPORTED)
        set_target_properties(TensorFlowLite::TensorFlowLiteGPU PROPERTIES
            IMPORTED_LOCATION "${TENSORFLOWLITE_GPU_LIBRARIES}"
        )
    endif()
  endif()
endif()
```

This extended module now includes `find_library` for the GPU delegate (`tensorflowlite_gpu`). If found, it sets `TENSORFLOWLITE_GPU_FOUND` and defines an imported target `TensorFlowLite::TensorFlowLiteGPU`. In the updated `CMakeLists.txt`, you can conditionally link with this target:

```cmake
# CMakeLists.txt (Extended)

cmake_minimum_required(VERSION 3.10)
project(MyTFLiteProject)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

find_package(TensorFlowLite REQUIRED)

add_executable(my_tflite_app main.cpp)

if(TENSORFLOWLITE_FOUND)
  target_link_libraries(my_tflite_app PRIVATE TensorFlowLite::TensorFlowLite)
  if(TENSORFLOWLITE_GPU_FOUND)
    target_link_libraries(my_tflite_app PRIVATE TensorFlowLite::TensorFlowLiteGPU)
    target_compile_definitions(my_tflite_app PRIVATE "TFLITE_USE_GPU")
  endif()
endif()

```

Now the executable is linked to `TensorFlowLite::TensorFlowLiteGPU` when available and a preprocessor definition (`TFLITE_USE_GPU`) is added which can be used in the code for conditional delegate initialization. Such granular handling helps in creating versatile builds tailored for specific needs.

For a comprehensive understanding of CMake itself, explore the official CMake documentation. For deeper dives into advanced TensorFlow Lite features, consult the TensorFlow Lite documentation and community forums. Specifically the TensorFlow Lite C++ API documentation provides essential knowledge on using the API from C++. In embedded development contexts where resources are often constrained, optimization strategies, documented in embedded-specific literature and tutorials will also greatly benefit. These resources collectively provide the foundations for effective CMake-based TensorFlow Lite integration.
