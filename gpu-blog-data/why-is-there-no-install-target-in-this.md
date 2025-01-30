---
title: "Why is there no install target in this CMake build?"
date: "2025-01-30"
id: "why-is-there-no-install-target-in-this"
---
The absence of an install target in a CMake build typically stems from a misconfiguration in the `CMakeLists.txt` file, specifically concerning the definition of installation rules using the `install()` command. I've encountered this issue several times during my projects, particularly when transitioning between basic compilation and more comprehensive deployment strategies. A simple build, concerned solely with generating executables or libraries, will not automatically include an install step. This requires explicit instructions.

The core of the problem resides in CMake’s design, which separates building and installing into distinct phases. Building, performed by `cmake --build`, compiles source code and links it into executables and libraries in the build directory. Installation, on the other hand, copies these built artifacts (and possibly other files like configuration files and header files) to a designated installation prefix, a directory meant to hold the final, deployable product. Without `install()` commands specifying which files to copy where, CMake has no instructions to create an install target, which is why no “install” action is present when using the default build system generators (such as Makefiles, Ninja, or Visual Studio).

To rectify this, the `install()` command needs to be applied to the relevant output targets. The most basic structure is:
```cmake
install(TARGETS <target>
    DESTINATION <destination>)
```
Here, `<target>` refers to the name of an executable or library defined in your `CMakeLists.txt`, and `<destination>` refers to the path within the installation prefix where these files should be placed.  The destination path is relative to the installation prefix, specified with the `-DCMAKE_INSTALL_PREFIX=<path>` during CMake configuration. Common subdirectories within the installation prefix include `bin` for executables, `lib` for libraries, `include` for header files and `share` for resources, although these are just conventions and could differ based on the project’s needs.

Let’s illustrate this with a series of examples. In my previous work on a small utility application, I had a `CMakeLists.txt` that initially only defined an executable:

```cmake
cmake_minimum_required(VERSION 3.10)
project(myUtility)

add_executable(myUtility src/main.cpp)
```

Executing `cmake --build .` in a build directory would compile the application into the `myUtility` binary, but there would be no install target. Adding the following `install()` line is required to create one.

```cmake
cmake_minimum_required(VERSION 3.10)
project(myUtility)

add_executable(myUtility src/main.cpp)

install(TARGETS myUtility
    DESTINATION bin)
```

Here, `install(TARGETS myUtility DESTINATION bin)` instructs CMake to copy the built `myUtility` executable to a `bin` subdirectory within the installation prefix. Now, after executing `cmake --build . --target install`, the `myUtility` executable will be found under `$CMAKE_INSTALL_PREFIX/bin/`. If the `CMAKE_INSTALL_PREFIX` was not set during initial configuration it will default to the equivalent of `/usr/local` for most Unix-like systems. The `bin` directory will also be created at that location if it does not already exist. This simple example shows the most basic form of adding an install target, handling only executables.

The second example focuses on installing libraries, including associated header files. While working on a common code repository that was being used by multiple projects, I had a library setup similar to this initial build step:
```cmake
cmake_minimum_required(VERSION 3.10)
project(myLib)

add_library(myLib src/mylib.cpp src/mylib.h)
```

This only created the library file. For others to use it we would also need to install the header files. This is achieved by adding specific install rules.

```cmake
cmake_minimum_required(VERSION 3.10)
project(myLib)

add_library(myLib src/mylib.cpp src/mylib.h)

install(TARGETS myLib
    DESTINATION lib)

install(FILES src/mylib.h
    DESTINATION include)
```

In this code example, `install(TARGETS myLib DESTINATION lib)` will copy the created library to the `lib` subdirectory of the installation prefix. Subsequently, `install(FILES src/mylib.h DESTINATION include)` will copy the header file to the `include` subdirectory. The destination directory has been changed to reflect the typical install path for libraries. This assumes that header files are part of the project source, and not generated files.

Finally, consider a more complex example incorporating multiple targets, as might be the case in larger systems. I was working on a project that had both a library and several executables depending on it:
```cmake
cmake_minimum_required(VERSION 3.10)
project(myComplexProject)

add_library(commonLib src/common.cpp src/common.h)
add_executable(app1 src/app1.cpp)
add_executable(app2 src/app2.cpp)

target_link_libraries(app1 commonLib)
target_link_libraries(app2 commonLib)
```

To provide a complete install, we would need to install the library along with both executables:

```cmake
cmake_minimum_required(VERSION 3.10)
project(myComplexProject)

add_library(commonLib src/common.cpp src/common.h)
add_executable(app1 src/app1.cpp)
add_executable(app2 src/app2.cpp)

target_link_libraries(app1 commonLib)
target_link_libraries(app2 commonLib)

install(TARGETS commonLib
    DESTINATION lib)

install(FILES src/common.h
    DESTINATION include)

install(TARGETS app1 app2
    DESTINATION bin)
```

The code now installs the library, the library header and the executables to their respective locations. It demonstrates handling multiple targets within a single `CMakeLists.txt`. It also shows that multiple `install` commands can be used, and that the same install command can accept multiple targets. All installed files are available under the installation prefix after building with the install target using `cmake --build . --target install`.

In summary, the absence of an install target in a CMake build is almost always attributed to the lack of explicit `install()` commands. One needs to specify what is to be installed and where it needs to be located relative to the install prefix, by target or file. Properly configuring these install steps is crucial for preparing a project for deployment. Failure to configure the install target results in a build environment with no way to deploy the created assets.

For further understanding, I recommend reviewing the CMake documentation on the `install` command, available in the official CMake documentation. Explore examples demonstrating installation of additional types of files like configuration files and resources. Look into packaging and distribution using tools that build on top of the install functionality of CMake such as CPack. Studying best practices related to the `CMAKE_INSTALL_PREFIX` variable is beneficial. The ‘Professional CMake’ book is another excellent resource providing extensive guidance on CMake best practices.
