---
title: "How can I add a project name to the 'include' directory during CMake installation?"
date: "2025-01-30"
id: "how-can-i-add-a-project-name-to"
---
The core issue lies in understanding CMake's variable scoping and the distinction between installation directories and build directories.  During the build process, CMake operates within a temporary build tree, while the installation stage defines the final location of your project's components.  Simply appending the project name to the `include` directory within the build tree won't persist after installation.  The solution necessitates manipulating CMake's installation rules.  Over the years, I've encountered numerous similar problems while working on large-scale embedded systems projects, and the consistent solution involved precisely controlling the `CMAKE_INSTALL_INCLUDEDIR` variable and utilizing target-specific installation commands.

My approach focuses on leveraging CMake's target-specific installation mechanisms rather than relying on global modifications to installation directories. This provides better control and avoids potential conflicts in multi-project builds.  The key is to specify the desired installation location for each library or header file independently.

**1.  Clear Explanation:**

The recommended methodology involves defining the installation path for header files during the `install(TARGETS ...)` command.  Instead of modifying the overall include directory (`CMAKE_INSTALL_INCLUDEDIR`), we explicitly set the destination directory for each target’s headers. This destination can include the project name, ensuring clear separation during installation and preventing naming clashes.

The `install(TARGETS ...)` command offers several arguments.  Crucially, the `DESTINATION` argument allows you to specify the location within the installation directory where the target's files (including headers) should reside.  By constructing a path that incorporates your project name, you effectively create a project-specific subdirectory within the installation's `include` directory.

Further, it’s beneficial to use a CMake variable to hold the project name, allowing for easy modification and consistent naming across the project. This improves maintainability and prevents hardcoding the project name in multiple locations.

**2. Code Examples with Commentary:**

**Example 1: Basic Installation with Project Name**

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyProject)

add_library(MyLib MyLib.cpp)

set(PROJECT_NAME "MyProject")  # Define project name variable

install(TARGETS MyLib DESTINATION include/${PROJECT_NAME})
```

This example demonstrates the fundamental approach.  The `PROJECT_NAME` variable is defined and used within the `DESTINATION` argument of the `install(TARGETS ...)` command.  After installation, the headers from `MyLib` will reside in `<installation_prefix>/include/MyProject`.


**Example 2: Handling Multiple Libraries and Header Files**

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyProject)

add_library(LibA LibA.cpp)
add_library(LibB LibB.cpp)

set(PROJECT_NAME "MyProject")

install(TARGETS LibA DESTINATION include/${PROJECT_NAME})
install(TARGETS LibB DESTINATION include/${PROJECT_NAME})
```

This example extends the concept to multiple libraries. Each library's headers are installed into the same project-specific subdirectory, maintaining organization even with a growing project.  This avoids potential conflicts if multiple libraries share common header file names.

**Example 3:  Advanced Scenario with Custom Include Paths**

```cmake
cmake_minimum_required(VERSION 3.10)
project(MyProject)

add_library(MyLib MyLib.cpp)
target_include_directories(MyLib PUBLIC include/MyLib) # Custom include path

set(PROJECT_NAME "MyProject")

install(TARGETS MyLib DESTINATION include/${PROJECT_NAME}
         ARCHIVE DESTINATION lib/${PROJECT_NAME}
         LIBRARY DESTINATION lib/${PROJECT_NAME})
```

This example showcases handling custom include directories within the library target itself, and then installs both the library and its associated headers into the project-specific subdirectory.  Note that we use explicit `ARCHIVE`, `LIBRARY` and `RUNTIME` to control the install location of each artifact type. This approach provides a cleaner structure, particularly beneficial when dealing with static and shared libraries.



**3. Resource Recommendations:**

I would strongly suggest consulting the official CMake documentation.  Pay close attention to the sections on installation commands, target properties, and variable scoping.  A deep understanding of these topics is crucial for effective CMake usage, particularly in complex projects.  Furthermore, reviewing examples from established open-source projects that use CMake extensively can provide valuable practical insights and demonstrate best practices.  Finally, familiarizing oneself with the concepts of build systems and the distinction between build and installation directories will solidify your understanding of the underlying processes.  This systematic approach is invaluable for tackling any CMake-related challenge.
