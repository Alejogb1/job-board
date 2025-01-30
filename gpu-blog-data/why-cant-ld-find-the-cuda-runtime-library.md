---
title: "Why can't ld find the CUDA runtime library?"
date: "2025-01-30"
id: "why-cant-ld-find-the-cuda-runtime-library"
---
The inability of the linker (`ld`) to locate the CUDA runtime library during compilation frequently arises from discrepancies in the compiler's search paths and the actual installation location of the CUDA Toolkit. Specifically, the linker relies on environment variables and configuration files to know where to look for necessary shared objects (.so files on Linux, .dylib on macOS, .dll on Windows), and when these are misconfigured, `ld` reports the dreaded “cannot find -lcudart” or similar error.

I've personally encountered this exact problem on numerous projects, ranging from small, experimental GPU kernel deployments to larger simulation platforms. Typically, the issue stems from one of three core areas: incorrect environment variables, missing symbolic links, or misconfiguration during the CUDA toolkit installation itself.

Let's first break down why the linker needs the CUDA runtime library in the first place. When you compile CUDA code, specifically, the device code (.cu files), the `nvcc` compiler generates intermediary object files. These objects, which might not contain machine code targeted for the GPU itself, reference functions provided by the CUDA runtime. Crucially, this runtime encompasses fundamental operations such as memory allocation on the GPU (`cudaMalloc`), kernel launches (`cudaLaunchKernel`), data transfers between host and device memory (`cudaMemcpy`), and other crucial elements needed for successful GPU operation. To create a final executable, these references must be resolvable by the linker.

The linker's job is to combine all the compiled object files, along with the needed library dependencies, into a single, coherent executable. This is why the `-lcudart` flag (or its platform-specific equivalent) is included during linking; it tells the linker to include the CUDA runtime library when creating the executable. If the linker cannot locate the `cudart` library (i.e. the libcudart.so/.dylib/.dll file), linking will fail.

The primary cause, in my experience, is a misconfiguration of environment variables such as `LD_LIBRARY_PATH` (on Linux), `DYLD_LIBRARY_PATH` (on macOS), or the `PATH` variable on Windows. These variables tell the operating system where to look for shared libraries at runtime and also, as important here, guide the linker during compilation. When the CUDA toolkit is installed, it typically provides its library location to these variables. However, discrepancies can emerge if the toolkit was installed in a custom location, if a different version of CUDA is active than the one being used for the compilation, or if the environment variable itself was modified independently.

Secondly, symbolic links can cause issues. If a symbolic link (symlink) is used to point to the CUDA library location, a broken symlink or one pointing to a different version of CUDA than the one desired will cause the error. The linker will follow the link and, if the file is missing or incorrect, will fail. This situation is particularly common in cloud environments where complex installation scripts are used.

Finally, the CUDA toolkit installation itself can sometimes be the cause. A faulty installation, or incomplete installation steps, can lead to the library not being located in the expected locations.

Here are some concrete code examples that demonstrate common ways to address this problem during compilation:

**Example 1: Explicit Library Path Specification**

Assume I'm using a compiler command similar to this:

```bash
g++ -o my_cuda_program my_cuda_program.o -lcudart
```

And it fails because `ld` can’t find `libcudart.so`. We can modify this to explicitly tell `ld` where to find the library file.

```bash
g++ -o my_cuda_program my_cuda_program.o -lcudart -L/usr/local/cuda-11.8/lib64
```

*Commentary:* Here, `-L/usr/local/cuda-11.8/lib64` specifies the directory where the CUDA library is located using the `-L` flag before specifying the library with `-lcudart`.  This assumes that the CUDA toolkit is installed in `/usr/local/cuda-11.8` with architecture-specific libraries in `/lib64`. This explicit approach is helpful when environment variables are not configured correctly or when you need to target a specific CUDA installation.

**Example 2: Setting `LD_LIBRARY_PATH`**

Instead of modifying the compilation command, we can modify the system environment:

```bash
export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH
g++ -o my_cuda_program my_cuda_program.o -lcudart
```

*Commentary:* Here, I first set `LD_LIBRARY_PATH` environment variable to include the correct library directory. Then I execute the normal compilation command. This tells the operating system where to find the shared libraries at runtime (which also helps the linker find them during compilation). If there are other libraries which already use the variable, we should append to it, not overwrite it. I have found this method to be more useful for making more flexible builds, as multiple projects or targets using the same system installation can each access the correct libraries via their own `LD_LIBRARY_PATH` variable. This approach is generally preferable to modifying the `Makefile` or build process for each individual project.

**Example 3: Using CMake for Build System Management**

Build systems like CMake can encapsulate complex library linking, making the entire process less prone to error.  Here is a fragment from a CMake file that handles CUDA library linkage:

```cmake
find_package(CUDA REQUIRED)

add_executable(my_cuda_program my_cuda_program.cu)

target_link_libraries(my_cuda_program PRIVATE ${CUDA_LIBRARIES})
```
*Commentary:* The `find_package(CUDA REQUIRED)` line in `CMakeLists.txt` attempts to locate the CUDA toolkit based on settings or environment variables, potentially using configurations specified during CMake execution. The found libraries are then placed in the variable `${CUDA_LIBRARIES}`.  The line `target_link_libraries` adds CUDA runtime library as a private dependency of `my_cuda_program`, avoiding propagation to targets that might depend on it.  This method allows the compiler to find the library correctly, without explicit command line arguments.

When debugging issues of this kind, the first step should be to verify that the CUDA Toolkit is correctly installed. The NVIDIA developer site provides tools and guides for toolkit installation verification. After verifying the base installation, I always check my environment variables, comparing them against the installation location of the CUDA toolkit.

For a more in-depth understanding of environment variables, system documentation for your specific operating system is invaluable. In addition, documentation on using specific build systems, such as CMake, is extremely helpful. I typically consult books related to system administration and development workflow to improve my understanding of the low-level system mechanics.  Finally, the official CUDA toolkit documentation is critical to understand the exact install locations for libraries.
