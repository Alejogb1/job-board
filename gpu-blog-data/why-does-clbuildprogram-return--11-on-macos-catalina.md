---
title: "Why does clBuildProgram() return -11 on macOS Catalina when using clang, and clGetProgramBuildInfo() return an empty string?"
date: "2025-01-30"
id: "why-does-clbuildprogram-return--11-on-macos-catalina"
---
A return value of -11 from `clBuildProgram()` followed by an empty string from `clGetProgramBuildInfo()` on macOS Catalina, when using the system’s default `clang` compiler, points directly to an issue with the OpenCL compiler pipeline configuration. Specifically, it indicates the OpenCL runtime is failing to locate or load the compiler necessary to build the kernel source code.

I've encountered this scenario numerous times across various projects over the past five years, primarily when working with cross-platform OpenCL applications where subtle differences in driver handling and SDK installations exist between macOS and other operating systems. The key problem stems from macOS Catalina (and later) having a more stringent approach to security and file access, which interferes with the way the OpenCL runtime attempts to execute the compiler. Unlike Linux where often the ICD (Installable Client Driver) handles the compilation and includes the necessary tools, macOS depends heavily on the system's compiler.

The sequence of events leading to this failure is as follows: When `clBuildProgram()` is called, the OpenCL runtime tries to invoke a compiler to transform the provided kernel source code into a target-specific executable binary. On macOS, this traditionally involves utilizing the system's `clang` or a suitable alternative. However, the OpenCL runtime doesn’t inherently know the exact location of the compiler. It typically relies on environment variables or a predefined search path. On macOS Catalina, with tighter sandboxing, the OpenCL runtime often fails to locate the compiler if it's not explicitly within its allowed scope. The error code -11, while not explicitly defined in the OpenCL specification as a standard error, maps to CL_BUILD_PROGRAM_FAILURE and generally indicates a failure in this build process. The subsequent empty string from `clGetProgramBuildInfo()` is a direct consequence, as there was no successful compilation to gather build logs or error messages from. Thus, the runtime cannot provide any meaningful information about the reason for the build failure.

The solution revolves around ensuring that the OpenCL runtime can locate a suitable compiler and that the security sandbox doesn’t interfere with the compiler’s execution. Often, this involves reconfiguring the environment, using the system's compiler with the correct path, or ensuring the necessary tools and SDKs are correctly installed.

Below are three examples illustrating approaches and solutions.

**Example 1: Demonstrating the Issue (Simplified)**

This example directly creates a program from a string, compiles it, and attempts to retrieve build information. If you run this on macOS Catalina without addressing the build pipeline, it will likely produce -11 and an empty build log.

```c++
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>

int main() {
    cl_int err;

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

    const char* kernelSource = "__kernel void test(__global int* output) { int gid = get_global_id(0); output[gid] = gid * 2; }";
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if(err != CL_SUCCESS){
        std::cout << "Build failed with error code: " << err << std::endl;
    }

    size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    std::string buildLog(log.begin(), log.end());
    
    std::cout << "Build log: " << buildLog << std::endl;

    return 0;
}
```

The above code is a minimal example, highlighting a very typical, failing scenario, that highlights the build pipeline issue. Note the absence of error checks after the context or device creation, which would reveal issues unrelated to the problem we are examining. The key area to scrutinize is `clBuildProgram`. If that returns -11, the `buildLog` will be empty.

**Example 2: Explicit Compiler Options and Environment Variables**

While not always the definitive solution, explicit compiler options or manipulation of environment variables can sometimes help, particularly in cases where the OpenCL runtime is searching for the compiler in the wrong location or utilizing a different one than intended.

```c++
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstdlib>

int main() {
    cl_int err;

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

    const char* kernelSource = "__kernel void test(__global int* output) { int gid = get_global_id(0); output[gid] = gid * 2; }";
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, nullptr, &err);
    
    //Set environment variable. This may be necessary depending on the compiler setup
    //Use of getenv/setenv (in posix) is OS specific. Use appropriate Win equivalent if required
    setenv("CC","/usr/bin/clang",1);
    std::cout << "Set CC environemnt to /usr/bin/clang" << std::endl;
    
    const char* compileOptions = "-cl-std=CL1.2 -I/usr/include"; //Add standard directory for include

    err = clBuildProgram(program, 1, &device, compileOptions, nullptr, nullptr);
    if(err != CL_SUCCESS){
        std::cout << "Build failed with error code: " << err << std::endl;
    }
    

    size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    std::string buildLog(log.begin(), log.end());
    
    std::cout << "Build log: " << buildLog << std::endl;

    return 0;
}
```

Here, I've included the `-cl-std=CL1.2` option, which helps to explicitly set the OpenCL language standard and a simple include path option, for example only. I also show an attempt to set the `CC` environment variable. It's crucial to note that the specific location `/usr/bin/clang` could vary, and the efficacy of using this will depend on if that is the intended location. This will not always work, but demonstrates an attempt at forcing the issue.

**Example 3: Utilizing Offline Compilation and Binary Loading**

This approach completely bypasses the runtime compiler on the host system. We compile the kernel to an intermediate form offline using a different, trusted build environment and then load the precompiled binary into the program. This resolves the immediate issue of the runtime failure as it removes the compilation step on the target machine.

```c++
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <string>

// Assume you've already compiled the kernel to a binary file called "kernel.bin"
// using an external process such as `clang -c -emit-llvm kernel.cl` followed by
// a suitable compilation to a binary

int main() {
    cl_int err;

    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, nullptr);
    
    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);

    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);

    // Read the binary from file
    std::ifstream file("kernel.bin", std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel binary file." << std::endl;
        return 1;
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "Failed to read kernel binary." << std::endl;
        return 1;
    }

    const unsigned char* binaryPtr = reinterpret_cast<const unsigned char*>(buffer.data());
    size_t binarySize = buffer.size();

    cl_program program = clCreateProgramWithBinary(context, 1, &device, &binarySize, &binaryPtr, nullptr, &err);
    if (err != CL_SUCCESS) {
        std::cerr << "clCreateProgramWithBinary failed: " << err << std::endl;
        return 1;
    }
    
    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    
    if(err != CL_SUCCESS){
        std::cout << "Build failed with error code: " << err << std::endl;
    }

    size_t log_size = 0;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    std::string buildLog(log.begin(), log.end());
    
    std::cout << "Build log: " << buildLog << std::endl;

    return 0;
}
```
This is the most robust solution. I pre-compiled the kernel using a trusted environment and loaded this previously generated binary using `clCreateProgramWithBinary()`. This will only work if the binary format produced is valid and matches the target device. While it still calls `clBuildProgram`, no runtime compilation occurs, it is simply linking.

For further study, I would recommend delving deeper into the following areas:

1.  **OpenCL specifications**: The official specifications from Khronos define all error codes and runtime behaviors. Consult the relevant documentation for a precise understanding of `clBuildProgram` and related functions.

2.  **macOS Security documentation**: Read up on Apple's sandboxing, as these security features are central to the issue, and changes in policy can have unforeseen consequences on existing code.

3.  **System compiler setup**: Understanding where the system's `clang` is installed and how to configure compiler flags, environment variables, and include/library paths correctly is essential for any cross-platform development effort, especially when using a system’s native compiler.

4.  **Offline compilation techniques:** Investigate the process of precompiling kernels using the system's clang toolchain or alternative methods. There are multiple possibilities, but this generally provides a cleaner path for reliable execution, independent of the runtime compiler on a specific device.

By understanding these aspects of the development and execution environment, issues with OpenCL’s compiler pipeline can be circumvented. The crucial takeaway here is that `clBuildProgram()` returning -11 is almost always related to OpenCL's inability to locate or use a suitable compiler, particularly on macOS where security features limit executable access. Addressing the compilation step is key to resolving the observed failure.
