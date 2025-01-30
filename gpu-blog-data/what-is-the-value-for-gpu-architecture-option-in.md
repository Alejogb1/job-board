---
title: "What is the value for 'gpu-architecture' option in nvcc, and why is 'sm_20' undefined?"
date: "2025-01-30"
id: "what-is-the-value-for-gpu-architecture-option-in"
---
The `gpu-architecture` flag passed to `nvcc`, NVIDIA's CUDA compiler, directly dictates the target instruction set for compiled code, influencing both performance and hardware compatibility. Misunderstanding its role often results in binaries that either fail to execute on the intended GPU or operate suboptimally. My experience porting a large-scale fluid dynamics simulation across various NVIDIA GPU generations highlighted the critical importance of precise architecture specification, revealing several common pitfalls related to this flag and its associated values. Specifically, the absence of a defined `sm_20` value is due to NVIDIA's shift in naming conventions following the Fermi architecture (Compute Capability 2.x), which `sm_20` refers to.

The `gpu-architecture` option instructs `nvcc` to generate device code, also known as PTX or binary cubin code, targeting a specific CUDA compute capability.  Each compute capability number signifies a distinct generation of NVIDIA GPU architectures, introducing new instruction sets, features, and performance characteristics. When developing a CUDA application, you're not solely writing code for the high-level interface (e.g., C++ with CUDA extensions). You're also implicitly dictating the low-level instructions executed by the GPU's Streaming Multiprocessors (SMs). Specifying the appropriate `gpu-architecture` is crucial because, unlike CPU code, where backward compatibility is generally more robust, GPU code compiled for a specific architecture will often refuse to run or exhibit unexpected behavior on GPUs with different architectures. Consequently, a mismatch between compilation target and target hardware manifests through run-time errors, incorrect results, or outright application crashes, particularly when the generated cubin files contain instructions the current device cannot understand.

The `sm_*` notation reflects the older naming scheme for compute capabilities before NVIDIA transitioned to more explicitly naming GPU architectures. The "sm" prefix historically designated "Streaming Multiprocessor" architecture, with a numerical suffix denoting version. `sm_20`, specifically, denotes the Fermi generation, which uses compute capability 2.0. Starting with the Kepler generation, NVIDIA moved to naming GPU architectures via the architecture name itself rather than their Streaming Multiprocessor designation (e.g. `compute_30` for Kepler).  Subsequently, NVIDIA introduced more comprehensive naming conventions, e.g., `compute_70` for Volta, or `compute_86` for Ampere. While older systems might accept `sm_20`, this is a compatibility layer, and targeting architectures through the explicit `compute_` naming convention improves code clarity and removes ambiguity. Additionally, NVIDIA prefers that when building for multiple architectures you use the virtual architecture flag `-arch=compute_xx -code=sm_xx`, which ensures the cubin will work on any device with compute capability `xx` or greater, and that at least one cubin is included in the binary file with architecture `sm_xx` to prevent run time errors.  The preferred approach is to not use `sm_*` naming in the `-arch` argument.  Using `-arch=compute_xx` only is generally sufficient. The newer naming conventions such as `compute_70` or `compute_86` or `-arch=compute_xx` will prevent error if a library or an application is compiled against an older compute architecture. This also ensures forward compatibility as the application will automatically target newer GPU architecture that meet or exceed the compute capability specified.

To demonstrate, consider a scenario where a developer is using a Pascal-based GPU (compute capability 6.0, represented by `compute_60`).

```cpp
// Example 1: Compilation targeting a modern architecture (Pascal)

// Assuming code.cu contains valid CUDA code
// Compile for Pascal architecture (compute_60)

//Command: nvcc -arch=compute_60 -o my_executable code.cu
```

In the above example, `nvcc` is instructed to compile the CUDA code, `code.cu`, to create `my_executable`. The `-arch=compute_60` flag ensures that the resultant binary is specifically optimized for the Pascal architecture. This executable is likely to perform optimally on Pascal GPUs, but is likely to encounter errors if executed on Fermi era GPUs.

```cpp
// Example 2: Compilation targeting an older architecture (Kepler), while compatible with Pascal

// Assuming code.cu contains valid CUDA code
// Compile for Kepler architecture (compute_35).  This will be usable by Pascal
//Command: nvcc -arch=compute_35 -o my_executable code.cu
```

Here, even though the developer might be using a Pascal GPU, code compiled with `-arch=compute_35` (Kepler) will still run on a Pascal GPU since Pascal’s compute capability is greater than Kepler’s.  However, performance is not likely to be optimal. When using a Pascal GPU, compiling for Kepler does not gain access to newer instructions or architecture optimizations present in the Pascal hardware. As such the user may opt to compile for the lowest supported compute capability. This is typically `compute_35` or `compute_37`.

```cpp
// Example 3: Compilation demonstrating use of both -arch and -code

// Assuming code.cu contains valid CUDA code
// Compile for Pascal architecture (compute_60), creating code for architecture sm_60

//Command: nvcc -arch=compute_60 -code=sm_60 -o my_executable code.cu

// Compiling for multiple architectures to provide a fat binary.
//Command: nvcc -arch=compute_35 -code=sm_35 -arch=compute_60 -code=sm_60 -o my_executable code.cu

```

In the final example, we see how to use both the `-arch` and `-code` options.   Using this option enables one to compile for multiple GPU architectures in a single binary.  In the later example, the single binary is capable of running on both Kepler and Pascal architecture GPUs. This binary includes multiple cubin files, targeting different `sm_*` versions, which will be chosen at run time. This ensures compatibility on the appropriate hardware.

My experience with cross-platform development has shown me that relying on a single hardcoded architecture can be a considerable limitation. Modern development practices often involve compiling for multiple architectures to support diverse user hardware. This approach allows applications to dynamically choose the best performing optimized kernel for the available GPU, thereby enhancing user experience. It also allows a single build to service users who have different generations of hardware.  This can be achieved through combining `-arch` flags as in the third example and using the `cubin` file with the correct `sm_*` for the current device.

To fully understand this aspect of `nvcc`, a thorough review of the NVIDIA CUDA documentation is recommended. The document dedicated to the CUDA compiler, `nvcc`, provides granular details on all architecture flags, including the corresponding compute capability for each architecture. The CUDA Toolkit documentation should also be consulted for a detailed explanation of compute capabilities, including the list of GPUs associated with each compute capability. Finally, the NVIDIA developer forums can be a very useful resource for finding solutions to problems and clarifying nuanced concepts.
