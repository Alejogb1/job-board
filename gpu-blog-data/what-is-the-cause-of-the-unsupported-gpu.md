---
title: "What is the cause of the 'Unsupported gpu architecture 'compute_86'' nvcc error?"
date: "2025-01-30"
id: "what-is-the-cause-of-the-unsupported-gpu"
---
The "Unsupported gpu architecture 'compute_86'" error encountered during nvcc compilation stems fundamentally from a mismatch between the target GPU architecture specified in your compilation command and the actual capabilities of your hardware.  This arises because NVIDIA's CUDA architecture is versioned, with each version ('compute_XX') representing a specific generation of GPU hardware and its associated instruction set.  Over my years working on high-performance computing projects, I've seen this error repeatedly, most often due to incorrect specification or a misunderstanding of the available compute capabilities.

**1. Clear Explanation:**

The `compute_XX` specification within your nvcc compilation flags dictates the lowest CUDA compute capability that the generated code should support.  For instance, `-gencode arch=compute_86,code=sm_86` indicates that the compiled code must run on GPUs with at least compute capability 8.6 or higher.  If your target GPU (or the GPU you intend to run the code on) possesses a lower compute capability, nvcc will fail with the "Unsupported gpu architecture" error.  The error specifically points to `compute_86`, implying your compilation command targets a GPU architecture that is unavailable on your system.  This isn't necessarily a problem with your code itself; rather, it reflects a configuration issue.

Several scenarios contribute to this error:

* **Incorrect `-gencode` specification:**  The most common cause is incorrectly specifying the compute capability in the `-gencode` flag. This often involves using a compute capability that is higher than the capabilities of the GPU actually present in your system.  This can occur due to a typo, outdated documentation, or confusion regarding the architecture of available hardware.

* **Out-of-date NVIDIA driver:** An outdated NVIDIA driver might not correctly report the GPU's compute capability to nvcc, leading to an incorrect assumption about supported architectures.  This is less frequent but still a plausible cause.

* **Incorrect CUDA Toolkit version:** Although less likely to directly cause this specific error, an incompatibility between the CUDA Toolkit version and the NVIDIA driver can indirectly lead to misreporting of capabilities.  A mismatched installation might not correctly recognize the hardware.

* **Virtual Machine configurations:**  In virtual machine environments, the host system might not properly expose or correctly identify the GPU resources to the virtual machine's guest operating system, resulting in a mismatch between the system's reported capabilities and the actual hardware capabilities accessible to the guest.

**2. Code Examples with Commentary:**

**Example 1: Incorrect Architecture Specification:**

```c++
// Incorrect compilation command, assumes compute capability 8.6
nvcc -gencode arch=compute_86,code=sm_86 myKernel.cu -o myKernel
```

This command will fail if the system's GPU only supports compute capability 7.5 or lower.  To resolve this, one must determine the correct compute capability of their target GPU through the `nvidia-smi` command or NVIDIA's documentation and adjust the `-gencode` flag accordingly. For instance, if the GPU supports compute capability 7.5, the correct command would be:

```c++
// Corrected compilation command for compute capability 7.5
nvcc -gencode arch=compute_75,code=sm_75 myKernel.cu -o myKernel
```

**Example 2:  Compilation for Multiple Architectures:**

If you need to support multiple GPU architectures, it is crucial to specify them all during compilation using the `-gencode` flag multiple times.  Suppose you wish to support both compute capability 7.5 and 8.6:

```c++
// Compilation for compute capability 7.5 and 8.6
nvcc -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 myKernel.cu -o myKernel
```

This generates two different versions of the kernel, one for each specified architecture.  At runtime, CUDA will select the appropriate version based on the capabilities of the GPU being used.  This approach enhances code compatibility across a range of hardware.  Note that compilation time will be longer.

**Example 3:  Using `-arch` (Deprecated but potentially relevant):**

While generally discouraged in favour of `-gencode`, the older `-arch` flag might be encountered in legacy code.  `-arch=sm_XX` directly specifies the target architecture.  However, `-gencode` is preferred as it allows for more fine-grained control and compatibility with newer CUDA toolkits.  For instance, the following would compile for compute capability 7.0:

```c++
// Using the deprecated -arch flag (avoid if possible)
nvcc -arch=sm_70 myKernel.cu -o myKernel
```

This command is functionally similar to `-gencode arch=compute_70,code=sm_70` but lacks the explicit support specification provided by the newer flag.  It's recommended to migrate to `-gencode` for improved clarity and compatibility.


**3. Resource Recommendations:**

I would recommend consulting the official NVIDIA CUDA documentation for the most up-to-date and accurate information on compute capabilities, compilation flags, and best practices.  Carefully review the system's specifications to verify the GPU's compute capability.  The NVIDIA website and the CUDA Toolkit documentation offer comprehensive guides on these topics, which are essential for successful CUDA development.  Furthermore, studying tutorials and examples focusing specifically on `-gencode` usage will prove beneficial.  Finally, pay close attention to any error messages generated during compilation; they often contain valuable hints pointing directly to the underlying cause of the issue.  The `nvidia-smi` command-line tool is invaluable for inspecting your system's GPU and its associated properties.
