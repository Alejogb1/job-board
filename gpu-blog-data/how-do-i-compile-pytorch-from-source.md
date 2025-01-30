---
title: "How do I compile PyTorch from source?"
date: "2025-01-30"
id: "how-do-i-compile-pytorch-from-source"
---
Compiling PyTorch from source offers granular control over the build process, enabling optimizations and customizations not achievable with pre-built binaries. I've found this invaluable for situations requiring specific hardware support, custom CUDA versions, or when contributing directly to the framework. The process, while initially daunting, breaks down into manageable steps, primarily involving dependency management, build configuration, and compilation itself.

First, let’s address dependencies. PyTorch relies on several core libraries, including NumPy, CMake, and a suitable C++ compiler (GCC or Clang are common). Furthermore, if GPU support is desired, a compatible CUDA toolkit and cuDNN library are necessary. These components must be installed and configured correctly before attempting a build. I recommend using a package manager like Conda or pip for installing Python dependencies, but system packages for core C++ tools. This creates a managed, reproducible environment. Ensure that the versions of dependencies align with the documentation in the PyTorch repository's README. Failure to do so is a primary source of compilation errors.

The second crucial element is the build configuration. PyTorch utilizes CMake to manage the build process. This involves specifying options to determine which features are included in the final binary. Common options include `USE_CUDA`, `USE_MKLDNN`, and `USE_DISTRIBUTED`, among others. These parameters enable or disable aspects of PyTorch that relate to, in the same order: NVIDIA GPU support, Intel Math Kernel Library for Deep Neural Networks, and distributed training functionalities. Specifying the right configuration parameters greatly impacts the size and performance of the resulting PyTorch build. I typically use a dedicated build directory, separate from the source tree, to maintain a clean environment and to allow multiple configurations to be tested independently.

Finally, the actual compilation process translates the source code into executable libraries. This process involves multiple steps, typically executed through a command-line interface. Depending on the system, the time required for compilation can range from several minutes to multiple hours. The build process generates a comprehensive suite of files, including the core PyTorch library, extensions, and testing utilities.

Here are some typical build scenarios and their respective command sequences:

**Scenario 1: CPU-Only Build**

This scenario is suitable for development environments or systems without GPU acceleration. In my work on embedded systems, I used this approach frequently. This scenario is the least complicated to set up since it doesn’t require additional NVIDIA components.

```bash
# Assuming a separate build directory called 'build'
mkdir build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DUSE_CUDA=OFF  \
  -DUSE_MKLDNN=ON # Optional: MKLDNN optimizations for CPU
make -j $(nproc)
make install
```

*Commentary:*
The `cmake` command configures the build. `-DCMAKE_BUILD_TYPE=Release` enables optimizations for production performance. `-DPYTHON_EXECUTABLE=$(which python3)` explicitly tells CMake which Python interpreter to use. `-DUSE_CUDA=OFF` disables CUDA compilation, making this a CPU-only build. `USE_MKLDNN=ON` enables Intel's optimized library for operations on CPU which can greatly impact performance for some systems. The `make -j $(nproc)` command starts the actual compilation using multiple processor cores, speeding up the build time. `make install` installs the built files to the specified Python environment.

**Scenario 2: CUDA-Enabled Build**

For projects involving large neural networks, GPU acceleration is vital. A correctly configured CUDA and cuDNN installation is a prerequisite for this approach. I have personally used this setup for model training.

```bash
# Assuming a separate build directory called 'build'
mkdir build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DUSE_CUDA=ON \
  -DCUDNN_INCLUDE_DIR=/path/to/cudnn/include \
  -DCUDNN_LIBRARY=/path/to/cudnn/lib64/libcudnn.so \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
make -j $(nproc)
make install
```

*Commentary:*
The key difference from the CPU-only build is the `-DUSE_CUDA=ON` option which enables CUDA support. `-DCUDNN_INCLUDE_DIR` and `-DCUDNN_LIBRARY` specify the location of the cuDNN library components. `-DCUDA_TOOLKIT_ROOT_DIR` indicates the location of the CUDA toolkit, typically located at `/usr/local/cuda`. These values must be accurate for a successful build. Specifying the wrong CUDA version here will cause an error during compilation. The rest of the commands are similar to the CPU build.

**Scenario 3: Custom CUDA Architecture Build**

Sometimes, when targeting specific GPUs or when using NVIDIA's newer features, compiling for specific CUDA architectures is required. This offers more control over the code generated. For this scenario, I have worked on projects optimized for NVIDIA's Turing and Ampere GPUs.

```bash
# Assuming a separate build directory called 'build'
mkdir build
cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DUSE_CUDA=ON \
  -DCUDNN_INCLUDE_DIR=/path/to/cudnn/include \
  -DCUDNN_LIBRARY=/path/to/cudnn/lib64/libcudnn.so \
  -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
  -DCUDA_ARCH_LIST="86;90" # Example for Ampere and newer
make -j $(nproc)
make install
```

*Commentary:*
This build adds `-DCUDA_ARCH_LIST="86;90"`. This option defines the compute capabilities for which the GPU kernel code will be compiled for. In this case, `86` corresponds to the Ampere architecture, and `90` to the newer Ada Lovelace architecture.  By specifying only architectures in the system, the generated binary is smaller. It can reduce the time needed to load GPU kernels at runtime. The list of architecture codes can be found in NVIDIA documentation.

Regarding resource recommendations, I advise consulting the official PyTorch documentation for installation instructions. The PyTorch GitHub repository contains detailed build information, including a list of build options and their impact. Additionally, NVIDIA provides comprehensive resources for setting up the CUDA toolkit and cuDNN. Furthermore, the CMake website offers a wealth of knowledge about build systems in general. These sources offer both theoretical explanations and practical advice. The PyTorch community forums and online forums are often useful when one encounters issues specific to hardware or dependency versions.

Successfully compiling PyTorch from source requires patience and attention to detail. The build process can be quite intricate, and encountering errors is not uncommon. Careful dependency management and correct build configuration greatly reduce the chance of encountering problems. Starting with a simple CPU build as a first step allows for a familiarization with the process before moving to more advanced GPU configurations. Through my experience, I've found that documenting build processes is beneficial for repeatability and troubleshooting. A record of which settings were used and with which version of dependencies can expedite fixes during subsequent attempts.
