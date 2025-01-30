---
title: "How do I compile and install TensorFlow 2.8 on aarch64/arm64 from source, considering its dependency on TensorFlow-IO?"
date: "2025-01-30"
id: "how-do-i-compile-and-install-tensorflow-28"
---
TensorFlow 2.8, while not officially providing pre-built binaries for aarch64/arm64 architectures, can be compiled from source, a process significantly complicated by its dependency on TensorFlow-IO, which also lacks straightforward aarch64 pre-built support. This necessitates a comprehensive understanding of the build process and a proactive approach to dependency resolution. My experience across several embedded Linux projects utilizing custom hardware has shown that the key is methodical environment setup, careful configuration of the build flags, and meticulous dependency management, particularly with TensorFlow-IO.

The core challenge stems from the limited availability of pre-compiled wheel files for the target architecture. The typical `pip install tensorflow` approach simply will not work; a full source build is necessary. The TensorFlow build system, Bazel, expects a specific environment setup, especially regarding the cross-compilation toolchain. Furthermore, TensorFlow-IO adds complexity because it also relies on native code, which must also be compiled from its source repository.

Here is a procedure to tackle this:

1.  **Environment Preparation**: I begin by ensuring a clean build environment on an x86_64 machine. Cross-compiling requires a host that can reliably execute the build process. I usually opt for a recent Ubuntu distribution, preferably 20.04 or later, with sufficient memory and disk space. Then, I install the build essentials and dependencies like `git`, `wget`, `python3-dev`, `python3-pip`, `bazel`, `swig`, `cmake`, and `autoconf`. Python version 3.7 or 3.8 is essential as some libraries may have compatibility issues with later versions at the time of writing. Ensure that the specific python3 being used is also registered as the primary python3 for the user.

2.  **TensorFlow Source Acquisition**: After that, I clone the TensorFlow 2.8 source code from the official GitHub repository. It is advisable to check out the specific branch or tag corresponding to version 2.8 rather than relying on the main branch to prevent unexpected issues during the build. Similarly, I acquire the TensorFlow-IO source code. It is crucial to check the compatible version of TensorFlow-IO for TensorFlow 2.8, as different versions may not work correctly. I carefully select the version based on the TensorFlow compatibility matrix, which is often available on the TensorFlow-IO GitHub repository. It’s important to clone both repositories into distinct locations, as I prefer not to modify any source files.
3.  **Bazel Configuration**: Now, I configure Bazel, the build system for TensorFlow. Within the TensorFlow repository, the `.bazelrc` file and the `configure.py` script guide the configuration process. I create a modified `.bazelrc` that specifies the aarch64 cross-compilation toolchain. This involves specifying the compiler (typically `aarch64-linux-gnu-g++`) and linker paths. I define the `--config=opt` flag to compile optimized release builds. Additionally, I set environment variables like `TF_NEED_CUDA=0` to disable CUDA support since I am cross-compiling for an ARM architecture. This is where the bulk of customisation lies. I run `python configure.py` and answer its prompts, carefully avoiding activating features not compatible with cross compilation or not needed for the target architecture. The key prompt is to answer "no" for the "Do you wish to build TensorFlow with XLA JIT support?" prompt, as it’s very difficult to make that work well with cross compilation.

4.  **TensorFlow-IO Dependency Management**: This step is very important. I create a virtual environment using the python3 executable within the build environment for the tensorflow folder and install the necessary dependencies for tensorflow-io using pip. This ensures that the dependencies are managed within an isolated build scope, preventing conflicts with the host system’s Python packages. Also, I must manually install all dependencies listed in TensorFlow-IO's requirements.txt file. I have seen some versions require older versions of some libraries (e.g. numpy, protobuf), so manually installing the correct versions will save time and trouble later. I also create a virtual environment for the tensorflow-io directory and use it in subsequent steps.

5.  **TensorFlow-IO Build Configuration**: I also need to adjust the bazel configuration in the tensorflow-io project. This process mirrors that in the tensorflow project, though with potentially different versions of the used Bazel libraries. I run the configure script for tensorflow-io and select the appropriate options for building on an ARM system. This part usually involves specifying the target architecture and the location of the bazel build output.

6.  **Compilation**: Finally, I execute the Bazel build command. For TensorFlow, this is typically `bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package`. For TensorFlow-IO, it will be something like `bazel build -c opt --config=monolithic //tensorflow_io/python/ops:tensorflow_io.so`. This step will trigger a long compilation process that will generate the necessary shared object files for the target architecture.
7.  **Package Creation**: After the compilation finishes, I create the pip package for each project using `bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg`. After building the tensorflow_io shared object, I also create the python package by using `python3 setup.py bdist_wheel --dist-dir /tmp/tensorflowio_pkg` from the tensorflow-io folder. These two commands create .whl files for both Tensorflow and Tensorflow-IO respectively.
8. **Installation on the Target**: Lastly, I transfer the generated .whl files to the target aarch64 device and install them using pip, specifically ensuring that both TensorFlow and TensorFlow-IO are installed with no conflict.

Here are three code examples that illustrate key aspects of this process:

**Example 1: Custom `.bazelrc` File**

```bash
# aarch64_bazelrc

build --config=opt
build --cpu=arm64
build --copt=-march=armv8-a
build --copt=-mtune=cortex-a72 # or target architecture

# Specify cross-compiler
build --compiler=aarch64-linux-gnu-g++
build --host_compiler=g++

# Linker and paths
build --action_env PATH="/usr/bin:$PATH"
build --action_env LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"

#disable features
build --define=tf_api_version=2
build --define=tf_need_cuda=0
build --define=tf_need_gcp=0
build --define=tf_need_aws=0
build --define=tf_need_hadoop=0
```

*Commentary*: This file defines the custom build configuration for Bazel, specifying the target CPU architecture (`arm64`), the compiler (`aarch64-linux-gnu-g++`), and additional compiler options for optimization. It also sets the paths for executables and libraries essential for the cross-compilation process, and explicitly disables features unnecessary for the ARM build.

**Example 2: Invocation of the Bazel Build Command for TensorFlow-IO**

```bash
# Build Tensorflow-IO shared object
bazel build -c opt --config=monolithic //tensorflow_io/python/ops:tensorflow_io.so
```

*Commentary*: This command instructs Bazel to compile only the specified target, in this case, the `tensorflow_io.so` shared object file. The `-c opt` flag specifies an optimized build, and `--config=monolithic` indicates that all dependencies should be statically linked. It is vital to select monolithic rather than dynamic linking for easier deployment on the target system.

**Example 3: Creating the Pip Package**

```bash
# Create the tensorflow wheel file
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
# Create the tensorflow-io wheel file
python3 setup.py bdist_wheel --dist-dir /tmp/tensorflowio_pkg
```

*Commentary*: These two commands trigger the creation of a Python package archive (wheel file) from the compiled artifacts. The first command uses the Bazel build output to generate the tensorflow package, while the second command uses the setup.py file within tensorflow-io to create the corresponding package. The wheel files are stored in the /tmp directories ready to be installed on the target device.

**Resource Recommendations**

For a more thorough understanding, I would recommend consulting the TensorFlow documentation on building from source. The TensorFlow repository also has numerous examples and scripts. Furthermore, the Bazel documentation provides in-depth insights into its build system configuration and usage. I also refer to various online forums and community discussions for troubleshooting complex issues. These discussions often contain helpful hints and potential workarounds for less obvious build-related problems. Finally, the Tensorflow-IO repository also contains a wealth of information that can be useful, in particular the `BUILD` and `WORKSPACE` files.

In conclusion, while challenging, compiling TensorFlow 2.8 for aarch64/arm64, while factoring in TensorFlow-IO, is achievable by carefully managing the build environment, understanding the dependencies, and using Bazel correctly. My experience has shown that meticulous attention to detail and proper configuration are paramount for a successful outcome. This ensures that the build is not only functional, but also optimized for the target hardware.
