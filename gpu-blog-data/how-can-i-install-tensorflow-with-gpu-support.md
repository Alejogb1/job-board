---
title: "How can I install TensorFlow with GPU support on macOS El Capitan?"
date: "2025-01-30"
id: "how-can-i-install-tensorflow-with-gpu-support"
---
MacOS El Capitan, released in 2015, presents a specific challenge when installing TensorFlow with GPU support. Its compatibility with later CUDA versions, which are typically required for NVIDIA GPU acceleration, is limited. The latest compatible CUDA Toolkit is 8.0, a prerequisite for using TensorFlow's GPU capabilities on El Capitan. Direct installation of current TensorFlow versions via pip will not enable GPU functionality due to dependency incompatibilities. Therefore, a specific approach, leveraging older TensorFlow builds compiled with CUDA 8.0, is required.

The primary issue lies in the operating system's age. Modern TensorFlow binaries are compiled against newer CUDA toolkits, typically 10.0 or higher. El Capitan lacks support for these later CUDA versions, creating a fundamental mismatch. The NVIDIA driver support and CUDA architecture requirements also significantly complicate matters. Newer GPU architectures (Pascal and later) might still function in a fallback mode, but full optimization and performance require more up-to-date libraries.

To address this, I’ve had to compile older TensorFlow versions from source, using specific CUDA and cuDNN versions compatible with El Capitan. While this is complex, it’s the most reliable method for enabling GPU support. Utilizing readily available pip packages for current TensorFlow releases will, at best, give you CPU-based TensorFlow computations on El Capitan. At worst, it will result in installation errors or segmentation faults when trying to utilize the GPU.

The process begins with ensuring a suitable environment for the build. El Capitan relies on a specific version of XCode’s command-line tools and a specific Python version. I typically advise users to leverage virtual environments for this.

The crucial steps involve:

1. **Installing CUDA 8.0 and cuDNN 5.1:** Download these directly from NVIDIA’s developer website (requires an account). Note that these versions are not readily available. This process is manual, involves unpacking the distribution files, and properly setting environment variables.

2. **Installing Bazel:** TensorFlow uses Bazel as its build system. An older Bazel version (e.g., 0.5.4) compatible with TensorFlow 1.x is necessary, installable via a precompiled binary.

3. **Cloning the Correct TensorFlow Branch:** Instead of utilizing the master branch, checkout a specific version, such as v1.10.0. This particular version will be tested and have the proper CUDA support enabled.

4. **Configuring the Build:** This involves executing the `configure` script within the TensorFlow source directory and providing the proper paths for CUDA and cuDNN. Choosing the correct GPU compute capability is critical during this step.

5. **Building TensorFlow:** This is the most resource-intensive process. The compilation can take several hours, depending on the machine. Build the pip package after the compilation succeeds.

6. **Installing the Compiled Package:** The generated whl file needs to be installed via pip, ensuring it takes precedence over any globally installed TensorFlow versions.

Here's a simplified illustration of the build process using code snippets within the command line.

**Code Example 1: Setting Environment Variables**

```bash
export CUDA_HOME="/usr/local/cuda-8.0"
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
```

*Commentary:* This bash code snippet demonstrates the crucial step of setting the environment variables needed for the compiler and the TensorFlow runtime. `CUDA_HOME` specifies the location of the CUDA 8.0 installation. `LD_LIBRARY_PATH` and `PATH` are amended to include the CUDA libraries and executables within the system's search path, respectively. Without these, the compiler and resulting TensorFlow library will be unable to locate necessary CUDA components.

**Code Example 2: Configuring TensorFlow Build**

```bash
./configure
# Follow on-screen instructions. Choose appropriate paths for CUDA and cuDNN.
# Select GPU support
# Select compute capability for GPU (e.g., 5.2 for Maxwell architecture)
```

*Commentary:* This script is run after cloning a compatible TensorFlow branch. The `configure` script interactsively prompts the user for build settings. Key aspects include correctly specifying the CUDA and cuDNN directory paths, enabling GPU support, and selecting the correct compute capability of the target GPU. The wrong compute capability will result in the library not being able to optimally utilize the GPU. This script must be executed *prior* to beginning the build itself.

**Code Example 3: Building and Installing TensorFlow**

```bash
bazel build --config=opt --config=cuda //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg
pip install /tmp/tensorflow_pkg/tensorflow*.whl
```
*Commentary:* These commands utilize Bazel to build the TensorFlow pip package. The first command initiates the compilation with optimizations and CUDA support enabled. The second command generates the actual wheel file into the `/tmp/tensorflow_pkg` directory. Lastly, pip installs the generated wheel, overriding existing installations if needed. The generated `.whl` is specifically built for your environment and will function correctly given all previous steps are followed exactly.

In my experience, successfully building TensorFlow on El Capitan with GPU support requires careful attention to the specific versions of dependencies, particularly CUDA, cuDNN and the correct branch of TensorFlow. The system’s age necessitates avoiding pip directly in favor of building from source. The build process is time-consuming, and any minor discrepancy in versions can lead to build errors or failure to utilize the GPU during TensorFlow computations. Once complete, the resulting package, built from source, will run seamlessly within El Capitan with full GPU acceleration.

Key resources for this process can be found on NVIDIA's developer website for the CUDA toolkit and cuDNN downloads. Older versions of Bazel are available from the Bazel GitHub releases. The TensorFlow GitHub repository also archives older versions and provides necessary documentation for building from source. Furthermore, a review of the TensorFlow’s official website documentation is invaluable in understanding build dependencies.
