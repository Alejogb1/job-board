---
title: "How to install TensorFlow with TensorRT from source using Bazel?"
date: "2025-01-30"
id: "how-to-install-tensorflow-with-tensorrt-from-source"
---
TensorFlow integration with TensorRT, when compiled from source using Bazel, requires meticulous management of build flags, dependency versions, and CUDA configurations. My experience over several projects reveals this process, while powerful, often presents unique challenges not encountered with simpler pip installations. The core issue stems from the need to precisely align the TensorRT libraries with the TensorFlow build, avoiding runtime incompatibilities. This alignment includes both the core TensorRT engine and its associated plugins.

The first critical step involves obtaining the correct TensorRT version. The version of TensorRT you use *must* be compatible with the specific TensorFlow version being built. I’ve learned the hard way that mismatches lead to obscure error messages related to shared library loading. These can be tricky to diagnose and frequently involve recompiling after carefully reviewing dependency matrices. For instance, TensorFlow 2.10 required a TensorRT version within a specific range (usually detailed in TensorFlow's documentation). TensorRT's installation involves its own procedure, often requiring separate downloads and configuration, usually involving unpacking to a user-defined path. It's crucial to note this path because we’ll need to tell Bazel where to find the necessary libraries.

Next, one must configure the Bazel build files. Specifically, `tensorflow/workspace.bzl` and `tensorflow/tools/bazel.rc` are modified to point to the installed TensorRT libraries. It is not enough to simply have these libraries available on the system's PATH; explicit linking is needed within the Bazel environment. This is done by setting environment variables and using Bazel flags to correctly resolve the library locations. Bazel leverages the `.bazelrc` file (often `tools/bazel.rc`) to pass command-line options to the Bazel build process. Therefore, specifying TensorRT-related paths and include directories within this file is essential for the build system to properly link against it. These settings often vary based on the TensorRT installation specifics, necessitating careful adaptation rather than blindly copy-pasting from generic examples.

The actual build command will also include flags which enable TensorRT and specifies the compute capability of the target GPU. Neglecting to properly specify these flags during the `bazel build` command will result in a build without TensorRT support or one that won’t properly utilize the target GPU. The most significant flag, beyond general CUDA flags, is `—define=tensorrt_root=/path/to/tensorrt`, which points Bazel to the root directory of the TensorRT installation. The compute capability needs to also be specified based on the specifics of the target GPU, a setting directly tied to the performance of TensorRT models. Another common area for errors involves mismatched CUDA toolkit versions, so consistent versions are a must across TensorRT, TensorFlow, and any other related components.

Here’s an example of the modified `tensorflow/workspace.bzl` file, focusing on a hypothetical TensorRT installation path. This code segment assumes TensorRT was installed under `/opt/tensorrt`:

```python
def workspace():
    # ... existing workspace configurations ...

    native.bind(
        name = "cublas",
        actual = "@local_cuda//:cublas",
    )
    native.bind(
        name = "cudart",
        actual = "@local_cuda//:cudart",
    )
    native.bind(
      name = "cuda",
      actual = "@local_cuda//:cuda",
    )
    native.bind(
        name = "cudnn",
        actual = "@local_cuda//:cudnn",
    )

    native.new_local_repository(
      name = "local_tensorrt",
      path = "/opt/tensorrt",
      build_file_content = """
package(default_visibility = ["//visibility:public"])

exports_files(glob(["lib64/*.so*","include/**/*.h"]))
""",
    )

    # ... rest of workspace configurations ...
```

This code adds a new local repository, named `local_tensorrt`, that points to the specified TensorRT installation path. The `build_file_content` creates a dummy `BUILD` file within the repository, making all its library files (.so) and include headers (.h) available to Bazel. This structure facilitates proper linking during the TensorFlow build process. I have found that directly relying on system-wide includes can sometimes fail, making this local repository approach much more robust.

The next example is a modified snippet from `tensorflow/tools/bazel.rc`, showcasing the crucial build flags. This part of the config file adds the correct TensorRT library locations, include paths and compute capability settings when building TensorFlow using Bazel:

```
# TensorRT and CUDA configuration

build --action_env PYTHON_BIN_PATH="/usr/bin/python3" # Or location of python 3
build --action_env TF_CUDA_COMPUTE_CAPABILITIES="7.5" # Specific Compute Capability of GPU
build --define=tf_cuda_build=true
build --define=tensorrt_root=/opt/tensorrt
build --copt=-I/opt/tensorrt/include
build --copt=-DGOOGLE_CUDA=1
build --linkopt=-L/opt/tensorrt/lib64

# ... other bazel configurations ...
```

Here, `--define=tf_cuda_build=true` activates the CUDA related components during the compilation. The compute capabilities (`TF_CUDA_COMPUTE_CAPABILITIES`) should be adjusted based on your target GPU architecture (check the CUDA documentation). The `-I` flag adds the TensorRT include directory, and `-L` tells the linker where to search for TensorRT libraries. The `python` binary path is also necessary because some build stages need python. Failing to set the compute capability correctly can result in runtime errors or performance degradation since the binaries would not be optimized for the architecture.

Finally, the command to initiate the build, demonstrating how these settings are used, should look similar to this:

```bash
bazel build //tensorflow/tools/pip_package:build_pip_package --config=opt --config=cuda
```

The `build_pip_package` target packages the TensorFlow build into a pip installable wheel. The `--config=opt` flag enables optimization and `--config=cuda` applies the CUDA configurations defined in `tools/bazel.rc`, including those pertaining to TensorRT. This command, when properly configured, will compile a TensorFlow package linked with TensorRT, using all the defined flags from the files above. It will take considerable time to complete. Successfully executing this command is an indication that your TensorRT configuration is correct.

In summary, successfully building TensorFlow with TensorRT from source using Bazel requires a methodical approach. Precise version control, careful configuration of Bazel build files, and proper flag specification are critical steps. Mismatches in version numbers or improper library paths usually lead to build failures or unpredictable runtime behavior. Resources such as the official TensorFlow documentation, CUDA documentation, and TensorRT installation guides provide specific details about compatible versions and installation procedures. Community forums also offer a wealth of specific use case experiences, which can help diagnose particular issues with individual setups. Further investigation into GPU architecture-specific options and build optimizations will be required if targeting maximum performance for specific hardware configurations.
