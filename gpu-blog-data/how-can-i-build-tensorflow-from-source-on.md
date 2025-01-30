---
title: "How can I build TensorFlow from source on macOS High Sierra?"
date: "2025-01-30"
id: "how-can-i-build-tensorflow-from-source-on"
---
Compiling TensorFlow from source on macOS High Sierra, especially with the nuanced challenges of older toolchains and specific hardware configurations, necessitates a meticulous approach. My experience, having wrestled with this process across several machines in a research lab setting, reveals a crucial starting point: the compatibility matrix between TensorFlow versions, Python versions, and Xcode. A mismatch here often results in obscure build errors, requiring significant troubleshooting time. Specifically, macOS High Sierra's age limits the Python and Xcode versions directly usable; a more current setup is not an option in this context.

The core challenge lies in ensuring a compatible environment before even initiating the TensorFlow build. I've observed that a clean build environment is paramount. This involves a dedicated Python virtual environment and careful version selection for bazel, TensorFlow's build tool. High Sierra doesn’t support the latest Bazel versions; a version close to the time of TensorFlow’s release is critical. The build process can be broken down into three main stages: dependency preparation, bazel configuration, and the actual build.

First, the virtual environment and Python version. Given High Sierra's constraints, Python 3.7, which may not be the latest now but offers good compatibility across TensorFlow versions from that era, typically works well. Create this using the following steps:
```bash
python3.7 -m venv tf_venv
source tf_venv/bin/activate
pip install -U pip
```
This ensures a clean sandbox to manage packages, avoiding conflicts with the system Python. The `-U pip` upgrades `pip` to the latest version within the virtual environment, a standard first step for any Python project. Next, TensorFlow’s specific requirements need to be met. I have frequently seen `numpy` and `wheel` dependencies cause issues. These are handled via:
```bash
pip install numpy==1.16.4
pip install wheel
```
The `numpy` version is critically important. TensorFlow expects a compatible ABI, and this specific version is often the sweet spot for older TensorFlow versions, based on my experiences.

Second, bazel configuration. I recall initially overlooking the significance of the correct bazel version and how it impacted build success or failure. Bazel is responsible for managing the complex dependencies inherent to TensorFlow. Download bazel using its official source and extract the binary. For older TensorFlow versions, I've found bazel version 0.26.1 to be stable. Place the bazel binary in a directory included in your system's `PATH`, for example `/usr/local/bin`. Ensure its executability with `chmod +x /usr/local/bin/bazel`.

Third, the TensorFlow build itself. We need the TensorFlow source code. Download the TensorFlow source archive from the official repository on GitHub. Once unzipped, navigate to the TensorFlow directory within the terminal. I typically create a configuration using the interactive script, but for clarity we will bypass the script here for this example:

```bash
./configure
```
This command launches the configuration wizard. When prompted, I have always found it optimal to answer the following questions:
* Python location: The path to your virtual environment’s Python executable, such as `tf_venv/bin/python`
* Bazel location: `/usr/local/bin/bazel`
* Select the default for all other options, including whether or not to build with XLA and other accelerated ops.

After answering the prompts, we can trigger the build process using bazel. This process is time consuming and may require significant system resources. Here I am going to build only the pip package as I found it to be less prone to error. This is accomplished using:
```bash
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
```
Here, `--config=opt` builds an optimized version of TensorFlow. The target is specified with `//tensorflow/tools/pip_package:build_pip_package`. This bazel command will compile the core TensorFlow library and all its required dependencies into a distributable pip package. The build process outputs the pip package in the bazel-bin folder. After the build, install the generated `.whl` file using pip. This is done using the following command, adjusting the path to the wheel file:
```bash
pip install bazel-bin/tensorflow/tools/pip_package/build_pip_package.whl
```
Post installation, a simple python check is appropriate to ensure functionality:
```python
import tensorflow as tf
print(tf.__version__)
```
This will output the version if the build was successful.

The code examples above reflect the core process. However, I encountered several issues while working through similar configurations on High Sierra. These issues were often due to underlying C++ library dependencies, Xcode configuration, and subtle Bazel configuration differences between TensorFlow versions. These are less frequent with the specific versions I mention here but may be encountered with slightly different selections of Python or bazel.

Beyond the basic steps, several resources proved crucial in my experiences. The official TensorFlow documentation for building from source, available from the TensorFlow website, although not tailored to specific system versions, does contain general principles. Moreover, the Bazel documentation offers insights into the build process. Thirdly, I've found the various Tensorflow community forums to be a useful resource for identifying specific compatibility issues and solutions, though relying entirely on community solutions is not advised as these may be out of date or not fully vetted.

In summary, building TensorFlow from source on macOS High Sierra is attainable with a systematic approach focusing on compatibility, version management, and careful execution of build instructions. By meticulously preparing the environment, configuring bazel appropriately, and diligently following the build instructions, it is possible to generate a functional TensorFlow library. My experience emphasizes the importance of clean build environments, compatible toolchains, and reliable version control, all crucial for navigating the intricacies of compiling complex scientific libraries on older systems. This methodology has, time and time again, allowed me to bring TensorFlow to life on legacy hardware.
