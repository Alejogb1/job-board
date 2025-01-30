---
title: "How can I build TensorFlow on Ubuntu 16.04?"
date: "2025-01-30"
id: "how-can-i-build-tensorflow-on-ubuntu-1604"
---
Successfully building TensorFlow on Ubuntu 16.04, while achievable, requires careful attention to dependency management and compilation flags due to the age of the operating system and its associated default software versions. I’ve encountered this scenario frequently, especially when working with legacy systems that haven't undergone regular upgrades. The core issue lies in the fact that TensorFlow relies on newer versions of compilers and libraries than what ships natively with 16.04. The following details the steps I've found most reliable for a successful build.

**Explanation of the Build Process**

The primary challenge when building TensorFlow from source on older systems is resolving the inevitable dependency conflicts. Ubuntu 16.04 includes versions of tools like `gcc`, `g++`, and Python that are often too old to seamlessly integrate with current TensorFlow releases. Further complicating matters is the need for a specific version of Bazel, TensorFlow's build system, as newer Bazel versions can exhibit compatibility issues. The build process, broadly, involves: setting up the correct development environment, cloning the TensorFlow repository, configuring the build through Bazel, and finally, building and installing the necessary binaries. This is not a simple `apt install` process. It's a hands-on endeavor requiring specific commands and configurations.

First, installing required dependencies is paramount. This includes Python 3, which is critical, as TensorFlow is generally optimized for Python 3 environments.  While Ubuntu 16.04 often includes Python 2.7 by default, this will not work. You will also need `pip3`, Python's package manager, and `virtualenv`, a utility for creating isolated Python environments. This isolation prevents conflicts between your system-wide Python setup and the specific requirements of TensorFlow.  Further crucial dependencies include header files for BLAS and LAPACK, which are used for optimized mathematical computations in the background.  Building on a system without these packages will result in link-time errors.

Next, Bazel needs to be installed. It is the tool that executes the compilation instructions to transform TensorFlow's source code into usable libraries and executables. It is essential to use a Bazel version known to work reliably with your chosen TensorFlow release. I have found inconsistencies when mixing versions. You should also confirm the correct version of g++ is being used, a particular sticking point, as newer versions usually lead to better compatibility. Then, after cloning the TensorFlow repository from GitHub, you'll configure the build using the configuration script `configure.py` (or an equivalent), which will allow specification of your desired build options, CPU/GPU support, optimization levels and other pertinent settings. The build is initiated by Bazel, which will then proceed to compile the entire TensorFlow code base. This step can take a significant amount of time, depending on your computer's CPU and memory.

Finally, after the build, the compiled libraries and executables need to be installed. This usually involves moving the generated files to a location where they can be found by Python (or other applications needing TensorFlow).

**Code Examples with Commentary**

The following code snippets illustrate core steps in the process. These are not complete scripts; they're representative commands demonstrating essential elements.

**Example 1: Environment Setup**

```bash
sudo apt update
sudo apt install -y python3 python3-dev python3-pip python3-virtualenv
sudo apt install -y software-properties-common
sudo apt update
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt update
sudo apt install -y g++-7  # Or appropriate g++ version for TF
sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-7 100

# Ensure g++-7 is default (use caution and test with another method)
# sudo update-alternatives --config g++ # Select correct g++ by number from list

sudo apt install -y libblas-dev liblapack-dev
```

*Commentary:* This code section focuses on setting up the basic Python environment, adding a newer g++ version, and installing required BLAS/LAPACK libraries.  The `update-alternatives` command is shown to ensure the `g++` executable points to the desired version. The PPA used here may have newer package versions. Careful attention to package choices is needed, to avoid destabilizing your system. It is not always advisable to use `-y`, but is included for illustration purposes.  The last line installs header files, required for linking during the build process.

**Example 2: Bazel Installation and TensorFlow Configuration**

```bash
wget https://github.com/bazelbuild/bazel/releases/download/0.26.1/bazel-0.26.1-installer-linux-x86_64.sh
chmod +x bazel-0.26.1-installer-linux-x86_64.sh
./bazel-0.26.1-installer-linux-x86_64.sh --user

git clone https://github.com/tensorflow/tensorflow.git
cd tensorflow

# Important: switch to compatible version, check with TensorFlow documentation for compatible Bazel versions
git checkout v2.1.0  # Or desired version
./configure.py
# Answer the prompted questions carefully, ensuring that Python is correctly located, and CPU/GPU acceleration is correctly configured.
```

*Commentary:*  Here, Bazel is installed from a specific version; version `0.26.1` is used in this example and is known to work with certain TF releases. Note that the specific Bazel version may vary based on the TensorFlow version you choose. Always consult the TensorFlow documentation for compatible versions. The `git clone` command downloads the TensorFlow repository and `git checkout` switch to a particular TensorFlow version.  The `./configure.py` script runs, prompting for various build settings; you will need to specify the path to your Python 3 binary, as well as indicating whether CUDA will be used for GPU acceleration, if available. Be very exact during this configuration step.

**Example 3: Building and Installing TensorFlow**

```bash
bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package

bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

pip3 install /tmp/tensorflow_pkg/tensorflow-*.whl

# Verify installation
python3 -c "import tensorflow as tf; print(tf.__version__)"
```

*Commentary:* This snippet illustrates the core build and installation phase.  `bazel build` is used with `config=opt` flag to include optimizations. The `//tensorflow/tools/pip_package:build_pip_package` argument specifies the target for Bazel.  This action compiles TensorFlow and creates a Python wheel package that is subsequently installed via `pip3 install`. The final command verifies successful installation using a simple Python program that prints the version of TensorFlow.

**Resource Recommendations**

When tackling this process, several resource types are invaluable. I’d strongly recommend the following:

*   **TensorFlow Official Documentation:** The primary resource for specific build instructions, especially concerning compatible versions of Bazel and other dependencies. The official documentation often contains a detailed troubleshooting section. Refer to the section for building from source.
*   **TensorFlow GitHub Repository Issues:** The issue tracker on GitHub can often reveal problems others have encountered and their solutions, particularly for specific configurations. It can be useful to search for issues pertaining to older Ubuntu versions or specific error codes.
*   **Stack Overflow:** A community-driven resource that provides solutions to common problems. Often, similar issues have been addressed, and a detailed search may uncover existing solutions. I have often found solutions to very obscure problems here.
*   **Ubuntu Server Forums:** Server-specific forums can offer insights into system-level issues which might interfere with the build process. These forums might contain information on system configuration and how to avoid common pitfalls.
*   **Developer Blogs:** Individual developers sometimes publish detailed walkthroughs on building TensorFlow on specific systems, which can be useful for getting a different perspective.

In conclusion, building TensorFlow from source on Ubuntu 16.04 is a demanding task that requires meticulous attention to detail.  The most critical steps are ensuring compatibility between TensorFlow, Bazel, compiler, and dependent libraries. While the process is lengthy and may require debugging, following the outlined steps and using the recommended resources can lead to a functional and optimized TensorFlow installation.  Careful planning and execution is key to success.
