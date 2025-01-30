---
title: "How do I install PyTorch on Jetson Nano with Ubuntu 18?"
date: "2025-01-30"
id: "how-do-i-install-pytorch-on-jetson-nano"
---
The successful deployment of deep learning models on edge devices like the NVIDIA Jetson Nano hinges on correctly configuring PyTorch, a task often complicated by hardware and software dependencies. Specifically, installing a suitable version of PyTorch on a Jetson Nano running Ubuntu 18.04 requires navigating pre-compiled binaries or, alternatively, undertaking a source build, each with its distinct challenges and trade-offs. My experience has shown that a meticulous, step-by-step approach, considering the specific CUDA architecture and resource constraints of the Nano, is paramount.

First, acknowledging the limitations, the Jetson Nano utilizes a Maxwell GPU with compute capability 5.3. Official PyTorch wheels from the project typically do not offer direct support for this architecture due to its age and lower compute capacity compared to newer NVIDIA hardware. Consequently, a naive `pip install torch` command often results in an unsuitable or non-functional installation, frequently returning errors related to missing CUDA libraries or incompatible binary formats. The most reliable route involves either using pre-built JetPack packages from NVIDIA or building PyTorch from source, each involving specific procedures and requirements.

The JetPack SDK, NVIDIA's platform software development kit for Jetson devices, is the quickest way to get a working PyTorch setup. When installed alongside the operating system, JetPack provides a compatible PyTorch distribution, including CUDA, cuDNN, and other necessary libraries. Although NVIDIA may not offer the most recent PyTorch version through JetPack, this approach guarantees stability and compatibility. The major benefit is ease of use, but it sacrifices control over the installed PyTorch version, often limiting the user to an older release, which might not support newer features.

Alternatively, building from source gives the most control, allowing for the installation of the latest PyTorch version, customized with specific build flags to optimize for the Nano. This process, however, demands careful configuration and considerable build time on the device itself, as cross-compiling is generally not straightforward. Resource limitations on the Nano, specifically memory and processing power, frequently necessitate using swap space and strategically managing build parameters. It's crucial to have sufficient storage space available on an external device.

Assuming a JetPack installation has not been performed, building from source is often the most flexible way to obtain the exact PyTorch version needed. Let’s begin with three crucial steps that I've found necessary to guarantee a functional compilation:

**Example 1: Setting up the Build Environment**

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-dev python3-pip cmake libopenblas-dev liblapack-dev gfortran
sudo pip3 install -U pip
sudo pip3 install numpy setuptools wheel
```

This code block establishes the fundamental prerequisites for source compilation. We start by updating and upgrading the system’s package manager, ensuring the most recent library versions are available. We install essential development tools, including `python3-dev`, `cmake`, and several math libraries that PyTorch relies upon. The `pip3 install -U pip` command upgrades the python package installer itself, and the final line installs packages like `numpy` required for a successful build. Neglecting this stage often leads to cryptic compilation errors due to unmet dependencies.

**Commentary:**

- `sudo apt update && sudo apt upgrade -y`: This standard command chain updates the local package index and upgrades all installed packages to the newest version. The `-y` flag allows the commands to proceed without manual confirmation.
- `sudo apt install -y python3-dev python3-pip cmake libopenblas-dev liblapack-dev gfortran`: These packages provide core building tools, Python headers, and linear algebra libraries. `gfortran` is a compiler used for Fortran code (often found in linear algebra implementations).
- `sudo pip3 install -U pip`: This is vital as older versions of pip can cause issues in installing the required packages.
- `sudo pip3 install numpy setuptools wheel`: Installs the `numpy` package (required for PyTorch) and other Python distribution tools, which are often needed later in the process.

**Example 2: Cloning and Configuring PyTorch**

```bash
git clone --recursive https://github.com/pytorch/pytorch.git
cd pytorch
git checkout v1.9.0 # or the desired version
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE=$(which python3) \
    -DCMAKE_C_COMPILER=/usr/bin/gcc-7 \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-7 \
    -DCAFFE2_USE_MSVC=0 \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -DUSE_CUDA=ON \
    -DUSE_CUDNN=ON \
    -DCUDNN_INCLUDE_DIR=/usr/include \
    -DCUDNN_LIBRARY=/usr/lib/aarch64-linux-gnu/libcudnn.so.7 \
    -DOPENBLAS_INCLUDE_DIR=/usr/include \
    -DOPENBLAS_LIB=/usr/lib/aarch64-linux-gnu/libopenblas.so \
    -DCPU_CAPABILITY=NEON
```

This code first clones the PyTorch repository and switches to the desired version tag. Creating a build directory and running `cmake` is the next step, specifying crucial compilation parameters. Note the use of GCC version 7, as this compiler version is generally more stable for older architectures. Also, CUDA paths and cuDNN locations need to be explicitly defined. `CPU_CAPABILITY=NEON` instructs the compiler to generate optimized code for ARM's NEON SIMD architecture, significantly improving performance on the Jetson Nano. The paths to CUDA libraries can vary, so it’s vital to verify their location prior to running this command.

**Commentary:**
- `git clone --recursive https://github.com/pytorch/pytorch.git`: This fetches the PyTorch source code including its submodules.
- `cd pytorch`: This navigates into the cloned repository.
- `git checkout v1.9.0`: Checkouts the specified PyTorch version. Users should replace with the desired version.
- `mkdir build && cd build`: This makes a separate build folder to keep the source directory clean.
- The `cmake` command configures the build process.
  - `-DCMAKE_BUILD_TYPE=Release`: Compiles for optimized, non-debug builds.
  - `-DPYTHON_EXECUTABLE=$(which python3)`: Tells CMake the location of the Python interpreter to use during compilation.
  - `-DCMAKE_C_COMPILER=/usr/bin/gcc-7` and `-DCMAKE_CXX_COMPILER=/usr/bin/g++-7`:  Specifies the gcc 7 compilers. Using a compiler other than version 7 may result in incompatibility.
  - `-DCAFFE2_USE_MSVC=0`: Sets the option for no MSVC support (irrelevant for Linux).
  - `-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda`: Specifies the location of CUDA drivers.
  - `-DUSE_CUDA=ON`: Enables CUDA support.
  - `-DUSE_CUDNN=ON`: Enables cuDNN support.
  - `-DCUDNN_INCLUDE_DIR=/usr/include`: Specifies the location of cuDNN header files.
  - `-DCUDNN_LIBRARY=/usr/lib/aarch64-linux-gnu/libcudnn.so.7`: Specifies the location of the cuDNN library.
  - `-DOPENBLAS_INCLUDE_DIR=/usr/include` and `-DOPENBLAS_LIB=/usr/lib/aarch64-linux-gnu/libopenblas.so`: Specifies the location of the OpenBLAS libraries.
  - `-DCPU_CAPABILITY=NEON`: Optimizes compilation for the Neon instruction set of the ARM processor.

**Example 3: Compiling and Installing PyTorch**

```bash
make -j$(nproc)
sudo make install
cd ../..
python3 -c "import torch; print(torch.__version__)"
```

The `make -j$(nproc)` command begins the compilation process. The `$(nproc)` argument instructs the compiler to use all available processor cores. The `sudo make install` command copies the built libraries and Python modules to the correct system directories, making PyTorch available for importing. The final line executes Python to verify successful installation. The compilation process itself is the most time-intensive part. It's advisable to monitor system temperatures during this phase.

**Commentary:**

- `make -j$(nproc)`: Initiates the build process using all available CPU cores.
- `sudo make install`: Installs the compiled libraries into the correct locations to use PyTorch from Python.
- `cd ../..`: Returns to the home directory.
- `python3 -c "import torch; print(torch.__version__)"`: Verifies the installation by importing PyTorch and printing its version.

Following the above procedure, you should have a fully functional, source-built PyTorch setup on your Jetson Nano. This approach allows for maximum control, albeit with a longer setup time. When choosing between this source-build method and relying on JetPack, I consistently recommend source compilation for precise version control, unless time constraints or a need for rapid prototyping outweighs the benefits of fine-grained control over library versions.

For those preferring JetPack, the download and install procedures can be located on the NVIDIA developer website, under Jetson software resources. Furthermore, the PyTorch documentation offers extensive insights into compiler options, build flags and compatibility information. Finally, community forums focused on Jetson devices frequently host discussions about build configurations and troubleshooting related to specific hardware configurations. Thoroughly understanding the requirements, limitations and procedures detailed above will enable consistent success when setting up PyTorch on the Jetson Nano.
