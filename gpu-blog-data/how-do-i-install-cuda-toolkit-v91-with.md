---
title: "How do I install CUDA Toolkit v9.1 with cuDNN v7 on macOS High Sierra 10.13.2 for TensorFlow?"
date: "2025-01-30"
id: "how-do-i-install-cuda-toolkit-v91-with"
---
Installing a specific combination of CUDA Toolkit and cuDNN versions, particularly older ones like v9.1 and v7, on macOS High Sierra for TensorFlow presents compatibility challenges primarily due to Apple’s transition away from NVIDIA’s GPUs and the evolving dependencies of the software ecosystem. This configuration, while functional, requires careful attention to version matching and manual configuration since it doesn't align with current standard installations on macOS.

The core issue is that macOS High Sierra predates the official deprecation of NVIDIA GPU support by Apple, which occurred in subsequent macOS releases. Consequently, drivers for NVIDIA GPUs were available for High Sierra, yet they are not maintained by Apple. Furthermore, installing specific older versions of CUDA and cuDNN requires navigating an environment where package managers and auto-installers aren't directly compatible. The installation process relies on downloading and manually placing files.

To proceed, we must first understand that CUDA v9.1, released in 2017, is incompatible with modern TensorFlow versions that expect CUDA 10.1 or later. Therefore, we will be targeting older TensorFlow builds which were compatible with this older infrastructure, which is critical for a successful integration. This also means one cannot directly utilize the `pip` package manager to install the TensorFlow build. Instead, one will require specific wheel files that are compatible.

Let's begin with the CUDA Toolkit installation itself. First, I recall downloading the CUDA Toolkit v9.1 package from NVIDIA’s archive. This is critical since the current driver download page will provide newer versions. Once downloaded, the .dmg image needs to be opened, and the installer run. The installer's default path is fine, but one needs to be vigilant about which components are installed. I opt for the full installation, including the drivers, development tools, and samples. While High Sierra should generally support CUDA, the graphics card model and specific driver compatibility issues need verification. One should confirm the GPU model is explicitly listed as supported in NVIDIA's release documentation for this specific CUDA version. Failure to do this may result in the CUDA drivers failing to load.

Once the CUDA Toolkit is installed, I'd usually verify it with a simple device query, which I'll cover later. The next critical step is installing cuDNN. With cuDNN v7, we again need to specifically download the compatible version from NVIDIA’s archives, making sure it's version 7 compatible with the CUDA v9.1 previously installed. This typically comes as a compressed tar.gz file. The cuDNN installation isn't a standard installer; rather, it involves extracting the contents and manually copying specific files into the correct CUDA directory. This manual step is where most common errors occur. The extracted archive usually contains three folders: `include`, `lib`, and `bin`. Within, `include` files are copied to the CUDA include path and `lib` files are moved to the CUDA library path, such as `/usr/local/cuda/include` and `/usr/local/cuda/lib`, respectively, using administrator privileges via `sudo`. The 'bin' directory is largely unnecessary here for cuDNN.

The core of this process is ensuring that the library is visible at run-time. This can be done by exporting the path to the CUDA library in the user's `.bash_profile` or `zshrc` for zsh shells. Specifically, I add the following export statement:

```bash
export DYLD_LIBRARY_PATH=/usr/local/cuda/lib:$DYLD_LIBRARY_PATH
```

This enables the dynamic linker to locate the relevant cuDNN libraries. Without this, TensorFlow will not see the GPU acceleration capability, and cuDNN will fail to load, usually generating a message about an incompatible version. Now for a practical verification: First, let's verify if CUDA itself can find the GPU. This can be achieved with a simple CUDA sample program compile, which is typically located in the cuda SDK sample directory. I can compile a basic deviceQuery executable via:

```bash
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

This `deviceQuery` tool will present detailed information regarding CUDA enabled GPUs. If the output lists the NVIDIA GPU and its properties, the installation is mostly successful. However, if this fails, re-installation of the CUDA driver and/or kernel extension loading becomes necessary. I would need to use `kextstat` command to check for errors in the kernel extensions related to NVIDIA drivers. If this is not found, that indicates a driver problem, requiring reinstall or checking the driver version against the GPU hardware.

The critical piece is installing the correct TensorFlow version. I am reminded of the challenge of using current TensorFlow versions; therefore, one must look at TensorFlow versions from around 2018 timeframe.  It is common to see `tensorflow-gpu` versions that claim compatibility in their package name. One typically will need to download the whl files specifically for the `tensorflow-gpu` library, which are matched with both the python version and the CUDA version used for compiling the package. For example:

```python
pip install tensorflow_gpu-1.12.0-cp36-cp36m-macosx_10_7_x86_64.whl
```

This command manually installs a specific wheel file targeted for python 3.6, and macOS 10.7, which often works with High Sierra, where the machine architecture is x86_64. It is essential to use the CPU version if the appropriate GPU version is not present, for example `tensorflow-1.12.0-cp36-cp36m-macosx_10_7_x86_64.whl`. Using the non-GPU version ensures TensorFlow functions properly, albeit without GPU acceleration. To confirm TensorFlow sees the GPU after the installation one can use code like this, which should reveal the presence of the GPU device:

```python
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

If the print statement reveals the GPU device, it indicates that the CUDA driver and the cuDNN files are correctly placed and configured. If the GPU device is missing, then the installation did not work properly. This failure can be attributed to various issues: incorrect driver versions, misplaced cuDNN files, insufficient environment settings, and incompatible TensorFlow wheel files. It is important to recheck the entire process, carefully verifying each step. Common errors are often the misplaced lib files of cuDNN and incorrectly set `DYLD_LIBRARY_PATH`.  

Given this situation, one often revisits several times over, ensuring every detail is matched correctly. This requires carefully checking the installation logs and error messages.

For additional resources, the NVIDIA CUDA Toolkit documentation should be consulted, paying close attention to the release notes for v9.1. The cuDNN documentation from NVIDIA, for version 7, would also be extremely helpful, particularly regarding manual installation instructions. Finally, a close review of older TensorFlow release notes is necessary to identify compatible versions. I would recommend focusing on community forums and archives for that information, since it is unlikely that current install instructions will work.  In summary, installing CUDA 9.1 and cuDNN 7 on macOS High Sierra for TensorFlow is possible, but requires a methodical approach with careful attention to versioning and manual configuration steps and using older compatible wheel files for TensorFlow rather than relying on direct `pip` installs.
