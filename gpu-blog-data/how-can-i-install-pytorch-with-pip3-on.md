---
title: "How can I install PyTorch with pip3 on Ubuntu 20.04?"
date: "2025-01-30"
id: "how-can-i-install-pytorch-with-pip3-on"
---
The successful installation of PyTorch via `pip3` on Ubuntu 20.04 hinges critically on satisfying PyTorch's complex dependency requirements and selecting the appropriate pre-built wheel package for your system's architecture and CUDA capabilities.  My experience working on high-performance computing projects has highlighted the common pitfalls surrounding this process, primarily stemming from inconsistencies in system configurations and neglecting CUDA compatibility.

**1.  Understanding PyTorch's Dependencies and Build Process:**

PyTorch, unlike many Python libraries, is not a purely Pythonic project.  Its core functionalities rely on highly optimized C++ and CUDA code (for GPU acceleration).  Consequently, a straightforward `pip3 install torch` often fails unless the prerequisite system libraries are installed. These include but are not limited to:

* **Basic Development Tools:**  `build-essential`, `cmake`, `wget`, `git` – these are necessary for compiling dependencies and managing the build process.
* **Python Development Libraries:** `python3-dev` (or equivalent) – crucial for integrating PyTorch with your Python environment.
* **CUDA Toolkit (Optional but Recommended):**  If you intend to leverage GPU acceleration, you must install the appropriate CUDA Toolkit version compatible with your NVIDIA GPU and PyTorch release. Incorrect CUDA version selection is a major source of installation errors.  Note that PyTorch wheels often specify CUDA version compatibility in their filenames.
* **cuDNN (Optional, dependent on CUDA):** If using CUDA, cuDNN (CUDA Deep Neural Network library) is usually required for optimal performance.  Its version must align with your CUDA Toolkit version.

Failure to install these dependencies correctly will lead to compilation errors and ultimately a failed PyTorch installation.  My past debugging efforts have shown this to be the most common reason for installation failures.

**2.  Installation Procedures and Code Examples:**

The preferred method, avoiding lengthy compilation times, is using pre-built PyTorch wheels directly from the PyTorch website.  However, understanding the process involving compiling from source can be helpful for troubleshooting.

**Example 1: Installation using a pre-built wheel (CPU-only):**

```bash
sudo apt update
sudo apt install build-essential python3-dev python3-pip

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This command first updates the package list and installs essential development tools and `pip3`.  Crucially, it then uses `pip3` to install PyTorch, torchvision (computer vision tools), and torchaudio (audio processing tools) using a specific index URL. The `cu118` portion suggests a CPU-only installation. Replace this with `cu117`, `cpu`, or other appropriate identifiers based on your CUDA version and PyTorch website instructions.  Always verify the appropriate wheel from the official PyTorch website.


**Example 2: Installation using a pre-built wheel (CUDA 11.8):**

```bash
sudo apt update
sudo apt install build-essential python3-dev python3-pip
#Install CUDA toolkit and cuDNN (versions as needed) - instructions omitted here for brevity.  Refer to NVIDIA documentation.

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This example is similar to the previous one but assumes you have already installed a compatible CUDA toolkit and cuDNN.  Ensure the CUDA version specified in the `--index-url` aligns with your installed CUDA Toolkit.  Mismatched versions will inevitably result in errors.


**Example 3:  Compiling from Source (Advanced, generally not recommended):**

While generally discouraged due to its complexity and longer installation times, compiling from source provides greater flexibility but requires a more in-depth understanding of system dependencies. I've utilized this only when very specific or experimental PyTorch versions are required.  Note that this approach requires a much deeper understanding of CMake and build systems.

```bash
sudo apt update
sudo apt install build-essential python3-dev python3-pip git cmake wget

git clone https://github.com/pytorch/pytorch
cd pytorch
# Configure the build (this step requires careful attention to specific options, including CUDA and other dependencies)
# ... (Configuration steps using CMake omitted for brevity) ...
make -j$(nproc)
sudo make install
pip3 install -e .
```


This demonstrates the general structure; the actual `cmake` configuration is highly specific and needs to be adapted based on your system and desired PyTorch features.  Refer to the official PyTorch documentation for comprehensive instructions on configuring the build process.  I would strongly advise against this unless thoroughly familiar with C++ build systems.



**3. Resource Recommendations:**

For comprehensive installation instructions, consult the official PyTorch website's documentation.  It provides detailed guidance on different installation scenarios and troubleshooting common problems.  The NVIDIA website offers resources concerning CUDA Toolkit and cuDNN installation.  Understanding CMake's documentation is beneficial when compiling from source. Finally, the official Python documentation aids in managing Python environments and virtual environments.  Thoroughly review these resources to ensure compatibility and avoid common errors.


**Conclusion:**

Installing PyTorch effectively requires careful attention to dependencies and selecting the appropriate wheel package.   Using pre-built wheels, as illustrated in Examples 1 and 2, is the recommended approach for most users.  Compilation from source (Example 3) should be reserved for advanced users with a deep understanding of build systems and the specific needs justifying this approach.  Remember to always cross-reference your CUDA and cuDNN versions (if applicable) with the PyTorch wheel you choose to ensure compatibility and avoid numerous potential pitfalls. My experience suggests proactive dependency management and meticulous version checking are key to a successful PyTorch installation.
