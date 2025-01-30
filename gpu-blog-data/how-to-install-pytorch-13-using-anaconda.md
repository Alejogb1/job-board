---
title: "How to install PyTorch 1.3 using Anaconda?"
date: "2025-01-30"
id: "how-to-install-pytorch-13-using-anaconda"
---
PyTorch 1.3’s installation via Anaconda warrants specific attention due to its dependency management nuances and the common pitfalls encountered when relying on default configurations. I’ve personally wrestled with these issues, having initially opted for haphazard installation methods during my work on a deep learning project involving recurrent neural networks for time-series forecasting in late 2019. This led to numerous dependency conflicts, ultimately requiring a complete environment rebuild. A focused approach using Anaconda's environment management is crucial for a stable and reproducible installation.

First, it's important to understand that PyTorch relies on specific CUDA driver versions, particularly its GPU accelerated variants. A mismatch between the installed CUDA toolkit and the PyTorch version can cause runtime errors that are difficult to diagnose. Furthermore, Python environments managed by Anaconda mitigate package conflicts. Therefore, the following steps prioritize these two aspects: creating a dedicated environment, and specifying the correct installation channels.

**Explanation**

The recommended way to install PyTorch 1.3 using Anaconda involves:

1.  **Creating an isolated environment:** This keeps dependencies specific to your PyTorch 1.3 project segregated from other projects or your base environment. This minimizes package conflicts which can cause unpredictable application behavior.

2.  **Selecting appropriate channel:**  The primary PyTorch package is not always found in the default Anaconda channels and often a specific channel needs to be specified to obtain that version. PyTorch's official site provides specific installation commands for various versions but relying on those specific commands is often unreliable. Therefore specifying known channels is often necessary.

3. **Specifying package variations:** Based on the intended compute environment either the CPU-only or the CUDA-enabled package needs to be installed. This is achieved by using either the `cpuonly` tag or explicitly selecting a `cuda` variant. Specifying the specific CUDA version is paramount, or the system might rely on defaults that are incompatible.

4.  **Verifying the Installation:** After installing the required packages, one must test if the installation was successful by verifying if the PyTorch library is importable and that the expected device is detected.

Failure to adhere to these steps frequently results in either PyTorch not being found, or CUDA support not being available. Specifically, neglecting the isolation of the environment often leads to dependency hell which causes time wasted resolving dependency issues.

**Code Examples**

I present three code examples, demonstrating different scenarios and the corresponding command-line instructions.

**Example 1: CPU-Only Installation**

This is the simplest case, suitable for development where GPU acceleration isn’t necessary. This approach can be used for example when building an environment to run tests or when only needing to test some CPU-bound preprocessing functionality.

```bash
conda create -n pytorch13_cpu python=3.7 # create an isolated environment using python 3.7
conda activate pytorch13_cpu # activate the created environment
conda install pytorch==1.3.1 torchvision==0.4.2 cpuonly -c pytorch # install CPU packages from the pytorch channel
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

*   `conda create -n pytorch13_cpu python=3.7`: creates a new Anaconda environment named `pytorch13_cpu` using Python 3.7. This specific Python version is selected as its known to be compatible with Pytorch 1.3.
*   `conda activate pytorch13_cpu`: activates the created environment, ensuring packages are installed within this isolated context.
*   `conda install pytorch==1.3.1 torchvision==0.4.2 cpuonly -c pytorch`:  installs PyTorch 1.3.1, TorchVision 0.4.2, specifying `cpuonly`, and getting those packages from the `pytorch` channel. Specifying the channel is crucial for getting the correct packages.
*   `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"`: A short script to verify the installation, by printing PyTorch version and confirming that no CUDA device is detected.

**Example 2: GPU Installation with CUDA 10.0**

This example assumes the system has CUDA 10.0 installed. Note that the correct CUDA driver needs to be installed and accessible before this command is run. This scenario would be appropriate if, for example, running training workloads on a machine with a graphics card that is compatible with this particular CUDA version.

```bash
conda create -n pytorch13_cuda10 python=3.7 # create an isolated environment using python 3.7
conda activate pytorch13_cuda10 # activate the created environment
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.0 -c pytorch # Install cuda 10.0 based pytorch from the pytorch channel.
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')" # A check to make sure the install was successful.
```

*   `conda create -n pytorch13_cuda10 python=3.7`: Creates the environment similarly to the CPU example, but the name reflects the CUDA context.
*   `conda activate pytorch13_cuda10`: Activates the environment.
*   `conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.0 -c pytorch`:  Here, besides the PyTorch and TorchVision packages, `cudatoolkit=10.0` is specified. This is imperative, ensuring that the CUDA enabled packages are correctly installed.
*   `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"`: Similar to Example 1, but also checks if a CUDA enabled device is detected, outputting the GPU's name if available.

**Example 3: Specific CUDA Version Not Explicitly Provided by PyTorch**

In scenarios where PyTorch doesn’t offer a pre-built package for the precise CUDA version, or if there is a specific CUDA requirement outside of what's provided. This method allows for installing a generic CUDA enabled Pytorch build, while ensuring the correct version of the CUDA runtime library is available in the environment. This is an advanced case and might require additional system level configurations. This might occur when targeting specific hardware that is not completely mainstream.

```bash
conda create -n pytorch13_cuda python=3.7
conda activate pytorch13_cuda
conda install pytorch==1.3.1 torchvision==0.4.2 -c pytorch
conda install cudatoolkit=10.1 # install the explicit cuda library
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

*   `conda create -n pytorch13_cuda python=3.7`: Creates the environment similar to the other examples.
*   `conda activate pytorch13_cuda`: Activates the environment.
*   `conda install pytorch==1.3.1 torchvision==0.4.2 -c pytorch`: Install Pytorch and torchvision without specifying the `cudatoolkit` version. This will get the generic CUDA enabled build.
*   `conda install cudatoolkit=10.1`: Then, a specific version of the CUDA toolkit is explicitly installed, which might not be a direct match with the Pytorch build. This will ensure the specific cuda libraries are present.
*  `python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"`: Verify the successful installation of Pytorch, and the CUDA device.

**Resource Recommendations**

To deepen your understanding and manage potential issues, I recommend reviewing these resources:

1.  **Anaconda Documentation:** Explore the official Anaconda documentation for in-depth information on environment management, package channels, and dependency resolution. It contains a full reference guide to the features and commands of the Anaconda ecosystem.
2.  **PyTorch Official Website:** While not a single installation guide for a specific version, the PyTorch website offers comprehensive documentation concerning the library's capabilities and installation instructions which should always be consulted, especially regarding CUDA version compatibility.
3. **CUDA Toolkit Documentation:** The official CUDA documentation from Nvidia offers a deep understanding of the underlying libraries and device drivers which are essential to using the GPU accelerated versions of Pytorch. It is particularly useful to understand dependency requirements.
4. **Stack Overflow:** Search Stack Overflow for PyTorch-related issues and solutions. Many users often encounter and resolve common problems, providing very good insights into common issues.

By adhering to the principle of creating isolated environments with specific channels for the desired packages, and ensuring the appropriate GPU device drivers and CUDA versions are installed, you should successfully install PyTorch 1.3 within an Anaconda environment, significantly reducing the risk of package conflicts and related runtime failures. I can personally attest to the effectiveness of these practices from my experience in developing and deploying deep learning models.
