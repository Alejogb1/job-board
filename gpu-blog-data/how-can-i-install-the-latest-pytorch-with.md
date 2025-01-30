---
title: "How can I install the latest PyTorch with CUDA enabled from the command line?"
date: "2025-01-30"
id: "how-can-i-install-the-latest-pytorch-with"
---
Achieving a CUDA-enabled PyTorch installation via the command line hinges on two key elements: having a compatible NVIDIA driver and using the correct `pip` command incorporating CUDA compatibility. Over my years working on deep learning projects, I've found that meticulous attention to these prerequisites avoids hours of troubleshooting. Incorrect driver versions or mismatched PyTorch builds are the most common culprits of installation failures. This response details the process, illustrating it with practical examples and pointing to additional resources.

First, confirm that your system has a CUDA-compatible NVIDIA GPU. The `nvidia-smi` command in your terminal (Windows, Linux, macOS) reveals this; if the command is not found or reports no NVIDIA driver, you will need to download and install the appropriate driver from NVIDIA's website before proceeding. This driver installation is platform-specific, requiring selection of the correct operating system and GPU model. Furthermore, verify the driver version; PyTorch compatibility with CUDA toolkits is version-specific, and the PyTorch installation command requires a matching CUDA toolkit version. The output from `nvidia-smi` also includes the driver version, which is crucial for the subsequent step.

With the driver installed and its version known, the next critical step is using `pip` to install the appropriate PyTorch package. PyTorch provides pre-built wheels that include CUDA support; the specific installation command must reflect your CUDA toolkit version and the operating system. The general structure of the `pip` command is as follows:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu<CUDA_VERSION>
```

The `<CUDA_VERSION>` placeholder represents your CUDA toolkit version, derived from the NVIDIA driver version you identified earlier. For example, if your `nvidia-smi` output indicates a driver that supports CUDA 11.8, the placeholder would become `118`. PyTorch's website maintains compatibility information for specific CUDA and PyTorch versions. Failing to match this version exactly will lead to runtime errors when utilizing CUDA functions.

Here are three concrete examples illustrating installation across different CUDA toolkit versions:

**Example 1: CUDA 11.7 installation**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
```

This command installs the PyTorch, torchvision, and torchaudio packages specifically built for CUDA 11.7. The `--index-url` flag directs `pip` to the PyTorch repository where compatible wheels are hosted. I've personally used this installation path in projects involving high-resolution image classification; the correct CUDA integration significantly accelerates training times, decreasing it from hours to minutes for large datasets. Post-installation, verification using a short Python script confirms CUDA usage.

**Example 2: CUDA 12.1 installation**

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

This variant targets CUDA toolkit version 12.1. In situations where my machine's driver was updated to leverage the latest CUDA features, this command was essential for correct PyTorch functionality. Notice that the structure remains consistent; only the CUDA toolkit version within the URL changes. It's crucial to be meticulous about the precise version, as mismatches can cause silent failures or incorrect execution paths within the PyTorch framework.

**Example 3: Specific PyTorch and torchvision versions with CUDA 11.8**

```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio --index-url https://download.pytorch.org/whl/cu118
```

This example demonstrates installing specific versions of PyTorch and torchvision while using CUDA 11.8. Occasionally, development workflows demand pinned version dependencies, and this command addresses that need. The `==` operator specifies the exact package version required. Such a situation arises frequently when maintaining compatibility across multiple environments or reproducing research results that rely on specific framework versions. Using version pinning prevents unexpected behavior from newer, possibly incompatible, releases of dependent libraries. It highlights the flexibility afforded by `pip` in managing versions and CUDA compatibility within the PyTorch ecosystem.

After successfully installing PyTorch with CUDA support, verification is a simple, but important, step. Within a Python interpreter, run the following:

```python
import torch
print(torch.cuda.is_available())
```

A result of `True` indicates that PyTorch detects and can utilize your NVIDIA GPU. If the result is `False`, double-check your driver installation, CUDA toolkit compatibility, and the PyTorch installation command. Reviewing the output of the `pip install` command for error messages can be very helpful in pinpointing problems.

Furthermore, a more comprehensive test uses the `torch.cuda.device_count()` function, which should return a number of GPUs found on the system. If it returns zero even when `torch.cuda.is_available()` returns `True`, it suggests there may be driver issues or the PyTorch installation is configured in a non-standard way.

Resource recommendations include the NVIDIA driver website, the official PyTorch documentation, and the `pip` documentation. These resources are all easily searchable and contain exhaustive explanations, example code, and troubleshooting steps. Specifically, the PyTorch website details compatibility information between PyTorch releases and CUDA versions. It also offers pre-built installation commands. Reading the release notes for both PyTorch and NVIDIA drivers before installation is recommended, as it addresses potential compatibility or bug issues. Consult the `pip` documentation for details on advanced features and troubleshooting specific to the package manager, such as how to clear the `pip` cache or specific ways to install from a local file.

In summary, installing CUDA-enabled PyTorch from the command line involves careful matching of NVIDIA drivers with the corresponding CUDA toolkit versions within the `pip install` command. Rigorous verification with simple test code ensures proper installation and functionality. Consulting the recommended resources provides more in-depth information and troubleshooting help. My experience has shown that time spent on precision during installation drastically reduces debugging time in the long run. The three examples provided represent common scenarios, but the general principles apply across different configurations and ensure a smooth and functional deep learning development environment.
