---
title: "How to use PyTorch CUDA on Ubuntu 20.04?"
date: "2025-01-30"
id: "how-to-use-pytorch-cuda-on-ubuntu-2004"
---
The effectiveness of deep learning models is often directly proportional to the availability of high-performance computing resources, and for many, this translates to harnessing the power of NVIDIA GPUs using CUDA. My experience developing complex neural networks for image segmentation has consistently highlighted the importance of a correctly configured PyTorch environment with CUDA support; a misstep here can mean hours of training time versus minutes. I will detail the proper configuration of PyTorch with CUDA on an Ubuntu 20.04 system, based on common pitfalls encountered.

The foundation for successful GPU utilization with PyTorch rests on several components: the correct NVIDIA GPU drivers, the CUDA Toolkit, and a PyTorch version built with CUDA support. Each must be installed and verified meticulously to avoid subtle compatibility issues that often manifest as cryptic error messages during model execution.

The initial step involves installing the appropriate NVIDIA GPU drivers for the specific hardware. Typically, these are available through Ubuntu's 'Software & Updates' application or by using the command-line tool `apt`. While the ‘Additional Drivers’ tab in the GUI provides a convenient way to install suggested drivers, a more controlled approach using the apt package manager is recommended to minimize unexpected updates or conflicts. First, update the package list with `sudo apt update` then install the recommended driver via `sudo ubuntu-drivers install`. Post installation, a reboot is required for the driver to fully load, and the `nvidia-smi` command should then display information about the installed driver and GPU status, verifying successful installation. If `nvidia-smi` returns an error, this indicates a problem with the driver installation that must be resolved before proceeding.

Following successful driver installation, the CUDA Toolkit must be downloaded and installed from the NVIDIA website. It is crucial to select the CUDA Toolkit version compatible with both the installed drivers and the desired PyTorch version. The NVIDIA website provides detailed instructions and an archive of historical releases. During the installation process, typically performed using a `.run` file obtained from their site, ensure that you specify the desired installation path, remembering that these settings affect the subsequent configuration for PyTorch. Moreover, the toolkit's installer often includes an option to update the system path and environment variables. Confirm that the `LD_LIBRARY_PATH` is correctly updated to include the location of the CUDA libraries. These environment variables are critical for PyTorch to find the necessary CUDA components. After installation, verify the installation with `nvcc --version`, which should display the version of the CUDA compiler. A failure here points to an incomplete installation or issues with environment path settings.

Finally, install PyTorch with CUDA support. This is best achieved using `pip`, referencing the specific index URL provided on the PyTorch website. Avoid the general PyTorch installation without the index URL, as this might result in a CPU-only version. Choose the installation command corresponding to your CUDA Toolkit version, as PyTorch releases are built against specific CUDA versions. Note that mismatch between the installed CUDA Toolkit version and the PyTorch CUDA build is a very common source of issues. A typical installation command looks like: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`. This command targets CUDA 11.8. Replace `cu118` with the appropriate `cuXXX` version based on your CUDA installation.

After completing the setup, confirm that PyTorch is able to use the CUDA-enabled GPU. Here are three illustrative code examples with explanations:

**Example 1: Basic CUDA device check:**

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"CUDA is available, using device: {device}")
    print(f"Number of CUDA devices: {torch.cuda.device_count()}")
    print(f"Current CUDA device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("CUDA is not available, using CPU.")

```

This code snippet performs a fundamental check to verify CUDA visibility. It uses `torch.cuda.is_available()` to ascertain CUDA support. If available, it prints information about the available CUDA devices including the device count, the currently selected device, and device name. Should CUDA be unavailable, it gracefully informs the user. This is a crucial sanity check to run first, ensuring the fundamental components are working. Failure here indicates problems with driver, CUDA toolkit, or PyTorch installation.

**Example 2: Moving tensors to GPU:**

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

tensor_cpu = torch.rand(3, 4)
print(f"Tensor on CPU: {tensor_cpu}")

tensor_gpu = tensor_cpu.to(device)
print(f"Tensor on GPU: {tensor_gpu}")

if tensor_gpu.is_cuda:
    print("Tensor is indeed on the GPU.")

tensor_cpu_again = tensor_gpu.cpu()
print(f"Tensor back on CPU: {tensor_cpu_again}")
```

This example demonstrates creating tensors on the CPU and moving them to the GPU for processing. This represents a typical usage pattern when training neural networks. The `.to(device)` operation transfers data to the specified device (GPU if available, otherwise CPU). The `is_cuda` method further verifies that the tensor is indeed located on a CUDA device. It also includes moving the data back to CPU, an operation required before numpy or other CPU-only libraries can be used. This example showcases practical data management using PyTorch with CUDA.

**Example 3:  Simple model training:**

```python
import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# Define a simple linear model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(10, 2)
    def forward(self, x):
        return self.linear(x)

# Move the model to the device
model = LinearModel().to(device)

# Sample input tensor on CPU
inputs_cpu = torch.rand(1, 10)
# Move tensor to device
inputs = inputs_cpu.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Fake Target tensor on device
targets = torch.rand(1, 2).to(device)

# Training
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, targets)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
print("Model training complete.")
```

This example presents a minimalist model training snippet, demonstrating the end-to-end process of preparing data, moving both the model and inputs to the GPU, performing forward and backward passes, and updating the model parameters.  This demonstrates practical application, showing how the model itself should be placed on the device. Loss value calculation ensures training works.  Without CUDA support, the `to(device)` commands will simply place all calculations on CPU, significantly slowing training.

For further learning about PyTorch and CUDA, I would recommend exploring the official PyTorch documentation, particularly the sections on CUDA semantics and GPU usage. Additionally, the NVIDIA developer website offers comprehensive documentation on the CUDA Toolkit and best practices for GPU programming.  Finally, online forums like the PyTorch discussion forum can provide insights into commonly encountered issues and troubleshooting techniques.
