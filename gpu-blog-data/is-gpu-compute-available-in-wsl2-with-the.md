---
title: "Is GPU compute available in wsl2 with the new build?"
date: "2025-01-30"
id: "is-gpu-compute-available-in-wsl2-with-the"
---
WSL2’s capability to leverage GPU compute resources, specifically for tasks like machine learning and scientific simulations, has evolved significantly. The initial release of WSL2 did not offer native GPU passthrough; however, subsequent builds, particularly those incorporating the Windows Display Driver Model (WDDM) and dedicated support for CUDA and DirectML, have changed this landscape considerably. In essence, yes, GPU compute is available in WSL2 with recent builds, but the implementation has nuances and dependencies I've observed first-hand during system configurations for various development teams.

The critical element enabling GPU compute in WSL2 is the introduction of a virtualized GPU. Previously, WSL2 operated with a hypervisor-based architecture where the Linux kernel and utilities ran inside a lightweight virtual machine, separated from the host hardware. This isolation meant that the Linux environment lacked direct access to the physical GPU. With the introduction of WDDM GPU virtualization, Windows now creates a virtual representation of the physical GPU that the WSL2 virtual machine can utilize. This process is not a direct passthrough, but rather, a mediated communication pathway between the Windows GPU driver and the virtualized device in WSL2.

Specifically, this involves using the Windows GPU driver stack to handle requests from applications running within WSL2. For NVIDIA GPUs, this requires installing the appropriate NVIDIA drivers on the host Windows system *and* ensuring that the necessary CUDA toolkit is installed within the WSL2 environment. Similarly, for AMD and Intel GPUs, DirectML is the primary path, and the corresponding drivers and libraries need to be present on both the host and within the WSL2 instance. This architecture means the performance isn't entirely on par with native Linux due to the virtualization overhead, however the productivity benefits of working within WSL2 are usually more relevant than a fractional percentage difference in raw speed.

The primary advantage of this setup is that it allows for using the same GPU drivers and infrastructure across both the Windows environment and your Linux development setup. This removes the requirement for dual booting or using separate, dedicated machines for GPU-intensive tasks. It also dramatically simplifies development environments, allowing me to configure them rapidly across different teams, minimizing platform incompatibilities.

Let's illustrate the typical usage of GPU compute in WSL2 with three examples using Python and libraries commonly employed in machine learning.

**Example 1: TensorFlow GPU check.**

This snippet verifies if TensorFlow can detect and utilize the GPU. The code uses TensorFlow’s built-in GPU discovery mechanism.

```python
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')

if len(physical_devices) > 0:
    print("GPU is available")
    print(f"Number of GPUs available: {len(physical_devices)}")
    for i, device in enumerate(physical_devices):
         print(f"  Device {i}: {device}")
else:
    print("GPU is not available")
```

In a correctly configured WSL2 environment, the output should show a list of the discovered GPU devices, with a corresponding device name (e.g., `/physical_device:GPU:0`). If the GPU is not available, it will explicitly state that. It is crucial that if you are using an NVIDIA GPU the correct CUDA and cuDNN libraries are installed in WSL2 for TensorFlow to function properly. I've found that even with the GPU drivers installed on Windows, TensorFlow will not recognize the GPU if these libraries are missing within the WSL2 instance. DirectML, on the other hand, can use the Windows driver more directly without external libraries, but its performance can vary depending on the specific GPU and task.

**Example 2: Simple PyTorch CUDA calculation.**

This example performs a basic matrix multiplication on the GPU using PyTorch, demonstrating the core functionality for many model training workloads.

```python
import torch

if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"PyTorch is using CUDA: {torch.cuda.get_device_name(0)}")
    a = torch.rand(1000, 1000).to(device)
    b = torch.rand(1000, 1000).to(device)
    c = torch.matmul(a, b)
    print("Resulting matrix is computed on GPU.")
else:
    print("CUDA is not available, computation will occur on CPU")

```
This code attempts to detect a CUDA-enabled GPU; it will print if CUDA is available and specify the device name, typically referencing the NVIDIA GPU. If CUDA is not detected, the calculations fall back to the CPU. It has been my experience that if CUDA is available but the code still runs on the CPU, you should carefully examine that PyTorch CUDA drivers align with the installed CUDA version. I’ve also found it’s beneficial to explicitly define the device at the start of a project using a variable; it helps avoid device errors later during development.

**Example 3: DirectML acceleration in TensorFlow**

This demonstrates a simple example of how a TensorFlow model can be accelerated using DirectML when working with AMD and Intel GPUs within the WSL2 environment. DirectML is a Microsoft API that provides low-level access to hardware acceleration.

```python
import tensorflow as tf

try:
    # Check if DirectML device is available
    devices = tf.config.list_physical_devices('DML')
    if devices:
      print("DirectML devices detected:")
      for i, device in enumerate(devices):
         print(f"  Device {i}: {device}")

      # Select the DirectML device to use
      tf.config.set_visible_devices(devices[0], 'DML')

      # A basic computation to verify acceleration.
      a = tf.random.normal((1000,1000))
      b = tf.random.normal((1000,1000))
      c = tf.matmul(a,b)

      print("Computation completed using DirectML accelerated device")
    else:
      print("No DirectML device was detected")
except tf.errors.NotFoundError as e:
  print(f"DirectML not found, likely not supported or correctly installed: {e}")
except Exception as e:
  print(f"Error during directML configuration: {e}")
```

This code snippet attempts to detect DirectML compatible devices. If successful, it will configure TensorFlow to utilise the found DirectML devices. The example performs a simple matrix operation and outputs that the work was done on a DirectML enabled device. This section is deliberately encased in a try/except block to catch both TensorFlow missing device errors as well as any other unexpected issues. During my initial implementations, I found that ensuring the DirectML libraries are updated and compatible on the Windows host and the WSL2 instance are crucial for successful operation.

For further exploration of this topic, I recommend referring to the official Microsoft documentation on WSL and GPU passthrough, specifically focusing on DirectML and CUDA support in WSL2. Additionally, the NVIDIA CUDA documentation provides detailed instructions on driver installation and the required software to use CUDA within a Linux environment. The PyTorch and TensorFlow documentation offers guidance on configuring GPU usage and troubleshooting errors that may arise. Consulting community forums related to WSL and machine learning with GPUs can also prove valuable, offering solutions to common configuration issues. Also be aware the required WSL build should be reviewed in the official documentation to ensure the correct features are enabled.

In conclusion, while the initial release of WSL2 did not support GPU compute, subsequent updates have enabled this functionality by implementing virtualized GPU support. Although performance is not identical to a native Linux system, the accessibility and productivity benefits make WSL2 a viable option for GPU-accelerated development work. Consistent maintenance of drivers and library versions on both the host and WSL2 environments are essential for reliable GPU utilization.
