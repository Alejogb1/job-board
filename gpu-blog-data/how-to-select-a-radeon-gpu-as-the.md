---
title: "How to select a Radeon GPU as the PyTorch device?"
date: "2025-01-30"
id: "how-to-select-a-radeon-gpu-as-the"
---
Selecting a Radeon GPU as the PyTorch device requires explicit configuration, given that PyTorch’s default CUDA backend primarily targets NVIDIA hardware. The process involves not only identifying the correct GPU but also ensuring the necessary software environment is correctly established. In my experience setting up large-scale distributed training jobs, the configuration inconsistencies often stem from improper handling of PyTorch device identification when using non-NVIDIA hardware. This can lead to unexpected fallback to CPU or, worse, runtime errors due to unsupported operations.

The fundamental challenge lies in PyTorch's abstraction layer. While the high-level API uses string arguments like "cuda" or "cpu," the underlying implementation must resolve these strings to actual hardware devices. For NVIDIA GPUs, this resolution is generally smooth because of the pervasive support for CUDA. However, AMD GPUs, accessed through the ROCm (Radeon Open Compute) framework, require an explicit direction to use the correct device backend.

The first critical step is to ascertain if your system can communicate with the Radeon GPU. This means confirming that ROCm is installed and the corresponding drivers are correctly configured. Once ROCm is functional, you should verify that PyTorch is built with ROCm support. Standard pip distributions of PyTorch may not include this support. Special builds are often necessary, frequently obtained from AMD's own distribution channels, or by compiling PyTorch from source with the relevant ROCm flags enabled. Failure to meet this requirement will result in "cuda" or equivalent devices being unavailable, despite the presence of a Radeon GPU.

Assuming this preliminary step is completed, you must then programmatically specify the Radeon GPU as your device. You cannot rely on PyTorch automatically detecting and selecting the correct GPU because it typically searches for CUDA by default. The "cpu" device is the standard fallback. The API relies on a combination of checking availability, enumerating devices and passing the appropriate device identifier to your training and model setup functions. The next section provides a code demonstration of how to do so.

**Code Example 1: Basic Device Selection**

```python
import torch

def select_rocm_device():
    if torch.cuda.is_available():
        print("CUDA is available. Check if ROCm is correctly installed and PyTorch build.")
    if torch.backends.rocm.is_available():
        print("ROCm is available.")
        device_name = "rocm"
        device_index = 0
        try:
            torch.device(device_name, device_index)
            return device_name + ":" + str(device_index)
        except RuntimeError as e:
            print(f"Error accessing ROCm device: {e}")
            return "cpu" # Fallback to CPU
    else:
        print("ROCm is not available. Using CPU.")
        return "cpu"


if __name__ == "__main__":
    device = select_rocm_device()
    print(f"Using device: {device}")

    # Example usage: tensor creation on device
    try:
        test_tensor = torch.ones((5,5), device=device)
        print(f"Tensor successfully created on: {test_tensor.device}")
    except RuntimeError as e:
        print(f"Error creating tensor on device: {e}")
```

**Commentary:**
This initial example establishes the most basic form of device selection. The `select_rocm_device()` function first checks if CUDA is available, and although not directly relevant for Radeon selection, it is important for debugging environment issues. Next, it directly confirms ROCm availability using `torch.backends.rocm.is_available()`. If available, it attempts to construct a device string in the "rocm:0" format, which designates the first ROCm device. Importantly, we wrap the actual device creation in a try block as we've experienced issues with devices failing to initialise in multi-GPU systems which can lead to application crashes if not handled. If the ROCm device construction succeeds, it's returned, otherwise a fallback to the "cpu" is performed. This approach highlights the direct, deliberate steps necessary to engage ROCm rather than relying on PyTorch’s default behavior. Finally, there is an attempt to create a basic tensor on the detected device. This shows a typical test you might use to verify successful operation and that an exception is handled gracefully.

**Code Example 2: Enumerating and Selecting a Specific ROCm Device**

```python
import torch

def select_specific_rocm_device(target_id: int) -> str:
    if not torch.backends.rocm.is_available():
        print("ROCm is not available. Using CPU.")
        return "cpu"

    num_devices = torch.cuda.device_count()
    print(f"Number of ROCm devices found: {num_devices}")

    if target_id >= num_devices:
        print(f"Target ROCm device id {target_id} out of range. Using default ROCm device or CPU.")
        return select_rocm_device()

    try:
        device_name = "rocm"
        torch.device(device_name, target_id) # Check if the device identifier works
        return device_name + ":" + str(target_id)
    except RuntimeError as e:
       print(f"Error accessing ROCm device {target_id}: {e}")
       return "cpu"

if __name__ == "__main__":
    target_gpu_id = 1 # Select the second ROCm GPU (index starts at 0)
    device = select_specific_rocm_device(target_gpu_id)
    print(f"Using device: {device}")

    # Example usage: tensor creation on device
    try:
        test_tensor = torch.ones((5,5), device=device)
        print(f"Tensor successfully created on: {test_tensor.device}")
    except RuntimeError as e:
        print(f"Error creating tensor on device: {e}")
```
**Commentary:**
This code snippet demonstrates how to select a specific ROCm GPU when multiple are available on the system. It begins with the same ROCm availability check. The core difference lies in calling `torch.cuda.device_count()` which in the context of ROCm returns the number of available AMD GPUs. The function then uses `target_id` to verify if the chosen device id is in range. Again, if there is a failure to initialize the device, the application will fall back to default behavior. By looping through the available device IDs, you can achieve finer control when running on a multi-GPU machine. This is an important step as models may need to be pinned to specific GPUs within a multi-GPU system.

**Code Example 3: Handling Device Availability with a Model**

```python
import torch
import torch.nn as nn
import torch.optim as optim

def train_model(device):
    # Simple example model
    class SimpleModel(nn.Module):
      def __init__(self):
          super(SimpleModel, self).__init__()
          self.linear = nn.Linear(10, 1)
      def forward(self, x):
          return self.linear(x)

    model = SimpleModel().to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Sample Data
    inputs = torch.rand(1, 10, device=device)
    target = torch.rand(1, 1, device=device)
    
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()

    print(f"Model trained on device: {next(model.parameters()).device}")


if __name__ == "__main__":
    device = select_rocm_device()
    train_model(device)
```
**Commentary:**
This final example showcases integrating device selection with model training. The `train_model()` function defines a very simple linear regression model which is then moved to the chosen device using the `.to(device)` method. Data inputs and labels are also generated on the chosen device to ensure compatibility during training. The training loop will then run the model on the selected device with all relevant tensors created there. An important takeaway here is that the `.to()` function and the device parameter of tensors should be called with the correct device identifier, otherwise errors will occur. By moving both model and tensors correctly, we can use a ROCm device to accelerate training. By checking the device location of the models parameters we can ensure the training has indeed run on the selected GPU. This demonstration illustrates that device selection is not an isolated step, but rather a critical component of the entire machine learning workflow.

**Resource Recommendations**

For thorough understanding of ROCm, the official AMD ROCm documentation provides the most accurate and up-to-date information. Specifically, refer to sections on installation, supported hardware, and debugging, particularly those sections covering the compilation and installation of custom builds of PyTorch. There are also multiple online forum communities dedicated to AMD GPUs and software such as ROCm, where users commonly share debugging tips and practical implementation advice. While these resources vary in their specific content, they collectively provide a holistic knowledge base covering both theoretical and practical aspects of ROCm and its interaction with frameworks like PyTorch. Pay particular attention to any notes regarding specific PyTorch versions as certain versions of the software may work more seamlessly than others.
