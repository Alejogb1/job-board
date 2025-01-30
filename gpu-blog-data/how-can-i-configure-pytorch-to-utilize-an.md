---
title: "How can I configure PyTorch to utilize an AMD GPU?"
date: "2025-01-30"
id: "how-can-i-configure-pytorch-to-utilize-an"
---
The practical challenge of utilizing an AMD GPU with PyTorch often stems from the distinct hardware and software ecosystem surrounding AMD's ROCm (Radeon Open Compute) platform compared to NVIDIA’s CUDA. Over the course of various projects, I’ve found that proper configuration hinges on a few key components: the right drivers, a compatible PyTorch build, and careful attention to environment variables. Failing in any of these areas typically leads to PyTorch defaulting to CPU usage, negating the benefits of GPU acceleration.

The primary point to understand is that unlike CUDA, which is tightly integrated with NVIDIA’s GPUs, ROCm is an open-source platform. This means the installation process isn't always as streamlined and may require more manual configuration. Furthermore, ROCm support across different PyTorch versions and operating systems varies, necessitating diligence in selecting the appropriate setup for your system.

The initial step is verifying ROCm compatibility with your specific AMD GPU. Not all AMD GPUs are supported by ROCm; typically, it's the higher-end discrete GPUs. The ROCm documentation provides an exhaustive list, which should be consulted first. Attempting to use an unsupported GPU will predictably fail. Also, the supported operating systems are often specific Linux distributions; support for Windows is limited and often experimental. For a stable development environment, a ROCm-supported Linux distribution is highly recommended.

Following hardware compatibility, the correct ROCm drivers need installation. These drivers are distinct from the standard graphics drivers provided by AMD and are essential for ROCm functionality. The specific method of driver installation depends heavily on your Linux distribution. Distributions like Ubuntu often require adding AMD's repositories and installing specific packages. It’s imperative to follow the AMD ROCm installation guide precisely for your OS and GPU, as inconsistencies during this step often cause downstream issues.

With the drivers correctly installed, the focus shifts to PyTorch itself. PyTorch wheels built specifically for ROCm are necessary to enable GPU acceleration. These wheels aren't always easily available through the standard pip index; they might have to be sourced from the PyTorch or AMD website, depending on your PyTorch version. Using a non-ROCm build, even if the drivers are correct, will force PyTorch to utilize the CPU.

Once the drivers and the appropriate PyTorch version are installed, environment variables come into play. Certain variables need configuration so that PyTorch can detect and properly use the available ROCm device. Specifically, `HSA_OVERRIDE_GFX_VERSION` and `ROCR_VISIBLE_DEVICES` variables are important for explicit device control. `HSA_OVERRIDE_GFX_VERSION` specifies the GPU architecture version, which can often be found using AMD system tools, and `ROCR_VISIBLE_DEVICES` lets you select the visible GPU devices for computation. In my experience, these variables are often a source of errors if left improperly configured. I'll now illustrate this setup with code examples.

**Code Example 1: Verification of ROCm Installation**

```python
import torch

def check_rocm():
    if torch.cuda.is_available():
        print("CUDA is available, but this is likely not what you want with AMD.")
        print("Check if ROCm was installed as well, as a driver could be pointing to the wrong device")
        return False

    if torch.version.hip is not None:
        print("PyTorch compiled with ROCm support detected.")
        device_count = torch.cuda.device_count()
        print(f"Number of available ROCm devices: {device_count}")
        if device_count > 0:
          for i in range(device_count):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
          print("ROCm is correctly configured and usable.")
          return True
        else:
          print("No ROCm devices found, although ROCm compilation is enabled. Check driver installation and variable settings")
          return False
    else:
        print("PyTorch is not compiled with ROCm support. Check PyTorch install")
        return False

if __name__ == "__main__":
    check_rocm()
```

This code snippet provides an initial verification step. It checks whether PyTorch was compiled with ROCm support (`torch.version.hip is not None`), then verifies that ROCm devices are detected. It avoids relying solely on CUDA, because systems can sometimes report that CUDA is available when ROCm is installed alongside it (although, if you're expecting ROCm, it's likely that a CUDA driver is also available). If a device is found it lists the number of devices and their names. If no ROCm devices are found, it flags the issue with specific instructions. This code is essential before executing any machine learning tasks to ensure the setup is functional.

**Code Example 2: Setting Environment Variables Programmatically**

```python
import os
import torch

def configure_rocm_environment():

    if os.environ.get("HSA_OVERRIDE_GFX_VERSION") is None:
        print("HSA_OVERRIDE_GFX_VERSION is not set. Set a suitable GPU version.")
        #This would normally be dynamically set or taken from user input
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0" #example
    else:
        print("HSA_OVERRIDE_GFX_VERSION variable is already configured")

    if os.environ.get("ROCR_VISIBLE_DEVICES") is None:
        print("ROCR_VISIBLE_DEVICES is not set. Setting to all visible devices")
        num_gpus = torch.cuda.device_count()
        os.environ["ROCR_VISIBLE_DEVICES"] = ','.join(map(str, range(num_gpus))) if num_gpus > 0 else ""
        print(f"ROCR_VISIBLE_DEVICES set to {os.environ.get('ROCR_VISIBLE_DEVICES')}")
    else:
        print("ROCR_VISIBLE_DEVICES variable is already configured")

if __name__ == "__main__":
  configure_rocm_environment()
  print("Environment variables check complete")
```

This code shows how to programmatically set the two critical environment variables. `HSA_OVERRIDE_GFX_VERSION` needs to match your GPU's architectural version. The example uses 10.3.0 as an example which should be replaced with the actual value for your setup. This is important as it often leads to issues if left unset. The `ROCR_VISIBLE_DEVICES` variable dynamically detects the visible devices via `torch.cuda.device_count()` and then sets it up for the system. It makes it simple to re-execute the script if more GPUs are added, as it will dynamically determine the devices present. The script verifies if these variables exist and only sets the values if they are not already present.

**Code Example 3: Basic PyTorch Operation on ROCm GPU**

```python
import torch

def run_rocm_operation():
  try:
    if torch.cuda.is_available():
      device = torch.device("cuda:0")
      print(f"Using device: {torch.cuda.get_device_name(0)}")
      tensor = torch.rand(2, 3).to(device)
      result = tensor @ tensor.T
      print(f"Result: {result}")

    else:
      print("ROCm device not found. Ensure device is available, and environment variables are set correctly.")
  except RuntimeError as e:
      print(f"Error during ROCm operation: {e}")
if __name__ == "__main__":
  run_rocm_operation()
```

This final example demonstrates a very simple matrix multiplication operation on the ROCm device. The `torch.device("cuda:0")` call selects the first available ROCm device for computation. It then creates a random tensor, moves it to the ROCm device, performs the multiplication, and prints the result. If no device is found, it prints an error message and reports the exception if the operation was unsuccessful. Successful execution of this code snippet implies that PyTorch is functioning correctly with the AMD GPU.

For further exploration of the topic, it's important to consult several crucial resources. First, the AMD ROCm documentation is the definitive source for specific installation and configuration details related to your hardware and operating system. Pay close attention to the installation instructions and compatibility charts. Second, the PyTorch website is essential for obtaining the specific PyTorch wheels compiled with ROCm support; navigate to the 'Previous versions' page to find past versions. Lastly, the official ROCm support forums are invaluable for addressing specific issues or errors you encounter in your setup and can be a good source for troubleshooting unique scenarios. Through a structured approach using these core principles, AMD GPU acceleration with PyTorch is highly achievable.
