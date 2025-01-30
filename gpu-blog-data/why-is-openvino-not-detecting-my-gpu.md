---
title: "Why is OpenVINO not detecting my GPU?"
date: "2025-01-30"
id: "why-is-openvino-not-detecting-my-gpu"
---
OpenVINO’s inference engine relies on explicit configuration and driver compatibility for GPU acceleration, and if the expected GPU device isn't being utilized, the issue typically lies within a few common areas. In my experience debugging this across various embedded and server deployments, I've found that proper environment setup, correct device specification, and appropriate driver versions are critical. Incorrectly configuring these elements often leads the inference engine to default to the CPU, regardless of the presence of a capable GPU. This response details the most likely culprits and how to address them.

The primary reason OpenVINO might fail to detect your GPU stems from the device specification when loading the model. OpenVINO needs to be explicitly told to use the GPU, and it doesn't automatically default to it just because one is present. This specification takes the form of a device string, usually passed to the `core.compile_model()` function. If this string is absent or set to a different device such as 'CPU' or 'AUTO', the inference engine will not attempt to load the model onto the GPU, even if GPU hardware and compatible drivers are installed. Furthermore, different types of GPUs require distinct device strings (e.g. 'GPU' for Intel Integrated graphics, 'GPU.0', 'GPU.1' for multi-GPU configurations or 'GPU.1.0', 'GPU.1.1' for a specific sub-device of a GPU, etc). Mistyping or using an incorrect string based on the target GPU is a common error. For example, 'GPU' alone might not suffice if the system has multiple GPUs, or if you intend to use a discrete GPU on a system with integrated graphics, which may require explicitly specifying the numerical identifier of the target GPU.

Another significant factor is the OpenVINO Runtime package itself. The correct runtime must be installed, and it must be compatible with the installed GPU drivers. A common error I’ve encountered involves inadvertently installing the CPU-only version of OpenVINO. The GPU drivers for Intel, NVIDIA or other vendors are not bundled into the core OpenVINO install and must be downloaded, installed, and often require explicit configuration via environmental variables to be recognized by OpenVINO. Without a GPU-supporting OpenVINO package and the appropriate drivers installed for the target GPU, the inference engine will fall back to the CPU. Furthermore, mismatches in the versions of the OpenVINO package and the GPU drivers can cause detection issues. For example, a recent update to the GPU driver might break compatibility with an older version of OpenVINO, requiring either the driver to be rolled back or OpenVINO to be updated. A good practice I follow is to test the driver versions against the Intel official compatibility guides to make sure I’m running a supported configuration.

Beyond the device specification and runtime configuration, the way model optimizations are handled can also lead to GPU detection failures. Certain model types or configurations are optimized for specific hardware, and if the compiled model is not targeted for the available GPU, it can fall back to the CPU execution. OpenVINO supports different inference precisions (e.g. FP32, FP16, INT8) which affect GPU compatibility. A model compiled for a precision not supported on the available GPU can lead to issues. For instance, some lower-end GPUs don’t support lower precision levels, and attempting to run a model with a precision these GPUs don’t support can cause OpenVINO to bypass the GPU, effectively defaulting to CPU inference. Furthermore, the `ovc` tool, used for converting and optimizing models to the OpenVINO IR format, needs the target device specified, especially when running custom optimizations targeted to a particular GPU architecture. A mismatch in targeted device during optimization and actual execution often leads to the described failure. I’ve found this especially true when moving across generations of GPUs, where an old model, still optimized for the old hardware, may fail to run on the latest available generation.

Here are three code examples that demonstrate the importance of device specification.

**Example 1: Basic Device Selection**

```python
from openvino.runtime import Core

core = Core()

# List available devices - this helps diagnose if the GPU is seen at all.
available_devices = core.available_devices
print(f"Available devices: {available_devices}")

# Incorrect device specification (defaults to CPU or AUTO)
# This will almost certainly NOT use the GPU, even if available.
model_path = 'path/to/your/model.xml'
compiled_model = core.compile_model(model_path)

# Correct device specification. Note: Use actual device string
# such as "GPU" or "GPU.0" based on your hardware and configuration
compiled_model_gpu = core.compile_model(model_path, device_name="GPU")

# Code to execute the model using the correct compiled_model_gpu model

```

In this example, the `available_devices` check is crucial to confirm whether OpenVINO has found the GPU, or if it only reports the CPU. The code highlights the difference between a generic model compilation which will likely default to CPU and an explicit GPU compilation using the `device_name` parameter.  If the available devices do not include a GPU the problem lies with system configuration, most likely the installation of the GPU drivers or OpenVINO runtime itself. The device names "GPU", "GPU.0" etc. will need to be adjusted for the target hardware and configuration.

**Example 2: Handling Multiple GPUs**

```python
from openvino.runtime import Core

core = Core()

# List available devices
available_devices = core.available_devices
print(f"Available devices: {available_devices}")

# Assuming multiple GPUs are detected (e.g., ['CPU', 'GPU.0', 'GPU.1'])
# The following allows to chose a specific GPU

#  Use the specific GPU ID:
if "GPU.1" in available_devices:
    compiled_model_gpu = core.compile_model("path/to/your/model.xml", device_name="GPU.1")
else:
    print("GPU.1 not found, using default device")
    compiled_model_gpu = core.compile_model("path/to/your/model.xml")

# Code to execute the model
```

This code snippet is particularly useful when dealing with systems that have multiple GPUs. It shows how to explicitly target a specific GPU device by its identifier. If 'GPU.1' is not available, the code gracefully falls back to the default behavior, but in a real application, a proper error would be raised. Checking `available_devices` is again important to ensure the specified GPU ID is available. Using the wrong ID for the target device will lead to inference on the CPU, or to outright errors if the user requests an unsupported identifier.

**Example 3: Checking Optimization Configurations**

```python
from openvino.runtime import Core
from openvino.runtime import properties

core = Core()
model_path = 'path/to/your/model.xml'

# Query available properties on a device
gpu_properties = core.get_property("GPU", "SUPPORTED_PROPERTIES")
print(f"GPU supported properties: {gpu_properties}")

# Check if the GPU supports the desired precision
if properties.device.capability.FP16_SUPPORTED in core.get_property("GPU", "SUPPORTED_PROPERTIES"):
    compiled_model_gpu = core.compile_model(model_path, device_name="GPU", config = {"INFERENCE_PRECISION_HINT":"f16"})
else:
    print("FP16 not supported on target GPU, defaulting to FP32")
    compiled_model_gpu = core.compile_model(model_path, device_name="GPU")
    
# Code to execute the model
```

This third example delves into optimization considerations. Specifically, it queries and checks if the target GPU supports FP16 inference precision, before requesting it. It will print the supported properties to standard output for further investigation. If FP16 is not supported, the code defaults to the FP32 precision. In a production system the precision would be chosen based on the target hardware and the required performance targets.  If there is a mismatch between a compiled model targeted precision, and the supported precision on the GPU, the inference can default to CPU execution, even when the user expects GPU acceleration.

To diagnose and resolve OpenVINO GPU detection issues, I would recommend consulting several resources. First, the official Intel OpenVINO documentation provides detailed explanations about device specification, supported precisions, optimization guides and trouble-shooting workflows. Pay special attention to their documentation regarding the "OpenVINO device plugin" architecture, and how it maps to your particular hardware. Furthermore, checking the specific GPU driver documentation often provides guidance regarding what configuration is necessary to correctly configure OpenVINO for target hardware.  Lastly, consulting the forums and community discussions associated with your vendor for OpenVINO may provide solutions or workarounds related to the specific hardware in use, especially concerning specific versions of the hardware drivers, the OpenVINO runtime packages and the operating system in use.
