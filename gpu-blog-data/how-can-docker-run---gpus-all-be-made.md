---
title: "How can `docker run --gpus all` be made to use the CPU if no GPUs are available?"
date: "2025-01-30"
id: "how-can-docker-run---gpus-all-be-made"
---
The `docker run --gpus all` command, while seemingly straightforward, exhibits brittle behavior in the absence of suitable GPU resources.  My experience troubleshooting GPU-dependent Docker deployments across diverse hardware configurations – from embedded systems with integrated graphics to high-performance computing clusters – highlights a fundamental limitation: this command, by design, prioritizes GPU allocation and fails gracefully only to a limited extent.  It does *not* automatically fallback to CPU execution.  This necessitates a more sophisticated approach involving conditional logic within the Docker image itself.

The core issue lies in how Docker interacts with the underlying CUDA/OpenCL runtime environments. When `--gpus all` is specified, the container attempts to access all available GPUs.  If none are detected, the container startup will either fail completely, or, depending on the Docker daemon and driver configuration, result in an application hanging indefinitely while awaiting unavailable resources.  A successful fallback requires the application logic to explicitly check for GPU availability and gracefully degrade to CPU processing.

This necessitates a two-pronged strategy: (1) building an image that can dynamically adapt to its runtime environment, and (2) employing a mechanism within the application itself to check for GPU presence and select an appropriate execution path.

**1.  Conditional Execution within the Docker Image:**

The most robust solution involves embedding logic within the Docker image to detect GPU presence at runtime and select the appropriate execution path. This avoids relying on external tools or scripts and enhances portability.  This typically involves using system calls or environment variables to detect hardware.

**2. Code Examples and Commentary:**

The following examples demonstrate different approaches, using Python (a popular choice for its versatility and readily available libraries) as a representative language.  Remember to adapt these examples to your specific application and libraries.

**Example 1:  Using `nvidia-smi` (Requires NVIDIA driver and utilities):**

```python
import subprocess
import os

try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    gpu_available = True
    gpu_name = result.stdout.strip()
    print(f"GPU detected: {gpu_name}")
except subprocess.CalledProcessError:
    gpu_available = False
    print("No GPU detected.")

if gpu_available:
    # Execute GPU-accelerated code here
    os.environ["CUDA_VISIBLE_DEVICES"] = "0" #Or a more sophisticated selection logic
    import my_gpu_module
    my_gpu_module.run_gpu_computation()
else:
    # Execute CPU-based code here
    import my_cpu_module
    my_cpu_module.run_cpu_computation()
```

This example leverages the `nvidia-smi` command to check for GPU presence.  The `check=True` argument ensures an exception is raised if the command fails, indicating the absence of a GPU.  The `CUDA_VISIBLE_DEVICES` environment variable restricts CUDA to utilize only the specified GPU(s); setting it to "" effectively disables CUDA.


**Example 2: Checking for CUDA-related environment variables:**

```python
import os

if "CUDA_VISIBLE_DEVICES" in os.environ and os.environ["CUDA_VISIBLE_DEVICES"]:
    gpu_available = True
    print("GPU detected (CUDA environment variable set).")
    # Execute GPU-accelerated code here
    import my_gpu_module
    my_gpu_module.run_gpu_computation()
else:
    gpu_available = False
    print("No GPU detected (CUDA environment variable not set).")
    # Execute CPU-based code here
    import my_cpu_module
    my_cpu_module.run_cpu_computation()
```

This is a simpler approach, relying on the presence of the `CUDA_VISIBLE_DEVICES` environment variable, commonly set by CUDA-aware applications and container orchestration tools.  The absence of the variable or its being set to an empty string suggests no GPU is available or usable.


**Example 3: Using a dedicated library (e.g., TensorFlow):**

Frameworks like TensorFlow often provide built-in mechanisms for checking GPU availability.

```python
import tensorflow as tf

if tf.config.list_physical_devices('GPU'):
    gpu_available = True
    print("GPU detected (TensorFlow detected GPU devices).")
    # Execute GPU-accelerated TensorFlow code here
    with tf.device('/GPU:0'):  # or a more sophisticated device selection
        # TensorFlow operations
        pass
else:
    gpu_available = False
    print("No GPU detected (TensorFlow found no GPU devices).")
    # Execute CPU-based TensorFlow code here
    with tf.device('/CPU:0'):
        # TensorFlow operations
        pass
```

This leverages TensorFlow's internal device detection capabilities, offering a more integrated and framework-specific solution. It's crucial to install the necessary CUDA libraries and configure TensorFlow accordingly if GPU usage is intended.


**3. Resource Recommendations:**

For in-depth understanding of CUDA programming, consult the NVIDIA CUDA C++ Programming Guide.  For efficient Python-based GPU programming, familiarize yourself with the documentation of relevant libraries such as PyTorch or TensorFlow.  Detailed information on Docker's GPU support is provided in the official Docker documentation.  Finally, explore system administration guides pertaining to your specific Linux distribution for managing GPU drivers and resource allocation.  These materials will offer practical guidance and advanced techniques beyond the scope of this concise response.
