---
title: "How do I use TensorFlow with a CPU in Python?"
date: "2025-01-30"
id: "how-do-i-use-tensorflow-with-a-cpu"
---
My primary experience lies in optimizing computational pipelines for deep learning models, frequently encountering the need to execute TensorFlow workloads on CPUs due to resource constraints or rapid prototyping phases. TensorFlow's default configuration often targets GPUs for accelerated computation, but harnessing CPU power is both feasible and sometimes preferable depending on the use case. Specifically, the process revolves around configuring TensorFlow to either exclusively utilize available CPUs or to prioritize them while allowing GPU access as a secondary option.

Fundamentally, TensorFlow's device placement logic determines where operations are executed – GPU or CPU. This placement can be controlled through environment variables, explicit function calls, and configuration settings. By default, if a compatible NVIDIA GPU is detected, TensorFlow will attempt to utilize it. To force TensorFlow to use a CPU, I typically employ one of two strategies: environment variable manipulation or explicit device specification within the code.

**Environment Variable Control:**

The simplest method involves setting the `CUDA_VISIBLE_DEVICES` environment variable to an empty string. This effectively disables TensorFlow's detection of any available GPUs. When running the Python script, TensorFlow, unable to locate a GPU, will fall back to using CPUs. This approach is useful when running many scripts sequentially without wanting to modify the core logic of each script. Consider this scenario:

```python
import os
import tensorflow as tf

# Set the environment variable to disable GPU visibility
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Check the available devices
physical_devices = tf.config.list_physical_devices()
print("Available devices:", physical_devices)

# Simple TensorFlow operation
a = tf.constant(2.0)
b = tf.constant(3.0)
c = a + b
print("Result:", c)

# Check device placement of operations
print("Device of 'a':", a.device)
print("Device of 'b':", b.device)
print("Device of 'c':", c.device)
```

In this example, before importing `tensorflow`, the `CUDA_VISIBLE_DEVICES` environment variable is set. Consequently, TensorFlow will not detect any GPU resources. When listing `physical_devices`, it will only display the available CPUs. Examining the output of `a.device`, `b.device`, and `c.device`, we should observe that these constants and their sum are placed and executed on a CPU device, usually `/device:CPU:0`. When you are conducting initial model architecture experimentation, this tactic allows for quick validation without immediate access to GPU resources.

**Explicit Device Placement with `tf.device()`:**

A more granular approach involves specifying the device within the code using `tf.device()`. This allows fine-grained control over where particular operations or variables are placed. This is crucial when you might want to mix CPU and GPU operations in the same script. I find it most relevant when portions of the processing pipeline are more suitable for one type of processor over the other. For instance, data pre-processing might benefit from CPU-bound libraries while the heavy model calculations use the GPU, although here we will focus on pure CPU processing. Observe the following code:

```python
import tensorflow as tf

# Explicitly define device placement for specific operations
with tf.device('/CPU:0'):
  a = tf.constant(2.0)
  b = tf.constant(3.0)
  c = a + b

print("Result:", c)

# Check device placement of operations
print("Device of 'a':", a.device)
print("Device of 'b':", b.device)
print("Device of 'c':", c.device)


# Another operation outside of the device context, should be on CPU as well
d = tf.constant(4.0)
e = d * 2
print("Device of 'd':", d.device)
print("Device of 'e':", e.device)
```

Here, the code creates a `tf.device('/CPU:0')` scope. Inside this scope, all TensorFlow operations will be placed on the specified CPU. Outside this context, any other operation will still be on the CPU since no GPU was detected by the system. This approach allows for more flexibility by precisely dictating where computations should occur. It prevents the accidental use of the GPU where the explicit intent was CPU-based processing, a common source of confusion during development.

**Configuring TensorFlow with `tf.config.set_visible_devices()`:**

While the previous methods achieve the goal, I've found it often more robust and clearer in the codebase to use `tf.config.set_visible_devices()` which allows to explicitly specify which devices to use, effectively controlling TensorFlow's device visibility similar to the environment variable approach, but without modifying the execution environment outside of the scope of the current script. This technique is particularly useful when you need to have both GPU and CPU execution available, but want to quickly test on only one device type, or when running in an environment where you don't have control over external environment variables. Observe:

```python
import tensorflow as tf

# Get list of physical CPU devices
cpus = tf.config.list_physical_devices('CPU')
print("CPU Devices before filtering: ", cpus)

# Configure TensorFlow to only use the first CPU device
if cpus:
    tf.config.set_visible_devices(cpus[0], 'CPU')
    print("CPU Devices after filtering: ", tf.config.list_physical_devices('CPU'))

# Check the available devices
physical_devices = tf.config.list_physical_devices()
print("Available Devices: ", physical_devices)


# Simple TensorFlow operation
a = tf.constant(2.0)
b = tf.constant(3.0)
c = a + b
print("Result:", c)

# Check device placement of operations
print("Device of 'a':", a.device)
print("Device of 'b':", b.device)
print("Device of 'c':", c.device)

```

This example uses `tf.config.list_physical_devices('CPU')` to obtain a list of available CPUs. Then it sets visible devices to only use the first available CPU in the list. This configuration persists for the duration of the program. Following that,  `tf.config.list_physical_devices()` shows that only the configured CPU is available to TensorFlow. This method provides clear, programmatic device management, preventing accidental use of unintended hardware resources, especially when debugging or collaborating.

**Resource Recommendations:**

For a comprehensive understanding of TensorFlow device placement, consulting the official TensorFlow documentation is essential. Specifically, review the sections pertaining to device management, configuration options, and the functions discussed previously. Additionally, studying examples from the TensorFlow tutorials and GitHub repositories can enhance your ability to configure and utilize CPUs with TensorFlow. Various online learning platforms also offer courses that explore this topic in greater detail with practical exercises. Finally, actively engaging in the TensorFlow community through forums or online help desks can offer answers to specific issues and provide nuanced insights to the community. This combination of studying official resources, reviewing established code examples, formal training, and community engagement, will help you effectively and confidently deploy your workflows.
