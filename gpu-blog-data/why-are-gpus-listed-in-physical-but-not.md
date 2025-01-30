---
title: "Why are GPUs listed in physical but not logical devices in TensorFlow?"
date: "2025-01-30"
id: "why-are-gpus-listed-in-physical-but-not"
---
TensorFlow’s device placement strategy prioritizes physical devices for computational resource allocation, which explains why GPUs are often explicitly referenced at the physical device level but not typically at a logical device level.  I’ve seen this confusion first-hand working on distributed training setups across multiple GPU servers. This design choice stems from the fundamental way TensorFlow interacts with hardware, aiming for direct control and optimal resource utilization.

**Understanding Physical and Logical Devices**

In TensorFlow, a *physical device* represents an actual hardware component capable of computation, such as a specific CPU core or a particular GPU card. These devices are concrete and directly tied to the underlying system architecture.  Each physical device is identified by a string, like `/physical_device:CPU:0` or `/physical_device:GPU:0`, which signifies the specific hardware instance. The operating system and low-level drivers manage these physical devices.

*Logical devices*, on the other hand, are abstract representations of computational resources within TensorFlow’s runtime. They don't directly correspond to a specific hardware entity but rather to a slice of the available hardware capacity.  Logical devices can be created by partitioning a single physical device to increase flexibility or for resource isolation. For example, you can create multiple logical GPUs from a single physical GPU for managing specific tasks. TensorFlow uses these logical devices to orchestrate computations without needing to reference physical hardware directly in every graph execution.

**Why GPUs are Primarily Listed as Physical Devices**

The discrepancy you observe arises from TensorFlow’s initialization process and the mechanisms through which it interacts with GPUs. Unlike CPUs, which can be easily abstracted into logical cores, GPUs necessitate a more direct form of control because of their highly specialized architecture.  GPUs need to be explicitly initialized and have their memory managed directly.

When TensorFlow starts, it queries the available hardware and registers all the physical devices. For GPUs, this typically involves interacting with the CUDA or ROCm driver API. Because GPU resources like memory are limited, TensorFlow needs to know the precise physical location of a GPU to properly allocate and manage its resources.

During typical operations, logical devices are the primary abstraction layer for computations. For instance, when creating a variable or running a computation, TensorFlow will usually allocate to a logical device – perhaps one that has been created automatically or specified by the user. However, behind the scenes, the allocation of that logical device will be mapped to a particular physical device to perform the underlying computation. While you can configure TensorFlow to create logical GPUs using API calls, TensorFlow typically manages these based on the physical GPUs that are discovered initially.

**Code Examples**

I'll illustrate this concept with a few scenarios:

**Example 1: Identifying Physical Devices**

```python
import tensorflow as tf

physical_devices = tf.config.list_physical_devices()
print("Physical Devices:")
for device in physical_devices:
    print(device)

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    print("GPUs Available:")
    for gpu in gpus:
      print(gpu)
      print(f"Device Name: {gpu.name}")
```

**Commentary:** This code snippet directly accesses physical devices. It uses `tf.config.list_physical_devices()` to retrieve a list of all physical computational units. We iterate through the list and print out the names for each physical device. Notably, this code explicitly looks for GPU devices which will appear in the physical device list. This is the level at which TensorFlow first encounters and identifies a GPU. The print out of the device's name shows a string such as `/physical_device:GPU:0` or `/physical_device:GPU:1`, demonstrating the physical device naming convention. This step is important before any task can be assigned to a GPU.

**Example 2: Creating Logical GPUs**

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Create two logical GPUs from the first physical GPU
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096),
         tf.config.LogicalDeviceConfiguration(memory_limit=4096)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print("Logical GPUs Created:")
    for gpu in logical_gpus:
      print(gpu)
      print(f"Device Name: {gpu.name}")

  except RuntimeError as e:
      # Virtual devices must be set before GPUs have been initialized
    print(e)
```

**Commentary:** This code demonstrates the creation of logical GPUs from a physical GPU. It first retrieves the list of physical GPUs. Then it configures the first physical GPU to allocate two logical GPUs, each with a memory limit of 4096MB.  After configuring the logical devices the `tf.config.list_logical_devices('GPU')` function is used to display the newly created logical GPUs. The names of these logical devices will likely be of the form `/device:GPU:0` or `/device:GPU:1`, reflecting their use within TensorFlow’s execution graph. The important takeaway is that the logical GPUs are created based on the configuration that references the underlying physical GPU. The runtime error check also shows that logical device configuration must happen before GPUs are initialized. The code has to occur directly after the list of physical devices is initialized.

**Example 3: Running computations on Logical Devices**

```python
import tensorflow as tf
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Create one logical GPU from the first physical GPU
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=4096)])

    logical_gpus = tf.config.list_logical_devices('GPU')

    if logical_gpus:
      with tf.device(logical_gpus[0].name):
          a = tf.constant(np.random.rand(10,10))
          b = tf.constant(np.random.rand(10,10))
          c = tf.matmul(a,b)
          print(c)

  except RuntimeError as e:
    print(e)
```

**Commentary:** This code illustrates how to execute a computation on a logical GPU. It first retrieves physical GPUs and configures a logical GPU from the first physical GPU (with a 4096 memory limit). A tensor operation (matrix multiplication) is executed within a `tf.device` scope, specifying the first logical GPU as the execution target. This demonstrates how computations are tied to logical devices, which in turn are associated with physical resources at a lower level. The output of this code is the matrix multiplication result. Although we are targeting a logical device, TensorFlow is ultimately utilizing the underlying physical device to perform the computation. We don't need to explicitly reference the physical device itself during graph construction.

**Resource Recommendations**

For deeper understanding of TensorFlow’s device management, I would suggest consulting TensorFlow's official documentation concerning device placement, device configurations, and distributed training strategies. Several online courses and textbooks covering deep learning with TensorFlow often include sections explaining hardware utilization and device management.  Additionally, examining the source code of TensorFlow itself, specifically the `tensorflow/core/common_runtime` directory, will expose the low-level implementations of device interaction. Look into discussions on forums and developer communities for specific problems.
