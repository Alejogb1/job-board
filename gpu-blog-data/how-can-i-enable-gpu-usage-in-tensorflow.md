---
title: "How can I enable GPU usage in TensorFlow on an ARM Mac?"
date: "2025-01-30"
id: "how-can-i-enable-gpu-usage-in-tensorflow"
---
The primary challenge in enabling GPU usage with TensorFlow on Apple Silicon arises from the fact that TensorFlow leverages the Metal Performance Shaders (MPS) framework for GPU acceleration on these architectures, rather than NVIDIA's CUDA which is used on traditional GPUs. Unlike CUDA, MPS support is integrated directly within specific TensorFlow builds and is not an external library requiring explicit driver installations. Therefore, the configuration process diverges significantly from that encountered on systems utilizing NVIDIA GPUs.

To illustrate, I recall struggling with this transition myself when I initially migrated my deep learning workflows from a dedicated CUDA-enabled Linux machine to my new M1 MacBook Pro. The error messages I encountered, typically pertaining to device unavailability, were initially perplexing. It became clear that the usual strategies involving `nvidia-smi` or explicit CUDA installations were completely irrelevant. The core issue was a misunderstanding of how Apple had chosen to integrate GPU acceleration.

Fundamentally, enabling GPU usage in TensorFlow on an ARM Mac necessitates three critical elements: a compatible TensorFlow build, proper device selection within the TensorFlow environment, and verification that the intended computations are indeed being offloaded to the GPU. The lack of any of these components will default to CPU execution, significantly impacting performance for deep learning operations.

Firstly, the installed TensorFlow package *must* be the `tensorflow-metal` variant, which incorporates MPS support. A standard CPU-only TensorFlow build, common for x86 architectures, will not utilize the GPU hardware even if the underlying system is capable. This requires a targeted installation process. Often, users accustomed to `pip install tensorflow` find that their system isn’t engaging the GPU. Thus, the specific command `pip install tensorflow-metal` (or equivalent with conda environments) is mandatory. This package provides the underlying code required to interact with the Metal framework. Without it, TensorFlow will only see the CPU.

Second, simply having `tensorflow-metal` installed does not automatically direct computations to the GPU. TensorFlow must be explicitly instructed to utilize the available MPS device. Within a TensorFlow session, the `tf.config.list_physical_devices('GPU')` function serves to identify available GPU devices. If this returns an empty list, it suggests that either the proper TensorFlow package has not been installed, or the system is failing to recognize the GPU for an unknown reason. Assuming a GPU is recognized, subsequent operations must be placed on the designated device, commonly achieved through either the `tf.device()` context manager or by explicitly specifying the device during tensor creation and computation. Neglecting this step is a common pitfall, even after correctly installing the package.

Thirdly, verification of successful GPU usage is paramount. Relying on the assumption that computations are accelerated is risky. The only effective approach is to monitor execution behavior, and profiling. Tools such as Apple's Instruments application can provide real-time insights into hardware utilization during TensorFlow operations. Additionally, within TensorFlow itself, performance metrics from tools like `tf.profiler.experimental.start` and `tf.profiler.experimental.stop` can shed light on whether computations are being performed on the CPU or the GPU. Slowdown when not using the GPU can be dramatic, leading to a substantial performance degradation.

Here are three illustrative code examples that highlight key concepts:

**Example 1: Identifying and utilizing the available GPU device.**

```python
import tensorflow as tf

# Check for available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU devices found:", gpus)
    # Assign calculations to the first GPU, if available
    gpu_device = gpus[0]
    with tf.device(gpu_device.name):
      a = tf.random.normal((1000, 1000))
      b = tf.random.normal((1000, 1000))
      c = tf.matmul(a,b)
      print(c)

else:
    print("No GPUs found. Using CPU.")
    # If no GPU found, execution on CPU (as fall back)
    a = tf.random.normal((1000, 1000))
    b = tf.random.normal((1000, 1000))
    c = tf.matmul(a,b)
    print(c)
```
*Commentary:* This snippet first checks if any GPUs are visible to TensorFlow. If found, it establishes a scope using `tf.device()` where subsequent tensor operations will be preferentially executed on the identified GPU. If no GPU is present, it defaults to a CPU-based computation. The `tf.random.normal` and `tf.matmul` operations represent typical tensor operations commonly encountered in machine learning. The output will differ from the output seen when not assigning operations to a GPU device, and performance will be severely decreased if done on a CPU.

**Example 2: Explicit device placement during tensor creation.**

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

if gpus:
    gpu_device = gpus[0]
    # Create the tensor directly on the GPU
    a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32, device = gpu_device.name)
    b = tf.constant([4.0, 5.0, 6.0], dtype=tf.float32, device = gpu_device.name)

    c = a + b
    print(c)


else:
   # Create on the CPU
   a = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
   b = tf.constant([4.0, 5.0, 6.0], dtype=tf.float32)
   c = a + b
   print(c)
```
*Commentary:* This example shows an alternative approach.  Instead of relying on a `tf.device()` scope, the `device` parameter is used at the point of tensor creation. By directly specifying the GPU during the instantiation of the tensors `a` and `b`, these tensors, and any related computations, will be executed on the GPU. The result, `c`, is a tensor computed on the GPU. Similarly to Example 1, if no GPUs are found, the example falls back to CPU processing. The performance impact when not assigning this to a GPU will be substantial.

**Example 3: Profiling GPU execution and performance comparison**

```python
import tensorflow as tf
import time

gpus = tf.config.list_physical_devices('GPU')

if gpus:
   gpu_device = gpus[0]

   # Start profiler for GPU execution
   tf.profiler.experimental.start('gpu_logdir')

   start_time = time.time()
   with tf.device(gpu_device.name):
      a = tf.random.normal((10000, 10000))
      b = tf.random.normal((10000, 10000))
      c = tf.matmul(a,b)

   end_time = time.time()

   gpu_time = end_time - start_time
   print("GPU matmul time: " , gpu_time)


   tf.profiler.experimental.stop()



   start_time = time.time()
   a = tf.random.normal((10000, 10000))
   b = tf.random.normal((10000, 10000))
   c = tf.matmul(a,b)

   end_time = time.time()
   cpu_time = end_time - start_time
   print("CPU matmul time: ", cpu_time)

   # Time comparison
   print("CPU/GPU execution time ratio: ", cpu_time/gpu_time)

else:
  print("No GPUs found, cannot profile. Defaulting to CPU")
  start_time = time.time()
  a = tf.random.normal((10000, 10000))
  b = tf.random.normal((10000, 10000))
  c = tf.matmul(a,b)
  end_time = time.time()

  cpu_time = end_time - start_time
  print("CPU matmul time: ", cpu_time)


```

*Commentary:* This snippet showcases profiling capabilities using `tf.profiler`, showing an example of measuring the execution time for large matrix multiplication both with and without GPU support. Profiling is crucial in verifying proper usage and identifying potential bottlenecks. This provides a real benchmark of the benefits of GPU acceleration. The ratio printed at the end highlights how much faster operations can be on a GPU as opposed to on the CPU. In this example, if no GPU is found, only a CPU matmul operation time is printed and the other tests are skipped. This should provide a demonstration for how much slower operations can be without a GPU.

For a thorough understanding of TensorFlow and GPU acceleration on Apple Silicon, I would recommend exploring several excellent resources. Start by studying Apple’s developer documentation regarding the Metal Performance Shaders framework, which provides a foundational understanding of the underlying technology. Secondly, the TensorFlow documentation itself has specific sections dedicated to MPS support, providing guidance on installation and configuration. Finally, examining the community support forums and knowledge bases can yield valuable troubleshooting tips and workarounds. These resources will provide the required expertise to effectively implement GPU acceleration in TensorFlow.

In closing, while the transition from traditional CUDA-based GPU programming to Metal on Apple Silicon may initially appear daunting, a careful adherence to the specific installation steps, proper device selection, and verification processes will allow you to take full advantage of the computational power of these systems. The examples above can serve as a useful starting point to accomplish this goal.
