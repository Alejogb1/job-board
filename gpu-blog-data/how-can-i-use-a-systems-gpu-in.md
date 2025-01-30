---
title: "How can I use a system's GPU in a Jupyter notebook?"
date: "2025-01-30"
id: "how-can-i-use-a-systems-gpu-in"
---
The efficient utilization of a Graphics Processing Unit (GPU) within a Jupyter Notebook environment typically requires careful configuration and awareness of the underlying libraries and hardware. Having spent considerable time optimizing numerical simulations and machine learning models, I've found that leveraging GPU capabilities is often crucial for achieving practical computation times. The most common approach involves employing libraries specifically designed for GPU acceleration, such as TensorFlow, PyTorch, or cuPy, while also ensuring the necessary CUDA drivers are correctly installed and recognized by the operating system and the kernel running the notebook.

The fundamental challenge lies in the fact that Jupyter notebooks operate within a Python kernel, which by default executes on the CPU. Therefore, explicitly instructing the library being used to perform computations on the GPU is necessary. This is typically achieved through API calls that allocate memory on the GPU and direct data and operations there. Incorrect configuration, missing drivers, or library missteps can result in computations reverting to the CPU without explicit notification, impacting performance dramatically.

Let's consider how this works in practice using a few concrete examples. First, we'll examine TensorFlow. A key concept in TensorFlow is the notion of a “device.” You need to specify that computations should be performed on a specific GPU instead of the CPU. If no GPU is available or drivers aren't properly installed, TensorFlow usually defaults to CPU execution, but this can be misleading. The following code snippet demonstrates explicitly requesting a GPU for computation:

```python
import tensorflow as tf

# Check if a GPU is available
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Attempt to set the first GPU as the active device
    tf.config.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print("Error setting GPU:", e)
else:
    print("No GPUs detected.")


# Define tensors
a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
b = tf.constant([[5.0, 6.0], [7.0, 8.0]])

# Perform matrix multiplication on the GPU if available, otherwise CPU
with tf.device('/GPU:0' if gpus else '/CPU:0'):
    c = tf.matmul(a, b)

print(c)
```
In this example, we first check for available GPUs using `tf.config.list_physical_devices('GPU')`. If GPUs are detected, we attempt to set the first detected GPU as the active device using `tf.config.set_visible_devices()`. This step is crucial because some environments may present multiple GPUs.  The `tf.device()` context manager then directs the matrix multiplication to the GPU if `gpus` has a value, and to the CPU otherwise. The explicit specification within the context manager ensures the operation is directed appropriately. If `tf.device('/GPU:0')` is used without a GPU, TensorFlow will silently revert to the CPU without throwing an error, which can lead to debugging challenges if performance is unexpectedly low. Thus, checking for availability is vital.

Next, let's consider PyTorch. PyTorch's mechanism for GPU utilization differs slightly but shares a similar core principle of moving tensors and computations to a specified device. The following code showcases this:

```python
import torch

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create tensors
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=device)
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]], device=device)


# Perform matrix multiplication
c = torch.matmul(a, b)

print(c)
```

Here, the code explicitly defines a `device` variable which is either “cuda” if a CUDA-enabled GPU is available, or "cpu" otherwise. All tensors are then created with the `device` parameter set appropriately, ensuring the data resides in the correct location. The `torch.matmul` operation subsequently executes on the same device as its inputs. The key difference compared to TensorFlow's approach is the specification of the device when creating the tensors, rather than using a context manager around the operation. Like TensorFlow, if `torch.device("cuda")` is used and CUDA is unavailable, an error will be generated.  `torch.cuda.is_available()` prevents this and gracefully falls back to CPU processing. This explicit device specification ensures predictable execution locations for both data and computations, crucial when optimizing for GPU performance.

Finally, let’s briefly discuss cuPy, a NumPy-compatible library that utilizes the GPU for numerical operations. This is often useful for workloads where the syntax of NumPy is more convenient than the tensor constructs of TensorFlow or PyTorch, and for rapid numerical prototyping.

```python
import cupy as cp
import numpy as np

# Check for GPU availability
try:
    device = cp.cuda.Device(0)
    print(f"Using device: {device}")
except cp.cuda.runtime.CUDARuntimeError as e:
    print(f"No CUDA device found: {e}")
    print("Falling back to NumPy for CPU operations.")
    device = None

# Create arrays
if device:
    a = cp.array([[1.0, 2.0], [3.0, 4.0]])
    b = cp.array([[5.0, 6.0], [7.0, 8.0]])

    # Perform matrix multiplication on the GPU
    c = cp.matmul(a, b)

    # Bring results back to the CPU if required
    c_cpu = cp.asnumpy(c)
    print(c_cpu)
else:
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([[5.0, 6.0], [7.0, 8.0]])
    c = np.matmul(a,b)
    print(c)

```

In the `cupy` code, we attempt to create a CUDA device using `cp.cuda.Device(0)`. If no CUDA device is available (which will raise a CUDARuntimeError), we revert to NumPy arrays. This method mirrors how the other frameworks gracefully handle unavailability. The key aspect here is that `cupy`’s arrays reside on the GPU if a device is available, and the functions in the `cupy` library operate on them directly. If one needs to bring data from GPU back to the CPU as NumPy arrays, the `cp.asnumpy()` call is employed.  This is essential since NumPy only processes data on the CPU. This illustrates the importance of being explicit about where data resides and on which device computations occur. It highlights that these frameworks have a slightly different philosophy regarding handling the CPU/GPU divide.

For additional information, consider consulting the official documentation for TensorFlow, PyTorch, and cuPy. Furthermore, materials related to CUDA driver installation and setup are highly recommended, as this is often the primary point of failure. Specifically, NVIDIA's documentation regarding driver versions and compatibility with CUDA versions is crucial. Several online courses covering deep learning and numerical computation will also touch on GPU utilization, which provides further context and practical examples.  Textbooks focusing on parallel computing, and those from the deep learning field will typically have chapters that can help consolidate the concepts of effective GPU usage. Finally, research papers directly relating to GPU architecture provide valuable insight for advanced optimisation scenarios, when memory layout and processing pipelines need to be carefully aligned with the specifics of your hardware.
