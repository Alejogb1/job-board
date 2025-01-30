---
title: "How should visible GPU devices be handled?"
date: "2025-01-30"
id: "how-should-visible-gpu-devices-be-handled"
---
The effective management of visible GPU devices, particularly in multi-GPU environments, hinges on a clear understanding of resource allocation and the drivers that facilitate access. It’s not simply a matter of selecting a GPU; it involves ensuring each process or task interacts with its allocated device without conflict, maximizing performance, and handling potential device failures gracefully. Over my years developing high-performance computing applications and deep learning models, I've observed the pitfalls of ignoring this aspect and the considerable benefits of proper GPU device handling.

The primary mechanism for controlling visible GPUs is often through environment variables and API calls specific to the compute framework or operating system being used. Typically, frameworks like TensorFlow and PyTorch build their abstraction layers atop the drivers, which in turn expose device identifiers. These identifiers can then be used to target computations towards specific GPUs. Crucially, the 'visibility' we’re addressing isn’t just a matter of identifying the GPU’s presence; it’s about dictating which devices are accessible to a particular process. Failure to adequately control visibility often leads to processes fighting over the same resources or, even worse, utilizing the wrong device resulting in suboptimal performance and potential out-of-memory errors. Proper handling, therefore, dictates not only identification but intentional selection and allocation.

One common approach to managing GPU device visibility utilizes the `CUDA_VISIBLE_DEVICES` environment variable (or analogous variables for other compute APIs, such as ROCm). This variable accepts a comma-separated list of device indices. By setting it prior to launching a process, you effectively restrict which GPUs that process can ‘see’ and utilize. I've employed this extensively when needing to concurrently run multiple instances of training or inference jobs each assigned to a different GPU on a server. Omitting the variable or leaving it undefined implies that all available devices are visible. This presents a significant risk of unintentional resource contention, especially when running multiple processes that utilize GPUs simultaneously. Careful and deliberate setting of this environment variable is crucial for resource management.

Let’s consider some specific examples. Suppose I have a server with four GPUs, indexed 0 through 3.

**Example 1: Running a training script on GPU 1.**

```python
import os
import torch

# Force process to only see GPU 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if torch.cuda.is_available():
    print(f"CUDA is available. Device name: {torch.cuda.get_device_name(0)}")
    device = torch.device("cuda:0")
    # Training code here, it is now utilizing GPU 1
    tensor = torch.randn(1000, 1000).to(device)
    result = torch.matmul(tensor, tensor)
    print(f"Successfully performed matrix multiplication on GPU: {torch.cuda.current_device()}")

else:
    print("CUDA is not available.")
```

In this Python snippet utilizing PyTorch, I've explicitly set `CUDA_VISIBLE_DEVICES` to "1" before any PyTorch CUDA operations. This directs the process to only recognize GPU 1 as available. Even though the server has multiple GPUs, the framework will only ‘see’ and utilize the device at index 0. Note that the indexing within the script starts at `0` representing the first and only *visible* device, despite the underlying machine's GPU index. The printing of the device name is also illustrative of the fact that the process sees GPU 1 as the available GPU 0. This is important to grasp as errors commonly originate due to a misunderstanding between the physical indices and the perceived device indices within a process.

**Example 2: Running inference on GPU 0 and 2 in parallel (using subprocesses in bash).**

```bash
# Run inference on GPU 0
CUDA_VISIBLE_DEVICES=0 python inference_script.py &

# Run inference on GPU 2
CUDA_VISIBLE_DEVICES=2 python inference_script.py &

wait
```
Here, I've used bash shell commands to illustrate running inference on distinct GPUs by launching two separate processes, each with restricted visibility. The `&` symbol allows the two processes to run in the background concurrently. The `wait` command ensures that the shell script waits for both background processes to finish before exiting. This pattern can be expanded for numerous concurrently run processes each targetting specific GPUs. This approach facilitates parallel execution of tasks that could otherwise saturate a single device leading to performance bottlenecks.

**Example 3: Checking for available visible GPUs in TensorFlow.**

```python
import os
import tensorflow as tf

#Force process to only see GPU 1 and 3
os.environ["CUDA_VISIBLE_DEVICES"] = "1,3"

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  print("Visible GPUs:")
  for i, gpu in enumerate(gpus):
    print(f"GPU {i}: {gpu.name}")
    with tf.device(gpu.name):
        matrix = tf.random.normal((1000, 1000))
        result = tf.matmul(matrix, matrix)
        print(f"Successfully performed matrix multiplication on visible GPU {i}")

else:
  print("No visible GPUs were detected")
```

This Python script utilizes TensorFlow to illustrate how to retrieve and utilize visible GPUs after setting the environment variable. Here, I've specified that the process should see only GPUs 1 and 3.  TensorFlow then enumerates the visible GPUs and provides their names and the corresponding operations have been run. This demonstrates the consistent use of visible indices for device selection, even if the machine contains additional, non-visible GPUs. In this example, the visible GPUs will be indexed as 0 and 1 respectively for the script despite being 1 and 3 in the machine.

It's also essential to note the importance of error handling when working with GPUs. A program should be equipped to handle the situation where a requested GPU isn’t available due to incorrect device selection. For example, if `CUDA_VISIBLE_DEVICES` is set to an index outside the valid range, some frameworks will raise an error. In other cases, the program may default to the CPU if a specified GPU is not found. Thus, validating the availability and accessibility of the targeted device is vital before beginning computationally intensive operations.

Beyond explicit environment variable settings, many frameworks also offer API calls for direct device selection. For instance, TensorFlow provides `tf.config.set_visible_devices()` while PyTorch uses `torch.cuda.set_device()`. Although these allow for programmatic selection, understanding the implications of their interaction with environment variables is crucial. While programmatic selection does not directly modify what is considered a visible device, it can further refine the usage pattern. For example, if the process has only GPU 1 and GPU 3 available (using `CUDA_VISIBLE_DEVICES`), `torch.cuda.set_device(1)` will map to using GPU 3. In other words, you are setting a target to be the second visible device after having configured the set of visible devices via the environment variable.

For further study on GPU device handling, I recommend referring to the official documentation provided by the specific framework you are utilizing. For CUDA-based frameworks, understanding the CUDA Toolkit documentation on device management is essential. For ROCm frameworks, consulting the ROCm documentation on device management can provide useful information. Additionally, exploring the driver-specific guides for your GPU provider will give you insights into how device enumeration and visibility are determined at the driver level. Furthermore, examining online code repositories within the open-source communities surrounding deep learning and high-performance computing also provides practical examples of different device handling techniques. The official documentation is typically quite robust and covers most of the nuances and complexities that you may encounter.
