---
title: "How can I select which GPUs my model uses in this code?"
date: "2025-01-30"
id: "how-can-i-select-which-gpus-my-model"
---
Multi-GPU training, particularly within deep learning frameworks like TensorFlow or PyTorch, is often managed through environment variables or specific API calls, rather than being automatically detected and utilized by default. I've spent considerable time optimizing model training pipelines in high-performance compute environments, encountering this precise issue multiple times with varying hardware configurations. The key isn't just about selecting *a* GPU, but ensuring your framework correctly initializes for parallel computation, effectively distributing the workload to the chosen devices.

The core challenge lies in instructing the deep learning framework which GPUs it can access, and furthermore, how to allocate model parameters and data across them for optimal processing. Failing to manage this explicitly can result in the model falling back to CPU usage or, worse, attempts to allocate memory on GPUs that are either unavailable or not intended for this process, leading to crashes and performance bottlenecks. It is a nuanced dance of device specification and data parallelization strategies.

The initial step often involves identifying the available GPUs on your system. Typically, these devices are assigned numerical indices starting from zero, which is then referenced when selecting them for computation. If you have four GPUs, for example, their identifiers would likely be 0, 1, 2, and 3. The framework's specific mechanisms for utilizing these indices vary, but they all revolve around this basic principle.

In TensorFlow, you generally interact with this device selection through the `tf.config.experimental` module. This approach allows you to enumerate devices and selectively assign operations to specific GPUs. The framework also incorporates the concept of 'visible devices' to limit what devices are available to the TensorFlow runtime, allowing for finer control over resource consumption in a multi-user environment.

```python
# TensorFlow example: Explicit GPU selection and device placement

import tensorflow as tf

# 1. List all available GPUs.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("Available GPUs:", gpus)

    # 2. Choose the specific GPU(s). Example: Use only GPU 0 and GPU 2
    selected_gpus = [gpus[0], gpus[2]]  # Selecting GPUs at index 0 and 2

    # 3. Restrict visibility to those devices.
    tf.config.set_visible_devices(selected_gpus, 'GPU')

    # 4. Configure memory growth, this helps with allocation on demand.
    for gpu in selected_gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # 5. Check visible GPUs.
    visible_gpus = tf.config.get_visible_devices('GPU')
    print("Visible GPUs:", visible_gpus)

    # 6. Example using a specific GPU for an operation.
    with tf.device('/GPU:0'):
        a = tf.constant([1.0, 2.0, 3.0])
        b = tf.constant([4.0, 5.0, 6.0])
        c = a + b
        print(c)


else:
    print("No GPUs detected, will use CPU")

```
In this TensorFlow example, we initially list all available GPU devices, then specify that we want to use GPUs at indices 0 and 2. The function `set_visible_devices` limits which GPUs TensorFlow will recognize, while `set_memory_growth` dynamically manages GPU memory usage as needed. Finally, the `tf.device('/GPU:0')` context manager demonstrates how you can force a specific operation to run on a given GPU. The print statement shows a vector addition computed on device 0.

PyTorch manages GPU access with similar logic but relies on its `torch.cuda` module. The primary difference lies in selecting devices by index directly through functions like `torch.cuda.set_device` or indirectly by moving tensors to a device object. You often utilize `torch.cuda.device` to specify the desired GPU. Similar to TensorFlow, PyTorch is capable of data parallelization with its `torch.nn.DataParallel` class.

```python
# PyTorch example: GPU selection and model placement

import torch
import torch.nn as nn

# 1. Check for GPU availability
if torch.cuda.is_available():
    print("CUDA Available")
    gpu_count = torch.cuda.device_count()
    print(f"Number of GPUs: {gpu_count}")
    
    # 2. Select specific GPUs (here we assume they exist)
    selected_gpus = [0, 2] # Use GPUs at index 0 and 2
    device_ids = [f"cuda:{i}" for i in selected_gpus]
    print(f"Selected GPUs are: {device_ids}")

    # 3. Create a model and wrap it for DataParallel
    model = nn.Linear(10, 2)  # Example model
    if len(selected_gpus)> 1 :
      model = nn.DataParallel(model, device_ids=selected_gpus)

    # 4. Choose device: use the first selected device for demonstration
    device = torch.device(device_ids[0])  # Choose the first in our list
    
    # 5. Move the model to the selected device(s)
    model.to(device)

    # 6. Create input and move to the device
    input_data = torch.randn(5, 10).to(device)

    # 7. Perform a forward pass
    output = model(input_data)
    print("Output Shape: ", output.shape)



else:
  print("CUDA not available, using CPU")

```

In this PyTorch example, we check for CUDA availability and enumerate the GPUs present, then select the GPUs with indices 0 and 2 as target GPUs and construct a list of devices from them. A model (a simple linear layer here) is then created, and if more than one GPU is selected, the model is wrapped within `nn.DataParallel` to perform data parallelization across those GPUs. We finally move the model and the input tensor to the chosen device for subsequent calculations. The output illustrates that the forward pass was executed on a GPU.

A common scenario involves selecting GPUs through environment variables. Often, these are the primary mechanism for influencing device selection at a system level, and they’re particularly relevant when running training processes through shell scripts or schedulers in a cloud or cluster environment. Both TensorFlow and PyTorch respond to these variables. Setting `CUDA_VISIBLE_DEVICES` affects which GPUs the framework will recognize.

```python
# Shell Script Example: Using Environment Variables (Bash)

# Set environment variable to use only GPUs at indices 1 and 3
# (Zero based indexing)
export CUDA_VISIBLE_DEVICES=1,3

# Run your PyTorch or TensorFlow training script here.
# Assuming a python script called "train.py"
# python train.py

# After the script completes (for illustrative purposes reset the environment)
unset CUDA_VISIBLE_DEVICES
```

This snippet demonstrates the use of `CUDA_VISIBLE_DEVICES` within a shell script, which allows you to influence which GPUs the program will see. By specifying `1,3` only GPUs at these indices will be available to the framework. The subsequent python script (represented here with a comment `#python train.py`) would then execute with that limitation in place. This setup is essential for controlling resource allocation in a multi-user system or when running multiple training processes simultaneously. The `unset` command demonstrates how you may wish to remove the environmental variable once you are done using it.

Beyond these basic mechanisms, selecting GPUs also involves considerations around data parallelization methods: distributing data across devices (data parallelism) versus distributing the model itself (model parallelism), though the latter is much more complex and often not suitable for the typical use case. For both frameworks, you can opt for distributed training techniques, especially when dealing with very large models or datasets that don’t fit into a single GPU’s memory, which might involve further environment configuration.

For further in-depth study, I recommend exploring the official documentation for TensorFlow and PyTorch, which contain comprehensive sections on GPU utilization and distributed training. Furthermore, the book "Deep Learning with PyTorch" by Eli Stevens, Luca Antiga, and Thomas Viehmann offers excellent explanations and best practices for utilizing GPUs effectively within PyTorch. For TensorFlow, the online guide “TensorFlow: Get Started” is often updated to keep current with its evolution. Reading material within each framework's community forums (e.g., StackOverflow, Reddit) may provide real-world examples and troubleshooting advice. Finally, if available, consult user manuals for high-performance computing infrastructures (clusters, cloud services), as they often outline specific configuration nuances. These materials provide a stronger understanding of both the technical mechanics and pragmatic application of GPU selection in deep learning.
