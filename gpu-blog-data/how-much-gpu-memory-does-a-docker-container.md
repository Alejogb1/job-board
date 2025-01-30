---
title: "How much GPU memory does a Docker container use?"
date: "2025-01-30"
id: "how-much-gpu-memory-does-a-docker-container"
---
GPU memory usage within a Docker container is not a straightforward allocation directly analogous to CPU RAM. Containers themselves do not inherently consume GPU memory. Instead, they leverage GPU resources made available by the host system via configured device drivers and the Docker runtime. Therefore, it’s more accurate to discuss how containerized applications request and consume host GPU memory.

The central concept is that a Docker container, even when configured to access the GPU, does not have its own isolated pool of GPU memory. The host system’s GPU and its associated memory are the definitive resources.  Docker acts as a conduit, allowing processes *inside* the container to communicate with the host's GPU via its drivers. When a containerized application initiates a process requiring GPU resources (e.g., a deep learning training script, a rendering task), that application’s demand directly translates into the host GPU memory consumption. If two containers simultaneously demand significant GPU memory, they are, in essence, competing for the same physical resource on the host.

The ability for containers to access the host's GPU is generally enabled using specific Docker runtime configurations and driver installations. Primarily, this involves the utilization of the `nvidia-docker` runtime (or its successor, the NVIDIA Container Toolkit) for NVIDIA GPUs, which manages the necessary mappings between containerized processes and the host’s NVIDIA driver. For AMD GPUs, the process involves different driver installations and potentially other specific container runtime configurations. These configuration options ensure the correct libraries and interfaces are available within the container to interact with the host GPU without exposing the complexities of driver management within the container itself.

Furthermore, resource limits set on a container’s CPU or RAM via Docker Compose or `docker run` commands *do not* automatically translate to limits on GPU memory usage.  Docker's primary focus is on resource allocation within the CPU and RAM space, and controlling GPU resources is treated differently, usually with device passthrough or utilization of specific environment variables. The containerized application itself usually has its own frameworks and APIs for handling GPU memory allocation (e.g., CUDA memory allocation in the case of NVIDIA GPUs). While the application runs inside the container, the framework's allocation process is, from the host's perspective, the actual mechanism of consumption.

It is imperative to recognize that the total amount of GPU memory available to all containers using a specific GPU on a host is limited by the physical GPU memory installed on that host. Over-subscribing this resource can lead to out-of-memory (OOM) errors for containers or cause serious performance degradation. Monitoring and careful management are necessary, particularly in multi-container GPU environments.

To illustrate how a container can access and potentially consume GPU memory, I will present a few examples using the `nvidia-docker` runtime. I have personally used variations of these approaches in my work on containerizing deep learning workflows.

**Example 1: Simple GPU Information Check**

This first example shows a basic container using `nvidia-smi`, an NVIDIA utility, to inspect the available GPU resources:

```dockerfile
FROM nvidia/cuda:12.0.0-base-ubuntu22.04
RUN apt-get update && apt-get install -y nvidia-utils
CMD nvidia-smi
```

In the Dockerfile, I start with a base NVIDIA CUDA image. I then install `nvidia-utils`, which includes `nvidia-smi`. The `CMD` specifies that `nvidia-smi` should be executed when the container starts. Building and running this image using `nvidia-docker run --rm my-gpu-check`  will output the current GPU utilization on the host within the container's terminal, demonstrating that the container has access to the host's GPU information.  The container itself has not allocated any memory here, instead it is passively observing resource usage on the host. The key takeaway is that this example reveals the container's ability to query the host GPU's status via the installed utilities without directly consuming GPU memory in any significant capacity.

**Example 2: GPU Memory Allocation using PyTorch**

The following example demonstrates how a containerized application can allocate GPU memory using a deep learning framework.

```dockerfile
FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip

RUN pip install torch torchvision torchaudio
COPY pytorch_gpu_test.py .

CMD python3 pytorch_gpu_test.py
```
And the `pytorch_gpu_test.py` script:
```python
import torch

if torch.cuda.is_available():
  device = torch.device("cuda")
  print("CUDA is available, using device:", device)
  try:
      tensor = torch.randn(1000, 1000, device=device)
      print("Successfully created tensor on GPU. This consumes a small amount of memory.")
  except RuntimeError as e:
      print(f"Failed to allocate memory on GPU: {e}")
else:
  print("CUDA is not available.")
```
Here, the Dockerfile installs Python and PyTorch and then copies a Python script into the image. The script checks for GPU availability, and if a GPU is available, creates a tensor on the GPU. Running this container (`nvidia-docker run --rm my-pytorch-test`)  will attempt to allocate a tensor on the GPU and thereby consume GPU memory.  The allocated memory is dynamic and depends on the tensor size and the application's requirements. Using `nvidia-smi` on the host will reveal an increase in the host's GPU memory usage while this container is running.  This clearly demonstrates how a containerized *application* initiates GPU memory allocation *on the host*.

**Example 3: Monitoring Memory Usage of a GPU Process**
This example highlights a situation where one can monitor the GPU usage of a process in the container.
```dockerfile
FROM nvidia/cuda:12.0.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip install torch torchvision torchaudio
COPY gpu_monitor.sh .
COPY pytorch_gpu_test.py .

CMD bash gpu_monitor.sh
```
And the `pytorch_gpu_test.py` script:

```python
import torch
import time

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available, using device:", device)
    try:
        while True:
            tensor = torch.randn(5000, 5000, device=device)  # Increased tensor size to consume more memory
            time.sleep(1)  # Keep the process alive
    except RuntimeError as e:
        print(f"Failed to allocate memory on GPU: {e}")
else:
    print("CUDA is not available.")
```
And the `gpu_monitor.sh` script:
```bash
#!/bin/bash
nvidia-smi &
python3 pytorch_gpu_test.py
```

Here, the Dockerfile installs the necessary tools. The `gpu_monitor.sh` script starts `nvidia-smi` in the background and then executes the PyTorch script that continually requests GPU memory. The `pytorch_gpu_test.py` is the same as the second example but with larger tensors and an infinite loop to simulate prolonged usage. This command `nvidia-docker run --rm  my-gpu-monitor ` will then reveal, via the output of `nvidia-smi` inside the container, the GPU memory consumption initiated from *within* the same container.  This demonstrates how a container can monitor its own GPU memory usage, reflecting the underlying host's GPU consumption.

Based on my experience working with GPU accelerated applications in containers, the core concept to remember is that containers themselves don’t ‘use’ GPU memory; they request it via the host GPU resources, facilitated by container runtimes like `nvidia-docker` and appropriate driver installations on the host. Container resource limits do not directly control GPU memory usage; that is managed primarily by the application. Careful monitoring and understanding are essential when deploying GPU accelerated workloads in Docker.

For further exploration, I recommend consulting resources such as the official Docker documentation, the NVIDIA Container Toolkit documentation, and specific deep learning framework documentation like PyTorch and TensorFlow concerning GPU usage. These resources offer in-depth information regarding container runtimes, driver setup, and specific APIs used for accessing GPU resources within containerized environments.  Additionally, monitoring tools like `nvidia-smi` and profiling utilities integrated with deep learning frameworks can be instrumental in understanding and optimizing GPU memory utilization within containers.
