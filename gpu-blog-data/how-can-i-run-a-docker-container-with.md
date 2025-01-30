---
title: "How can I run a Docker container with specified GPUs using the Docker SDK for Python?"
date: "2025-01-30"
id: "how-can-i-run-a-docker-container-with"
---
The core challenge when leveraging GPUs within Docker containers using the Python SDK stems from the need to not only specify which GPUs should be accessible to the container but also to ensure the necessary NVIDIA Container Toolkit components are installed and properly configured on the host system. This toolkit bridges the gap between the Docker engine and the NVIDIA drivers, allowing containers to access host GPU resources efficiently.

I've found that managing this process with the Docker SDK for Python involves a few key steps: configuring runtime options, understanding device requests, and carefully parsing the output to confirm successful GPU access. The `docker.client.DockerClient` object provides the necessary methods for container creation, but the correct specification of GPU resources requires knowledge of the underlying mechanics.

Firstly, let’s consider the fundamental requirement: specifying the runtime configuration that includes GPU access. This is achieved through the `runtime` and `device_requests` arguments within the `client.containers.run()` method. The `runtime` must be set to `"nvidia"` to signal the usage of the NVIDIA Container Toolkit. Subsequently, `device_requests` allows for the precise specification of which GPUs are to be made available. These requests are generally constructed as JSON-like dictionaries detailing resource capabilities.

```python
import docker

client = docker.from_env()

try:
    container = client.containers.run(
        "nvidia/cuda:11.8.0-base-ubuntu20.04",
        command="nvidia-smi",
        runtime="nvidia",
        device_requests=[
            docker.types.DeviceRequest(
                count=-1,
                capabilities=[["gpu"]]
            )
        ],
        detach=True
    )
    
    container.wait()
    logs = container.logs().decode('utf-8')
    print("Container logs:\n", logs)

except docker.errors.ContainerError as e:
    print(f"Error running container: {e}")
finally:
   if 'container' in locals():
        container.remove()
```

In this first example, we're instructing the Docker daemon to launch an NVIDIA CUDA base image and execute `nvidia-smi` inside the container. This command is invaluable for verifying GPU presence and health. `count=-1` requests all available GPUs. `capabilities=[["gpu"]]` specifies that we are requesting GPUs. Crucially, the `runtime` is set to `"nvidia"`. The output of the `nvidia-smi` command is captured in the logs, providing immediate verification of GPU accessibility. Error handling is also present, in the event of failure to create the container. Post-execution cleanup ensures the container is removed after use, a recommended best practice for resource management.

Often, more granular control is desired, such as specifying a subset of available GPUs. This is where the index property of the `DeviceRequest` becomes vital. If a system has multiple GPUs, the index can be used to target a specific card.

```python
import docker

client = docker.from_env()

try:
    container = client.containers.run(
      "nvidia/cuda:11.8.0-base-ubuntu20.04",
      command="nvidia-smi",
      runtime="nvidia",
      device_requests=[
          docker.types.DeviceRequest(
              device_ids=["0"],
              capabilities=[["gpu"]]
           )
      ],
      detach=True
    )
    
    container.wait()
    logs = container.logs().decode('utf-8')
    print("Container logs:\n", logs)

except docker.errors.ContainerError as e:
    print(f"Error running container: {e}")

finally:
    if 'container' in locals():
        container.remove()
```
This second snippet demonstrates the targeting of a specific GPU, in this case, the device with an index of '0'. The `device_ids` parameter takes a list of device indices. This approach allows for targeted resource allocation, which is important when managing multiple concurrent GPU-intensive workloads or when a particular job requires a specific device.  The underlying NVIDIA Container Toolkit mechanisms ensure only the requested device is visible within the container. As with the previous example, the container's output is captured and printed to verify device visibility.

Furthermore, I've encountered scenarios where applications require very specific driver capabilities, beyond simple GPU access. For instance, when using CUDA-specific features like Unified Memory or CUDA Graphs, specialized capabilities may need to be requested in the `capabilities` list.

```python
import docker

client = docker.from_env()

try:
    container = client.containers.run(
       "nvidia/cuda:11.8.0-base-ubuntu20.04",
       command="nvidia-smi",
       runtime="nvidia",
       device_requests=[
           docker.types.DeviceRequest(
                count=-1,
                capabilities=[["gpu","compute","utility"]]
            )
        ],
       detach=True
    )

    container.wait()
    logs = container.logs().decode('utf-8')
    print("Container logs:\n", logs)


except docker.errors.ContainerError as e:
    print(f"Error running container: {e}")
finally:
    if 'container' in locals():
        container.remove()

```

In this final illustration, we extend the `capabilities` parameter beyond the bare minimum `gpu` entry. Here, the list includes  `["gpu", "compute", "utility"]`, which requests additional CUDA capabilities such as compute-specific features. This is essential for ensuring the application has access to required functionality. The exact composition of the `capabilities` list depends on the specific needs of the application. Note the consistency of  the  `nvidia-smi` command, which continues to provide a standard, convenient way to check the resulting configuration.  As before, error handling and post-execution cleanup remain integral.

Several resources can deepen one’s understanding of GPU management within Docker. The official NVIDIA Container Toolkit documentation provides essential technical details about the inner workings of GPU virtualization and device drivers. Documentation for the Docker SDK for Python thoroughly covers API details, specifically on the use of `runtime` and `device_requests`. Finally, research into the specific versions of the NVIDIA drivers and associated CUDA versions is critical for matching host capabilities to those within the containerized applications.
These resources, combined with practical experimentation and careful debugging, have proven crucial to implementing a stable and reliable solution for GPU acceleration using Docker containers.
