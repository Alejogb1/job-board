---
title: "How do Docker Compose GPU access methods differ?"
date: "2025-01-30"
id: "how-do-docker-compose-gpu-access-methods-differ"
---
Having spent considerable time orchestrating machine learning pipelines, I've encountered various approaches to granting Docker containers access to GPUs, especially within the Docker Compose framework. The core difference lies in how the host’s GPU resources are exposed and mapped into the container environment. This variation stems from the need to balance ease of use, security, and flexibility. Fundamentally, it’s about moving from a default CPU-centric container runtime to one that’s cognizant of, and can effectively leverage, the underlying GPU hardware.

The fundamental problem addressed by GPU access methods is that Docker containers, by default, operate in isolation and are unaware of host hardware beyond the abstracted CPU. Without explicit configuration, a Docker container would be unable to detect or use any available GPUs. The methods for enabling this access fall into two primary categories: leveraging runtime flags at container creation and more sophisticated configurations through driver support and device mapping. The efficacy of each method is highly dependent on the host environment, container runtime, and the software libraries within the container that are used to access the GPU.

The most straightforward approach involves using runtime flags within a Docker Compose file. Typically, this is achieved by specifying the `--gpus` flag to the `docker run` command; however, within Docker Compose, the equivalent directive is contained under the `deploy.resources.reservations.devices` configuration. This allows a user to specify which GPU devices are available to the container. This method works by explicitly passing device nodes into the container. This is commonly used for single GPU or targeted GPU selections.

```yaml
version: "3.9"
services:
  my-gpu-app:
    image: my-gpu-image
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["0"]
```

In this example, the `my-gpu-app` service, based on the `my-gpu-image` image, is configured to have access to GPU device `0` on the host. The `nvidia` driver indicates that the GPU is an NVIDIA card, which is crucial for the underlying container runtime and the device mapping. The `device_ids` list can be expanded to include multiple GPUs if needed, or replaced with `all` for all devices to be accessible. This approach relies on the NVIDIA container runtime, typically included with the NVIDIA Container Toolkit, which intercepts calls to the CUDA or other GPU drivers within the container and maps them to the host system's drivers.

A significant consideration is the driver version compatibility. The NVIDIA container runtime provides driver mapping, meaning the driver version on the host and within the container should be compatible. In the absence of a proper driver, an error will occur, as the runtime will not be able to properly connect the host GPUs and the container-based programs. This also means that the base image needs to contain libraries that use GPU resources.

The second approach uses the Docker Compose `runtime` field, specifically when working with NVIDIA GPUs, to declare the usage of `nvidia` as a runtime. This method implicitly informs the Docker daemon and NVIDIA container runtime that GPU support is required for this service. This can be simpler, and is often used alongside other resource limitations or settings for a container. This method relies on the host system having installed the proper NVIDIA drivers and NVIDIA container runtime.

```yaml
version: "3.9"
services:
  my-gpu-app:
    image: my-gpu-image
    runtime: nvidia
```

In this second example, we simply specify that the `my-gpu-app` service uses the `nvidia` runtime. This approach means that the Docker daemon utilizes NVIDIA-specific hooks during container creation. It automatically handles the device mounts and ensures that the required environment variables are set to allow GPU access inside the container. This configuration method simplifies the Docker Compose configuration as it removes the need for explicit device reservation mapping. If multiple GPUs are present on the host, this method will generally expose all of them, which might be useful but less controlled than explicitly mapping each device as illustrated in the previous example.

The advantage of this method is that it’s less verbose and, as a user, allows the container runtime to handle more of the necessary configurations. However, it lacks the fine-grained device control offered by using `deploy.resources.reservations.devices`. Furthermore, the same issue of driver compatibility needs to be considered.

A third, slightly more advanced technique, is explicitly mapping the device nodes for more granular control, especially when using specific device types or if you are operating outside of a standard NVIDIA setup. This method involves using the `devices` field within the Docker Compose service configuration, where device node paths from the host are directly mapped into the container.

```yaml
version: "3.9"
services:
  my-gpu-app:
    image: my-gpu-image
    devices:
      - "/dev/nvidia0:/dev/nvidia0"
      - "/dev/nvidiactl:/dev/nvidiactl"
      - "/dev/nvidia-uvm:/dev/nvidia-uvm"
      - "/dev/nvidia-modeset:/dev/nvidia-modeset"
```

This example is more verbose but can be necessary when you need precise control over which device nodes are exposed. The four nodes shown, while not exhaustive, represent commonly needed node paths for NVIDIA hardware. The `/dev/nvidia0` node is a representative device node; this would be `/dev/nvidia1`, etc, for other GPUs. Similarly, `/dev/nvidiactl`, `/dev/nvidia-uvm` and `/dev/nvidia-modeset` are critical for NVIDIA driver functionality. This configuration is suitable when the `nvidia` runtime is unavailable, for instance with alternate container runtimes, or when you need finer control over device visibility for security or debugging purposes. Note, the exact device paths can change depending on the host system. This method bypasses the nvidia container runtime, and assumes the host system is setup correctly for direct node use.

Choosing between these methods hinges on the complexity of the application, the level of control needed, and the host system configurations. When beginning, using the `runtime: nvidia` directive is generally sufficient. However, if you need more granular control, especially in systems with multiple GPUs or specialized hardware, the `devices` field or `deploy.resources.reservations.devices` offer significant advantages, albeit with slightly more intricate configurations. Regardless of method, maintaining compatibility between host and container driver versions is always critical.

For additional resources, consult the documentation for Docker Compose and your specific GPU vendor’s container runtime toolkit. The NVIDIA Container Toolkit documentation offers a deep dive into the nuances of GPU access within Docker containers and NVIDIA specific configurations. Books on containerization or specifically targeted at GPU-accelerated computing also provide comprehensive details on the underlying technical challenges and solutions. Furthermore, exploring communities centered on machine learning deployment often surfaces best practices and addresses various issues related to GPU access within containers.
