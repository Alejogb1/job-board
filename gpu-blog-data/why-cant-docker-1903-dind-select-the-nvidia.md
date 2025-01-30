---
title: "Why can't Docker 19.03 DIND select the NVIDIA GPU driver?"
date: "2025-01-30"
id: "why-cant-docker-1903-dind-select-the-nvidia"
---
The core issue with utilizing NVIDIA GPUs within a Docker 19.03 DIND (Docker-in-Docker) setup stems from how Docker containers interact with host hardware resources, particularly device nodes exposed through `/dev`. Docker 19.03, while possessing the `--gpus` flag, was designed primarily to operate on a host-level basis, directly interfacing with host-installed NVIDIA drivers. When Docker runs within another Docker container (DIND), this assumption breaks down. The inner Docker daemon, attempting to utilize the `--gpus` option, finds itself within a namespace that lacks direct access to the necessary host device files and library dependencies needed for NVIDIA GPU functionality.

Specifically, Docker relies on a combination of kernel driver interfaces exposed as character devices (typically under `/dev/nvidia*`) and user-space libraries (primarily `libnvidia-*`). When a standard Docker container is launched with `--gpus all` (or `--gpus device=...`), the Docker daemon on the host system mounts the requisite `/dev/nvidia*` device nodes into the container's namespace and ensures that the appropriate NVIDIA libraries from the host system are accessible inside the container. This process involves direct interaction with the host's file system and the driver modules loaded into the host kernel.

In a DIND configuration, this relationship is disrupted. The outer Docker container (the DIND host) runs its own Docker daemon, which has access to the host's hardware and drivers. However, the inner Docker daemon executes within its own isolated environment. The inner container's namespace is not automatically equipped with either the `/dev/nvidia*` device nodes or the NVIDIA user-space libraries from the *outer* container, let alone from the host itself. This results in the inner Docker daemon attempting to interface with a non-existent NVIDIA driver. Consequently, the `--gpus` flag within a DIND context simply doesn’t function as expected. The error messages often seen reflect this, ranging from device node access failures to library loading errors.

Let’s consider several scenarios with illustrative code examples, highlighting the challenges.

**Example 1: Basic DIND Attempt**

```dockerfile
# Dockerfile for the outer (DIND host) container

FROM docker:19.03

RUN apk add --no-cache docker-compose bash && \
    dockerd --host=tcp://0.0.0.0:2375 --host=unix:///var/run/docker.sock &

CMD ["/bin/bash"]
```

```bash
# Execute on host

docker build -t dind .

docker run -d --privileged -v /var/run/docker.sock:/var/run/docker.sock -p 2375:2375 --name dind dind
```

This sets up the DIND environment. We can now attempt to use the inner docker.

```dockerfile
# Dockerfile for inner container attempt, expecting GPU to be present

FROM nvidia/cuda:11.8.0-base-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

CMD ["python3", "-c", "import torch; print(torch.cuda.is_available())"]
```

```bash
# Inside outer DIND container

docker build -t inner-cuda-fail -f inner.dockerfile . # Build inside DIND container

docker run --gpus all inner-cuda-fail # Attempt to run inside DIND container

```
This inner container, even though using the nvidia/cuda image, will report `torch.cuda.is_available()` as `False`. The `--gpus all` flag is effectively ignored within the DIND environment because the inner Docker daemon doesn't have the required access to the host GPU resources. The privileged flag is essential for enabling inner dockerd to even start and does not resolve the GPU access problem.

**Example 2: Attempt to Mount `/dev/nvidia*` Inside DIND**

```dockerfile
# Dockerfile for modified outer DIND container

FROM docker:19.03

RUN apk add --no-cache docker-compose bash && \
    dockerd --host=tcp://0.0.0.0:2375 --host=unix:///var/run/docker.sock &

CMD ["/bin/bash"]
```

```bash
# Execute on host

docker build -t dind-dev .

docker run -d --privileged -v /var/run/docker.sock:/var/run/docker.sock -v /dev:/dev -p 2375:2375 --name dind-dev dind-dev
```
```dockerfile
# Dockerfile for modified inner container attempt
FROM nvidia/cuda:11.8.0-base-ubuntu20.04

RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

CMD ["python3", "-c", "import torch; print(torch.cuda.is_available())"]
```

```bash
# Inside outer DIND container

docker build -t inner-cuda-mount -f inner.dockerfile .

docker run --gpus all inner-cuda-mount
```
While we've now exposed the host's `/dev` directory (including `/dev/nvidia*`), this is not a complete solution.  The inner container does not have the correct versions of the `libnvidia-*` libraries that match the host's drivers, leading to loading issues. Additionally, even if versions aligned perfectly, the kernel interfaces are direct. The device nodes alone are not sufficient for proper function.

**Example 3: Simulating Library and Device Node Mounting for DIND**

This illustrates the complexity further, without providing a direct, workable solution.  A full solution requires more advanced techniques.

```dockerfile
# Dockerfile for simulated DIND with libraries and device nodes
FROM ubuntu:20.04

#Simulate nvidia driver packages (Not working code)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvidia-common-470 libnvidia-compute-470 libnvidia-decode-470 libnvidia-encode-470 \
    libnvidia-gl-470 libnvidia-ml-dev-470 nvidia-utils-470

RUN mkdir /dev/nvidia && \
    mknod /dev/nvidia/nvidia0 c 195 0 && \
    mknod /dev/nvidiactl c 195 255 && \
    mknod /dev/nvidia-uvm c 195 254 && \
    mknod /dev/nvidia-modeset c 195 253
    
RUN apt-get update && apt-get install -y python3 python3-pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

CMD ["python3", "-c", "import torch; print(torch.cuda.is_available())"]
```

```bash
docker build -t simulated-dind-gpu .

docker run --privileged --gpus all simulated-dind-gpu
```

Here, I attempted to simulate mounting device nodes and installing a plausible set of user-space NVIDIA libraries *inside* the container. This will still fail to work correctly due to the complex interaction with the host kernel that a real driver requires. This demonstrates the futility of merely replicating libraries and device files within a container. These libraries are built for specific kernel versions and are intimately tied to host-level configurations, making it difficult, even with this privileged mode, to emulate proper behavior.

To overcome these limitations, techniques like using the NVIDIA Container Toolkit within a non-DIND environment and utilizing container images specifically designed to execute in such environments would be preferred. The complexity arises from the strict isolation enforced by containerization that interferes with the direct host-hardware access, required by the NVIDIA drivers.

**Resource Recommendations**

To gain deeper knowledge regarding these issues, one should explore documentation on the NVIDIA Container Toolkit, especially its utilization within Kubernetes and container orchestration frameworks. The official Docker documentation concerning the use of GPUs, particularly the nuances of the `--gpus` parameter, provides fundamental understanding. Researching the complexities of Linux device files and their handling within different namespaces is also beneficial. Furthermore, studying the inner workings of the NVIDIA driver stack in Linux environments is essential to fully grasp why this specific configuration fails. I have found that information concerning the nuances of user-space driver libraries, their interaction with kernel modules, and container runtime specifications are vital for troubleshooting and proper GPU enablement within more complex containerized scenarios.
