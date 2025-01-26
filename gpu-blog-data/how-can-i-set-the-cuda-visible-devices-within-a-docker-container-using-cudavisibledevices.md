---
title: "How can I set the CUDA visible devices within a Docker container using `CUDA_VISIBLE_DEVICES`?"
date: "2025-01-26"
id: "how-can-i-set-the-cuda-visible-devices-within-a-docker-container-using-cudavisibledevices"
---

The `CUDA_VISIBLE_DEVICES` environment variable, when utilized within a Docker container configured for GPU access, provides a critical mechanism for selectively exposing specific physical GPUs to applications running inside the container. This granular control is paramount in environments with multiple GPUs, allowing resource allocation and preventing interference between concurrently running tasks.

**Understanding the Mechanism**

Docker containers, by default, do not inherit the host's environment variables. Furthermore, even if they did, directly exposing all host GPUs to a container may not be desirable due to resource constraints or job isolation needs. The `nvidia-container-runtime`, a specialized runtime for Docker, bridges this gap, enabling the use of `CUDA_VISIBLE_DEVICES` to control which GPUs within the host system are made accessible to the container.

When a container is initiated with the `--gpus` flag (or the older `--runtime=nvidia` flag), the `nvidia-container-runtime` intercepts the Docker runtime, examining the `CUDA_VISIBLE_DEVICES` environment variable during the container creation process. It translates the values specified in this variable to the appropriate device nodes within the container's file system. As a result, the NVIDIA driver inside the container will only "see" the GPUs identified by the values set in the environment variable. Crucially, the container's internal view of GPUs (e.g., device index 0, 1, etc.) will be a mapping to those specified by the `CUDA_VISIBLE_DEVICES` and will not necessarily align with the physical IDs of the GPUs on the host.

**Code Examples and Commentary**

Let's illustrate these principles with several practical scenarios:

**Scenario 1: Exposing a Single Specific GPU**

Imagine a host system with four GPUs, indexed 0, 1, 2, and 3. I've been working on a deep learning model development project that specifically needs access to GPU 2. To expose only GPU 2 to a container, I use the following command:

```bash
docker run --gpus all -e CUDA_VISIBLE_DEVICES=2 -it my_cuda_image:latest /bin/bash
```

*   **`docker run`**: Initiates the creation and execution of a Docker container.
*   **`--gpus all`**: This essential flag signals to Docker to use the `nvidia-container-runtime`, enabling GPU support within the container. Alternatively, `--runtime=nvidia` can be used on older Docker versions, but `--gpus all` is the more modern and recommended approach for most use cases.
*   **`-e CUDA_VISIBLE_DEVICES=2`**: The core of this example. This sets the `CUDA_VISIBLE_DEVICES` environment variable within the container to `2`. The container will map its internal device index 0 to the physical device 2 of the host. Within the container, GPU 2 is represented as the index 0.
*   **`-it my_cuda_image:latest`**: Specifies the Docker image to be used. I use this image that is preconfigured with CUDA drivers and my project's specific environment.
*   **`/bin/bash`**: Launches an interactive bash shell inside the container.

Once inside the container, running the `nvidia-smi` command (part of the NVIDIA driver toolkit) will reveal that only one GPU is visible with index `0`. This confirms that the container has successfully been limited to the specified host GPU.

**Scenario 2: Exposing Multiple Specific GPUs**

In a different project, I needed to parallelize training across two GPUs, specifically host GPUs 0 and 3. I used the following command:

```bash
docker run --gpus all -e CUDA_VISIBLE_DEVICES=0,3 -it my_cuda_image:latest /bin/bash
```

The primary change is the `CUDA_VISIBLE_DEVICES` value which has been changed to `0,3`. This instructs the runtime to expose host GPUs 0 and 3 to the container. Inside the container, the device index `0` will map to host GPU `0`, and device index `1` to host GPU `3`. The container will not be aware of any other available GPUs. Again `nvidia-smi` will show two visible GPUs numbered `0` and `1`, showing an ordered mapping to the provided host GPUs.

**Scenario 3: Disabling GPUs**

On occasion, I have needed to run CPU-bound tasks inside a container originally intended for GPU workloads. To explicitly disable all GPU access, I would set the following:

```bash
docker run --gpus all -e CUDA_VISIBLE_DEVICES="" -it my_cuda_image:latest /bin/bash
```

The empty string `""` for `CUDA_VISIBLE_DEVICES` effectively disables all GPU visibility within the container. This ensures that no GPU resources are consumed, even if the container's image has CUDA drivers and potentially attempts to use a GPU. Upon starting this container, attempts to leverage CUDA code will fail or run on the CPU depending on how the program is coded. `nvidia-smi` will show no NVIDIA devices recognized, confirming that no GPU is available to the container.

**Important Considerations**

It's crucial to understand that the `CUDA_VISIBLE_DEVICES` variable defines *which* GPUs are accessible, and also re-indexes them within the container. The container's internal device index is not the same as the physical ID on the host. This is a source of confusion for new users. Always use `nvidia-smi` within the container to understand the mapping between internal indexes and the specified host GPUs. Additionally, ensure the container has the correct NVIDIA drivers and CUDA toolkit versions that are compatible with the host system's drivers. Conflicts in driver versions will lead to errors or unexpected behavior.

**Resource Recommendations**

To deepen understanding, I recommend exploring the official NVIDIA documentation on `nvidia-container-runtime` and Docker GPU support. The documentation for the NVIDIA driver provides insights into device discovery and environment variable usage. Additionally, I find it helpful to review examples and discussions in the developer forums related to containerized GPU workloads. These resources contain detailed explanations of the underlying concepts and practical use cases. Lastly, several blog posts and tutorials cover similar concepts and may offer different viewpoints on advanced container setups for GPU usage.
