---
title: "Can Docker Swarm utilize GPU devices?"
date: "2025-01-30"
id: "can-docker-swarm-utilize-gpu-devices"
---
Docker Swarm, while primarily known for orchestrating CPU-bound workloads, can indeed leverage GPU devices for specialized tasks, albeit with some nuanced configuration and limitations compared to solutions like Kubernetes. My experience deploying machine learning models across heterogeneous clusters has highlighted the practicalities and challenges involved. The core mechanism enabling GPU utilization in Swarm involves the use of Docker's runtime options to expose host-level GPU resources to containers. This is accomplished using the `nvidia-container-runtime` and specific device mappings, rather than Swarm having any intrinsic, dedicated GPU scheduling.

The crucial element to understand is that Swarm does not inherently possess awareness of GPU availability or capability at a node level. It relies on Docker’s low-level features to manage access to the GPU, much like it manages other host devices. Consequently, scheduling decisions are not made based on GPU demand, unlike some more sophisticated orchestrators. Instead, a service is scheduled on a node, and *if* that node possesses the necessary GPU drivers and has been configured to expose its GPUs, the container will have access to them as specified in the service definition. This means we're essentially relying on local resource management at the node level with Swarm managing the overall service placement and scaling.

The primary mechanism for GPU support involves using the `docker run` option `--gpus` and its associated syntax within a `docker-compose` file or a Swarm service definition. The `--gpus` flag, when used with `all` will make all detected GPUs available within the container. You can also specify individual devices using their indices, UUIDs, or vendor-specific capabilities. The key is that these options are interpreted by the Docker daemon on the specific node, not by Swarm itself, which handles task placement based on general resource constraints (like CPU and memory) and defined placement constraints.

To effectively utilize GPUs, certain prerequisites must be met on each node intended to run GPU-enabled services. These include the installation of NVIDIA drivers appropriate for your GPU hardware, and crucially, the `nvidia-container-runtime`. Without the runtime, standard Docker images will not be able to access GPU functionality. The runtime provides an intermediary that maps the host's NVIDIA drivers to the container environment, ensuring the libraries necessary for GPU processing are available inside the container.

Here are code examples to demonstrate various scenarios, along with the necessary explanations:

**Example 1: Basic GPU access with `docker-compose`:**

```yaml
version: "3.7"
services:
  gpu_app:
    image: nvidia/cuda:11.8.0-base
    deploy:
      resources:
        reservations:
          cpus: '1'
          memory: 1G
    runtime: nvidia
    command: ["nvidia-smi"]
    # Explicitly expose GPUs on the host (assuming all GPUs available)
    deploy:
        placement:
          constraints:
          - node.labels.gpu_enabled == true
```

*   **Commentary:** This `docker-compose.yml` file demonstrates a very basic setup. First, we are specifying a NVIDIA base image with a CUDA toolkit that provides the libraries required to execute commands on NVIDIA hardware. The important part here is `runtime: nvidia`, which is essential. Without this line, `nvidia-smi`, which reports on the status of available GPUs, will not be able to function correctly inside the container. The `command` in this case will simply run `nvidia-smi`, verifying the GPU is accessible. Additionally, I've added a `placement constraint` in the deploy section to ensure the service only runs on nodes with a `gpu_enabled` label set to true.  This label would need to be manually set on the Docker Swarm node using the command `docker node update --label-add gpu_enabled=true <node-name>`.  This prevents services from deploying to non-GPU nodes.

**Example 2: Specific GPU selection with `docker service create`:**

```bash
docker service create \
  --name gpu_service \
  --image nvidia/cuda:11.8.0-base \
  --runtime nvidia \
  --gpu '"device=1"' \
  --deploy-label gpu_enabled=true \
  --limit-cpu 1 \
  --limit-memory 1G \
  nvidia-smi
```

*   **Commentary:** This demonstrates using `docker service create` on the command line. The `--gpu '"device=1"'` option instructs Docker to make only GPU device with index 1 available to the container. If the host has multiple GPUs, each identified by their respective index, this would specifically choose the second GPU for the process. This granular selection can be essential when managing workloads across nodes with varying GPU configurations, allowing for the precise control of resource allocation at the device level. In this example, we also included the `--deploy-label gpu_enabled=true` option, which is necessary when deploying to a swarm if your worker nodes have a label set as in the example above.

**Example 3: Using a composite device spec with environment variables:**

```yaml
version: "3.7"
services:
  ml_app:
    image: my_ml_image
    environment:
     - CUDA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          cpus: '2'
          memory: 2G
    runtime: nvidia
    deploy:
      placement:
        constraints:
        - node.labels.gpu_enabled == true
      #Composite device selection using a custom capability
        --gpus '"capability=compute,count=2"'
```

*   **Commentary:**  Here we show another common use case, specifically requesting two devices that support the `compute` capability. Notice also the addition of the environment variable, `CUDA_VISIBLE_DEVICES`. Although the `--gpu` argument is enough to provide device access, some libraries and applications expect the use of the `CUDA_VISIBLE_DEVICES` variable to dictate what devices the application can use, so you should generally add it in your configurations. Also note we’re using a hypothetical `my_ml_image` which would need to have the appropriate libraries and dependencies to interact with NVIDIA devices. It’s also important to note that device selections based on capabilities are vendor specific, and need to be supported by the driver or runtime.

**Limitations and Considerations:**

While Swarm can utilize GPUs, its approach differs significantly from Kubernetes' GPU scheduling. In Kubernetes, the scheduler is aware of the number and types of GPUs available on each node, allowing for more sophisticated scheduling strategies based on application needs and node capabilities. Swarm, on the other hand, relies entirely on manual or scripted placement based on labels and lacks this native awareness of GPU specifics. This means you need to implement your own resource-aware scheduling to manage applications needing to leverage GPUs effectively across diverse hardware.

Swarm provides no real concept of resource quotas for GPUs, which may be a limitation if you need fine-grained control on how much GPU resource a service can consume. The GPU becomes accessible to the container if the `nvidia-container-runtime` is in place, which can be a challenge in complex multi-tenant environments.

**Resource Recommendations:**

*   **Docker Documentation:** Comprehensive explanations regarding the `--gpus` option, specifically for Docker runtime configuration and usage. This includes different syntax options, such as selecting based on UUID or specific capabilities. Pay special attention to how container runtimes are specified and configured, as that is how this mechanism is actually enabled.

*   **NVIDIA Container Toolkit Documentation:** Essential information and tutorials on installing and configuring the `nvidia-container-runtime`. It includes detailed information regarding GPU driver installation and the use of container runtime hooks necessary to make GPU devices available in a container environment. The specific instructions will depend on the host operating system.

*   **Community Forums:** Actively engage with communities, such as the Docker subreddit, Stack Overflow, or similar forums, as they often provide real-world advice and troubleshooting suggestions when dealing with GPU-accelerated workloads on Docker Swarm. Check for common issues related to driver compatibility, runtime configuration, or user-level access problems.
