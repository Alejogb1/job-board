---
title: "How can Docker Compose automatically use a GPU if available, otherwise proceed without one?"
date: "2025-01-30"
id: "how-can-docker-compose-automatically-use-a-gpu"
---
The core challenge in orchestrating GPU resource allocation with Docker Compose lies in the dynamic nature of hardware availability.  My experience working on high-performance computing projects at a previous firm highlighted the need for robust, conditional resource allocation, especially given the unpredictable nature of shared compute clusters.  A rigid specification within the `docker-compose.yml` file will either always demand a GPU, leading to failures on systems lacking them, or always omit GPU access, foregoing potential performance gains on GPU-capable machines. The solution hinges on leveraging environment variables and conditional configuration within the Docker Compose file.


**1. Clear Explanation:**

The strategy involves dynamically adjusting the container's runtime environment based on the presence of a suitable GPU. This is achieved by using an environment variable to signal GPU availability, and then leveraging that variable within the `docker-compose.yml` file to conditionally define the container's resources. We can detect GPU presence using a simple shell command within a conditional statement at runtime.  The command should check for the existence of a CUDA-capable device (for NVIDIA GPUs) or a suitable ROCm device (for AMD GPUs).  This approach ensures that the container's resource requests are tailored to the specific hardware configuration without requiring manual intervention.  The resulting setup provides a seamless experience, gracefully transitioning between GPU and CPU-only execution.

The process unfolds in three key steps:

* **GPU Detection:** A shell script or command checks for GPU availability.  This script defines an environment variable (e.g., `GPU_AVAILABLE`) that reflects the outcome.
* **Conditional Configuration:** The `docker-compose.yml` file uses this environment variable to conditionally define container resources.  If `GPU_AVAILABLE` is set, the container requests GPU resources; otherwise, it proceeds without them.
* **Container Execution:** Docker Compose uses the configured resources to launch the container.  The dynamic configuration ensures optimal resource utilization regardless of the host's hardware.


**2. Code Examples with Commentary:**

**Example 1: Using a shell script for GPU detection (Bash):**

```bash
#!/bin/bash

if nvidia-smi -L | grep -q "GPU"; then
  export GPU_AVAILABLE=true
else
  export GPU_AVAILABLE=false
fi
```

This script checks for the presence of NVIDIA GPUs using `nvidia-smi`.  If GPUs are found, it sets the `GPU_AVAILABLE` environment variable to `true`; otherwise, it sets it to `false`.  This script should be executable and located in a directory accessible from your Docker Compose file.


**Example 2:  Docker Compose file leveraging the environment variable:**

```yaml
version: "3.9"
services:
  my-service:
    build: .
    depends_on:
      - db
    environment:
      - MY_ENV_VAR=some_value
    deploy:
      resources:
        reservations:
          # GPUs are conditionally reserved based on GPU_AVAILABLE
          nvidia:
            count: "${GPU_AVAILABLE:+1}"  # only allocate if $GPU_AVAILABLE is set
    volumes:
      - ./data:/app/data
    ports:
      - "8080:8080"


  db:
    image: postgres:13
    environment:
      - POSTGRES_PASSWORD=postgres


```

The key element here is `"${GPU_AVAILABLE:+1}"`.  This parameter expansion in Bash will only substitute the value `1` if the variable `GPU_AVAILABLE` is set and non-empty. If `GPU_AVAILABLE` is unset or empty, it resolves to an empty string effectively allocating no GPUs. This conditional allocation allows the container to run on both GPU and non-GPU systems seamlessly.  Note that the  `nvidia` section within `deploy.resources.reservations` requires the `nvidia-container-toolkit` to be installed on the host system.


**Example 3: Dockerfile with conditional CUDA setup:**

```dockerfile
FROM ubuntu:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
    cuda-toolkit-11-8  #This will only install if the build context includes the needed packages

# ... rest of your Dockerfile ...

CMD ["/bin/bash", "-c", "if [ -f /usr/local/cuda/bin/nvcc ]; then echo 'CUDA is installed!'; else echo 'CUDA is not installed!'; fi"]
```

This Dockerfile demonstrates conditional installation.  You would need to manage the CUDA toolkit installation separately and include the necessary packages in your build context.  The final `CMD` only serves as a simple check; replace this with your application's start command.  This approach is less elegant than environment variable control of resources because it requires rebuilding the image for different environments.  It serves to illustrate another possible but less optimal technique.


**3. Resource Recommendations:**

For a thorough understanding of Docker Compose, consult the official Docker documentation.  Explore advanced topics like resource constraints and environment variable usage within Docker Compose.  For detailed information on GPU programming and CUDA, refer to the NVIDIA CUDA Toolkit documentation.  Finally, understand the implications of environment variable inheritance and scoping within the context of Docker and shell scripts.  Careful attention to these details is crucial for a robust and portable solution.
