---
title: "How can CUDA be used within a GitLab CI Docker executor?"
date: "2025-01-30"
id: "how-can-cuda-be-used-within-a-gitlab"
---
The seamless integration of CUDA-accelerated workloads within a GitLab CI Docker executor hinges on a crucial understanding:  the Docker image itself must contain the necessary CUDA toolkit and drivers.  Simply installing these components within the CI job's runtime environment is insufficient; the underlying Docker image must already possess them. This is because the GPU driver is fundamentally a component of the operating system within the container, not an easily installed library at runtime.  This fundamental point, often overlooked, is the source of many integration failures.  In my experience supporting high-performance computing teams, this misunderstanding is the most prevalent obstacle.

My approach to resolving this involves a three-pronged strategy: meticulously crafting a base Docker image, employing appropriate build arguments, and implementing efficient job execution within the CI pipeline.

**1. Crafting the CUDA-Enabled Docker Image:**

This step is paramount. Building a custom Docker image with the appropriate CUDA toolkit version, drivers matching your target GPU architecture, and necessary libraries is critical.  Generic CUDA images are often available on Docker Hub, but their applicability depends on the specific GPU hardware available in your GitLab CI runners and the CUDA version compatibility of your project's dependencies.  I've often found that pre-built images lack specific dependencies needed for complex projects.

The Dockerfile should follow a structured approach. First, it needs a base image compatible with your hardware architecture (e.g., `nvidia/cuda:11.8.0-base`). This base image already incorporates the CUDA toolkit and drivers.  Next, install any additional system libraries or dependencies your project requires.  Finally, copy your application code and any necessary configurations.

**Code Example 1: Dockerfile for CUDA-enabled image**

```dockerfile
# Use a base image with CUDA
FROM nvidia/cuda:11.8.0-base

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install project-specific dependencies (example)
RUN pip3 install numpy cupy

# Set the working directory
WORKDIR /app

# Copy the application code
COPY . .

# Build the application
RUN cmake . && make

# Expose the application port (if necessary)
EXPOSE 8080
```

This Dockerfile demonstrates a basic setup.  Adjust the `nvidia/cuda` tag to match the specific CUDA version and base image necessary for your project's compatibility requirements. The `apt-get` commands handle system dependencies, while `pip3` installs Python-based libraries. Remember to replace the example dependencies with your actual project requirements. The `WORKDIR`, `COPY`, and `RUN` commands facilitate the build process within the container. The `EXPOSE` command is conditionally included based on the need to expose application ports.


**2. Utilizing Build Arguments and GitLab CI Configuration:**

Leveraging build arguments allows flexibility in tailoring your Docker image for various CUDA versions or other specific configurations. For instance, you can pass the CUDA version as a build argument to select the appropriate base image dynamically.  In your `.gitlab-ci.yml` file, you'll define your build stages, including the Docker build stage and the subsequent execution stage.

**Code Example 2: GitLab CI configuration (.gitlab-ci.yml)**

```yaml
stages:
  - build
  - test

build_image:
  stage: build
  image: docker:latest
  services:
    - docker:dind
  variables:
    CUDA_VERSION: "11.8.0"
  script:
    - docker build --build-arg CUDA_VERSION=$CUDA_VERSION -t my-cuda-image:latest .

test_application:
  stage: test
  image: my-cuda-image:latest
  script:
    - ./my_cuda_application
```

This example uses `docker:dind` (Docker in Docker) service to allow the build process inside of the runner. The `CUDA_VERSION` variable is passed to the `docker build` command. The `script` section of the `test_application` job then executes your CUDA application using the custom built image.  The Dockerfile would need to be modified to accept `$CUDA_VERSION` as a build-time argument to be dynamic.


**3. Implementing Efficient Job Execution:**

Once the CUDA-enabled image is built and pushed to your container registry, the CI pipeline can seamlessly execute your CUDA application.  This requires careful attention to resource allocation.  The GitLab CI runner needs to have the correct GPU drivers installed and assigned to the Docker container.  The `.gitlab-ci.yml` file can specify resource requests, but this heavily depends on your GitLab runner's configuration.

**Code Example 3:  Dockerfile modification for argument usage and resource requirements**

```dockerfile
ARG CUDA_VERSION
FROM nvidia/cuda:${CUDA_VERSION}-base
# ... rest of Dockerfile remains the same ...
```

```yaml
test_application:
  stage: test
  image: my-cuda-image:latest
  resources:
    limits:
      memory: 8Gi
      cpus: 4
    request:
      nvidia.com/gpu: 1  # Request one GPU
  script:
    - ./my_cuda_application
```

This demonstrates using a build-argument `CUDA_VERSION` in the Dockerfile, allowing flexibility for different CUDA versions. In the `.gitlab-ci.yml`, the `resources` section is essential for GPU resource allocation –  `nvidia.com/gpu: 1` requests one GPU.  Adjust the values (`memory`, `cpus`) to match your application's requirements.


**Resource Recommendations:**

For further learning, I suggest consulting the official Nvidia CUDA documentation, the Docker documentation, and the GitLab CI documentation.  These resources provide extensive information on CUDA programming, Docker image creation and management, and configuring GitLab CI pipelines.  Focusing on best practices for Docker image optimization, specifically around layer caching, is also crucial for efficient pipeline execution.  Furthermore, understanding the intricacies of your GitLab runner’s hardware and its configuration is crucial for avoiding common pitfalls.  The GitLab runner's capabilities and limitations will directly dictate your approach to GPU resource allocation.
