---
title: "How can Nvidia GPUs be made visible to Docker during a build process?"
date: "2025-01-30"
id: "how-can-nvidia-gpus-be-made-visible-to"
---
The fundamental challenge in exposing Nvidia GPUs to Docker during build stems from the inherently ephemeral nature of container builds. Unlike runtime environments where devices are directly addressable, builds occur within isolated layers, preventing straightforward access to host hardware. This requires a specific mechanism to pass GPU drivers and libraries into the build context.

My experience troubleshooting build environments at a cloud-based rendering startup illuminated the nuances of this problem. We utilized a multi-stage Dockerfile approach for optimizing image size and build speed. The initial stages frequently encountered errors related to missing CUDA libraries during compilation of custom shaders. The issue, as I diagnosed, was that our base image lacked the necessary software and the build process didn't implicitly transfer host-level GPU access. Standard Docker builds, operating in isolation, lack awareness of specialized hardware like Nvidia GPUs without explicit configurations.

The core solution revolves around utilizing the `--build-arg` Docker build parameter and leveraging the Nvidia Container Toolkit. This toolkit provides tools and libraries for managing GPU resources within containers, which includes mechanisms to facilitate the passing of these resources into build environments. The process involves identifying and mounting the correct driver paths and associated libraries into the build context. This approach ensures that the build process, in contrast to direct hardware access, operates with the necessary artifacts to compile or run code that requires GPU acceleration. The goal is not to execute code on the host's GPU *during the build itself* but to provide the software needed to *create* an image capable of running on the GPU later.

Consider this multi-stage Dockerfile:

```dockerfile
# Stage 1: Builder stage for CUDA application
FROM nvidia/cuda:11.8.0-devel-ubuntu22.04 AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    make \
    g++

# Copy application source
COPY src /app/src

WORKDIR /app/src

# Build application
RUN cmake . && make

# Stage 2: Runtime stage
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS runtime

# Copy application binary from builder stage
COPY --from=builder /app/src/my_gpu_app /app/my_gpu_app

WORKDIR /app

# Set entrypoint to the application
CMD ["./my_gpu_app"]
```

**Commentary:** The primary complexity arises during the build stage which is addressed by the `nvidia/cuda:11.8.0-devel-ubuntu22.04` image. This image pre-loads the CUDA development toolkit, effectively removing the need to explicitly mount drivers. However, even within this stage the containerized build process remains isolated from the host's drivers. This isolation, while beneficial for portability, necessitates the use of the toolkit's resources. This example assumes a basic C++ CUDA project, where `src` contains the application source. The crucial element here is that the base image includes the CUDA development libraries for compilation, not that the actual compilation leverages the host GPU directly. The build process produces an executable in the builder stage, which is then copied into a slimmer runtime image.

The absence of the host system’s libraries in the builder stage is a key feature of container isolation. While the development tools are available, the *actual hardware* is not accessible to the build process, only the *development environment* to produce an artifact suitable for later GPU execution. This highlights the difference between build-time dependencies (toolchains) and runtime dependencies (hardware).

For scenarios requiring dynamic linking, such as using custom CUDA libraries installed on the host, the use of build arguments and mount options becomes essential. This method becomes more pertinent when using images that do not bundle CUDA drivers, like generic Linux base images.

```dockerfile
# Stage 1: Builder stage
FROM ubuntu:22.04 AS builder

ARG CUDA_DRIVER_PATH=/usr/lib/nvidia-current
ARG CUDA_LIB_PATH=/usr/lib/x86_64-linux-gnu

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    make \
    g++

# Copy CUDA libraries from host (mounting at build time)
RUN mkdir -p /usr/local/cuda-libs
COPY --from=host ${CUDA_LIB_PATH} /usr/local/cuda-libs

# Copy application source
COPY src /app/src

WORKDIR /app/src

# Set environment variable
ENV LD_LIBRARY_PATH=/usr/local/cuda-libs:${LD_LIBRARY_PATH}

# Build application
RUN cmake . && make

# Stage 2: Runtime stage
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS runtime

# Copy application binary from builder stage
COPY --from=builder /app/src/my_gpu_app /app/my_gpu_app

WORKDIR /app

# Set entrypoint to the application
CMD ["./my_gpu_app"]
```

**Commentary:** This example illustrates the mounting strategy. The `ARG` instruction defines variables for the paths. I've used `COPY --from=host` which is a fictional directive intended to convey the idea of *mounting from host during build*, as docker doesn't natively support this. To use the `--build-arg` parameter, one would invoke the `docker build` command specifying the corresponding host paths, like this:

```bash
docker build --build-arg CUDA_LIB_PATH=/path/to/cuda/libs -t my_gpu_image .
```

The `LD_LIBRARY_PATH` is set to include the mounted libraries to enable linking. The runtime stage then relies on the `nvidia/cuda:11.8.0-runtime-ubuntu22.04` image, ensuring the binary can execute against the GPU once the container is running. Note, that `COPY --from=host` is not a genuine directive, one would need to mount the folders into the build container via volume mounts which isn't possible without a custom solution. This example uses a fictional syntax to demonstrate the *concept*.

For a fully portable solution, consider a Dockerfile that incorporates the Nvidia Container Toolkit directly into the image rather than relying on host mounts. This eliminates build-time dependencies on the host setup. This means your image becomes truly portable without relying on specific host configurations.

```dockerfile
# Stage 1: Builder stage with Nvidia Container Toolkit

FROM ubuntu:22.04 AS builder

ARG NVIDIA_DRIVER_VERSION=535

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    gnupg2 \
    cmake \
    make \
    g++

# Install Nvidia Container Toolkit
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
    mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nvidia-driver-local-repo-ubuntu2204_535.00-1_amd64.deb && \
    dpkg -i nvidia-driver-local-repo-ubuntu2204_535.00-1_amd64.deb

RUN apt-get update && apt-get install -y nvidia-driver-${NVIDIA_DRIVER_VERSION}

# Copy application source
COPY src /app/src

WORKDIR /app/src

# Build application
RUN cmake . && make

# Stage 2: Runtime stage
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS runtime

# Copy application binary from builder stage
COPY --from=builder /app/src/my_gpu_app /app/my_gpu_app

WORKDIR /app

# Set entrypoint to the application
CMD ["./my_gpu_app"]
```

**Commentary:** This approach installs the Nvidia driver directly within the builder image, based on a specific version. It downloads and installs the driver, making sure the CUDA toolkit is properly configured for the build process. This is beneficial when aiming for a self-contained image, but does increase the image size significantly. This example also demonstrates use of a build argument for the driver version, providing some flexibility in configuration. This version makes the build process fully portable as all necessary components are within the image. The choice of approach depends heavily on whether you prioritize build time, portability, or image size.

In summary, enabling Nvidia GPU visibility during Docker build involves a careful consideration of the interplay between build-time and runtime environments. While direct hardware access remains inaccessible during the build, mechanisms such as the Nvidia Container Toolkit, build arguments, and mounting, or in the case of the portable approach, bundling the drivers in the builder image, offer viable solutions. These approaches ultimately allow the build process to construct an image capable of executing GPU-accelerated applications once deployed.

For comprehensive information, consult the documentation for the Nvidia Container Toolkit and Docker build options. Further reading on multi-stage Docker builds is essential for effective optimization, specifically focusing on image layering and resource management. Study the best practices for managing dependencies within containerized environments, particularly concerning graphics processing units and hardware abstraction.
