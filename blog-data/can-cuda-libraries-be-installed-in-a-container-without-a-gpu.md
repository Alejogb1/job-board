---
title: "Can CUDA libraries be installed in a container without a GPU?"
date: "2024-12-23"
id: "can-cuda-libraries-be-installed-in-a-container-without-a-gpu"
---

, let's talk about CUDA and containers. It’s a fairly common question, and I've definitely seen my share of headaches trying to sort this out over the years, particularly back when we were first adopting containerization at that old robotics startup.

The short answer is, yes, you absolutely *can* install CUDA libraries within a container even if the host machine—the one running your container—lacks a physical GPU. Now, the *why* and *how* of that is where things get interesting. It's less about magically making GPU computation happen where there's none and more about preparing the environment for when a GPU *is* available. Think of it like building a Lego set; you can lay out all the pieces and instructions even if you don't have the final baseplate to build on.

The key concept to understand here is the separation between compilation, linking, and runtime. CUDA libraries consist of two primary components: the *development libraries* (headers, static libraries) and the *runtime libraries* (dynamic shared libraries). During the build process inside the container, you need access to the development components to compile CUDA code. The runtime components are what the actual executables depend on when they're executed, ideally on a system with a GPU. If you are building docker images that will be used on both CPU and GPU environments, the separation is very important to consider.

Here's a practical situation I ran into once: we were deploying an AI model training pipeline using containers. Our developers were building and testing on their local machines, which were often CPU-only, but the target deployment environment had beefy GPUs. We had to ensure our containers had all the CUDA dependencies *without* needing a GPU for initial setup. The goal was to enable seamless transitions and ensure that the images created on CPU based environments worked correctly when deployed on GPU environments.

Let's explore the "how" with a few concrete examples. I’ll demonstrate this using Dockerfiles, as it’s a fairly standard method for container image creation.

**Example 1: Using a CUDA Base Image (and its drawbacks)**

A seemingly straightforward approach is using a CUDA-enabled base image. NVIDIA provides these, often named something like `nvidia/cuda:<version>-devel-<os>`. You might think this is the complete solution, and indeed, it's very convenient. However, it comes with several drawbacks, which I will demonstrate here:

```dockerfile
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git python3 python3-pip

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
RUN cmake . -B build && cmake --build build

CMD ["./build/your_cuda_executable"]
```
This Dockerfile pulls the full CUDA development environment which includes the compiler, libraries, and drivers. This will build successfully on any environment, but it will result in a huge container image, as it's bundling a whole lot of stuff that you probably will not use if you don't have a GPU in the runtime. Also, it will expect that the drivers are installed in the environment that it runs on, which is not ideal, and might lead to conflicts with the system ones.

**Example 2: Using a Multi-Stage Build and CPU-based Base image**

A better approach, in my view, is a multi-stage build. We use a CUDA-enabled base image for the compilation stage and then copy only the required libraries to a smaller CPU-based image which will contain the application, thus making the end image smaller and less dependent on CUDA drivers. Let's see an example:

```dockerfile
# --- Build Stage ---
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git python3 python3-pip

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
RUN cmake . -B build && cmake --build build

# --- Runtime Stage ---
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
   libcublas-dev libcudart-dev  && \
   mkdir -p /usr/local/cuda-12.2

COPY --from=builder /app/build/your_cuda_executable /app/your_cuda_executable
COPY --from=builder /usr/local/cuda-12.2/lib64/ /usr/local/cuda-12.2/lib64/
COPY --from=builder /usr/local/cuda/include /usr/local/cuda/include

CMD ["/app/your_cuda_executable"]
```
Here, we define the `builder` stage, using a CUDA image to compile. Then, we create a lean runtime image based on `ubuntu:22.04`. We copy the compiled executable and the CUDA runtime libraries only, as well as needed includes to be available at runtime, into the final image. This results in a much smaller image and we are reducing driver dependencies in the environment.

**Example 3: Minimalistic Approach using Runtime dependencies**

An even more refined approach would be to only use the minimum runtime dependencies needed, and do not carry any include headers with our images. This minimizes even more the final image size and reduces potential conflicts. We can install only what is strictly required at runtime:

```dockerfile
# --- Build Stage ---
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 as builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git python3 python3-pip

WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

COPY . .
RUN cmake . -B build && cmake --build build

# --- Runtime Stage ---
FROM ubuntu:22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
   libcublas12 libcudart12 && \
   mkdir -p /usr/local/cuda-12.2/lib64

COPY --from=builder /app/build/your_cuda_executable /app/your_cuda_executable
COPY --from=builder /usr/local/cuda-12.2/lib64/libcudart.so.12 /usr/local/cuda-12.2/lib64/
COPY --from=builder /usr/local/cuda-12.2/lib64/libcublas.so.12 /usr/local/cuda-12.2/lib64/

ENV LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:${LD_LIBRARY_PATH}"

CMD ["/app/your_cuda_executable"]

```

This example is further minimized by only copying needed runtime libraries (`libcudart.so.12` and `libcublas.so.12`) and their dependencies, instead of the full `lib64` directory. We also set up the `LD_LIBRARY_PATH` to ensure the libraries are found at runtime. Remember that the specific libraries might change, depending on your application requirements, as well as on the CUDA version you're using. Inspect your dependencies to make sure to include everything needed.

**Important Considerations**

Several factors play into this process beyond what I have covered here.
Firstly, the CUDA driver compatibility is something that needs to be taken in account. Container images built on a specific CUDA version might be incompatible with other versions of the driver. The best way to approach this, is to use specific base images that match the targeted environment, or to test your docker images thoroughly to ensure that no runtime errors occur.

Secondly, make sure you're familiar with the nuances of your chosen container runtime (Docker, containerd, Podman, etc.) and its integration with NVIDIA Container Toolkit. This toolkit provides the necessary interface between the container and the host GPU drivers, when you eventually do run your container on a GPU equipped system.

For deeper learning, I'd highly recommend these resources:

*   **NVIDIA's official CUDA Documentation:** This is the bible for all things CUDA. You can find details on library versions, installation procedures, and much more.
*   **"Programming Massively Parallel Processors: A Hands-on Approach" by David B. Kirk and Wen-mei W. Hwu:** This book provides a very in-depth look into GPU architectures, CUDA programming model, and optimization techniques.
*   **Docker Documentation on Multi-stage Builds and NVIDIA GPU support:** The official Docker docs are very clear about how multi-stage builds work and how to run GPU containers using the correct nvidia runtime flags.
*   **NVIDIA Container Toolkit Documentation:** This contains very valuable information on how containers can interact with GPUs.

In conclusion, while you *can* indeed install CUDA libraries within a container without a GPU being present, it’s crucial to understand the distinction between build-time and runtime dependencies. By leveraging multi-stage builds and only including essential runtime libraries, you can build lightweight, portable containers that are ready to execute when the GPU finally comes into play. It's a practical approach that has saved us many headaches and made our deployment pipelines more efficient. Remember always to properly test your builds to catch any errors in the build or execution phase.
