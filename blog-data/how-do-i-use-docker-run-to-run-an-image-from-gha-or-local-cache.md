---
title: "How do I use `docker run` to run an image from (gha or local) cache?"
date: "2024-12-23"
id: "how-do-i-use-docker-run-to-run-an-image-from-gha-or-local-cache"
---

,  Getting `docker run` to use cached images efficiently, whether from GitHub Actions (gha) or your local environment, is a common pain point, and I’ve definitely spent my fair share of time optimizing this. It’s less about a single magic command and more about understanding Docker’s layer caching mechanism and how it interacts with different contexts. So, let's break it down.

Fundamentally, Docker uses layers to build and store images. Each instruction in a Dockerfile creates a new layer. These layers are cached, and Docker reuses them whenever possible to speed up subsequent builds and runs. The core principle to efficient caching lies in structuring your Dockerfile so that frequently changing instructions are at the bottom, and static instructions sit at the top. The `docker run` command itself doesn't directly interact with these caches; rather, it relies on the image being present, and that image, in turn, is the result of cached build layers.

First, let’s consider the local scenario. When you build an image using `docker build`, the daemon checks if there are any existing layers that match the instructions in your Dockerfile. If there are, it reuses them; otherwise, it creates new ones. Thus, the key to leveraging the local cache is to ensure that your Dockerfile is constructed logically. Let's illustrate with a simple example. Suppose I'm developing a python application. I generally would structure the Dockerfile something like this:

```dockerfile
# Stage 1: Builder
FROM python:3.9-slim-buster AS builder

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN python setup.py sdist

# Stage 2: Final Image
FROM python:3.9-slim-buster

WORKDIR /app

COPY --from=builder /app/dist/*.tar.gz .
RUN pip install --no-cache-dir *.tar.gz

COPY . .
CMD ["python", "app.py"]
```

In this example, we have a multi-stage build. The first stage, "builder," installs the dependencies based on `requirements.txt`. If that `requirements.txt` doesn't change between builds, `RUN pip install --no-cache-dir -r requirements.txt` is reused from the cache. The second stage then copies the output of the builder and executes the app. This approach minimizes the amount of rebuilds.

When I run `docker run <image>`, docker checks if the image is present in local storage. If so, it uses it. The layers are already established from building, so caching is a consideration for builds, not runs, in this specific case. This is a commonly overlooked point. `docker run` uses an existing image. The cache is used during the build to produce that image. To ensure this is working, make changes to your source code, leaving the `requirements.txt` untouched. Build the image again. The build should be remarkably faster compared to a build with an altered `requirements.txt`.

Now, let's move onto GitHub Actions, where the situation is a little more nuanced. In a gha workflow, containers are spun up in new environments each time you run it, making local caches irrelevant. This means each job starts without access to previously built layers by default.

You can still take advantage of caching, but you need to explicitly persist the Docker layer cache. There are two primary methods here: using the `docker/build-push-action` with the correct parameters, or using the built in caching features of GHA. Let’s look at how I would approach this in a gha workflow.

```yaml
name: Docker Build and Push

on:
  push:
    branches:
      - main

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    - name: Build and Push Docker Image
      id: docker_build
      uses: docker/build-push-action@v3
      with:
        context: .
        push: true
        tags: ${{ secrets.DOCKER_USERNAME }}/my-app:latest
        cache-from: type=gha
        cache-to: type=gha,mode=max
```

Here, the `cache-from` and `cache-to` parameters are critical. They instruct the build action to pull existing layers from gha cache and to store any new layers in gha cache. In this configuration, I've opted to store the cache locally on GitHub Action's infrastructure. Notice that the build action is separate from the run. The gha build action will create the image using gha's cache if possible, and then push that image to docker hub, or a similar registry. Later you might run this image elsewhere. The `docker run` command will leverage your system's local cache for the image pulled, if present, when the image tag is resolved.

A further optimization is using buildx with a dedicated docker cache volume in GHA. This would look something like this in gha:

```yaml
name: Docker Build and Push with Buildx Volume Cache

on:
  push:
    branches:
      - main

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_USERNAME }}
        password: ${{ secrets.DOCKER_PASSWORD }}
    - name: Build and Push Docker Image
      id: docker_build
      uses: docker/build-push-action@v3
      with:
        context: .
        push: true
        tags: ${{ secrets.DOCKER_USERNAME }}/my-app:latest
        cache-from: type=local,src=/tmp/.buildx-cache
        cache-to: type=local,dest=/tmp/.buildx-cache
    - name: Save buildx cache
      uses: actions/upload-artifact@v3
      with:
          name: buildx-cache
          path: /tmp/.buildx-cache
    - name: Restore buildx cache
      uses: actions/download-artifact@v3
      with:
          name: buildx-cache
          path: /tmp/.buildx-cache
```

This approach leverages a filesystem based cache. It explicitly saves and restores the `/tmp/.buildx-cache` directory to artifact storage between runs, which further accelerates the build process. `docker run` itself still just consumes the resulting image.

In summary, `docker run` itself doesn't use a cache directly. It relies on pre-built images that benefit from effective layer caching during the build process. When using local builds, ensure that your Dockerfile is optimized for layering. For gha, leverage the caching capabilities either built into the `docker/build-push-action` or via dedicated volume caching through `buildx` and artifact storage. It’s all about understanding how the caches interact at the build stage to provide the most efficient images to the `docker run` command.

To deepen your understanding, I’d recommend reading "Docker Deep Dive" by Nigel Poulton which goes into the internal mechanics of docker caching in great detail. Another excellent source is "Using Docker" by Adrian Mouat which covers docker build strategies and workflows in detail. The official docker documentation also provides comprehensive guidance on docker build caching. Studying these should provide a much deeper understanding of effective layer caching with docker.
