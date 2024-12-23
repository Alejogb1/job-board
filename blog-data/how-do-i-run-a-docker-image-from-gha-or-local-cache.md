---
title: "How do I run a docker image from GHA or local cache?"
date: "2024-12-23"
id: "how-do-i-run-a-docker-image-from-gha-or-local-cache"
---

Alright,  I recall a project a few years back where we were heavily relying on docker for microservice deployments and, simultaneously, were working towards drastically reducing build times within our GitHub Actions (GHA) pipelines. The challenge of efficiently utilizing cached docker images, whether from local development or within GHA, was paramount. Getting this wrong can lead to painfully slow deployments and wasteful resource consumption. It's more about finesse than brute force, focusing on smart strategies that leverage docker’s layer caching mechanism.

The core problem stems from the way docker images are built and stored. They are composed of layers, and if a layer hasn't changed, it doesn’t need to be rebuilt or re-downloaded. Understanding this is crucial. When we talk about running a docker image from a GHA cache or local cache, we’re essentially focusing on reusing existing layers and avoiding redundant operations. We need to tell docker and GitHub Actions to look for those previously built or pushed layers. This approach speeds things up dramatically.

From a local perspective, it's mostly straightforward. As long as you have built an image, docker will retain the layers unless you specifically remove them. The command to run an image is always `docker run`, irrespective of whether it’s freshly built or pulled from a local cache. For instance, if you have built an image tagged as `my-app:latest`, running it locally would look like this:

```bash
docker run -d -p 8080:8080 my-app:latest
```

This would start your container in detached mode, mapping port 8080 on your host machine to port 8080 inside the container. If you've pulled this image from a registry, or built it previously, docker will reuse the locally cached layers. That's the magic of docker at a local level.

Now, GHA introduces another layer of complexity. GHA runners are generally ephemeral. This means the environment where the action runs is spun up for each workflow and torn down afterward. Any local changes, including docker image layers, are lost. This is where GHA's caching mechanism comes into play. We need to explicitly tell GHA to cache specific docker layers. This requires saving and restoring the docker image or specific layers across different workflow runs.

The way to accomplish this is by leveraging the docker build command and carefully constructing the action steps. For instance, let's say we have a dockerfile and we build it with a tag `my-image:latest` during a github workflow. We can use the cache functionality using a unique cache key. Note that depending on your GHA runner environment, Docker-in-Docker might be required in your workflow which impacts performance. For illustration purposes, I'm using the default docker setup. I’ll be using the `actions/cache` action in the following examples. Here’s a simplified example of how you can implement this in a GHA workflow:

```yaml
name: Docker Build and Cache

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Set up Docker Buildx
        id: buildx
        uses: docker/setup-buildx-action@v2

      - name: Cache Docker Image Layers
        id: cache-docker
        uses: actions/cache@v3
        with:
          path: /tmp/.buildx-cache
          key: ${{ runner.os }}-docker-image-${{ hashFiles('Dockerfile') }}
          restore-keys: |
            ${{ runner.os }}-docker-image-

      - name: Build Docker Image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: false
          tags: my-image:latest
          cache-from: type=local,src=/tmp/.buildx-cache
          cache-to: type=local,dest=/tmp/.buildx-cache
```

In this example, the `actions/cache` action is used to cache the `/tmp/.buildx-cache` directory, where buildx (specified in the `docker/setup-buildx-action`) stores its build cache. The key is composed of the runner operating system and a hash of the Dockerfile which helps to trigger a cache miss only if the Dockerfile is modified. The `cache-from` and `cache-to` arguments in the `docker/build-push-action` instruct docker buildx to use the cached layers during builds and update the cache after a build. This ensures the next run can efficiently build. Note that `push: false` is used in this example, to focus on local caching; if you want to push the image to a registry, you should set it to true and add the appropriate registry configuration.

Another approach is to use image tags to implement versioning and, instead of caching just intermediate layers, we can cache built images in a dedicated registry which can be useful to share images across different projects and workflows. The following example demonstrates this idea. In this case, we're building a new image each time and pushing to a registry (you should replace with your registry).

```yaml
name: Docker Build, Tag and Cache

on:
  push:
    branches:
      - main

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v3

      - name: Log in to registry
        uses: docker/login-action@v2
        with:
            registry: ghcr.io
            username: ${{ github.actor }}
            password: ${{ secrets.GITHUB_TOKEN }}

      - name: Set up Docker Buildx
        id: buildx
        uses: docker/setup-buildx-action@v2

      - name: Build and Push Docker Image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: ghcr.io/${{github.repository}}/my-app:${{ github.sha }}
          cache-from: type=registry
          cache-to: type=registry
```

In this example, we build the image, tag it with the commit sha and push to ghcr.io. This allows the `docker/build-push-action` to use registry based caching, so if the same image with the same commit is built again from a workflow or locally, the layers will be pulled from the registry instead of being built.

For more intricate situations where you have specific intermediate layers that are shared between projects, it’s worth delving into tools such as buildkit’s cache exporters or using dedicated layer caching solutions. Buildkit provides flexible options for exporting cache layers to various storage locations, which can further enhance caching efficiency.

For deeper understanding, I recommend exploring the official docker documentation specifically on layer caching and also diving into the buildx documentation. Another excellent resource is "Docker in Action," by Jeff Nickoloff and Karl Matthias, as it provides thorough details on how to structure dockerfiles and implement complex build strategies. I would also advise reviewing the GHA documentation regarding the `actions/cache` action and the various nuances surrounding caching in CI/CD environments. Finally, the official Docker documentation provides the most up-to-date information on new features and best practices regarding image building and caching.

My experiences have shown that efficient docker image caching isn't a 'set it and forget it' solution. It requires careful consideration of your specific build process and an understanding of how docker's layer system works. By thoughtfully applying the techniques described and referring to the mentioned resources, you can significantly optimize build times, and create a more efficient development and deployment pipeline.
