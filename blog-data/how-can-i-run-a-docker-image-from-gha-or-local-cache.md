---
title: "How can I run a docker image from GHA or local cache?"
date: "2024-12-16"
id: "how-can-i-run-a-docker-image-from-gha-or-local-cache"
---

Alright, let's tackle this one. I've been through this dance with docker images and GitHub Actions (GHA) countless times, and optimizing the process is critical, especially when dealing with large projects. The issue of pulling images repeatedly, even when they haven’t changed, can slow down your pipelines dramatically. So, let’s break down how to leverage caching, both locally and within GHA, to minimize those redundant pulls.

The core concept here revolves around understanding how docker’s layer architecture works and how both your local docker daemon and GHA runners handle caching. Docker images aren’t monolithic entities; they are composed of layered filesystems. Each instruction in your dockerfile typically generates a new layer. This layering allows docker to only download the layers that have changed since the last build. Similarly, GHA runners can cache these layers to avoid pulling the same unchanged image multiple times in a workflow.

Let's start with the local scenario. Suppose you're in the development phase and making iterative changes to your docker application. Each time you build with `docker build`, if you’re not careful, you’ll essentially rebuild the image from scratch, which is wasteful if only a few things have changed. The first thing I check in a situation like that is whether or not I'm utilizing docker's buildkit. Buildkit is a replacement for the older builder that is significantly faster at performing builds due to concurrent execution of steps and an enhanced caching mechanism. To use buildkit, ensure you’ve set the `DOCKER_BUILDKIT=1` environment variable.

Now, if you already are using buildkit and you are still seeing a lot of rebuilding, chances are it's due to a poor dockerfile design. To understand why, let's look at an example of a problematic dockerfile:

```dockerfile
FROM ubuntu:latest

COPY ./app /app
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install -r /app/requirements.txt
CMD ["python3", "/app/main.py"]
```

In this scenario, if you change any file in your `/app` folder, the copy instruction changes, and then every instruction *after* the copy becomes invalidated – meaning the `apt-get`, the `pip3 install`, and everything else will have to run again from scratch!

The key to preventing this is to place the least-changing instructions first, and put the most frequently-changing code last. A better approach would be:

```dockerfile
FROM ubuntu:latest

RUN apt-get update && apt-get install -y python3 python3-pip

WORKDIR /app
COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY ./ .

CMD ["python3", "main.py"]
```

Now, changes to your application source code in the second `COPY` instruction will be cached until requirements change, saving a great deal of build time. Notice also the use of `WORKDIR` to reduce clutter and improve clarity. This technique, known as caching layer optimization, is crucial for local development. You can delve deeper into this approach in the official docker documentation, specifically the section on optimizing dockerfiles. For a more comprehensive understanding, I recommend “Docker Deep Dive” by Nigel Poulton.

Now, let's move to GHA workflows. It's essential to understand that GHA runners are typically ephemeral; each workflow runs on a clean virtual machine. This means that by default, every time your workflow runs, it must download all the required docker layers, similar to that initial slow local build we wanted to prevent. This can add significant overhead.

GitHub provides the actions/cache action which is not specifically for docker caching, but can be used for that as well. However, while it works, it’s not very docker-aware. Therefore I tend to prefer using the official docker/build-push-action instead with its built-in caching mechanism. Let me illustrate a basic example of a GHA workflow that utilizes this action to leverage docker caching:

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
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Login to Docker Hub (Optional)
        uses: docker/login-action@v2
        with:
          username: ${{secrets.DOCKER_USERNAME}}
          password: ${{secrets.DOCKER_PASSWORD}}

      - name: Build and Push Docker Image
        uses: docker/build-push-action@v4
        with:
          context: .
          push: true
          tags: your-dockerhub-username/your-image-name:${{ github.sha }}
          cache-from: type=gha,scope=${{ github.workflow }}
          cache-to: type=gha,scope=${{ github.workflow }}

```

In this workflow, the `docker/build-push-action` does the heavy lifting. Note the `cache-from` and `cache-to` attributes. By specifying `type=gha`, you are telling buildkit to use GHA's built-in caching backend and therefore, it stores built layers and image manifests associated with the specific workflow. This ensures that successive workflow runs on the same branch will skip rebuilding unchanged layers of your docker image. The scope, `${{ github.workflow }}`, creates a cache entry that is specific to the particular workflow definition, meaning that other workflows won't interfere with each other's cache. Also, by using the git sha as a tag, we have a unique identifier for each build which is helpful for version control.

Finally, I'd like to mention that if your project is more complicated, you may require more advanced build configurations. For example, using a docker compose file to create multi-container images or using a more powerful buildx builder that utilizes more resources by running it in docker. Let's examine a slightly more advanced example.

```yaml
name: Advanced Docker Build and Push

on:
  push:
    branches:
      - main

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        with:
          driver-opts: |
              image=moby/buildkit:master
              network=host #optional, if needed

      - name: Login to Docker Hub (Optional)
        uses: docker/login-action@v2
        with:
          username: ${{secrets.DOCKER_USERNAME}}
          password: ${{secrets.DOCKER_PASSWORD}}

      - name: Build and Push Docker Image using Docker Compose
        uses: docker/build-push-action@v4
        with:
          push: true
          file: docker-compose.yml
          load: false
          build-args: |
            APP_VERSION=${{ github.sha }}
          tags: your-dockerhub-username/your-image-name:${{ github.sha }}
          cache-from: type=gha,scope=${{ github.workflow }}
          cache-to: type=gha,scope=${{ github.workflow }}
```

Here, the key differences are the use of a `docker-compose.yml` file via the `file` argument and the `load` property set to `false`. This allows for more complex builds and pushes while still using the caching mechanisms we talked about. Using `load:false` reduces the overhead of pushing the built image to the local docker daemon inside the GHA runner. This is particularly relevant when the only intention is pushing to a remote registry. I recommend taking a look at the Docker documentation for the `buildx` command, as well as reading the build-push-action documentation to see all the configuration options available.

In short, efficiently caching docker images locally and in GHA comes down to a combination of mindful dockerfile design and the correct configuration of build tools. By paying attention to the layers and the provided caching mechanisms in both environments you can significantly decrease your build times and speed up your development pipeline. Don't hesitate to experiment with the settings provided to find the combination that suits your specific application. Remember, there is no one-size-fits-all approach when dealing with build infrastructure.
