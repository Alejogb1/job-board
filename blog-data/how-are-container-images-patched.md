---
title: "How are container images patched?"
date: "2024-12-23"
id: "how-are-container-images-patched"
---

Let's tackle container image patching; it's a crucial aspect of maintaining secure and reliable applications in today's landscape, and it's something I've personally spent a fair amount of time navigating during my tenure in cloud infrastructure. The process isn't exactly straightforward, and understanding its nuances is vital for anyone deploying containerized workloads.

The core issue stems from the immutable nature of container images. Once built, an image is intended to be a static snapshot of your application and its dependencies. Direct modification isn’t really an option; you can’t just ‘patch’ a container image in place like you would a traditional operating system. This is intentional; it ensures consistency and reproducibility. The solution, therefore, involves creating new, updated images that include the necessary changes. This might sound inefficient, but the trade-off is a significantly more reliable and predictable deployment pipeline.

So, how *do* we go about it? Well, there are primarily two strategies: rebuilding the entire image and using layer-based updates, and occasionally, a more granular approach for specific use cases.

The most common method is to rebuild the image from scratch. You’d start with your base image, which could be an operating system image like `ubuntu:latest` or `alpine:latest`. Then, in your Dockerfile, you’d typically add your application code, install dependencies, configure environment variables and so forth. When a vulnerability is discovered in one of your base layers, or in a dependency you include, you would update the Dockerfile accordingly (e.g., update package versions, modify configurations). Then you simply rebuild the image using `docker build`, resulting in a new image with all of the corrected changes. It's thorough, and straightforward enough; the entire stack is rebuilt ensuring no overlooked corners, but it also means a potentially more time-consuming build process and a larger image than necessary.

Here is a simplified example of a Dockerfile rebuild workflow:

```dockerfile
# Dockerfile - Initial version
FROM ubuntu:20.04

RUN apt-get update && apt-get install -y some-application=1.0
COPY ./app /app
CMD ["/app/run"]

# ... later, a vulnerability is found in some-application

# Dockerfile - Updated version
FROM ubuntu:20.04
RUN apt-get update && apt-get install -y some-application=1.1 # Updated to a patched version.
COPY ./app /app
CMD ["/app/run"]
```

After building this new Dockerfile (using `docker build . -t my-patched-app:latest`) , you would deploy the new `my-patched-app:latest` instead of the original. Note that this new image will contain all updated layers even though only `some-application` changed; this process builds the entire image from the ground up. This works well for small-scale projects; for larger projects that need quicker deployment times, this process may be too verbose.

The other approach takes advantage of the layer architecture of container images. Every instruction in your Dockerfile typically creates a new layer, each based on the layer below it. Docker images are built from these layers; each layer is immutable and can be cached. This layered approach is crucial for optimizing image patching. The goal is to only rebuild the layers that have changed, thereby significantly reducing build times and minimizing image size updates.

To illustrate this, consider this workflow:

```dockerfile
# Dockerfile - Initial version
FROM ubuntu:20.04 as builder

RUN apt-get update && apt-get install -y some-builder-tool=1.0

COPY ./app-source /app-source
RUN some-builder-tool /app-source /app

FROM ubuntu:20.04
COPY --from=builder /app /app
RUN apt-get update && apt-get install -y some-app-dependency=1.0
CMD ["/app/run"]


#  ... later, a vulnerability is found in some-app-dependency

# Dockerfile - Updated version
FROM ubuntu:20.04 as builder

RUN apt-get update && apt-get install -y some-builder-tool=1.0

COPY ./app-source /app-source
RUN some-builder-tool /app-source /app


FROM ubuntu:20.04
COPY --from=builder /app /app
RUN apt-get update && apt-get install -y some-app-dependency=1.1 # Updated dependency.
CMD ["/app/run"]
```

In this example, we are taking advantage of multi-stage builds. The changes were limited to the installation of `some-app-dependency` in the final stage. Therefore, docker should be able to skip building the first `builder` stage as long as `some-builder-tool` version and the input `/app-source` did not change. We have optimized the build time by only recreating the necessary layers. This also reduces the amount of data required to be uploaded to the image registry. These optimizations are especially important when building frequently.

More complex scenarios sometimes require more targeted approaches, typically involving package managers inside the container. These are far less common and can be harder to maintain consistency with. Suppose you had a very specific library in a container that required a patch. Rather than rebuilding the entire image, it could be tempting to shell into a running container and apply the patch, committing this as a new layer. This has many drawbacks. This approach violates the principles of immutability and reproducibility. It also adds a new layer onto your existing image with the change you made. This updated image lacks a reproducible pipeline that tracks the change you just made. It is generally better to avoid this situation by making any required changes at the dockerfile level.

Here is an example demonstrating the *wrong* way of handling updates, showing how attempting to patch a running container can become problematic:

```bash
# Assume a running container called "my-container"

docker exec -it my-container bash

# Inside the container, try updating a vulnerable package (e.g., a specific python library)
apt-get update
apt-get install -y python3-specific-lib=1.2 # Imagine version 1.1 was vulnerable.

exit # Exit the shell

# Commit this change into a new image:
docker commit my-container my-patched-container:some-tag

```

While this seems like a quick fix, the changes are not recorded in the Dockerfile, making it hard to replicate the process later or to track the changes. This approach will quickly devolve into a confusing state. For this reason, the best solution is to build a new container image from an updated Dockerfile rather than trying to directly modify existing images.

For anyone looking for deeper insights into container technologies and their security aspects, I'd recommend several resources. "Docker Deep Dive" by Nigel Poulton is an excellent starting point for anyone looking to understand docker concepts. For more security-focused reading, "Container Security: Practical Techniques to Defend Your Cloud Native Infrastructure" by Liz Rice is very useful. The official docker documentation itself is a goldmine of information and contains many best practices and tutorials to explore.

In conclusion, container image patching isn't about modifying existing images but about generating new ones based on changes to their source files, whether through rebuilding the image from scratch or by leveraging layers in an optimized way. Careful selection of strategies, a solid understanding of your application dependencies, and proper use of Dockerfiles are all keys to an efficient and reliable container patching process. Understanding these nuances will save you time, resources, and potential headaches when maintaining your containerized applications.
