---
title: "How can I create a light weight docker image using a redhat based base image?"
date: "2024-12-23"
id: "how-can-i-create-a-light-weight-docker-image-using-a-redhat-based-base-image"
---

Okay, let’s tackle this. I’ve spent a considerable amount of time optimizing container images, particularly those derived from red hat distributions, and let me tell you, it’s a topic where a bit of strategic thought really pays off. The goal, of course, is a small, secure, and performant image, and the journey to that ideal can be more involved than one might initially expect. I'm going to walk you through it based on experience, not theory, focusing on techniques that have consistently delivered results for me.

The common pitfalls we encounter when building from a red hat base are usually around unnecessary bloat – think bundled utilities, development tools, and accumulated package caches. These add significant overhead to the image size, and hence, increase distribution time and resource usage. My early projects had images easily clocking in at multi-gigabyte sizes, a painful learning curve, and it really emphasized the necessity for a disciplined approach to layer creation.

Let's first talk about the base image selection itself. If you’re starting from, say, `registry.redhat.io/rhel8/rhel-init`, or a similar ‘full’ version, be prepared for an uphill battle against excess baggage. These are designed to accommodate a broad range of functionalities, not necessarily tailored for lean containerization. A better strategy is often to move towards ‘minimal’ variants. For example, consider using `registry.redhat.io/ubi8/ubi-minimal` or similar if you’re using rhel 8. These stripped down images offer a significantly smaller starting point. Remember, you’re building an application container, not a complete operating system environment.

After choosing your minimal base, focus on efficient layering. Each line in your `dockerfile` that results in a filesystem change contributes to a layer in your final image. The order matters. Place infrequent changes at the start of the dockerfile. Changes such as installing package dependencies should ideally be at the start as those will rarely change if your code does.

Here's the first key technique: minimize package installation. Instead of blindly installing a 'recommended' package set, only install the absolute bare minimum needed by your application. And where possible, perform multi-stage builds. That is where you build your application using a larger build environment, copy out the application binaries/artifacts and then copy them to the final light weight image, avoiding any unnecessary compilers, development libraries and so on in the final container.

For illustration, let’s assume we’re building a simple python application. Here’s how you might approach the `dockerfile` using a multi stage build technique:

```dockerfile
# Stage 1: Build environment
FROM registry.redhat.io/ubi8/python-38 as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Stage 2: Final image
FROM registry.redhat.io/ubi8/ubi-minimal
WORKDIR /app
COPY --from=builder /app/app.py .
COPY --from=builder /app/venv /venv
COPY --from=builder /app/requirements.txt .

ENV PATH="/venv/bin:$PATH"
EXPOSE 8080
CMD ["python", "app.py"]
```

In this example, the first stage, named `builder`, leverages a larger python base image to install requirements and build artifacts. Critically, this stage doesn't end up in the final image. The second stage starts with a slimmed-down `ubi-minimal` image and *only* copies over the necessary files (application code, dependencies, and necessary environment). This avoids bloating the final layer with build tools and libraries you don't need at runtime. The `--no-cache-dir` in the pip install command helps keep the temporary data from polluting your intermediate build layer.

Next, let's look at package management. When installing packages with `yum`, always use the `--setopt=tsflags=nodocs` and `--setopt=clean_requirements_on_remove=1` flags. The `nodocs` option prevents the unnecessary installation of documentation files, and the `clean_requirements_on_remove` ensures that redundant dependencies are removed, further reducing the size footprint. You can also use the `yum clean all` instruction at the end of a package install step to clear cached data. Here’s an example illustrating this point:

```dockerfile
FROM registry.redhat.io/ubi8/ubi-minimal

RUN yum -y install --setopt=tsflags=nodocs --setopt=clean_requirements_on_remove=1 openssh-server && yum clean all

RUN mkdir /var/run/sshd
EXPOSE 22
CMD ["/usr/sbin/sshd", "-D"]
```

In this case, we’re installing `openssh-server`, a very common use case, and you can see the use of `nodocs` and `clean_requirements_on_remove` as well as `yum clean all`. A minor optimization is the usage of the short form of `&&` command chaining, instead of a verbose `RUN`.

Finally, be extremely mindful about what you copy into your image. Never copy entire directories unless absolutely necessary, and always prioritize the explicit inclusion of specific files. This goes hand-in-hand with using a `.dockerignore` file. This file prevents docker from copying unnecessary files into the build context, which can affect the build process and the size of the image. Create one in your project root that excludes everything that is not needed in your image.

Here's an example of a more complex copy operation demonstrating the specificity principle:

```dockerfile
FROM registry.redhat.io/ubi8/ubi-minimal

WORKDIR /app
#COPY . . #avoid this
COPY app/app.py .
COPY app/config.json .
COPY assets/logo.png ./assets

RUN ls -la

CMD ["python", "app.py"]
```
In this example, we are selecting only specific files/folders that are needed for the application. As you can see, we avoid a blind copy of the entire project by using `COPY <source> <destination>` instructions per file or directory. The `ls -la` command gives a very simple sanity check of the final image contents (this should be removed once happy with the image).

Let's talk resources. For a deep dive on best practices in container image building, you should consult “Docker in Action” by Jeff Nickoloff, a comprehensive guide that covers not only image building but also advanced topics on orchestration. “Programming Kubernetes” by Michael Hausenblas and Stefan Schimanski gives a strong background in container runtime environments and should provide additional context around containerized applications. The official docker documentation is, of course, a valuable resource. Red Hat's documentation on ubi images is also essential, as it contains specific guidelines for using their base images, which is particularly useful for red hat based images. Also, reviewing the official documentation on the specific version of red hat you are using for any further specific optimizations would also be beneficial.

In summary, crafting lightweight docker images using a red hat base image isn’t simply a matter of luck, it’s a methodical exercise in precise layering, minimalistic package usage, smart multi-stage builds, specific file copies, and understanding of underlying principles. It’s a skill that develops with practice, so don’t be discouraged if your first attempts are less than perfect. The crucial thing is to constantly review each step of the process to identify and eliminate inefficiencies. Each small adjustment contributes to the end goal of lean, efficient and high-performing containerized applications.
