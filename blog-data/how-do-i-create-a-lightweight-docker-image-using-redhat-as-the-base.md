---
title: "How do I create a lightweight Docker image using RedHat as the base?"
date: "2024-12-23"
id: "how-do-i-create-a-lightweight-docker-image-using-redhat-as-the-base"
---

, let's talk about lightweight Docker images based on Red Hat. It's a topic I've spent considerable time on, especially back when we were optimizing container deployments for our microservices architecture. The bloat in some base images was becoming a significant drag, impacting both build times and resource utilization in our Kubernetes clusters. It’s a challenge I've seen pop up frequently.

So, the core issue boils down to this: standard Red Hat base images, while robust and reliable, often include a lot of packages that are simply not needed for specific applications. They’re designed for general purpose use, not for minimal container footprints. We aim to reverse that and achieve what I consider ‘lean’ container images. We're not just talking about shaving off a few megabytes, but potentially orders of magnitude, impacting your infrastructure footprint significantly.

The approach isn’t particularly revolutionary but demands precision and an understanding of what's truly essential. This process, in my experience, can be broken down into a few critical steps. First, meticulous selection of your base image is paramount. Instead of a generic `redhat/ubi8` image, you might consider `redhat/ubi8-minimal`. That switch alone reduces the image size considerably. The `minimal` images are designed for containerized workloads and strip out much of the extraneous overhead.

Second, you need to perform careful package management within your image. It's a common trap to include development tools or debugging utilities within the final image. We're aiming for a runtime-only environment. This usually involves using `dnf` or `yum` (depending on your Red Hat version) to install *only* what your application *absolutely* needs. Think dependencies, runtime libraries, and possibly a few essential utilities like `curl` or `wget` if required. Forget compilers, debuggers, documentation, or any unnecessary add-ons.

Third, multi-stage builds are essential. This technique allows you to leverage one image for the build process, and then copy *only* the needed artifacts into a final, much smaller runtime image. This approach significantly reduces the amount of accumulated junk in your final container image. It’s often where the most significant size reductions come from.

Let’s illustrate with some code examples.

**Example 1: Using `ubi8-minimal` as a starting point and installing minimal dependencies.**

```dockerfile
# Stage 1: Runtime Image
FROM redhat/ubi8-minimal:latest AS runtime

# Update package lists and install only what is needed.
RUN microdnf update -y && \
    microdnf install -y --setopt=tsflags=nodocs \
    libstdc++ \
    ca-certificates && \
    microdnf clean all

# Copy in the application executable
COPY my-application /app/my-application

# Set the working directory and command
WORKDIR /app
CMD ["./my-application"]
```
In this basic example, we start with `ubi8-minimal`, update the package lists using `microdnf`, install only the `libstdc++` and `ca-certificates` packages (an application's common dependencies), clear the cache and then copy our application executable. Note the usage of `--setopt=tsflags=nodocs`. This flag is critical for keeping documentation packages out of the final image. We're setting a clean runtime image with just the essential files for our application to run.

**Example 2: Multi-stage build for Java applications**

```dockerfile
# Stage 1: Build Stage
FROM redhat/ubi8-openjdk-17 AS builder

WORKDIR /build

COPY pom.xml .
COPY src ./src

RUN mvn -B -e -s settings.xml dependency:resolve

RUN mvn -B -e -s settings.xml package

# Stage 2: Runtime Image
FROM redhat/ubi8-minimal:latest AS runtime

# Install jre and dependencies only for the runtime.
RUN microdnf update -y && \
    microdnf install -y --setopt=tsflags=nodocs \
    java-17-openjdk-headless  \
    ca-certificates && \
    microdnf clean all

WORKDIR /app
COPY --from=builder /build/target/*.jar ./app.jar
CMD ["java", "-jar", "app.jar"]
```
Here's a more complex example involving a java application. The first stage uses `redhat/ubi8-openjdk-17` as the basis for the build. It resolves dependencies, compiles the code and creates a `jar` file. The second stage utilizes `ubi8-minimal` and copies *only* the compiled jar and necessary java runtime elements. The important piece here is that the *entire build environment* including the development tools is discarded, leaving us with an image that contains only the bare minimum to run the Java application. The output of one stage is effectively passed to the next. We install `java-17-openjdk-headless` as this is the JRE, not the JDK, making this image much smaller.

**Example 3: Utilizing a custom user and security hardening**

```dockerfile
# Stage 1: Runtime Image
FROM redhat/ubi8-minimal:latest AS runtime

RUN microdnf update -y && \
    microdnf install -y --setopt=tsflags=nodocs \
    libstdc++ \
    ca-certificates && \
    microdnf clean all

# Create a dedicated user for security purposes.
RUN groupadd -r appuser && useradd -r -g appuser appuser

#Copy the application executable.
COPY --chown=appuser:appuser my-application /app/my-application

# Set the working directory, use the dedicated user, and set the command.
WORKDIR /app
USER appuser
CMD ["./my-application"]
```
This example builds upon the initial example but adds security considerations by adding a custom user ‘appuser’ and then ensuring that the `/app` directory is owned by that user. Running as a non-root user within the container is an essential security best practice. Notice that we use the `COPY --chown=` flag to ensure proper permissions after copying the application executable, and the `USER appuser` command specifies the non-root user that will run the application.

Beyond these examples, there are further strategies we used successfully. Layer ordering is vital. Place frequently changing layers (such as your application) *after* layers that are more static, such as base dependencies. This allows Docker to utilize its caching effectively which speeds up subsequent builds dramatically. Also, ensure your application is optimized itself; minimize external dependencies, use efficient coding practices, and reduce the footprint of all included files.

For further reading, I highly recommend “Docker Deep Dive” by Nigel Poulton as it provides in-depth understanding about Docker internals. For more on Linux best practices, especially around minimal images, "Linux System Programming" by Robert Love is incredibly insightful. Red Hat’s own documentation on UBI (Universal Base Images) on their website is also a valuable resource. Don't forget to explore the official documentation of `dnf` and `microdnf` too, as they are the fundamental tools you'll be using.

I found, from past experience, that meticulous attention to these steps really does make a difference in resource utilization, deployment speed and overall container management. It can feel a bit tedious at first but the benefits are significant over time. Keep in mind that there are no magic bullets in this process; it’s an exercise in careful planning and attention to detail.
