---
title: "How do I create light weight docker images based on redhat images?"
date: "2024-12-16"
id: "how-do-i-create-light-weight-docker-images-based-on-redhat-images"
---

Alright, let's talk about lightweighting those redhat-based docker images. It's a situation I've faced countless times in my career, especially when dealing with resource-constrained environments or aiming for faster deployment pipelines. We often start with these incredibly feature-rich, yet bulky, redhat base images and find ourselves needing to trim the fat. It's rarely a single-step solution, but a combination of thoughtful techniques that work together to achieve the desired outcome.

My experience dates back to a large-scale microservices project where we initially used standard RHEL images for each service. The deployment times were atrocious and resource consumption was through the roof. We had to aggressively optimize our approach. The core issue is that many pre-built images come packed with tools and libraries that are simply not needed for the specific application you are deploying. They're designed for general use, which is understandable, but it’s up to us as engineers to tailor them for our narrow use case.

The primary goal is to minimize the image size, which directly impacts build and deploy times, resource consumption, and even security surface. Let’s explore the key strategies that proved most effective.

First and foremost, we must adopt a **multi-stage build process**. This technique is a game-changer. Think of it as creating two separate dockerfiles – one for building your application and the other for producing the final, minimal image. In the first stage, you build everything including all the necessary build tools, dependencies, etc. Then, in the second stage, you copy *only* the compiled artifacts and the runtime dependencies to a much smaller base image. This separation ensures your production image doesn’t carry unnecessary development baggage.

Here's a basic example using a hypothetical java application:

```dockerfile
# stage 1: build stage
FROM maven:3.8.6-openjdk-17 AS builder

WORKDIR /app

COPY pom.xml .
COPY src ./src

RUN mvn clean install

# stage 2: final image
FROM registry.access.redhat.com/ubi8/ubi-minimal:latest

WORKDIR /app

COPY --from=builder /app/target/*.jar ./app.jar

CMD ["java", "-jar", "app.jar"]
```

In this example, the first stage uses a full `maven` image for building. The second stage, however, leverages `ubi-minimal`, a greatly reduced version of the Red Hat Universal Base Image, and only copies the generated jar file. Notice how we use `COPY --from=builder` to pull the artifact from the builder image rather than introducing a new step.

Secondly, **careful package selection** in the second stage is crucial. Even within `ubi-minimal`, you should explicitly install only what your application needs. Avoid a blanket approach of simply copying everything. Use your application runtime error logs as well as ldd and other introspection tools to identify exact dependencies and ensure you are not pulling in superfluous libraries.

Let's elaborate with an example for a node.js application:

```dockerfile
# Stage 1: build
FROM node:18 as builder
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

# Stage 2: final stage
FROM registry.access.redhat.com/ubi8/ubi-minimal:latest
WORKDIR /app
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/package*.json ./
RUN npm install --omit=dev
EXPOSE 3000
CMD ["node", "dist/index.js"]
```

In this Node.js example, we explicitly use the `--omit=dev` flag during `npm install` in the final stage to exclude development dependencies, reducing the final image size. This meticulous approach to package installation makes a huge difference. Also, note that we are using `ubi-minimal` instead of `ubi8` image which is smaller and therefore faster to deploy.

Third, we should look into **optimizing the base image**. While `ubi-minimal` is a great starting point, we can refine it further. For instance, if your application doesn’t require a shell, you could use a scratch image (though that’s more for extreme cases) or a pre-built image with even less included packages. In addition, removing unneeded files or executables that came with the base image will also contribute to smaller images. This requires investigation into the contents of the redhat base images, using tools like `docker history` and exploring the directories.

To illustrate the concept of optimizing the base image itself, we will build from a base `ubi8/ubi-minimal` image and install only essential components and perform a simple cleanup procedure.

```dockerfile
FROM registry.access.redhat.com/ubi8/ubi-minimal:latest

RUN microdnf update -y
RUN microdnf install -y tar gzip which

# this would depend on your app, but we can remove some basic things from ubi-minimal that are often not required
RUN microdnf remove -y nano vim ed man less info findutils shadow-utils bzip2 grep

# optional steps to further optimize - requires advanced know-how and knowledge of the base image
# RUN rm -rf /usr/share/doc/*
# RUN rm -rf /var/cache/*
# RUN microdnf clean all

WORKDIR /app
COPY my-app /app/my-app
CMD ["/app/my-app"]

```
In this example, we are removing some utilities and text editing programs, which might not be required in a production application container, and installing those which might. The important point to notice is that the specifics of the `remove` commands would depend on your target application requirements and understanding of the image. Using `microdnf` instead of `dnf` also has its impact on image size. The optional steps (commented out) show advanced steps like removing documentation which requires very careful consideration but can further reduce the image size.

Beyond these three core techniques, other strategies include using optimized file formats like `.tar.gz` to reduce the size of copied files within the container image, carefully ordering docker commands to optimize layer caching and ensure only changed layers are rebuilt, and regularly reviewing and updating your base images to get the benefits of any optimization from Red Hat.

It’s essential to mention, though, that pursuing minimalism must be balanced with functionality. Overly zealous optimization might inadvertently break your application. Therefore, testing is critical after any image optimization.

For further deep-dives, I would recommend examining the official docker documentation, specifically the sections related to multi-stage builds and image layer caching. Additionally, the OCI (Open Container Initiative) specification document provides a comprehensive overview of container image specifications which is helpful for advanced usage. “Docker Deep Dive” by Nigel Poulton is an excellent resource covering docker internals and best practices, including image optimization techniques, as well as the detailed usage of OCI specifications. Lastly, I would strongly recommend reviewing the Red Hat Universal Base Image (ubi) official documentation as well as the release notes to keep up to date with any changes.

Ultimately, building lightweight docker images is an iterative process. It requires a solid understanding of docker concepts, the redhat base image you're using, and a detailed understanding of your application and its requirements. There is no "one size fits all" approach, rather you should adjust your optimization strategy to fit your application and needs. I hope these strategies, combined with the given resources will prove useful to you.
