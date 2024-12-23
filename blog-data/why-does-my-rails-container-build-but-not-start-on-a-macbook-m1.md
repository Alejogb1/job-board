---
title: "Why does my Rails container build but not start on a macbook M1?"
date: "2024-12-23"
id: "why-does-my-rails-container-build-but-not-start-on-a-macbook-m1"
---

Let’s tackle this. I’ve seen this particular issue a few times over the past few years, often leaving developers scratching their heads, especially when moving to the arm64 architecture of the M1 series. You’re experiencing a container that builds successfully but then refuses to run – a frustrating scenario, indeed. The core of the problem frequently lies in architecture mismatches or dependency inconsistencies between the build and runtime environments. In practical terms, the image you're creating is likely optimized or built for a different CPU architecture than your macbook’s.

The first thing to consider is the base image you’re using in your Dockerfile. If you’re pulling an image that’s explicitly built for `amd64` (the traditional x86 architecture), your M1, which is arm64 based, will struggle to execute it properly. It might build because docker can emulate the architecture during the build phase, but that emulation falls apart when you try to *run* it, leading to those dreaded start failures. You need to ensure that your base image is either explicitly built for `arm64` or, more ideally, is a multi-architecture image that will gracefully adapt to your macbook’s processor. Docker itself has added a lot of tooling to help with that.

Let’s explore some concrete code examples to illustrate these points.

**Scenario 1: Explicitly specifying the architecture:**

Let's say you’re using an old Dockerfile that looked something like this:

```dockerfile
FROM ruby:3.1.2
# Rest of the Dockerfile here...
```

This implicitly assumes the architecture of the provided ruby image is what’s right for the host system. On older systems, you might get away with it. However, on M1, that ruby image, if it's only available for `amd64`, won't run smoothly. A more robust approach, especially if you're unsure about the image's supported architectures, is to explicitly request the right version.

Here's how you would modify the Dockerfile to handle this:

```dockerfile
FROM --platform=linux/arm64 ruby:3.1.2
# Rest of the Dockerfile here...
```

By adding `--platform=linux/arm64`, you're explicitly instructing Docker to pull the arm64-compatible version of the ruby:3.1.2 image. This small change can be the difference between a container that runs and one that does not. Now, this particular example forces the architecture and makes certain you get an arm64 build, which is good for your M1 macbook. But what if you need the same image to run on `amd64` in the cloud?

**Scenario 2: Multi-architecture builds**

A more flexible approach is to use multi-architecture images. These are specially built images that contain executables for various architectures, allowing your container to run on different hardware. In this case, you wouldn’t need to specify the platform explicitly because Docker will select the correct architecture automatically if the image supports it. Let’s demonstrate using a different example, with a more complete Dockerfile:

```dockerfile
# Dockerfile
FROM ruby:3.1.2-slim
WORKDIR /app
COPY Gemfile Gemfile.lock ./
RUN gem install bundler && bundle install
COPY . .
EXPOSE 3000
CMD ["bundle", "exec", "rails", "server", "-b", "0.0.0.0"]
```

This is a fairly common Rails app dockerfile. To ensure that this image is built in a way that will work both on an M1 mac and older intel-based cloud servers, the ruby image used should be compatible with multiple architectures. Now, the `-slim` variant of `ruby:3.1.2` should come with that compatibility built-in. Let's verify, though. We can build this image using the `--platform linux/arm64` argument for the local M1 mac, or we can use the `--platform linux/amd64` argument when building the same image for a cloud server. If the underlying image is multi-architecture, we can achieve that with one command:

```bash
docker buildx build --platform linux/arm64,linux/amd64 -t my-rails-app . --push
```

`docker buildx` is an extension of `docker` designed to handle multiple platforms. This command builds images for both `arm64` and `amd64` and pushes the multi-architecture build to the container registry, making it easy to deploy wherever your needs take you. The underlying image must also support both architectures. If you were not using a multi-architecture ruby image, this would still fail to start on your M1.

**Scenario 3: Dependency issues and platform inconsistencies**

Sometimes, the issue isn’t directly with the base image, but with native dependencies your application needs. These dependencies often have platform-specific builds, and it's not uncommon for these native gems to be compiled incorrectly during the build phase, leading to runtime failures. This can be tricky to diagnose.

For example, imagine a gem needs a specific C library. If that library is not available for arm64, or if the version linked during the build process was for a different platform, your container might build successfully because the build environment has libraries available, but crash during execution, when the library isn’t there or the compiled binary doesn’t match.

Consider a hypothetical case where your Gemfile relies on a database adapter which has native extensions:

```ruby
gem 'pg', '~> 1.0'
```

The `pg` gem often has native binaries that must be compatible with the system. If the build phase occurs in an environment that is `amd64` (possibly due to the base image or the platform setting of the build agent or your own setup) and then it gets deployed to your arm64 mac, the application may crash on startup when it tries to use the native extensions linked for `amd64`.

To address this, you have a couple options:

1.  Ensure you build inside a container that matches the target architecture. You could do this by building on your M1 mac, or using the `buildx` argument above to target both `arm64` and `amd64`.
2.  Consider using versions of gems that have pre-compiled binaries for your target architecture.
3.  Rebuild your gems with the correct architecture for your host, or use build phases to recompile the gem in a container that matches the target.

To help troubleshoot these issues further, you should look at the following resources:

*   **Docker documentation:** The official docker docs have extensive coverage of multi-architecture builds, buildx, and platform-specific image handling. Start there to make sure you have a handle on the foundational parts of containers and the ecosystem.

*   **“Effective DevOps” by Jennifer Davis and Katherine Daniels:** This book, while not specifically about Docker, covers the architectural considerations that underpin many of these kinds of issues. It is a good resource for understanding the broader context of software architecture.

*   **“Programming Phoenix” by Chris McCord, José Valim, and Bruce Tate:** Although specifically about the Elixir programming language, chapter 15 covers a wide range of deployment issues in detail, and has a discussion about using Docker in production. These discussions about deployments can give insight into troubleshooting errors related to building and running.

These three examples cover the core issues that I've encountered over the years. Your problem likely stems from one of these. The most frequent offender is that mismatch between build and run architectures, but also be cognizant of native library issues that can present as container build success, yet start failure. In general, ensuring you use multi-arch images when possible and explicitly specifying the platform during builds when needed should solve your problem.
