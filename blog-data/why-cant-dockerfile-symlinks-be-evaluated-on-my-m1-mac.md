---
title: "Why can't Dockerfile symlinks be evaluated on my M1 Mac?"
date: "2024-12-16"
id: "why-cant-dockerfile-symlinks-be-evaluated-on-my-m1-mac"
---

Alright, let's dive into this one. I recall a particularly frustrating sprint a couple of years back; we were migrating our build pipelines to arm64 architecture, specifically using M1 Macs, and hit this exact symlink issue with Dockerfiles. It wasn't pretty, but it did lead to a good understanding of what's happening under the hood.

The core problem you're encountering isn't a bug in Docker per se; it’s fundamentally about how Docker's build process interacts with the filesystem and the architecture differences between x86-64 and arm64. The crucial element here is that Docker build contexts are often copied – not directly referenced - into a build environment. When a symlink exists within your build context, its *target* (what the symlink points to) might not be present or accessible in the copied environment, or may have a different structure from your host system.

Think of it this way: when you initiate a `docker build .` command, everything in that current directory (the build context) is bundled and potentially transferred to a Docker daemon, which could be running on a completely different operating system or architecture. During this process, the symlinks are copied, but they're not automatically resolved in the context of the build environment. This is crucial. Docker's build engine operates as a distinct entity, insulated from the host system's direct file structure.

On x86-64 systems, this might appear to work "fine," primarily because the build environment and host filesystem are often very similar, often also using x86-64 architecture and similar directory structures. It's less about the CPU architecture per se, and more about architectural homogeneity and path equivalence. If your host and the docker daemon share the same architecture and a similar directory layout, a symlink to, say, `../data/config.json`, might happen to resolve correctly in the docker container build environment, purely by chance.

However, on M1 Macs, which are arm64 based, and especially with Docker Desktop using virtualized environments to emulate x86 if you are using x86 images, this potential for path equivalence breaks down. Symlinks that might have coincidentally worked on your previous x86 system, now point to paths within the *host's* file system that are absolutely *not* present or accessible within the docker container's build process. Moreover, if the daemon is running inside a virtual machine for compatibility purposes, the pathing and architecture differences may cause issues even with other arm64 images.

This issue highlights a core principle: Dockerfiles should avoid relying on symlinks within the build context. It's inherently fragile and leads to these types of frustrating build failures. The build context needs to be self-contained. Let me show some practical examples and how we addressed this:

**Example 1: The Problem**

Let’s say we have the following structure:

```
project/
├── data/
│   └── config.json
├── src/
│   └── app.py
└── docker/
    ├── Dockerfile
    └── symlink_to_config -> ../data/config.json
```
And a simplistic Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .

RUN cat symlink_to_config

CMD ["python", "src/app.py"]
```

When building this on an M1 Mac, you're likely to see a "no such file or directory" error when trying to read `symlink_to_config` during the `RUN` command. Even though it exists within the build context, the docker engine isn’t resolving it relative to the *host's* `../data/config.json`. It’s looking for `../data/config.json` *within* the docker container build environment, and it doesn't exist.

**Example 2: Solution with Copying the Source**

A more robust solution is to directly copy the target of the symlink into the docker context. In our example, that would be directly copying `config.json`. This makes the build context self-contained and predictable.
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY src/ src/
COPY data/config.json data/config.json

CMD ["python", "src/app.py"]
```
In this modification, we directly copy `config.json` into the root directory of the container’s `/app/` folder. Now the container has direct access to the config, without any reliance on symlinks.

**Example 3: Solution Using a Build Arg**

Sometimes, it’s simply not desirable to include potentially sensitive configurations directly in the Dockerfile. In this case, we can leverage a build argument, though it doesn't eliminate copying as in solution 2.

First, let's assume our configuration is stored outside the current build context to address sensitivity concerns.
```
project/
├── src/
│   └── app.py
└── docker/
    ├── Dockerfile
```
And our Dockerfile might look like this:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

ARG CONFIG_FILE
COPY src/ src/
COPY $CONFIG_FILE config.json

CMD ["python", "src/app.py"]
```

Then, when building, specify the build argument:
```bash
docker build --build-arg CONFIG_FILE=../data/config.json -f docker/Dockerfile .
```
This will take the path specified in the build argument and copy that file during the build step. Again, we are still copying files rather than symlink evaluation, but here we explicitly control and define the config source via the `CONFIG_FILE` variable. This avoids directly embedding the configuration file path within the Dockerfile.

**Key Takeaways and Recommendations**

*   **Avoid symlinks:** The key takeaway is to avoid relying on symlinks within the docker build context. It leads to brittle builds and makes them architecture-dependent, something we should avoid as much as possible. Copying data directly is a more reliable solution.
*   **Build contexts:** Remember that the build context gets copied to the Docker daemon. Pathing relative to your host machine is not maintained. Ensure everything the container build requires is present within the copy.
*   **Self-contained contexts:** Always aim for self-contained build contexts. The build should operate predictably regardless of the machine that is executing the build command, whether that be your M1 mac or a cloud-based build agent.
*   **Build arguments:** Leverage build arguments for flexible configurations, as demonstrated in Example 3, but remember the build stage will still involve a copy operation of files specified.

For further reading, I'd recommend diving into the Docker documentation itself; the section on build contexts and Dockerfiles provides significant insight. Additionally, reading sections about build processes and filesystem interactions within books on Linux containerization such as "Docker Deep Dive" by Nigel Poulton can add to a deeper understanding of these behaviors. "Programming in the UNIX Environment" by W. Richard Stevens is also valuable for understanding how filesystem operations such as symlinks behave. Understanding the underlying mechanics helps make solutions more apparent.

In my experience, adopting these practices dramatically reduces issues related to subtle file system interactions and significantly contributes to consistent and reproducible builds. It's tempting to take the easy route and use symlinks, but the time saved is not worth the frustration it introduces further down the line, especially when dealing with differing architectures.
