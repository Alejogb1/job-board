---
title: "Why can't symlinks in Dockerfiles be evaluated on M1 Macs?"
date: "2024-12-23"
id: "why-cant-symlinks-in-dockerfiles-be-evaluated-on-m1-macs"
---

Ah, symlinks in Dockerfiles on M1 Macs. It’s a topic that, from my past encounters, can certainly lead to some head-scratching moments, especially if you're used to the more straightforward behavior on x86 architectures. Let me break down why this occurs and how to navigate it, drawing from my own experiences battling build issues across different platforms.

The core of the issue lies within the architectural differences and how Docker handles build contexts and the subsequent image creation. Traditionally, Docker builds on x86 (like Intel or AMD chips) rely on a Linux-based daemon that operates natively on that architecture. When you execute a `docker build` command, the daemon essentially copies the build context – all the files in your directory – to a temporary location within the daemon. During the build process, commands in your Dockerfile are executed by this daemon. Crucially, the daemon sees the symbolic links (symlinks) as regular filesystem objects within the context and resolves them *at build time* using its native architecture and file system. That resolution, on an x86 system, results in the symlink destination being part of the image.

Now, enter M1 Macs (Apple Silicon). Docker Desktop on M1 machines uses a virtualization layer to run a Linux virtual machine (VM), which then houses the Docker daemon. This is where the disparity begins. When you execute `docker build` on an M1 Mac, the build context, which originates on the macOS file system, gets transferred to the Linux VM. Now, this transfer process *does not resolve the symlinks on the macOS file system*. Instead, it copies the symlink *as a symlink*. This key distinction changes everything.

Within the Linux VM, when the Dockerfile instructions run and attempt to access a file through a symlink, the path lookup is performed by the Linux guest OS within the VM, not the macOS filesystem it originally referenced. However, because the target of the symlink (the original path that the symlink pointed to on the macOS file system) isn't actually within the *Linux guest file system*, the symlink target cannot be resolved, resulting in build failures or unexpected behavior.

To illustrate this, let's consider a common scenario. Let's say you have a directory structure like this:

```
project/
├── src/
│   ├── actual_file.txt
│   └── my_symlink -> ../../another_directory/target_file.txt
├── another_directory/
│   └── target_file.txt
└── Dockerfile
```

And, let’s say your Dockerfile includes instructions to copy and use ‘my_symlink.’ Here's what happens, and why it fails on M1 Macs but works on x86:

**Scenario 1: Working (x86)**

On an x86 machine, the build process operates directly on a linux based Docker daemon. Thus, the symlink gets resolved, and the content of `target_file.txt` is included in the image

**Scenario 2: Failing (M1 Mac)**

On an M1 Mac, the symlink is not resolved before being transferred to the VM. Within the Docker build process inside the VM, the path `../../another_directory/target_file.txt` does not exist, leading to the symlink failing at build time, even though both the symlink and the original file existed in the build context on the host.

Let's get more concrete with some code. Here’s a simple Dockerfile designed to demonstrate this:

```dockerfile
# Example 1: Demonstrating symlink failure

FROM alpine:latest

WORKDIR /app

COPY . .

RUN cat my_symlink # This would succeed on x86 but will fail on M1
```
*Code Snippet 1*

This Dockerfile (assuming the folder structure provided above), if run on an x86-based Docker daemon, would likely succeed because the symlink would be resolved during the `COPY` phase *on the host x86 machine*. The `cat` command would operate against the resolved path. However, on an M1 Mac, the `cat` command would fail during the build phase because the file linked to does not exist within the VM's filesystem.

Now, let’s consider a second, more common scenario: the inclusion of a virtual environment in a Python project.

```dockerfile
# Example 2: Python venv issues

FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

RUN source venv/bin/activate && python script.py # This would work on x86 but cause potential issues on M1
```
*Code Snippet 2*

Here, if the `venv` directory is symlinked to another location outside the project during its creation, you could encounter issues during the `RUN source venv/bin/activate` step, especially if the python virtual environment `venv` folder wasn't copied over as a real directory into the root of the `app` directory because of a symlink existing within the directory structure being copied.

The reason is that `venv/bin/activate` is a bash script, and if `venv` is symlinked, the shell will resolve the symlink *inside* the linux vm and the target file may not be there, especially if the target exists outside of the build context. The `RUN` directive is executed within the Linux VM.

Finally, lets consider a case that might be successful on the x86 machine, but would not be consistent across systems, making it hard to build the same docker image on M1 and x86:

```dockerfile
# Example 3: Unintended Symlink resolution at Copy Stage

FROM ubuntu:latest

WORKDIR /app

COPY bin/* /app/bin/ # Assuming bin/target_link exists as a symlink on the host.

RUN ls -l /app/bin # List the directory content after copy.

```
*Code Snippet 3*

In this snippet, if bin/target_link is a symlink on the host and targets a location on the host, the copied file in /app/bin/ could have entirely different targets when the same instruction was run on a different build system, introducing subtle differences into the final built images.

So, what's the solution? Here are a few approaches that, based on my own trials and tribulations, tend to be quite reliable:

1.  **Avoid Symlinks in Your Build Context:** This is the most direct and reliable method. If feasible, replace symlinks with actual directories or files. This typically means restructuring project directories in the case of symlinked venvs or copying the target directory instead of the symlink. In more complex scenarios, scripts could be used to create the directories and files required rather than relying on symlinks.

2.  **Use Docker’s `COPY --chown`**: While this won't solve the root problem, you can use this option to potentially bypass the problem by ensuring that any file being copied has the correct permissions.

3.  **Re-evaluate Your Directory Structure**: Perhaps the symlinks were a byproduct of a particular development workflow or environment. Refactor the structure to avoid the necessity of symlinks.

4.  **Use BuildKit Features**: Docker's BuildKit engine, enabled via environment variables or the `DOCKER_BUILDKIT=1` environment variable, offers more control and potentially different behaviors around context handling. While it doesn’t directly fix symlink resolution on M1 Macs, it’s worth investigating whether alternative build strategies or caching behaviors resolve issues on a case-by-case basis.

To further enhance your understanding of Docker internals and file system interactions, I recommend studying *“Operating System Concepts” by Silberschatz, Galvin, and Gagne* for a solid foundation on operating system fundamentals. For a deep dive into Docker, consider "Docker in Action" by Jeff Nickoloff and “The Docker Book” by James Turnbull. These resources provide invaluable knowledge that clarifies the complexities behind the simple commands you use on a daily basis. Understanding these aspects of the underlying infrastructure allows for greater control and a more nuanced understanding of issues like symlink management.

In conclusion, while the behavior of symlinks in Dockerfiles on M1 Macs can be perplexing initially, it’s ultimately a result of the virtualization layer and the separation between the host filesystem and the Linux VM running the docker daemon. The key is to acknowledge the difference in resolution during the build process, and then adapt your build strategy. By restructuring your projects, removing symlinks in the build context, and understanding where your instructions actually execute, you can build resilient and portable Docker images across architectures.
