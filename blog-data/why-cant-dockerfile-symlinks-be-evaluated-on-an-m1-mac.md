---
title: "Why can't Dockerfile symlinks be evaluated on an M1 Mac?"
date: "2024-12-16"
id: "why-cant-dockerfile-symlinks-be-evaluated-on-an-m1-mac"
---

Alright, let's talk about Dockerfile symlinks on M1 Macs. I've personally spent more than a few late nights debugging similar issues, and it's a nuanced problem that boils down to a combination of factors, not just some isolated quirk. It’s less about the M1 itself, and more about how the Docker daemon interacts with different underlying architectures and file system behaviors.

The core issue stems from how Docker builds images. When you use the `COPY` or `ADD` instructions within a Dockerfile, the daemon doesn't simply copy files verbatim. It creates a snapshot of the source context, and that snapshot is what gets layered into the image. Now, when it encounters a symbolic link (symlink), it needs to decide what to do with it. On Intel-based machines, or x86_64 architecture, the Docker daemon usually follows the symlink and copies the target of the link, not the link itself. This works fine if the target is within the build context.

However, on M1 (arm64 architecture) machines, you'll notice that symlinks often don't get evaluated correctly. Instead, you might see the symlink itself copied into the image, or even worse, the build process might simply fail if the symlink points outside the build context. It's not a bug, per se, but a consequence of different operating systems and architecture interpretations. This arises primarily from how filesystem namespaces and emulation are handled, especially during the virtualization process that happens under the hood within Docker Desktop. The build process is happening in a virtualized environment using the qemu emulator on the M1 chip. The interaction with the host system's file structure, including symlinks, becomes sensitive to differences between these architectures.

Think of it this way: the build context is like a sandbox. Symlinks within the sandbox (your build directory) often work as expected because the Docker daemon can readily interpret the target paths. But when the symlink's target falls outside this sandbox or, more critically, relies on host architecture-specific features that aren't present within the container's environment, things go sideways. The emulation layer might not be able to accurately translate those symbolic link paths across architectures and operating systems.

Let me provide a few examples to make it clearer.

**Example 1: Symlink Within the Build Context (Generally Works)**

Consider the following directory structure and `Dockerfile`:

```
├── app
│   └── my_file.txt
├── symlink_to_app -> ./app
└── Dockerfile
```

The `Dockerfile` would be simple:

```dockerfile
FROM ubuntu:latest

COPY . /app

RUN ls -al /app

CMD ["cat", "/app/app/my_file.txt"]

```

When you build this (`docker build -t example1 .`), it will *typically* work on an M1, *assuming the target exists at build time and within the build context*. The symlink `symlink_to_app` gets copied, and when you list contents of the `/app` directory within the container, you'll see it (as a link). The subsequent `cat` command would resolve through the symlink. This works because the *target* of the symbolic link is within the same build context, and the filesystem semantics within the Docker image are relatively straightforward in this case.

**Example 2: Symlink Pointing Outside the Build Context (Will Fail or Be Ignored)**

Now, let's look at a case that typically fails:

Let's say you have a symlink that points to a directory that exists on your host system, *but isn't part of your Docker build context*. Let's assume you have the structure as follows:

```
├── data_dir
|   └── real_file.txt
└── my_symlink -> /Users/yourusername/data_dir/real_file.txt
└── Dockerfile
```

The `Dockerfile` is as follows:
```dockerfile
FROM ubuntu:latest

COPY . /app

RUN ls -al /app

CMD ["cat", "/app/my_symlink"]
```
When you build (`docker build -t example2 .`) and run this, you’ll likely find that the `COPY . /app` either ignores the symlink, or the `cat` command will error out because the target is outside the image's file system scope. Docker doesn't follow symlinks that point outside the build context for security and isolation purposes. It’s trying to encapsulate the application and all its dependencies into a discrete package, and following arbitrary paths outside the build context would violate that.

**Example 3: The Solution: Copying Files Directly**

The proper solution is to usually avoid relying on symlinks directly during the build process by adjusting the Dockerfile itself. A common workaround is to `COPY` the actual target files or directories. For instance, in the example two, you can modify the Dockerfile to:

```dockerfile
FROM ubuntu:latest

COPY data_dir /app/data_dir

RUN ls -al /app

CMD ["cat", "/app/data_dir/real_file.txt"]
```

Now, the `data_dir` is explicitly copied into the container image, sidestepping the issue with symlink evaluation entirely. The `cat` command would be successful in this case. This approach ensures the image contains what's required and is not dependent on paths outside of the container environment or on host-specific symlink behaviors.

In summary, the trouble with symlinks and M1 Macs during docker builds isn't really about the M1 chip specifically, but about how Docker handles file system abstraction between different architectures, and how the emulation layer interacts with the host’s file system. The behavior of symlinks in docker is often not architecture independent and can change dramatically when moving across Intel and ARM architecture. Docker build context limitations and the way images are structured as layers of file system differences are also key. Therefore, you should refrain from using symlinks across the build context, and where required, try to copy the target of the symlinks explicitly into the image.

For further depth on this, I highly recommend reading the Docker documentation on build context. Also, delving into resources that cover qemu emulation and its interaction with Docker would be insightful. Specifically, look at papers discussing filesystem virtualization in container environments and the challenges of maintaining consistent behavior across different architectures. Lastly, researching the principles behind Docker layering and its implications on file system access will greatly enhance your understanding of these issues. Specifically, you'd find value in works like "Understanding and Deploying Docker Containers" by David Clinton or "Docker in Action" by Jeff Nickoloff which often have sections dedicated to build context and its specifics. There are also numerous excellent academic papers about the details of containerization at the operating system level, published by institutions working on virtualization technologies. These generally provide more fundamental background on the technology at play.
