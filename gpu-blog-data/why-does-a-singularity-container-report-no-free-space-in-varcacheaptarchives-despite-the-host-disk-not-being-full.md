---
title: "Why does a singularity container report 'no free space' in /var/cache/apt/archives despite the host disk not being full?"
date: "2025-01-26"
id: "why-does-a-singularity-container-report-no-free-space-in-varcacheaptarchives-despite-the-host-disk-not-being-full"
---

Singularity containers, by default, operate with a layered file system overlay. This mechanism, while efficient for distributing immutable application environments, can lead to discrepancies between the reported free space within the container and the actual free space available on the host system, particularly regarding directories like `/var/cache/apt/archives`. The apparent “no free space” error arises not from a host-level disk capacity issue, but from how Singularity manages container storage and how changes are handled within the overlay.

The fundamental issue resides in the writeable overlay applied on top of the base read-only image. Singularity typically uses a temporary directory on the host for these changes. When the container is instantiated, a temporary directory (often within `/tmp` or a user-specific `.singularity` directory) is created. Any file modifications, including downloads via `apt`, are written to this temporary directory within the overlay. This temporary directory has an associated size limit, which is separate from the host’s filesystem capacity. If the accumulated writes exceed this allocated space within the overlay, regardless of host disk space, the container will report a full disk error in areas like `/var/cache/apt/archives` where `apt` stores downloaded packages. This error message is therefore not an indication of overall host disk exhaustion, but rather the ephemeral, isolated storage for the running instance being filled.

The overlay system operates on the concept of copy-on-write. When a container needs to modify a file in the base image, a copy of the file is made in the writeable overlay, and the change is applied there. The base image remains unmodified. Consequently, even if the base image has a very large `/var/cache/apt/archives` directory, the container’s local version starts as a blank directory in the overlay. Then, as `apt` downloads packages, these are written to the overlay. Because the overlay is usually small, it quickly fills if many large packages are cached by `apt`.

Here are some examples and associated analyses:

**Example 1: Basic apt update and install**

```bash
# Host system
singularity shell my_container.sif

# Inside the container shell
apt-get update
apt-get install some-package
```

In this common scenario, `apt-get update` downloads the package lists, and `apt-get install some-package` retrieves the requested package and its dependencies. These operations place the downloaded `.deb` files within `/var/cache/apt/archives`. If the total size of downloaded files exceeds the allocated space for the overlay, even when installing a single moderate-sized package, it will result in an error even if the host system has ample disk space. The size of the writeable overlay, which defaults to a modest value, is a critical factor in this process and the reason for the error message.  The container experiences limited space because it's within the temporary overlay rather than the host storage.

**Example 2: Examining Overlay Space via df**

```bash
# Host system
singularity shell --writable-tmpfs my_container.sif

# Inside the container shell
df -h /var/cache/apt/archives
df -h /tmp
```

The `--writable-tmpfs` flag when executing the container forces Singularity to mount a memory-backed tmpfs as the overlay instead of using disk. This makes it explicit and obvious that the temporary nature of the overlay is the source of the error.  This command, run inside the container, shows the disk usage specifically for `/var/cache/apt/archives` and for the `/tmp` directory where the container is often staged by Singularity. The size displayed for `/tmp` (which is the mount point in this case for the ephemeral overlay) is small and distinct from the host’s `/tmp`, making it obvious the limited space is coming from the overlay mount, not the host. It will likely show very different results compared to the host. The size allocated to the tmpfs is determined by the operating system’s default or specific Singularity runtime configurations and it is usually bounded.

**Example 3: Cleaning the cache within the container**

```bash
# Host system
singularity shell my_container.sif

# Inside the container shell
apt-get clean
```

After receiving the error, the command `apt-get clean` is often the initial recourse. This command removes the package archives within `/var/cache/apt/archives`, thus freeing up space within the container’s overlay. This will temporarily resolve the problem, but only until the cache fills again. While cleaning the archive will resolve the lack of space, this doesn't alter the underlying issue of the limited size of the container's writeable layer, and the problem is likely to occur again if multiple or larger packages are installed. This is because the fundamental space constraint is in the overlay and not the host, which is again not transparent.

To address this “no free space” situation within Singularity, several strategies should be considered. One common approach is to use the `--writable-tmpfs` flag, as shown above, which makes the overlay a memory-backed mount instead of one based on the disk which will implicitly limit its size to the system memory and is a better solution than using disk backed overlays. Another approach would be to build custom base images that include necessary packages and dependencies, minimizing the need to use `apt` within the running container. This will move the overhead from runtime to build time. Careful planning should be applied to identify which approach is more appropriate given the constraints of the system.  It's also good practice to utilize container orchestration tools which can automatically clean these types of overlay directories and handle the issue of limited space on a wider scale. The appropriate solution really depends on balancing container immutability, development iteration speed and resource usage.

In summary, the “no free space” reported by Singularity containers within `/var/cache/apt/archives` is misleading. The error is not an issue of host disk capacity, but rather the result of the layered file system overlay, which has a limited temporary storage. Understanding the copy-on-write behavior and the associated limitations of the overlay is crucial for troubleshooting such issues and finding appropriate solutions.  Container management tools will need to be appropriately configured to resolve the problem, not the host system.

For further understanding of Singularity container storage, I recommend consulting documentation related to Singularity's storage options. In addition, studying general concepts in container file system management, such as copy-on-write layers and temporary storage, would enhance one's comprehension. Lastly, reviewing the `apt` documentation, particularly its usage of `/var/cache/apt/archives`, will help in understanding why this particular location experiences this issue within containers. These resources collectively provide a thorough base for addressing the complexities of container file systems and their behavior within isolated environments.
