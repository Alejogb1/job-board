---
title: "Why is LVM mapping visible on the host but not inside Kubernetes containers?"
date: "2024-12-23"
id: "why-is-lvm-mapping-visible-on-the-host-but-not-inside-kubernetes-containers"
---

Alright, let’s tackle this LVM visibility conundrum. I’ve seen this exact scenario play out more times than I care to remember, often when setting up persistent storage for stateful applications in Kubernetes. It’s a classic example of how containerization introduces layers of abstraction that can sometimes catch you off guard. The short answer is that containers operate within an isolated environment, typically at a process and filesystem level, which by default doesn't expose the host's underlying device mappings, including those managed by the logical volume manager (LVM).

When you create an LVM logical volume on a host, you're essentially manipulating block devices at the kernel level. The kernel, through the device mapper, creates these abstract block devices that applications see as regular filesystems. When a container starts up, however, it’s not simply running directly on top of this kernel. Instead, it's placed into a confined environment controlled by the container runtime (e.g., Docker, containerd) using features like namespaces and cgroups. Namespaces provide the isolation, and in this case, the most relevant are the mount namespace and the network namespace.

The mount namespace is key here. It provides each container with its own private view of the filesystem hierarchy. By default, the mount namespace will *not* inherit the host's mount points, including the block devices exposed by LVM. Therefore, a container will not ‘see’ the `/dev/mapper` devices or any other LVM related device nodes that are visible to the host operating system. What the container *does* see is determined by the configurations of the container runtime when launching a container, and the images it's built from.

Think of it like this: the host OS is the primary stage, where LVM is operating as a part of the infrastructure. The containers are individual theatre sets placed onto this stage. By default, they don’t see beyond their own set. They don’t have a direct view of what is occurring backstage, namely, the LVM activity. This separation is crucial for the security and stability of the system. We wouldn’t want a compromised container to have unfettered access to underlying devices and potentially disrupt other workloads, or the host, itself.

Now, let's illustrate this with code and explore how you can actually get LVM volumes into containers, if the need exists, because we have had such needs in the past.

**Example 1: Default Container Behavior – No LVM Visibility**

Let’s start by demonstrating the default behavior: a container not seeing LVM devices. First, I’ll show you what exists on the host. Let's assume I've created a logical volume named `my_lv` within a volume group named `my_vg` and made a file system on it. In my past, I would normally create these with commands like `vgcreate` and `lvcreate`.

Here’s what that might look like on the host:

```bash
# On the Host
sudo lvs # You'll see something like my_vg/my_lv
sudo ls -l /dev/mapper/my_vg-my_lv # Should be a block device
```
The command `lvs` displays existing logical volumes, and the `ls` command displays the symbolic links that point to a block device.

Now, let's run a simple container and attempt to access these devices:

```bash
# On the Host
docker run -it --rm ubuntu:latest bash

# Inside the Container
root@<container-id>:/# ls -l /dev/mapper
ls: cannot access '/dev/mapper': No such file or directory
root@<container-id>:/# lvs
Command 'lvs' not found, but can be installed with:
apt install lvm2
```
As you can see, the container does not have access to `/dev/mapper` and can’t execute `lvs` without additional tooling, which it doesn't possess by default. The command `lvs` requires the installation of the `lvm2` package, and even then, it would likely not be able to view the existing volumes as it does on the host.

**Example 2: Mounting a Host Path into a Container**

One of the most straightforward methods for making host resources accessible is by bind mounting a directory path. However, we need to be careful with this approach since it doesn't provide device-level access, it only provides access to whatever is *mounted* at that path. Let's imagine we have a directory on the host at `/mnt/my_lv_mount` which is mounted from our logical volume. We can mount this into the container:

```bash
# On the Host, mount the logical volume somewhere
sudo mkdir -p /mnt/my_lv_mount
sudo mount /dev/mapper/my_vg-my_lv /mnt/my_lv_mount

# On the Host, run the container and mount the path
docker run -it --rm -v /mnt/my_lv_mount:/container_mount ubuntu:latest bash

# Inside the Container
root@<container-id>:/# ls /container_mount
# You'll see the files located on the mounted volume.
```

This example shows the container accessing the data stored within the logical volume. The container has access to the *contents* at the mount point, not to the underlying block device and the LVM structure.

**Example 3: Passing Devices via --device (Less Recommended)**

A less common and often less recommended, but still valid solution for some specific situations is passing device nodes directly. This is generally not something I would advise unless you have a very specific reason and have the understanding to manage it correctly, but it illustrates another approach:

```bash
# On the Host
# Note: Please use with caution, as this can have implications on the security of the container and host OS.
docker run -it --rm --device /dev/mapper/my_vg-my_lv:/dev/my_lv ubuntu:latest bash

# Inside the Container
root@<container-id>:/# ls -l /dev/my_lv
# Should be a block device inside the container.
# You can try commands like mount /dev/my_lv /mnt.
```
Here, the container is given access to the LVM device as if it were directly connected. You would have to mount it or perform any operations you need inside the container. It is important to note that while this example works, exposing devices in this manner requires careful consideration of the security and reliability implications. Generally, using volumes and persistent storage solutions that Kubernetes provides is a much safer approach.

The key takeaway here is that containers, by design, offer isolation from the host OS. This isolation is essential for security and portability. Directly exposing LVM volumes or device nodes can be convenient in some situations, but can quickly become complex, less portable, and have security implications.

For a better approach, Kubernetes provides its own abstraction layers for persistent storage through PersistentVolumes (PVs) and PersistentVolumeClaims (PVCs), which abstract away the underlying storage infrastructure. In my experience, it's far more maintainable to leverage these higher-level abstractions.

For a deeper dive into containerization, namespaces, cgroups, and the underlying mechanics of Linux, I highly recommend reading *Linux Kernel Development* by Robert Love. It's a detailed look at how the kernel functions. For understanding the intricacies of Kubernetes storage, the official Kubernetes documentation is an excellent resource, especially the section dedicated to storage concepts. And for a good practical dive into LVM from first principles, check out the *Linux Storage Administration* books by Sander van Vugt, which is comprehensive and very hands-on.

Ultimately, the visibility of LVM volumes inside containers is a design choice by the container runtimes for enhanced security and better management. While there are ways to bypass this design, understanding these mechanisms and using the Kubernetes native resources like PVs and PVCs is the generally recommended approach for managing persistent storage in production environments.
