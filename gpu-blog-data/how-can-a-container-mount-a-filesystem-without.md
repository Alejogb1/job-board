---
title: "How can a container mount a filesystem without root privileges?"
date: "2025-01-30"
id: "how-can-a-container-mount-a-filesystem-without"
---
A container, by its nature, operates within a namespace, a form of process isolation, that restricts its access to host resources. Traditionally, mounting a filesystem, particularly one intended for read-write operations, necessitates root capabilities. However, several modern Linux features, combined with careful container configuration, allow for mounting filesystems without the container process having the `CAP_SYS_ADMIN` capability, often considered a root-equivalent privilege within the container namespace. This is achieved by leveraging user namespaces and specific mount options, allowing a fine-grained control over permissions and minimizing potential security vulnerabilities.

The core mechanism lies in the interplay of user namespaces and the `fuse` filesystem driver. User namespaces permit mapping a user identifier (UID) and group identifier (GID) inside the container to a different UID/GID on the host. When a container process attempts to perform a mount operation, the kernel checks the permissions against the mapped user credentials on the host, rather than the user credentials *inside* the container. This allows for operations, that would normally require root privileges *within* the container, to be executed with the mapped host user's permissions. This concept combined with `fuse` gives the necessary primitives. The `fuse` filesystem driver, short for Filesystem in Userspace, operates entirely in user space, facilitating the creation of custom filesystems without relying on direct kernel modules. It acts as an intermediary, passing filesystem calls from the kernel to a user space process that implements the actual filesystem logic.

Crucially, while the mount operation still requires privileged access initially, it's the *host* user or, in some cases, a specifically configured service outside the container, that handles the initial mount, and subsequently the container process interacts with the mounted filesystem using the mapped user identity. This carefully structured delegation of privilege is the crux of mounting filesystems in a non-root container.

The `fuse` filesystem, once mounted by a privileged entity, can then be accessed and modified by the container process using its mapped user identity. This process of user mapping allows the container to access the filesystem with permissions that are defined on the host machine, rather than the root privileges normally required by traditional mounting. Consequently, the container is not able to escalate privileges by directly invoking system calls to perform mount operations or gain access to parts of the host filesystem it shouldn’t be able to access.

Below are three illustrative code examples, demonstrating different aspects of this technique.

**Example 1: Host-level FUSE Mount with User Mapping**

This example showcases a basic `fuse` mount made outside the container, with the intent to access it from a container after mounting. The core idea is that a host process with adequate permissions mounts the filesystem, and then maps user access within the container to access it.

```bash
# On the host machine:
# Create a directory for testing
mkdir /mnt/test_fuse

# Create a file system using fuse driver
# The "hello.py" is a fuse implementation, it is
# not shown here for the scope of this example.
python3 hello.py /mnt/test_fuse

# Now, create a container using docker:
# Mount the host fuse filesystem into the container
docker run -it --user 1000:1000 --mount type=bind,source=/mnt/test_fuse,target=/mnt/container_fuse ubuntu bash

# Inside the container (bash prompt):
ls -l /mnt/container_fuse
# You should be able to see the contents of the fuse filesystem
# And interact with it using user ID 1000

# To unmount fuse filesystem, exit the container, and on the host
fusermount -u /mnt/test_fuse
```

In this setup, the `python3 hello.py /mnt/test_fuse` command on the host creates and mounts a `fuse` filesystem. The critical part is the docker command, with ` --user 1000:1000` which maps UID/GID 1000 within the container to the same UID/GID outside the container. The `--mount` flag simply makes the mounted directory accessible inside the container.  The container process running as user ID 1000 can now interact with the filesystem in `/mnt/container_fuse`. Note that if the file ownership of that folder on the host is not for user ID 1000, the container user would not be able to interact with the filesystem. This highlights the importance of user mappings for achieving seamless interaction.

**Example 2: Container-Level User-Namespaced Mount with a Helper Daemon**

In this case, mounting is initiated by a helper process outside the container, but user namespaces are leveraged to create a secure mapping. This is a more complicated scenario, usually implemented using a sidecar container, or a separate process, as it requires the helper to have elevated privileges to mount the filesystem, but is more practical for production environments.

```bash
# Host-level helper process (e.g., a go program):
# Pseudo-code, this assumes a helper go daemon runs on the host:
# 1. Receive mount request with target path, uid, and gid mapping
# 2. Create a mount with target path
# 3. Adjust the ownership of target path using chown
# 4. Launch the container with --mount type=bind,source=/path/mounted,target=<container_path> and
# --user <user_id>:<group_id>

# Inside container after mounting process has been executed:
# In the container, you can now mount filesystems, assuming they fall within the
# permissions set by the helper.
mount -t tmpfs tmpfs /mnt/container_tmpfs

# Then the user inside the container can modify the mounted directory
touch /mnt/container_tmpfs/new_file.txt
```
Here, a hypothetical host process takes the responsibility for making the initial mount, but the container does not need `CAP_SYS_ADMIN`.  The helper daemon on the host would receive the request, perform the `fuse` mount, change ownership to the target uid/gid from container, and then pass it on to the container. The container can now use `mount` command to mount directories within the container's namespace without needing privileges. This example illustrates how a helper program acting as an intermediary can manage file system mounts.

**Example 3: Using `overlayfs` in conjunction with a lower-dir mounted using the techniques explained above.**

This example leverages the capability of mounting a filesystem and then layering it to provide a read-write layer. `overlayfs` is an interesting option since it works well inside a container, when only the upper directory needs to be mounted by the host, and the container does not need any privileges to mount the overlayfs itself.
```bash
# Host machine:
# Mount the lower read-only directory first, with the techniques described above.
mkdir /mnt/ro_dir
# Create some files in the folder
echo "hello" > /mnt/ro_dir/file1.txt

# Use similar techniques to create the folder /mnt/upper_dir, and allow write access for the container user
mkdir /mnt/upper_dir

# Container launch command:
# The container mounts the host-mounted directories. The overlayfs is then mounted by the container itself.
docker run -it --user 1000:1000 --mount type=bind,source=/mnt/ro_dir,target=/mnt/ro_dir --mount type=bind,source=/mnt/upper_dir,target=/mnt/upper_dir ubuntu bash

# Inside the container:
mkdir /mnt/work
mount -t overlay overlay -o lowerdir=/mnt/ro_dir,upperdir=/mnt/upper_dir,workdir=/mnt/work /mnt/merged
ls /mnt/merged
# You can modify /mnt/merged. Changes will be saved inside /mnt/upper_dir.
echo "world" >> /mnt/merged/file1.txt
```

In this example, the host prepares both read-only and upper directories using the host mount mechanisms. The container maps these directories into its namespace, then the container itself proceeds to mount an overlayfs. Changes made to `/mnt/merged` will be transparently stored in `/mnt/upper_dir` on the host. The container can therefore modify the mounted directory without any extra privileges, while the root mount operation is still performed by the host.

In summary, achieving filesystem mounts without root in containers is not a singular solution, but a combination of approaches revolving around user namespace mapping and carefully leveraging helper processes to manage the initial mount with appropriate permissions, thereby limiting the attack surface.

For further study, the following resources are recommended, offering in-depth explanations of these concepts: Kernel documentation pertaining to user namespaces, FUSE filesystem specifics, and various resources focusing on security best practices in container environments would provide a solid foundation. Information regarding container orchestration systems, and how they leverage similar techniques for dynamically provisioning volumes would also deepen understanding. Also, reviewing the documentation on security capabilities within Linux namespaces can provide a broader perspective.
