---
title: "What causes Docker's 'failed to create shim' error during OCI runtime creation?"
date: "2025-01-30"
id: "what-causes-dockers-failed-to-create-shim-error"
---
The "failed to create shim" error encountered during OCI runtime creation in Docker typically stems from inconsistencies between the Docker daemon's configuration, the host system's kernel capabilities, and the container's runtime environment.  My experience troubleshooting this issue across diverse Linux distributions, from CentOS 7 to Ubuntu 20.04, within large-scale container orchestration systems, reveals that the root cause is rarely a single, easily identifiable problem. Instead, it's often a confluence of factors that require a systematic diagnostic approach.

**1.  Clear Explanation**

The Docker daemon relies on a container runtime (like containerd, runc, or CRI-O) to manage the lifecycle of containers.  The "shim" acts as an intermediary between the container runtime and the kernel. It's responsible for managing the container's process, handling signals, and generally ensuring the container runs securely and predictably.  The "failed to create shim" error signifies a failure at this crucial interface. This failure can be triggered by various reasons, including:

* **Insufficient Kernel Capabilities:** The host kernel might lack necessary features or security modules required by the runtime or the shim.  This often manifests when working with older kernels or customized kernels lacking essential security features.

* **Incorrect Runtime Configuration:** Docker's configuration files (`daemon.json`, for example) might contain incorrect paths to the runtime binary, its dependencies, or required system libraries.  Typographical errors or outdated settings are common culprits here.

* **Corrupted Runtime Installation:** The container runtime itself (e.g., runc) might be corrupted, incomplete, or improperly installed. This can result from failed package updates, incomplete installations, or system-level inconsistencies.

* **Resource Exhaustion:**  In heavily loaded systems, insufficient memory or file descriptors might prevent the creation of the shim process. This is particularly likely in environments with many containers running concurrently.

* **SELinux or AppArmor Conflicts:** Security modules like SELinux or AppArmor, if improperly configured, can prevent the Docker daemon from accessing necessary resources or creating the shim process.  Denial of permissions at a security module level is a common cause, often overlooked.

* **Incompatible Runtime and Kernel Version:** The runtime might not be compatible with the host kernel's version.  Although less frequent with modern runtimes, mismatch between major versions can lead to subtle compatibility errors.


**2. Code Examples and Commentary**

The following examples demonstrate diagnostic approaches to identify and resolve the "failed to create shim" error. These are illustrative; specific commands might vary depending on your distribution and setup.

**Example 1: Verifying Kernel Capabilities**

```bash
# Check if required kernel modules are loaded (example: aufs, overlay)
lsmod | grep aufs
lsmod | grep overlay

# Inspect system logs for kernel errors or warnings related to containerization
journalctl -xe | grep docker
dmesg | grep docker
```

*Commentary:* This example focuses on verifying the presence of essential kernel modules, often required for containerization technologies like overlayfs.  Examining system logs helps pinpoint potential kernel-level issues hindering the shim's creation.  The `journalctl` command is preferred over `dmesg` for modern systems due to its superior filtering capabilities.


**Example 2: Inspecting Docker Configuration**

```bash
# Examine the Docker daemon configuration file
cat /etc/docker/daemon.json

# Verify the runtime path specified in the daemon configuration
grep "runtimes" /etc/docker/daemon.json
```

*Commentary:* This checks the Docker daemon configuration file (`daemon.json`) for potential errors in the runtime settings.  This file dictates which runtime Docker uses; incorrect paths or missing entries will directly lead to errors during shim creation.  Inspecting this file thoroughly and comparing it against the default configuration can identify discrepancies.


**Example 3: Reinstalling the Container Runtime**

```bash
# Stop the Docker daemon
systemctl stop docker

# Remove the existing container runtime (adjust package names as needed)
sudo apt-get remove runc  # For Debian/Ubuntu
sudo yum remove runc  # For CentOS/RHEL

# Reinstall the container runtime
sudo apt-get install runc  # For Debian/Ubuntu
sudo yum install runc  # For CentOS/RHEL

# Start the Docker daemon
systemctl start docker
```

*Commentary:* If issues persist despite checking kernel and configuration files, reinstalling the runtime (in this case, `runc`) might resolve corruption issues.  Always ensure the Docker daemon is stopped before performing these operations. The package manager commands are distribution-specific.  Consider using a dedicated package manager for container runtimes to ensure consistency and avoid conflicts with the system's package manager.


**3. Resource Recommendations**

The official Docker documentation, particularly sections on installation and troubleshooting, should be your primary resource.  Consult the documentation for your specific container runtime (e.g., containerd, runc, CRI-O) as well.  Familiarize yourself with your Linux distribution's system administration guides to understand kernel modules, package management, and system logging.  Understanding SELinux and AppArmor configuration and troubleshooting is also crucial for advanced diagnostics, particularly in production environments.  Finally, thorough examination of the Docker daemon and container logs, employing appropriate filtering and searching techniques, is indispensable in pinpointing the precise source of the error.
