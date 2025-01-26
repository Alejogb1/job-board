---
title: "Can containerized applications run SystemTap user-space probes?"
date: "2025-01-26"
id: "can-containerized-applications-run-systemtap-user-space-probes"
---

User-space probing with SystemTap within containerized environments presents specific challenges and requires careful configuration. The fundamental issue stems from the isolation mechanisms containers employ, particularly namespaces and cgroups, which restrict a process's ability to interact with the host kernel and other processes. The SystemTap tool relies on accessing kernel-level debugging interfaces, and the container's isolated environment often limits its reach into the necessary parts of the host operating system.

To provide a concise explanation: User-space probes, by their nature, operate within the address space of a specific process. When that process is containerized, the address space, memory layout, and access permissions are governed by the container runtime and, ultimately, the underlying Linux kernel's namespace isolation. While SystemTap *can* technically instrument user-space applications within a container, this is not a straightforward 'out-of-the-box' experience, and there are critical requirements to enable successful probing.

The primary impediment lies in the way SystemTap interacts with debug symbols and memory mappings. SystemTap relies on reading process memory and debug information (often present in DWARF format) to identify function entry points and extract argument values. Within a container, the process's address space is relative to its isolated view of the filesystem and memory space. Furthermore, SystemTap needs to access the host's kernel modules and debug symbols, access that is not inherently provided to a containerized process.

Several crucial prerequisites exist. First, the kernel needs to have been compiled with CONFIG_DEBUG_INFO enabled; this typically is not an issue with most distributions, but verification is crucial. Second, the container runtime (e.g., Docker, containerd) must be configured to allow the necessary access to the host filesystem and debug infrastructure. Third, SystemTap itself must be available and correctly installed *within* the container image. Lastly, the target application or process inside the container needs to be compiled with debugging symbols present.

I have encountered this situation on multiple occasions, notably while attempting to diagnose performance bottlenecks in several microservices deployed within a Kubernetes cluster. The initial attempts to use SystemTap from a host machine to probe a container were met with errors relating to symbol resolution and memory access failures. Successfully overcoming these issues requires several configuration modifications.

Let's examine a series of examples, including commentary on the approach taken.

**Example 1: Minimal User-space Probe within a Simple Container**

First, let's consider a minimal example demonstrating the fundamental problem. Suppose we have a simple C program called `test_app.c`:

```c
#include <stdio.h>

void my_function(int arg) {
    printf("Value: %d\n", arg);
}

int main() {
    for (int i = 0; i < 5; i++) {
        my_function(i * 10);
    }
    return 0;
}
```

We can compile it (with debugging symbols) as follows:
`gcc -g test_app.c -o test_app`

Now, assume this `test_app` is included in a simple Dockerfile and built into an image. If I attempted to run a SystemTap script from the host like so:

```stap
probe process("./test_app").function("my_function") {
    printf("Entered my_function with argument: %d\n", $arg);
}
```

I would typically encounter errors, such as:

```
WARNING: /home/user/test_app.ko not found; falling back to host (unreliable) symbol lookup.
process("./test_app"): No such file or directory.
```

This occurs because SystemTap, running on the host, is looking for the executable in the host filesystem context, while `./test_app` resides *within the container's filesystem context*.

**Example 2: Containerized SystemTap with Host Mount**

To address the above, one approach involves installing SystemTap inside the container image and then mounting a directory from the host into the container. This enables SystemTap within the container to access the target executable directly. Further, the `/proc` mount, which gives view to the running processes, needs to be made available.

First, build a container image with SystemTap installed and including the source and the compiled binary (test_app)

```Dockerfile
FROM ubuntu:latest

RUN apt-get update && apt-get install -y systemtap gcc gdb

COPY test_app.c test_app

RUN gcc -g test_app.c -o test_app

ENTRYPOINT ["/bin/bash"]
```

Then, run the container mounting the /proc directory and the test app directory

```bash
docker run --rm -it  --mount type=bind,source=$(pwd),target=/work -v /proc:/proc  your_image_name
```

Now, inside the container shell, I can run the following SystemTap script:

```stap
probe process("/work/test_app").function("my_function") {
    printf("Entered my_function with argument: %d\n", $arg);
}
```
This will correctly probe `my_function` because SystemTap, now running *inside* the container, can correctly locate and access the debug symbols. Note that the program must also be run within the container environment.

This solution assumes the kernel modules required by SystemTap are available within the host and are accessible within the container's view of `/proc`.  It does require access to the host kernel by mapping /proc and is likely an approach that may not be compatible with hardened security requirements for container deployments.

**Example 3:  Kubernetes Container Probing with Privilege and Host PID**

Finally, let's consider a scenario more directly related to Kubernetes. If I wanted to debug a pod using SystemTap, the approach is similar, but some container configuration adjustments are needed. It is *crucial* to understand that these configurations *significantly weaken security isolation* and therefore should only be used for debug purposes, ideally within a development environment.

A pod definition that permits such probing would resemble this:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: debug-pod
spec:
  hostPID: true  # Allows access to the host's process table
  containers:
    - name: target-container
      image: your-image-name  # Image containing test_app (as per Example 2)
      securityContext:
        privileged: true # Allows access to kernel modules
      volumeMounts:
        - name: proc
          mountPath: /proc
  volumes:
  - name: proc
    hostPath:
      path: /proc

```

The `hostPID: true` field provides visibility into host processes, the `privileged: true` flag enables access to kernel modules. The mounting of /proc also assists the resolution of process information. Using kubectl, deploy the pod and then exec into the `target-container`. Once inside the container, the previously used SystemTap script (from example 2, replacing the /work mount point with the actual path inside the container) would then successfully probe the user-space application within that container.

**Resource Recommendations**

For further study of container internals and security, several general resources exist. Documentation concerning Linux namespaces and cgroups is essential to understanding the isolation mechanisms at play. Further, general information about SystemTap’s functionality and limitations in accessing kernel debugging interfaces is beneficial. Specific container runtime documentation (Docker, containerd) offers details on runtime configurations related to mounting host filesystems and setting process privileges. Finally, if using Kubernetes, reviewing their security contexts and host-level networking options is critical.
