---
title: "Can crictl display the size of running container ephemeral layers?"
date: "2025-01-30"
id: "can-crictl-display-the-size-of-running-container"
---
The core challenge lies in how `crictl`, the command-line interface for interacting with CRI (Container Runtime Interface) compatible container runtimes, exposes information about container image layers. It does not directly provide a command to show the size of ephemeral layers of running containers. I've encountered this limitation frequently while debugging storage consumption in various Kubernetes environments. `crictl` is primarily focused on container lifecycle management (creating, starting, stopping, inspecting basic metadata) and lacks fine-grained filesystem layer analysis tools. Instead, the runtime itself manages these layers and, while the information *exists*, it's not easily exposed through `crictl` alone. We need to understand the underlying storage drivers (e.g., overlay2) and their mechanics to truly grasp and access ephemeral layer sizes.

The ephemeral layer, also commonly called the container's writable layer, is a crucial aspect of containerization. It represents the difference between the read-only image layers and any modifications made during the container's runtime. All file system changes, new files, or edits, happen within this ephemeral layer. This layer is also crucial to understand storage consumption because any data written to the container during its execution contributes directly to the layer's growth, and this growth is *not* typically accounted for in static image size information. A primary reason `crictl` doesn’t display these sizes directly is that the implementation details vary considerably between container runtimes, storage drivers, and even underlying operating systems. This lack of a unified interface makes a consistent display within `crictl` exceptionally difficult without making assumptions about the underlying system.

Furthermore, the ephemeral layer’s size is dynamic. It changes as the container operates, requiring monitoring, rather than a single-point-in-time display. `crictl`, being a primarily static information provider, wouldn't be the ideal tool for this dynamic observation. More advanced tools for storage analysis, or the runtime's own internal tools, are better suited. `crictl` *can* help in identifying the container's ID which is essential for accessing the ephemeral layer information, but it's not the entire solution.

Here are three examples highlighting why `crictl` alone is insufficient and how I would approach the problem:

**Example 1: Using `crictl` to Obtain Container ID, then Utilizing Container Runtime specific tooling**

```bash
# Step 1: Find a container to investigate using crictl
crictl ps -a

# Sample output might be:
# CONTAINER           IMAGE               CREATED             STATE               NAME                      ATTEMPT
# a1b2c3d4e5f67       my-app:latest     3 minutes ago       Running             my-app-container      0

# Step 2: Extract the container ID: a1b2c3d4e5f67 (This will be unique for your system)
CONTAINER_ID="a1b2c3d4e5f67"

# Step 3:  Now, if using docker as the container runtime, use docker's tooling to inspect the container:
# (assuming a Linux host, and "docker" as the runtime behind cri-dockerd)
sudo docker inspect $CONTAINER_ID  | jq  '.[].GraphDriver.Data.UpperDir'

# Output might look like:
# "/var/lib/docker/overlay2/12345abcdef67890/diff"

# Step 4: Then, use du to estimate the size of this diff directory
sudo du -sh  /var/lib/docker/overlay2/12345abcdef67890/diff
```

**Commentary for Example 1:** This example demonstrates that while `crictl` provides essential container identification information, I have to switch to the underlying Docker engine (in this case) using `docker inspect`. We utilize `jq` to parse the JSON response to obtain the specific diff directory associated with the container's writable layer. Then, `du` helps to estimate the size using operating system facilities. This is specific to the Docker runtime, another runtime would use its own specific commands. This process is considerably more manual and complex than what a simple `crictl` command could provide. Furthermore, the "diff" directory size is an estimate and doesn’t capture any files stored outside of the diff directory.

**Example 2:  Using `ctr` for a container with a containerd runtime**

```bash
# First, crictl ps provides the container ID (same as example 1)
CONTAINER_ID="a1b2c3d4e5f67"

# If the runtime is containerd, we would use the containerd cli ctr
# Find the container name from crictl:
crictl inspect $CONTAINER_ID | grep "name:"
# Output is like "name: my-app-container"
CONTAINER_NAME="my-app-container"

# Now use ctr to examine it, using the container name. The "container" namespace is needed here.
sudo ctr -n k8s.io containers ls | grep $CONTAINER_NAME | awk '{print $1}'
CONTAINER_ID_CTR="0079d75368b72b4365905c017951441e81586339b4d37026141d5f998a5a9916"

# Use ctr image info command to see the size of the writable layers
sudo ctr -n k8s.io  containers inspect $CONTAINER_ID_CTR | jq .snapshotter | xargs  sudo ctr -n k8s.io  snapshots info  | grep "size"

```

**Commentary for Example 2:** This example highlights the differences when the underlying runtime is `containerd`. `crictl` still yields the container ID; however, accessing the writable layer size data now requires `ctr`, containerd's command-line tool, and an understanding of namespaces, specific to the container runtime. There are multiple commands here, because information is only exposed at the image level, not the container level. Note that we now need the container name as well as the ID. We then again parse the JSON output. Again, this process involves more steps and familiarity with the specific runtime and its internal operations.

**Example 3: Using a volume mount instead of the ephemeral layer and how `crictl` won't show volume sizes**

```bash
# crictl shows the container is running with volume mounts
crictl inspect $CONTAINER_ID | grep mounts

# Output may include something like:
# "mounts": [
#      {
#       "containerPath": "/data",
#       "hostPath": "/mnt/host-data",
#        "readOnly": false
#     }
#  ],

# crictl doesn't offer the size of host volumes.
# To check the size of data written to the volume:
sudo du -sh /mnt/host-data

```

**Commentary for Example 3:** This is to highlight another area where `crictl` is not intended to show size information. In a production environment, many applications utilize persistent volumes instead of relying solely on the ephemeral container layer. These volumes can be mounted as directories within the container's filesystem. `crictl` can display the volume mounts but it won’t show the size of the mounted data in the host file system. This data is outside of the container's ephemeral layer and must be assessed directly using host file system tools. I’ve used these volume mount locations to persist data, and so monitoring both the container's ephemeral layer, and any associated volume mounts becomes crucial to effective storage management, which is not handled by `crictl`.

In conclusion, while `crictl` is an invaluable tool for managing container lifecycles, it does not inherently provide information about the size of running container ephemeral layers. Obtaining that data requires working directly with the container runtime's underlying infrastructure.  The method varies significantly based on which runtime is used (Docker, containerd, etc). This process also highlights the separation of concerns, with `crictl` focusing on the CRI level, and runtime-specific tools providing detailed information. Monitoring and storage analysis require using multiple technologies and understanding how storage drivers and the file system interact.

For further learning, I would recommend exploring resources on the following:
* Container Runtime Interface (CRI) specifications. This will assist in understanding the limitations imposed by CRI on tooling like `crictl`.
* Documentation for specific container runtimes (Docker Engine, containerd, CRI-O). This is crucial for the specific tooling and internal implementation details.
* Concepts of storage drivers (overlay2, aufs) and their behavior related to ephemeral layers. This provides the foundational knowledge for how ephemeral layers function.
* Tools for analyzing disk space usage on Linux systems (`du`, `df`, etc.). These are essential for measuring disk usage after identifying the correct directory.
* Resources on managing persistent volumes within container orchestration platforms like Kubernetes, especially with container storage interface (CSI) driver details, to comprehend where data persists outside the ephemeral layer. This helps in understanding the different types of storage containers may use.
