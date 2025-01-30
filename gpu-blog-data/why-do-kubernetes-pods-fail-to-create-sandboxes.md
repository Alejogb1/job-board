---
title: "Why do Kubernetes pods fail to create sandboxes on a Linux/Windows cluster?"
date: "2025-01-30"
id: "why-do-kubernetes-pods-fail-to-create-sandboxes"
---
A primary reason Kubernetes pods fail to create sandboxes on mixed Linux/Windows clusters stems from fundamental mismatches in the container runtime's expectations versus the underlying node operating system's capabilities. I've personally encountered this when managing a hybrid environment where inconsistent configurations often resulted in seemingly inexplicable pod failures. Specifically, the CRI (Container Runtime Interface) requests a sandbox environment – basically a container's namespace and resources – that the node's operating system can't adequately provide due to discrepancies in the requested runtime.

This process typically involves Kubernetes’ kubelet making a request to the node’s container runtime (e.g., Docker, containerd) via the CRI. The container runtime, in turn, attempts to create the sandbox, which is a set of isolated operating system resources that the containerized application will operate within. In a homogenous Linux cluster, this generally works seamlessly since the requested specifications directly map to the underlying Linux kernel's namespace mechanisms and resource control groups (cgroups). However, with a mixed OS environment, the divergence in how Linux and Windows handle containerization introduces multiple points of potential failure.

The primary mismatch centers on the differences between Linux namespaces and Windows jobs/containers. Linux uses namespaces for isolating processes, mount points, network interfaces, users, etc. Windows achieves process and filesystem isolation through different mechanisms, utilizing job objects and Hyper-V containers respectively. Critically, Windows containerization also often relies on specific kernel versions and required host components. Therefore, a CRI request designed for a Linux sandbox (e.g., leveraging Linux namespaces) will fail to execute on a Windows node, or vice versa. If a deployment manifest lacks specific node selectors or tolerations, or if the node label is not correctly defined, the scheduler may incorrectly place Linux-targeted pods onto Windows nodes, or vice versa, resulting in the failure to create a sandbox. This situation is exacerbated when using runtime classes. If a pod specifies a runtime class that targets only Linux, Kubernetes should ideally only schedule such pods onto Linux nodes. Yet, incorrect configuration of runtime classes or incorrect node labeling can lead to this error.

Additionally, the container runtime itself must be correctly configured and compatible with the host operating system. For example, Docker Desktop on Windows uses the Hyper-V hypervisor, and a misconfiguration in Hyper-V or the Docker daemon can hinder sandbox creation. Similarly, if the container runtime on the Windows node is not the most recent stable release and contains bugs related to Windows containers, sandbox creation failures can occur. In my experience, ensuring that both Kubernetes and the underlying container runtime are configured identically across all nodes, as much as the OS allows, minimizes unexpected failures.

The container image itself also plays a role in sandbox creation failures. Attempting to run a Linux-based image within a Windows sandbox, or a Windows-based image on a Linux sandbox, will consistently fail at the CRI level. This is due to fundamental differences in the executable formats, system calls, and dependencies packaged into the respective images. The container runtime expects the image to be compatible with the host operating system of the created sandbox. A Kubernetes deployment manifest’s image attribute is an indicator of the intended target operating system, and the Kubernetes scheduler ought to take this into account, which is further informed by node labels and tolerations.

Let's illustrate this with a few code examples.

**Example 1: Incorrect Node Selector**

Here is an example of a deployment manifest where the `nodeSelector` is missing or incomplete:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-web-app
  template:
    metadata:
      labels:
        app: my-web-app
    spec:
      containers:
      - name: web
        image: nginx:latest # This is a Linux image
        ports:
        - containerPort: 80
```

This manifest attempts to deploy three replicas of the NGINX image, which is Linux-based. If the Kubernetes cluster includes Windows nodes and there isn't any node selector or toleration to guide the scheduler, it is possible that the scheduler may attempt to place these pods on Windows nodes. Since Windows nodes will not be able to create a Linux container sandbox, pod creation fails. To solve this, a node selector specifying Linux should be included within the pod spec:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-web-app
  template:
    metadata:
      labels:
        app: my-web-app
    spec:
      nodeSelector:
        kubernetes.io/os: linux
      containers:
      - name: web
        image: nginx:latest
        ports:
        - containerPort: 80
```

This updated example uses `nodeSelector` to target nodes labeled with `kubernetes.io/os: linux`, thereby ensuring the pod is scheduled on an appropriate node. If an appropriate label is absent from your target node, this pod will remain in a `Pending` state.

**Example 2: Runtime Class Mismatch**

Incorrect use or setup of runtime classes can also lead to problems. Consider a case where a runtime class is specified, but it does not map correctly to the underlying operating system.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-web-app
  template:
    metadata:
      labels:
        app: my-web-app
    spec:
      runtimeClassName: linux-container # Assuming a class meant for Linux
      containers:
      - name: web
        image: my-windows-app:latest # This is a Windows image
        ports:
        - containerPort: 80
```

In this example, although the scheduler may place the pod on a Windows node, the specified runtime class is configured for a Linux environment. Even if the scheduler manages to place it on a Windows node, the CRI will attempt to create the container using the defined linux-container runtime and ultimately fail to create the sandbox.

The proper configuration would require specifying a Windows runtime class when using a Windows image:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-web-app
  template:
    metadata:
      labels:
        app: my-web-app
    spec:
      runtimeClassName: windows-container
      containers:
      - name: web
        image: my-windows-app:latest
        ports:
        - containerPort: 80
```

Here, we’ve specified `windows-container` which should align with the windows specific image.

**Example 3: Incorrect Node Labels**

Let's say that you intend to schedule Linux workloads to nodes with a label of `os:ubuntu` and Windows workloads to `os:windows`. In this case, mislabeling a Linux node with the `os:windows` or vice versa will lead to the failure to schedule or successfully create a sandbox. Node labels are key components in routing workloads to an appropriate node.

To diagnose sandbox creation failures, I’d recommend investigating the logs of the kubelet, the container runtime, and even the scheduler if the error is related to scheduling. Kubernetes event logs using `kubectl get events` can provide additional clues. For further understanding of the internals, the Kubernetes documentation on node selectors, tolerations, and runtime classes is essential. The CRI specification documentation also outlines the expected behavior between Kubernetes and the container runtime.

For diagnosing Windows containers specifically, Microsoft's documentation on Windows containerization best practices is recommended. Container runtime specific documentation (Docker, containerd, etc.) is equally crucial. I often find consulting these resources to be invaluable, allowing me to delve deeper into a specific problem and isolate the source of the failure. Maintaining a detailed understanding of these layers is imperative for any cluster operator managing a complex mixed OS Kubernetes environment.
