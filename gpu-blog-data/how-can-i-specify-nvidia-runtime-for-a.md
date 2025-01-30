---
title: "How can I specify NVIDIA runtime for a GPU application in a k3s pod YAML deployment?"
date: "2025-01-30"
id: "how-can-i-specify-nvidia-runtime-for-a"
---
Ensuring a Kubernetes pod running a GPU-accelerated application utilizes the NVIDIA runtime requires meticulous configuration within the pod's YAML specification. Without explicitly defining this, the container runtime might default to a CPU-only environment, negating the performance gains expected from a GPU. I've encountered this numerous times when initially deploying machine learning models and simulations within k3s clusters, and the solutions typically converge to a specific set of configurations.

The critical component is the `runtimeClassName` field within the pod's specification. This field directs Kubernetes to use a specific container runtime, which in our case needs to be the NVIDIA Container Runtime. However, simply specifying this isn't always enough. We also need to ensure the underlying node has the correct NVIDIA drivers and the NVIDIA Container Toolkit installed, and that a corresponding `RuntimeClass` resource has been configured in the cluster. This `RuntimeClass` resource acts as a pointer that instructs Kubernetes to use the NVIDIA runtime whenever a pod references it. Ignoring these prerequisites will lead to pod failures or silently running containers on the CPU.

First, I’ll provide a basic illustration. Consider a situation where you've deployed the NVIDIA Container Toolkit, drivers and configured a `RuntimeClass` resource named `nvidia` as advised. The below example shows how to specify that this particular `RuntimeClass` should be used with a basic PyTorch image.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-gpu-pod
spec:
  runtimeClassName: nvidia
  containers:
  - name: pytorch-container
    image: nvcr.io/nvidia/pytorch:23.11-py3
    resources:
      limits:
        nvidia.com/gpu: 1
```

Here, `runtimeClassName: nvidia` is the crucial line. It tells Kubernetes to use the `RuntimeClass` named `nvidia`, which I previously assumed is configured to use the NVIDIA Container Runtime. This setup ensures the container has access to the designated GPU within the node via the device plugin and necessary libraries. The `resources.limits` section, especially the `nvidia.com/gpu: 1` line, requests one GPU, and while this resource directive is not directly tied to runtime specification, its presence is essential for GPU applications. Without this, Kubernetes might schedule the pod onto a node without the capability, which is important in environments with both CPU-only and GPU nodes.

Now, let’s consider a scenario where we require a sidecar container for monitoring alongside the main application. This slightly more advanced case requires the same `runtimeClassName`, but demonstrates its usage in a multi-container scenario, which I often use to handle logging and metrics collection alongside computation.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: multi-container-gpu-pod
spec:
  runtimeClassName: nvidia
  containers:
  - name: pytorch-container
    image: nvcr.io/nvidia/pytorch:23.11-py3
    resources:
      limits:
        nvidia.com/gpu: 1
  - name: prometheus-sidecar
    image: prom/prometheus:latest
    ports:
    - containerPort: 9090
```

In this example, `runtimeClassName: nvidia` applies to *all* containers within the pod. This ensures that both the `pytorch-container` and the `prometheus-sidecar` are launched under the NVIDIA runtime. While the `prometheus-sidecar` doesn't use the GPU, its inclusion serves to demonstrate that the `runtimeClassName` applies at the pod level, not the container level, influencing how Kubernetes manages resource allocation for all containers within. This is important, as unexpected behaviour may arise if containers run using different runtimes, even if only one of them actually uses a GPU. I’ve seen this before when a logging sidecar wasn't properly accounted for, and the application log was not visible.

Finally, I'd like to address the use of a deployment. Deployments add further complexity. The `runtimeClassName` has to be specified within the pod template, not directly under the `spec`. The Deployment is responsible for managing replicas and updates, and each pod created via it should still receive the same runtime specification. Let's look at a practical example.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: pytorch-gpu-deployment
spec:
  replicas: 2
  selector:
    matchLabels:
      app: pytorch-gpu-app
  template:
    metadata:
      labels:
        app: pytorch-gpu-app
    spec:
      runtimeClassName: nvidia
      containers:
      - name: pytorch-container
        image: nvcr.io/nvidia/pytorch:23.11-py3
        resources:
          limits:
            nvidia.com/gpu: 1
```

Here, we see the familiar `runtimeClassName: nvidia` within `spec.template.spec` not just `spec` as with Pod objects. Every pod spun up by the Deployment controller will respect this setting, ensuring each replica utilizes the NVIDIA runtime for the GPU computation. The labels in metadata and selector are essential for Kubernetes to manage the deployment’s pods correctly, but are not strictly related to specifying the runtime. Misunderstanding where this runtime configuration should be specified within Deployment YAML is a frequent cause of failures when first working with Kubernetes, especially when migrating from standalone pod definitions.

For further investigation and deeper understanding, I would recommend reviewing the official Kubernetes documentation concerning `RuntimeClass` resources, the NVIDIA Container Toolkit documentation and any reference material related to your chosen NVIDIA base image. Also, explore some tutorials on Kubernetes scheduling for resource-intensive applications and specifically those that address GPU management. Lastly, understanding the specifics of how your cluster's CRI interacts with `RuntimeClass` is essential to resolve complex issues.

In summary, properly specifying the NVIDIA runtime for a GPU application in a k3s pod deployment hinges upon using the `runtimeClassName` within the pod specification. The `RuntimeClass` itself must be configured, and the node must be equipped with the necessary NVIDIA tools and drivers. I hope this practical experience helps you avoid some of the pitfalls that I, and I’m sure many others, have encountered along the way.
