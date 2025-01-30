---
title: "Why is GPU workload unschedulable in GKE node pool?"
date: "2025-01-30"
id: "why-is-gpu-workload-unschedulable-in-gke-node"
---
I've encountered this particular headache numerous times, and while the underlying cause often stems from a combination of configuration oversights, the core issue in GKE (Google Kubernetes Engine) regarding unschedulable GPU workloads usually traces back to how Kubernetes interprets resource requests in relation to the underlying node hardware and the availability of NVIDIA device plugins. It’s not typically a fault in GKE itself, but a mismatch between what’s requested in a Kubernetes manifest and what’s actually present and accessible.

Firstly, understanding how Kubernetes handles GPUs is vital. Kubernetes doesn’t directly interact with GPUs. Instead, it relies on a device plugin, specifically the NVIDIA device plugin, to discover and expose these resources to the cluster. This plugin, running as a DaemonSet on each node equipped with GPUs, enumerates the available GPUs and reports their capacity to the Kubernetes API server. Only after this registration process can pods request GPU resources. If the plugin is not functioning properly, or not deployed at all, Kubernetes will be unaware of any GPU capabilities on the node, and consequently, no GPU-requiring workload will be schedulable.

Secondly, the manner in which a pod requests GPUs within its specification plays a crucial role. The `resources.limits` and `resources.requests` fields in the container's spec, specifically the `nvidia.com/gpu` key, inform Kubernetes of the pod's GPU demands. If these values are set incorrectly, or if the sum of requested GPUs exceeds the actual number available on a node, scheduling will invariably fail. For instance, a common pitfall occurs when the `requests` value is set equal to or greater than the `limits` value, which can lead to unforeseen contention and scheduling failures, especially in multi-GPU configurations.

Thirdly, node auto-scaling can inadvertently complicate GPU workload scheduling. GKE auto-scaling strives to maintain optimal resource utilization based on overall cluster demand. However, during periods of high demand, it might rapidly add nodes that lack GPU hardware or don't have the device plugin deployed. If a GPU-requesting pod attempts to schedule on such a newly provisioned node, it will predictably fail, even though the cluster as a whole has capacity. The delay in node provisioning and device plugin startup can lead to temporary scheduling failures. Careful consideration of node pool configurations and auto-scaling parameters is therefore paramount.

Let's look at some code examples to illustrate these scenarios.

**Example 1: Correct GPU Resource Request:**

This YAML defines a simple pod requesting a single GPU. This assumes that the NVIDIA device plugin is deployed and functioning correctly.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-test-pod
spec:
  containers:
  - name: gpu-container
    image: nvidia/cuda:11.8.0-base-ubuntu20.04
    resources:
      limits:
        nvidia.com/gpu: 1
      requests:
        nvidia.com/gpu: 1
    command: ["nvidia-smi"]
```

*Commentary:* This configuration correctly specifies that one GPU is required using the `nvidia.com/gpu` key in both the `limits` and `requests` fields. The `nvidia/cuda` image is an example of an image that depends on having CUDA capabilities and thus, is a good testing image. `command` executes `nvidia-smi`, a command to check if the GPU is correctly initialized. If the pod fails to schedule, verify if the NVIDIA device plugin is operational on the target nodes.

**Example 2: Incorrect GPU Resource Request (Request > Limit):**

This demonstrates a common error where `requests` are greater than `limits`.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: faulty-gpu-test-pod
spec:
  containers:
  - name: gpu-container
    image: nvidia/cuda:11.8.0-base-ubuntu20.04
    resources:
      limits:
        nvidia.com/gpu: 1
      requests:
        nvidia.com/gpu: 2
    command: ["nvidia-smi"]
```

*Commentary:* Here, the pod requests two GPUs but is limited to one. Although seemingly counterintuitive, this can lead to a pod that may never be scheduled. Kubernetes scheduler can be confused when allocating this pod as a limit of one will never satisfy the request of two. While this particular example might seem obvious, subtle misconfigurations, particularly with fractional GPU requests or within complex applications, may lead to similar issues.

**Example 3: NodeSelector for GPU-Enabled Nodes:**

To prevent scheduling on non-GPU nodes during auto-scaling, node selectors can be utilized.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-specific-pod
spec:
  nodeSelector:
    cloud.google.com/gke-accelerator: nvidia-tesla-t4
  containers:
  - name: gpu-container
    image: nvidia/cuda:11.8.0-base-ubuntu20.04
    resources:
      limits:
        nvidia.com/gpu: 1
      requests:
        nvidia.com/gpu: 1
    command: ["nvidia-smi"]
```

*Commentary:* The `nodeSelector` ensures that this pod only schedules on nodes labeled with `cloud.google.com/gke-accelerator: nvidia-tesla-t4`. Replace `nvidia-tesla-t4` with the actual label corresponding to your GPU hardware. Without this, the auto-scaling system could create new nodes that aren't intended for GPUs. Utilizing `nodeSelector` to specify GPU-capable hardware effectively isolates GPU workloads from incompatible nodes.

In my experience, debugging these issues requires a systematic approach. Start by validating that the NVIDIA device plugin is correctly deployed across all target nodes with `kubectl get daemonset -n kube-system nvidia-device-plugin-daemonset`. Check logs for the plugin using `kubectl logs -n kube-system <pod-name>`. If you find errors there, that is your first diagnostic area. Confirm that all nodes that are meant for GPU workloads have correct GPU labels and resources by inspecting their node object via `kubectl get nodes -o yaml`. Ensure that the resource requests and limits within your pod specifications align with the available GPU resources. Examine GKE’s auto-scaling configurations, if applicable, to see if nodes without GPU resources are being provisioned into the pool of candidate nodes.

Finally, consulting documentation from both Google Cloud Platform and NVIDIA is paramount. The Google Kubernetes Engine documentation provides comprehensive information regarding node pools, auto-scaling, and GPU support. NVIDIA’s documentation covers the specifics of their device plugin and its configuration. These resources offer in-depth explanations and debugging steps that often resolve these complex scheduling problems. Specifically, the documentation covering GKE node pools, autoscaling options and GPU driver deployment is crucial for successful GPU workload management. They provide the foundation needed to correctly configure and deploy the infrastructure necessary to make these workloads schedulable.
