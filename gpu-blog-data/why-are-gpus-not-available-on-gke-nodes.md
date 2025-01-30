---
title: "Why are GPUs not available on GKE nodes despite being present in the NodePool?"
date: "2025-01-30"
id: "why-are-gpus-not-available-on-gke-nodes"
---
The discrepancy between GPU presence in a Google Kubernetes Engine (GKE) NodePool configuration and their actual availability within deployed Pods stems from a crucial layer of orchestration: resource scheduling. While a NodePool declaration instructs GKE to create instances equipped with GPUs, the Kubernetes scheduler is not inherently aware of this specialized hardware. I've encountered this directly on several projects where complex deep learning workloads failed to leverage the intended GPU capacity, highlighting the need for explicit configuration.

The core issue revolves around Kubernetes’ resource model. By default, Kubernetes understands resources in terms of CPUs and memory. GPUs, considered a type of extended resource, require additional configuration for proper recognition and allocation. The NodePool, during its creation, provisions compute instances with GPUs, which are reflected in the node's metadata. However, this is merely hardware provisioning; it doesn't inform the Kubernetes scheduler about the GPU's presence or its specific type. The scheduler needs to be explicitly instructed to recognize GPUs as schedulable resources, allowing it to place Pods requiring GPU compute on appropriate nodes. Without this, the scheduler treats all nodes identically regarding resources except for CPU and RAM, leading to deployments failing to utilize available GPUs.

To bridge this gap, we utilize the NVIDIA GPU device plugin, a DaemonSet that runs on each node equipped with a GPU. This plugin performs two critical functions: Firstly, it discovers the installed GPUs and reports them as an extended resource to the Kubernetes API server. Typically, the extended resource is named `nvidia.com/gpu`. Secondly, it handles the lifecycle of these GPUs, allowing access to them by containers running within pods. Without this plugin, the Kubernetes scheduler has no knowledge of the GPU availability; the hardware remains effectively invisible from the perspective of the scheduler. Therefore, deploying the NVIDIA device plugin is crucial for enabling GPU support within Kubernetes. The plugin itself consists of an executable, which registers the device with kubelet and monitors the state.

In the context of GKE, these components operate within a managed environment, yet the underlying Kubernetes mechanisms remain consistent. GKE’s NodePool configuration provides the hardware, while the NVIDIA device plugin and the Pod definitions, are the key pieces for enabling GPU access to workloads. I have frequently found that even with a correctly configured NodePool, a failure to deploy the device plugin or to correctly specify the Pod resource requirements results in underutilization of expensive GPU resources.

Here are three practical examples illustrating common scenarios and their solutions.

**Example 1: Basic GPU request within a Pod definition**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-pod-example
spec:
  containers:
  - name: gpu-container
    image: nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
    resources:
      limits:
        nvidia.com/gpu: 1
      requests:
        nvidia.com/gpu: 1
```

This YAML file defines a simple pod requesting one GPU, specified using `nvidia.com/gpu`. The crucial part here is `resources.limits.nvidia.com/gpu` and `resources.requests.nvidia.com/gpu`. The limits section ensures that no more than one GPU is allocated to the container, while the requests part ensures the scheduler only places the pod on nodes which have at least one GPU. When I deployed this without the NVIDIA device plugin running, the pod would remain in the `Pending` state indefinitely. This occurs because the scheduler could not find a node with an advertised `nvidia.com/gpu` resource. Once the NVIDIA device plugin was installed, and reported this extended resource on nodes with GPUs, the pod scheduled properly. Without the resources declarations, the scheduler would not know the pod needs a GPU and can incorrectly place it on a node without them. The image used, `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04`, is a convenient base image with CUDA drivers, for a quick start.

**Example 2: Deploying the NVIDIA Device Plugin as a DaemonSet**

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  labels:
    k8s-app: nvidia-device-plugin
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      k8s-app: nvidia-device-plugin
  template:
    metadata:
      labels:
        k8s-app: nvidia-device-plugin
    spec:
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
      containers:
      - name: nvidia-device-plugin
        image: k8s.gcr.io/nvidia/k8s-device-plugin:v0.14.1
        resources:
          limits:
            memory: "512Mi"
          requests:
            cpu: "100m"
            memory: "256Mi"
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            add:
              - SYS_ADMIN
        volumeMounts:
          - mountPath: /var/lib/kubelet/device-plugins
            name: device-plugin
      hostNetwork: true
      nodeSelector:
        accelerator: nvidia-tesla-p100  # Match specific GPU label on node
      volumes:
        - name: device-plugin
          hostPath:
            path: /var/lib/kubelet/device-plugins
```

This YAML file provides the configuration for deploying the NVIDIA device plugin. This manifest utilizes a `DaemonSet`, ensuring that one instance of the plugin runs on every node in the cluster. Critically, it specifies `hostNetwork: true` to correctly access system devices, and a `nodeSelector` can be added to match nodes with a specific label indicating the presence of GPU resources; I've done this with `accelerator: nvidia-tesla-p100`. The image `k8s.gcr.io/nvidia/k8s-device-plugin:v0.14.1` is the official container from NVIDIA, and it is responsible for discovering and reporting the GPU resources available on the node. The toleration prevents the plugin from scheduling on nodes without GPU hardware, reducing resource usage. Without the plugin the `nvidia.com/gpu` resource will not be exposed to the Kubernetes scheduler.

**Example 3: Troubleshooting missing GPU allocation**

```bash
kubectl describe pod gpu-pod-example
kubectl describe node <nodename>
kubectl logs -n kube-system -l k8s-app=nvidia-device-plugin
```

These commands provide essential troubleshooting capabilities. The first, `kubectl describe pod gpu-pod-example` will show why a Pod is not being scheduled, and what resources it requires. For instance, it would show if it has a resource request that cannot be fulfilled by the cluster. The second command, `kubectl describe node <nodename>`, outputs details of the node including allocated resources and extended resources, if they have been reported by the plugin. Examining the output of `kubectl describe node` will reveal if the `nvidia.com/gpu` resource is present. Finally, `kubectl logs -n kube-system -l k8s-app=nvidia-device-plugin` will show the logs of the plugin, which are useful for identifying issues such as initialization problems or failed communication. On many occasions, I have had to review the plugin's logs to see if the driver installation is working, or if there are permissions problems preventing the plugin from registering the GPU with the kubelet. These commands, used in sequence, provide a systematic way to diagnose common issues relating to GPU unavailability.

For further understanding and advanced configuration, I recommend consulting the Kubernetes documentation on extended resources, focusing on device plugins and resource scheduling. The NVIDIA documentation for the device plugin and their GPU cloud platform is also invaluable. Textbooks on Kubernetes, especially those covering resource management, and any material focusing on the underlying mechanisms rather than specific tools or platforms, are also highly beneficial. Lastly, experimenting with small local Minikube deployments before deploying to production helped significantly in my troubleshooting experience, which provides a low-risk environment for configuration verification.
