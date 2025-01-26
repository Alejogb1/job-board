---
title: "Why isn't the GPU node scaling up in my AWS EKS managed node group?"
date: "2025-01-26"
id: "why-isnt-the-gpu-node-scaling-up-in-my-aws-eks-managed-node-group"
---

A frequent bottleneck in scaling GPU workloads within Amazon Elastic Kubernetes Service (EKS) managed node groups stems from insufficient configuration beyond simply provisioning GPU instances. Over the years I've encountered this situation repeatedly, observing that while users often ensure their EC2 instances are GPU-equipped, the Kubernetes infrastructure itself often lacks the necessary settings to properly utilize and discover these resources. Specifically, correctly configuring the Nvidia device plugin, understanding resource requests and limits, and using the appropriate autoscaling policies are crucial.

**Understanding the Root Causes**

The core problem arises from the disconnect between AWS’s infrastructure and Kubernetes’s resource management. Simply spinning up a `p3.2xlarge` instance, for example, does not automatically make its GPU available to Kubernetes as a schedulable resource. Kubernetes, by default, doesn't recognize GPU devices without specific extensions. The critical component enabling this is the Nvidia device plugin. This daemonset, deployed within the cluster, discovers the GPU hardware on each node and advertises it as a resource (typically `nvidia.com/gpu`) that pods can then request.

Absent this plugin, pods requiring GPU resources remain in a `Pending` state, unable to be scheduled because Kubernetes is unaware of available GPUs. Further, resource requests and limits specified in pod definitions must align with what the Nvidia plugin advertises. If a pod requests `nvidia.com/gpu: 1`, and the node’s resources only indicate `nvidia.com/gpu: 0` (because the plugin isn't functioning or installed), the scheduling will again fail.

Finally, autoscaling mechanisms must be configured correctly to ensure that when the current GPU node capacity is fully utilized, new nodes are added to accommodate pending pods requesting GPU resources. Misconfigured autoscalers, improperly set scaling policies, or insufficient resource reservations on the node itself can all impede proper scaling. These factors can combine to create a situation where the node group seems to be “stuck” at its current size.

**Code Examples and Explanation**

The following examples demonstrate key configurations needed for a working GPU-enabled Kubernetes cluster and how they impact node scaling.

**Example 1: Nvidia Device Plugin Manifest**

This YAML manifest defines a DaemonSet for the Nvidia device plugin. It ensures that one instance of the plugin runs on each node that has an Nvidia GPU, enabling the GPU resources to be advertised to the Kubernetes scheduler.

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nvidia-device-plugin-daemonset
  namespace: kube-system
spec:
  selector:
    matchLabels:
      name: nvidia-device-plugin-ds
  template:
    metadata:
      labels:
        name: nvidia-device-plugin-ds
    spec:
      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
      containers:
      - name: nvidia-device-plugin
        image: nvcr.io/nvidia/k8s-device-plugin:v0.14.1
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            add: ["MKMNT"]
        volumeMounts:
        - name: device-plugin
          mountPath: /var/lib/kubelet/device-plugins
      volumes:
      - name: device-plugin
        hostPath:
          path: /var/lib/kubelet/device-plugins
```

*   **`tolerations`**: This ensures the daemonset is scheduled on nodes that have the `nvidia.com/gpu` label applied. The label is applied automatically to GPU instances if their associated node group launch template has necessary settings to detect that they are GPU instances.
*   **`image`**: Specifies the container image for the device plugin, pulled from NVIDIA's container registry.
*   **`securityContext`**: The plugin needs the `MKMNT` capability in order to create device plugin socket files on the host.
*   **`volumeMounts`**: Mounts the `device-plugins` directory into the container to communicate with the kubelet.

Without this DaemonSet correctly deployed and running, Kubernetes will be completely unaware of any GPU resources. The associated node groups will therefore remain at their current size regardless of any pending pods.

**Example 2: Pod Definition with GPU Resource Request**

This manifest illustrates a sample pod definition that requests a GPU resource. Crucially, the `resources` section defines the amount of `nvidia.com/gpu` required for the pod to be schedulable.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: gpu-workload
spec:
  containers:
  - name: gpu-container
    image: nvidia/cuda:12.3.2-base-ubuntu22.04
    resources:
      limits:
        nvidia.com/gpu: 1
      requests:
        nvidia.com/gpu: 1
    command: ["nvidia-smi"]
```

*   **`resources.limits.nvidia.com/gpu: 1`**: The pod requests exclusive use of a single GPU.
*   **`resources.requests.nvidia.com/gpu: 1`**: The pod requests allocation of a single GPU resource.

If a node does not have a GPU, or if the device plugin is not configured to expose the GPU resource, the pod will remain in a `Pending` state. It’s vital to remember that the `limits` and `requests` must align with what the Nvidia device plugin is advertising on the node. If the request exceeds the node capacity or if there is no available `nvidia.com/gpu` resource, the cluster auto-scaler might trigger the addition of more nodes to accommodate the pod, assuming the autoscaling settings allow it to do so. However, if the node is created without correctly applying the `nvidia.com/gpu` label, or if the plugin is not operational, it cannot accept the pending workload.

**Example 3: Cluster Autoscaler Configuration**

This example is a partial manifest demonstrating how the cluster autoscaler is used within an EKS cluster. This configuration will scale nodes based on requests for resources such as `nvidia.com/gpu`. This particular example focuses on relevant autoscaling annotations:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: cluster-autoscaler
  namespace: kube-system
spec:
  template:
    spec:
      containers:
      - name: cluster-autoscaler
        args:
        - --v=4
        - --stderrthreshold=info
        - --cloud-provider=aws
        - --skip-nodes-with-local-storage=true
        - --expander=least-waste
        - --node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled
        - --balance-similar-node-groups
      volumes:
      - name: ssl-certs
        hostPath:
          path: /etc/ssl/certs
      terminationGracePeriodSeconds: 10
      nodeSelector:
         kubernetes.io/os: linux
```

*   **`--node-group-auto-discovery=asg:tag=k8s.io/cluster-autoscaler/enabled`**: The autoscaler uses this parameter to discover which node groups to scale using a specific tag. For managed node groups this is set as a tag applied to the EC2 autoscaling group that backs the managed node group.
*   **`--balance-similar-node-groups`**: This can help to make sure that if multiple ASGs are available for the same purpose they have roughly the same number of active nodes.
*   **`expander=least-waste`**: This option specifies that the autoscaler will select the node group with the least wasted CPU and memory capacity.

While the autoscaler is crucial for cluster scaling, its efficacy is dependent on correctly configured node groups that expose their available GPUs. The autoscaler expands the node group if it identifies unschedulable pods due to a lack of resource capacity, provided that it has been configured with the appropriate options to discover the node group and that the node group has an ASG backed by appropriately sized GPU instances.

**Recommendations**

When encountering scaling issues with GPU nodes, consider the following:

1.  **Verify the Nvidia Device Plugin:** Ensure the Nvidia device plugin is running correctly on all your GPU nodes and has successfully advertised `nvidia.com/gpu` as a schedulable resource by describing a node using `kubectl describe node <node-name>`. Check the logs of the device plugin pod for any errors using `kubectl -n kube-system logs -l name=nvidia-device-plugin-ds`.
2.  **Review Pod Resource Requests:** Confirm that your pod definitions accurately request `nvidia.com/gpu` and specify appropriate limits. Insufficient requests will lead to pods not utilizing the GPUs, whereas excessive limits may make the pod unschedulable.
3.  **Examine Autoscaler Configuration:** Scrutinize your cluster autoscaler configuration, ensuring that it is correctly discovering your GPU-enabled node groups using the tag `k8s.io/cluster-autoscaler/enabled`, and the relevant tags have been applied to the ASG backing your EKS managed node group. Also, confirm that the autoscaler has sufficient permissions to manage the node group's scaling activities and that the configured scaling policies have appropriate parameters.
4.  **Check Cloud Provider Metadata:** Ensure your nodes correctly report available resources to the Kubernetes API. This can be verified by inspecting the node's description. If `nvidia.com/gpu` is not listed as an allocatable resource, the issue lies in the device plugin's setup or the node's labeling.
5. **Review your Launch Template/Launch Configuration:** The EKS managed node group’s launch template or configuration should have a valid UserData script which installs required NVIDIA packages and ensures the nodes join the cluster with the appropriate labels. Ensure that a valid security group is being used as well as any custom AMI settings are configured correctly.

By systematically investigating each of these components, I have found that a root cause for seemingly inexplicable GPU node scaling problems can be identified. Addressing the above points will not only enable scaling but will also lead to a more predictable and reliable GPU workload deployment environment. Ignoring these aspects leads to unexpected behavior and resource wastage.
