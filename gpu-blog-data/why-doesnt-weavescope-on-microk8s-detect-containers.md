---
title: "Why doesn't Weavescope on Microk8s detect containers?"
date: "2025-01-30"
id: "why-doesnt-weavescope-on-microk8s-detect-containers"
---
Weavescope's inability to detect containers within a Microk8s environment often stems from a misconfiguration of the Weave Net plugin, specifically concerning its integration with the Kubernetes CNI (Container Network Interface).  My experience troubleshooting this across various projects, including a recent deployment for a high-frequency trading firm, highlights the crucial role of proper network namespace setup and the impact of conflicting CNI plugins.  The issue isn't necessarily a fundamental incompatibility, but rather a common oversight in configuration.

**1.  Clear Explanation:**

Weavescope relies on the Weave Net plugin to inject its observability agents into the network namespaces of containers.  Microk8s, being a lightweight Kubernetes distribution, requires careful management of its CNI. If another CNI plugin is active or improperly configured alongside Weave Net, it can interfere with Weave Net's ability to properly attach to the containers' namespaces. This interference manifests as Weavescope failing to detect or properly track the containers' network activity.  Further, incorrect installation or incomplete configuration of Weave Net itself can prevent it from correctly registering with the Kubernetes cluster, rendering its monitoring capabilities ineffective.  This can involve issues with the `kubelet` configuration, the `weave` daemon, or incorrect pod networking settings.  Finally, firewall rules, both within the host and potentially across the network, can block the necessary communication channels Weave Net needs to monitor the containers.

Several scenarios can lead to this problem:

* **Conflicting CNI plugins:**  If another CNI, such as Calico or Flannel, is already active and configured as the primary network provider in Microk8s, it may prevent Weave Net from taking control of the network namespaces. This results in Weavescope not seeing the containers.  Microk8s' streamlined nature makes this a frequent source of errors.
* **Incomplete Weave Net Installation:**  A seemingly successful installation might still be missing crucial components or configurations. This includes the proper deployment of the Weave Net daemonset and the correct configuration of the `kubelet` to utilize Weave Net.
* **Network Policy Restrictions:**  Stringent firewall rules or network policies, either on the host machine itself or on the network infrastructure, can block the necessary communication between Weave Net, the containers, and Weavescope.
* **Incorrect Kubernetes Cluster Configuration:**  Problems with the underlying Kubernetes cluster, such as misconfigurations in the kube-proxy or network settings within the control plane, can hinder Weave Net’s functionality.

Addressing these issues requires a systematic approach involving verification of CNI plugin configuration, checking the status of the Weave Net daemonset, and inspecting network policies and firewall rules.

**2. Code Examples and Commentary:**

The following examples illustrate potential problem areas and troubleshooting steps.  These examples are conceptual representations and may require minor adjustments depending on your exact Microk8s and Weave Net versions.

**Example 1: Verifying CNI Plugin Configuration (kubectl):**

```bash
kubectl get pods -n kube-system | grep weave
kubectl get cni -n kube-system
kubectl describe node <node-name> | grep CNI
```

**Commentary:** The first command checks for Weave Net pods in the `kube-system` namespace.  If they're not running or showing a `Ready` status, it indicates an installation or configuration problem. The second command lists all CNI plugins installed. If other plugins are present besides Weave Net, they may be conflicting. The third command examines the node's configuration to confirm Weave Net's role as the CNI.  Look for output explicitly mentioning `weave`. Absence of Weave Net or presence of conflicting CNIs indicates a configuration conflict.


**Example 2: Checking Weave Net DaemonSet Status (kubectl):**

```bash
kubectl get daemonset weave -n kube-system -o yaml
```

**Commentary:** This command retrieves the YAML configuration of the Weave Net daemonset. Examine the `status` section for any errors or indications that pods are not running correctly.  Pay close attention to the number of desired, current, and ready replicas.  Discrepancies indicate potential issues with the deployment or node connectivity.  This gives more detailed insight than a simple `kubectl get daemonset` command.


**Example 3: Inspecting Network Policies (kubectl):**

```bash
kubectl get networkpolicies -n <namespace>
kubectl describe networkpolicy <networkpolicy-name> -n <namespace>
```

**Commentary:** These commands retrieve network policies that may be affecting Weave Net's ability to communicate with containers within a specified namespace.  Check for policies that might overly restrict network traffic to or from containers or the Weave Net daemonset pods.  Examine the `spec.ingress` and `spec.egress` rules for overly restrictive policies.  If excessively strict, these policies need review and potential modification.


**3. Resource Recommendations:**

I recommend reviewing the official documentation for both Microk8s and Weave Net. Pay close attention to the sections on CNI plugin configuration and troubleshooting network connectivity issues.  Consult the Weaveworks community forums and support channels for assistance with specific problems.  Consider examining system logs for the Kubernetes control plane components, the `kubelet`, and the Weave Net daemon for potential clues regarding error messages or warnings.  Familiarity with Kubernetes networking concepts and troubleshooting techniques is vital for effective resolution.  Finally, a solid understanding of network fundamentals, particularly IP addressing, routing, and firewalls, will expedite diagnosis.
