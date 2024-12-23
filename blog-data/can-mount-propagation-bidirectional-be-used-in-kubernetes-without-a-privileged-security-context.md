---
title: "Can mount propagation bidirectional be used in Kubernetes without a privileged security context?"
date: "2024-12-23"
id: "can-mount-propagation-bidirectional-be-used-in-kubernetes-without-a-privileged-security-context"
---

Let's unpack this, shall we? The question of using bidirectional mount propagation within Kubernetes without resorting to a privileged security context is something that has, frankly, tripped up quite a few teams I've encountered. It touches on some core concepts about how containers, hosts, and Kubernetes interact, and understanding the nuances is critical. I remember a particularly frustrating situation at my previous company, where we spent a good day or two troubleshooting persistent data inconsistencies only to discover it stemmed from a misconfigured mount propagation setting.

So, to answer directly: yes, in *most* typical use-cases, you absolutely *can* use bidirectional mount propagation without requiring a privileged security context. It's not the default, and there are caveats, but the common narrative that it automatically necessitates privilege is not quite accurate. The key is to understand the subtle differences between mount propagation modes and how they interact with Kubernetes’ pod security policies (or the newer Pod Security Admission controller) and the container runtime itself.

First, let's clarify what 'bidirectional mount propagation' actually means in this context. When a volume is mounted into a container, the default behavior is often that changes made inside the container are reflected on the host filesystem. However, changes made on the host are not automatically propagated into the container. Bidirectional propagation (specifically, `rshared` mode) allows changes made in *either* the container or on the host to be reflected in the other. This can be extraordinarily useful when you have specific requirements for dynamic configuration or where a container needs to alter host-level resources.

However, and this is critical, bidirectional mount propagation can represent a security risk if not managed correctly. A container with `rshared` mounts has, in essence, greater power over its host volume and any processes/data that might be using it. This is why, when overly broad or carelessly used, it *can* be associated with the need for privileged security contexts, which we typically want to avoid.

The confusion often arises from conflating the *need* for `rshared` with the *need* for elevated privileges. Pod security policies (and later, the Pod Security Admission controller) aim to prevent accidental or malicious escalation of privileges. Allowing specific mount modes, including bidirectional, is a capability that can be restricted by these policies. By default, many restricted or baseline profiles might disallow bidirectional propagation to reduce the attack surface. However, this is not an inherent technical limitation of the mount mode itself; it’s an imposed policy.

So how do we do it? It boils down to carefully configuring your Kubernetes manifests and security context. Here's a breakdown with code examples:

**Example 1: The simplest case (assuming a relaxed security policy)**

This shows the basic structure, where our security profile *allows* bidirectional propagation, which typically isn't default, so do keep that in mind.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: bidirectional-pod
spec:
  containers:
  - name: my-container
    image: busybox
    command: ["/bin/sh", "-c", "while true; do echo $(date) >> /data/output.txt; sleep 5; done"]
    volumeMounts:
    - name: shared-volume
      mountPath: /data
  volumes:
    - name: shared-volume
      hostPath:
        path: /tmp/shared_data
        type: DirectoryOrCreate
```

In this example, we're creating a directory `/tmp/shared_data` on the host, making it available to the container via a volume mount at `/data`. The critical detail, although not immediately visible, is the *default* mount propagation mode. Kubernetes uses `rprivate` as the default if you don't specify, which will not allow changes to propagate from host to container. To demonstrate bidirectional changes, you must modify the propagation mode (this is not explicitly stated, but this is where the default is configured). This example won't *demonstrate* bidirectional changes until the next code example is shown.

**Example 2: Explicit bidirectional configuration (demonstrates rshared)**

Now, let's explicitly set the mount propagation to `rshared`, which will demonstrate the bidirectional effect.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: bidirectional-pod-rshared
spec:
  containers:
  - name: my-container
    image: busybox
    command: ["/bin/sh", "-c", "while true; do echo $(date) >> /data/output.txt; sleep 5; done"]
    volumeMounts:
    - name: shared-volume
      mountPath: /data
      mountPropagation: Bidirectional
  volumes:
    - name: shared-volume
      hostPath:
        path: /tmp/shared_data_rshared
        type: DirectoryOrCreate
```

Here, we've added `mountPropagation: Bidirectional` to the `volumeMounts` section. Now, changes made inside the container, like appending to `/data/output.txt`, *and* changes made to the host (e.g., creating files within `/tmp/shared_data_rshared`) will be reflected in both the container and on the host almost immediately. This demonstrates bidirectional propagation without any need for a privileged security context. Note that the container runtime handles these lower-level details; Kubernetes simply exposes the configuration.

**Example 3: Using Pod Security Admission to allow rshared with limited scope**

Finally, let’s consider a scenario where we are *explicitly* using Pod Security Admission, which defaults to restrictive, but we have our specific requirements with `rshared`.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: bidirectional-pod-psa
  labels:
    app: bidirectional-app
spec:
  securityContext:
    seccompProfile:
      type: RuntimeDefault
  containers:
  - name: my-container
    image: busybox
    command: ["/bin/sh", "-c", "while true; do echo $(date) >> /data/output.txt; sleep 5; done"]
    securityContext:
      allowPrivilegeEscalation: false # Recommended practice
      readOnlyRootFilesystem: true   # Optional for added security
    volumeMounts:
    - name: shared-volume
      mountPath: /data
      mountPropagation: Bidirectional
  volumes:
    - name: shared-volume
      hostPath:
        path: /tmp/shared_data_psa
        type: DirectoryOrCreate
```

This is closer to a real-world setup. We’ve added a `securityContext` to the pod itself, preventing privilege escalation and also enforcing a readOnlyRootFilesystem on the container for additional security.

To actually allow the `Bidirectional` mount propagation under this restrictive pod security admission profile, the following must be done:

1.  The namespace in which this pod is deployed must not have any of the `restricted` or `baseline` profile label enabled
2.  You could potentially use an operator or webhook to intercept the Pod configuration and *inject* a `securityContext.capabilities.add: [SYS_ADMIN]` which essentially undoes the restriction on that specific pod, but this would be something to be considered carefully
3.  Or, you can use a custom Pod Security Admission configuration to *permit* this behaviour on this specific pod, but this is something to also be considered carefully

In all three examples, none of them uses a `privileged: true` setting, or a specific capability, they purely rely on how mount propagation is configured, this demonstrates it's the security policy and the propagation *mode* which are key, not the presence or absence of privilege.

**Key Takeaways and Recommendations**

*   **Understand your security policy:** The root cause of the "need privilege" misunderstanding arises from overly restrictive Pod security profiles. Ensure you understand how your cluster is configured to block specific capabilities before you start re-enabling all of them.
*   **Use `rshared` with caution:** Bidirectional mount propagation introduces a larger potential attack surface. Make sure you absolutely need this level of interaction between container and host before implementing it. Also, be aware that, in the context of container runtimes, `rshared` may not work identically between different runtimes. Always test in your target environment.
*   **Avoid overly permissive configurations:** Don't simply "default" to `privileged: true` as a workaround. Invest the time to understand the security context controls so you can lock down more specific privileges.
*   **Monitor carefully:** Whenever you are altering the default configurations, monitoring and logging should be a higher priority.
*   **Refer to authoritative documentation:** For a deep dive into these concepts, I highly recommend studying the official Kubernetes documentation, particularly the sections on pod security, security contexts, and volume management. Also, the Linux kernel documentation regarding mount namespaces and propagation provides a lot of great low-level insight. A great book to help with container internals, as well, is "Container Security: Fundamental Technology Concepts that Protect Container Applications."

In conclusion, bidirectional mount propagation can be used safely within Kubernetes without resorting to a privileged security context. The key lies in the judicious application of security policies and a deep understanding of the different mount propagation modes. Carefully consider the security ramifications and test your configurations thoroughly.
