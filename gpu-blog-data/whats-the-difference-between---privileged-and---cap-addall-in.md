---
title: "What's the difference between `--privileged` and `--cap-add=all` in Docker?"
date: "2025-01-30"
id: "whats-the-difference-between---privileged-and---cap-addall-in"
---
The core distinction between Docker's `--privileged` and `--cap-add=all` flags lies in their scope and mechanism for granting elevated privileges to containers. While both aim to circumvent the inherent security restrictions of containerization,  `--privileged` offers a significantly broader, less granular control, effectively granting near-root access, whereas `--cap-add=all` provides a more controlled approach, adding all Linux capabilities to the container.  This difference has significant implications for security and maintainability, a lesson I learned firsthand during a critical incident involving a misconfigured Kubernetes deployment.

My experience highlighted the potential pitfalls of overly permissive container settings. We were utilizing a containerized machine learning application requiring access to certain hardware features.  Initially, opting for `--privileged` seemed the simplest solution. However, this resulted in a security vulnerability exploited by a rogue process within the container, which gained unauthorized access to the host system.  Switching to a more precise approach using `--cap-add` and explicitly specifying required capabilities mitigated this risk considerably.


**1. Clear Explanation:**

The `--privileged` flag grants the container almost all the capabilities of the host system. This essentially removes most of the security isolation provided by Docker.  The container's process will run with the same privileges as the Docker daemon, effectively bypassing most Linux kernel security mechanisms.  This is exceptionally risky and should be avoided whenever possible.  The implications extend beyond mere access to hardware resources.  A compromised container running with `--privileged` can potentially access and modify the host filesystem, network configuration, and other critical system components.

Conversely, `--cap-add=all` operates on a finer level of privilege management through Linux capabilities. Capabilities are a fine-grained mechanism for controlling system privileges. Instead of granting complete root access, `--cap-add` adds specific capabilities to the container's process.  `--cap-add=all` adds *all* defined Linux capabilities. While seemingly equivalent to `--privileged`, it is crucial to understand the subtle differences.  While `--cap-add=all` provides broad access, it doesn't automatically grant access to all devices or bypass all security mechanisms in the same way that `--privileged` does. Certain privileges, such as those related to direct kernel interactions, may still remain restricted, even with all capabilities added.

In essence, `--privileged` is a blunt instrument, granting broad access without precision, while `--cap-add=all` provides granular control but still grants a significant level of privilege.  The choice should always favor the more specific and restrictive approach whenever possible, prioritizing security and minimizing the attack surface.


**2. Code Examples with Commentary:**

**Example 1: Using `--privileged` (Highly discouraged)**

```bash
docker run --privileged -it ubuntu bash
```

This command runs an Ubuntu container in interactive mode (`-it`) with the `--privileged` flag.  This grants the container virtually complete access to the host system's resources and kernel functionalities.  The potential security risks associated with this are substantial.  Consider this only as a last resort for tasks requiring extremely deep system access, and even then, carefully evaluate the necessity and implement compensating security controls.


**Example 2: Using `--cap-add=all` (More controlled, but still potentially risky)**

```bash
docker run --cap-add=SYS_ADMIN --cap-add=NET_ADMIN --cap-add=DAC_OVERRIDE -it ubuntu bash
```

This command runs an Ubuntu container with specific capabilities added. `SYS_ADMIN`, `NET_ADMIN`, and `DAC_OVERRIDE` are examples of capabilities that might be required for certain tasks, such as managing system processes, networking configurations, or manipulating file permissions.  Instead of `--cap-add=all`, selectively adding needed capabilities reduces the security risks compared to using `--privileged` or `--cap-add=all`.  This approach demands a precise understanding of the required capabilities.


**Example 3: A more secure approach –  Minimizing capabilities**

```bash
docker run --cap-add=NET_ADMIN -it ubuntu bash
```

This example demonstrates a best practice. Instead of adding all capabilities or relying on `--privileged`, only the absolutely necessary capabilities are added. In this case, `NET_ADMIN` is added, which might be required for network-related operations within the container.  This drastically limits the potential impact of a compromise, as the container's access is restricted to only network-related functions.  This targeted approach is always preferable to using either `--privileged` or `--cap-add=all`.


**3. Resource Recommendations:**

For a deeper understanding of Linux capabilities, I recommend consulting the official Linux documentation.  The Docker documentation, particularly sections on security best practices and containerization fundamentals, is also essential reading.  A comprehensive guide on system administration principles will provide a strong foundational understanding for making informed decisions about privilege management. Finally, a text on security hardening techniques will provide valuable context on mitigating vulnerabilities within containerized environments.  These resources, studied together, will build a strong understanding of security implications and help you make the best choice between `--privileged` and `--cap-add`.
