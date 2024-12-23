---
title: "Should customer containers be separated into specific namespaces?"
date: "2024-12-23"
id: "should-customer-containers-be-separated-into-specific-namespaces"
---

,  This topic brings back some interesting memories from a large-scale cloud migration project I was involved in a few years ago, dealing with a multi-tenant platform. The question of whether to isolate customer containers using namespaces isn't a trivial one, and it’s definitely something I’ve seen play out both ways, with varying degrees of success and accompanying headaches. In my experience, the short answer is: absolutely, yes, *whenever* feasible and practical, and the long answer delves into why and how, which I'm happy to outline here.

The core idea behind using namespaces for customer containers revolves around **isolation**. Think of namespaces as creating distinct virtual environments within the same kernel. Without this separation, all containers share the same kernel resources – networking, process ids, mount points, user ids, and inter-process communication (ipc) mechanisms. This shared environment introduces potential vulnerabilities, commonly referred to as "container escape" issues, where a compromised container could theoretically impact or access resources of other containers running on the same host, potentially belonging to different customers. Such a scenario is a recipe for disaster in a multi-tenant system.

For a deeper dive into this concept, consider reading “Understanding the Linux Kernel” by Daniel P. Bovet and Marco Cesati. This book explains Linux kernel mechanisms at a low level, which is fundamental to grasp the importance of namespace isolation. Furthermore, “Docker Deep Dive” by Nigel Poulton provides a practical look at how these concepts translate to real-world container environments.

Now, let's break down *why* we want separate namespaces, going beyond just the broad statement of "security". Each type of namespace addresses specific isolation concerns:

*   **pid namespaces:** This namespace isolates the process id space. Meaning, process ids inside a container are distinct from those outside and from other containers within different pid namespaces. Without this, one container might be able to send signals to processes in another container, leading to unexpected behavior or even denial-of-service attacks.
*   **net namespaces:** Isolates networking configurations. This allows each container to have its own virtual network interface, routing tables, firewall rules, etc. This prevents containers from directly listening on the same ports and, coupled with appropriate network policies, helps contain network traffic and minimizes lateral movement potential.
*   **mnt namespaces:** Isolates the mount point space. Each container can have its own view of the file system. One container cannot directly access the mount points of another container, preventing file system traversal attacks.
*   **uts namespaces:** Provides hostname and domain name isolation. Without it, each container would share the same hostname, making them harder to differentiate.
*   **ipc namespaces:** Isolates inter-process communication (ipc) mechanisms like System V ipc and POSIX message queues. Without this, one container could potentially communicate with processes in another container using ipc, a significant security risk.
*   **user namespaces:** Perhaps one of the more complex, and often overlooked, areas. This provides user and group ID isolation. A user within the container can have a different set of user and group ids than outside, limiting damage should a vulnerability be exploited.

During that aforementioned migration project, we initially used a single namespace per node, grouping customer containers together, which, in hindsight, was a pretty terrible idea given the scale and sensitivity of the data. We encountered various issues ranging from unintended port conflicts to surprisingly tricky debugging issues tracing signals across different tenant containers. It wasn't *entirely* catastrophic, largely due to the additional layers of security we had, but it definitely was a major catalyst to re-engineer the entire setup towards more granular isolation.

Let me illustrate this with some code snippets using docker (as it's widely used). Assume each of these commands are executed in separate terminals:

**Snippet 1: No Namespaces Isolation**
This example shows a scenario where we aren’t using separate namespaces for a basic web service.

```bash
# Terminal 1
docker run -d -p 8080:80 nginx
docker ps
# Note the container id, let’s say it's 1234abcd

# Terminal 2
docker run -d -p 8081:80 nginx
docker ps
# Note this container's id, let’s say it's 5678efgh

# Now, both containers are running on host network
# and both are listening on port 8080 and 8081 respectively,
# but sharing the same network namespace
```

In the example above, the containers run in the default network namespace (usually the host network). Although they have different external ports mapped (8080 and 8081) they are still susceptible to potential attacks if any of the internal processes becomes compromised, as they share some kernel-level resources.

**Snippet 2: Namespaces Isolation Using Docker**

Now, let's demonstrate a more secure approach, using network namespaces.

```bash
# Terminal 1
docker run -d --net=none --name customer1 nginx
docker network create customer1-net
docker network connect customer1-net customer1

# Terminal 2
docker run -d --net=none --name customer2 nginx
docker network create customer2-net
docker network connect customer2-net customer2

# Now, customer1 and customer2 have their own network namespaces
# and we can use the docker network inspect command to see the details
# of the containers, and see that they are on different subnets and
# have completely separate network interface setups.
```

In the above scenario, each container lives in its own, completely separate network namespace, which provides better isolation.

**Snippet 3: Example of User Namespaces**

User namespaces are harder to illustrate via command-line alone but this will highlight a simple use-case.
```bash
# run container with user namespace enabled, maps the user 1000 in container to 10000 outside
docker run -it --user 1000 --userns-remap=10000:10000 --entrypoint bash ubuntu
# Inside the container, check the current user id
whoami # Returns the user within the container
id -u  # Returns 1000
# Exit container

# Now, create a file outside the container
echo "hello" > file.txt
# chown file.txt to user 10000 outside the container
sudo chown 10000:10000 file.txt

# If we attempt to access the file from inside the container,
# we'll find it's owned by user 1000 inside the container.
# If we ran the container without user namespace it would appear owned
# by root user inside the container, but not with user namespaces enabled.
```
This highlights that user namespaces add an extra layer of access control, with container users being mapped to different external ids.

Moving to separate namespaces for customer containers significantly increased the system's robustness. Although it introduced some complexity in areas like networking configurations, the enhanced security and reduced blast radius made it well worth the effort. The added complexity was offset by tools designed to manage container networking such as Calico, Cilium, or Flannel, which are well worth exploring if you’re tackling multi-tenant container deployments.

In short, isolating customer containers with namespaces is not just a good practice; it is generally a *necessity* when handling sensitive data or shared infrastructure. While it adds initial setup overhead, it is vital to provide a more secure and resilient environment. Therefore, any real-world multi-tenant container deployment should seriously consider the use of granular namespaces as a fundamental design decision. The resources mentioned and other related materials should provide a better understanding of the depth this topic has.
