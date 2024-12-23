---
title: "How does DIND on containerd address the deprecation of DockerShim in Kubernetes?"
date: "2024-12-23"
id: "how-does-dind-on-containerd-address-the-deprecation-of-dockershim-in-kubernetes"
---

Okay, let's talk about dind on containerd in the context of the dockershim deprecation. This isn't some abstract theoretical concept for me; I've spent considerable time wrestling (okay, *working*) with various container runtimes over the years, including a fairly hairy transition period when we had to migrate our k8s clusters away from the standard docker runtime. So, I have a deep understanding of the real-world practicalities involved.

The move from Docker as the container runtime in Kubernetes (k8s) to container-native solutions like containerd or CRI-O was, honestly, a long time coming. Docker, in its early days, was fantastic for developers—simple to grasp and use. However, as k8s matured, the overhead of having a full docker daemon involved became a clear bottleneck. Docker itself is a comprehensive platform, encompassing build tools, image management, and networking alongside the core runtime component. Kubernetes, however, only needs that latter bit, and forcing k8s to interact with the full Docker suite using the DockerShim created an unnecessary layer of complexity and performance limitations. This is the problem that the dockershim deprecation addressed.

Enter containerd, and its more minimal, targeted design. Containerd is a container runtime that directly aligns with the Container Runtime Interface (CRI) standard used by k8s. This eliminates the need for the shim layer, allowing k8s to interact directly and efficiently with the container runtime. But, and here's the crucial piece, sometimes you *need* docker itself, especially when developing or debugging containers. This is where the "dind on containerd" approach (docker-in-docker) becomes relevant.

The underlying concept of dind, in a nutshell, involves running a container that hosts a complete docker daemon inside. Now, think about it in this new context. We can run a container with Docker inside, all managed and orchestrated by containerd as the base runtime on the node, which k8s is orchestrating. This is very different from having Docker as the *host* runtime for k8s. We are *nesting* our docker instance inside a container managed by the containerd. So when we're using dind on containerd, we're not re-introducing the dockershim. We're just leveraging a docker container for those instances when the full docker CLI and image management capabilities are needed—and crucially, those instances remain *isolated* from the k8s orchestration layers.

How does it work? Basically, you create a Docker image that contains the docker daemon and client. This image gets deployed as a pod using containerd as the runtime managed by k8s. Inside the pod, you can then use the docker CLI to build, run, and manage docker images and containers—all within that specific containerized environment.

Let's consider three different scenarios with code examples:

**Scenario 1: Basic dind Deployment**

This simple snippet shows a basic k8s pod definition to deploy a dind container:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: dind-pod
spec:
  containers:
  - name: docker
    image: docker:dind
    securityContext:
        privileged: true
    volumeMounts:
    - name: docker-storage
      mountPath: /var/lib/docker
  volumes:
  - name: docker-storage
    emptyDir: {}
```

Here we are launching a pod containing a docker daemon inside, using the standard `docker:dind` image. The key here is setting `securityContext: privileged: true`. This is needed because the docker daemon inside the container needs access to kernel capabilities. The `emptyDir` volume ensures the containers persist across restarts, which is not ideal for production. For production scenarios, you would typically use a persistent volume claim.

After applying this configuration, you can `kubectl exec -it dind-pod -- bash` to get inside the pod and use the `docker` command.

**Scenario 2: Running Containers Inside dind**

Let's extend the first example and run a simple nginx container inside the dind container. This is the crux of why using dind is helpful.

```bash
kubectl exec -it dind-pod -- bash
# Inside the dind container
docker run -d -p 8080:80 nginx
docker ps # you should see your nginx container
```

Here we’ve successfully launched an `nginx` container entirely inside the docker daemon running in the `dind-pod` container. Remember, that *entire* container stack is being managed by containerd on the node and orchestrated by k8s. Kubernetes itself isn't aware of this nginx container. It's only aware of the container running the docker daemon.

**Scenario 3: Accessing the Docker API remotely with DIND**

Now, let’s showcase how to access the docker api remotely when using dind. First, inside the `dind` container start docker with a TCP port open to the host using a port forwarding configuration.

```bash
kubectl exec -it dind-pod -- bash
# Inside the dind container
dockerd -H tcp://0.0.0.0:2375 &
export DOCKER_HOST=tcp://127.0.0.1:2375
```

Now, we use `kubectl port-forward` to expose this local port to the host:
```bash
kubectl port-forward dind-pod 2375:2375
```

Then, in a separate terminal you should be able to interact with docker via the exposed tcp port using:

```bash
export DOCKER_HOST=tcp://localhost:2375
docker ps # you should see the containers inside your dind instance
```

So, the critical point here is that containerd manages the 'dind' container just like any other k8s container, and the docker daemon within is isolated. You’re using the full Docker feature set, but not at the host runtime level that k8s depends on. This bypasses the dockershim issue entirely.

This dind-on-containerd approach gives you a way to use Docker during development or debugging, but without reverting to using Docker as the primary k8s container runtime and, therefore, without relying on the deprecated dockershim.

For more in-depth knowledge on container runtimes, I strongly recommend diving into "Programming Kubernetes" by Michael Hausenblas and Stefan Schimanski. It goes deep into the core mechanics of k8s, including how CRI and containerd work. Another great resource would be “Docker Deep Dive” by Nigel Poulton to really grasp the specifics of the Docker architecture. Understanding both the higher level orchestration and the specific runtime details is key.

In summary, dind on containerd offers a pragmatic solution for scenarios where you still need access to the full Docker feature set, while maintaining the performance and architectural benefits of using a container-native runtime like containerd within a k8s environment. It's an isolation technique, and when used correctly, it avoids the pitfalls of the deprecated dockershim entirely. It's not ideal for production workloads in a normal circumstance due to the additional overhead, but for specific workflows, it remains an incredibly valuable pattern.
