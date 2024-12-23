---
title: "Do pod replicas share a read-only filesystem?"
date: "2024-12-23"
id: "do-pod-replicas-share-a-read-only-filesystem"
---

Let's tackle this one. I’ve seen a fair number of deployments where this question comes up, usually mid-troubleshooting when things aren't quite behaving as expected. So, do pod replicas share a read-only filesystem? The short answer is: *typically*, no, they do not share a read-only filesystem directly between them in the way one might initially assume. There's more nuance involved, and understanding it is critical for designing robust and predictable containerized applications.

When we talk about containerized applications in kubernetes, particularly when using pod replicas, each replica effectively gets its own ephemeral filesystem. This is foundational to how containerization provides isolation. The base image pulled for the container, as declared in your container manifest, does often present as read-only, but that’s more about image immutability during the container creation process than anything else. The containers, and by extension, their pods, get a *copy* of this base image layered with a writable layer on top. This copy is separate for each pod replica, not shared.

I remember one particular incident where this caused a lot of headaches. We had a logging service running in kubernetes, where the configuration file was being altered programmatically, using a volumeMount, at pod startup. The assumption, based on a less-than-perfect understanding of kubernetes mechanics, was that since the underlying image was ‘read-only’ (as advertised), all pods would share the same altered configuration after our initial modification. We quickly found that wasn't the case, and every pod ended up with its own slightly different configuration after modifying the shared config file. The issue boiled down to each replica having its copy, and the shared volume only provided the initial state. This experience really underscored the importance of understanding the difference between base images, container layers, and shared volumes.

, so where does the idea of read-only come from? It primarily stems from how container images are structured and used. Docker, and by extension, Kubernetes’s container runtime interface (CRI), employs a layered filesystem approach. Each layer in the image is, essentially, immutable. The base layers comprise the foundation of the operating system, libraries, and application binaries. When a container starts, these immutable layers are stacked, and a writable layer is placed on top for that container instance to perform its operations. This implies that any changes your application makes are local to *that* particular container's writable layer. If you want the changes to persist beyond that container's lifecycle, you will explicitly have to persist them in a persistent volume or some other external storage.

It's also crucial to distinguish between the container's *image* as read-only and filesystem access from within the container. The application, by default, is often not running in a read-only filesystem *inside* the container. This is configured through kubernetes security context and the readOnlyRootFilesystem field on the securityContext definition, which is usually false. Even without explicitly setting readOnlyRootFilesystem to 'true,' the base layers of the image remain immutable. It is the writable layer on top that allows for changes within the container.

Let's look at a few code snippets to illustrate this. Firstly, the very simple case, just to show that each pod gets a copy of the same configuration.

**Snippet 1: Showing Independent Filesystems**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: config-test
spec:
  replicas: 2
  selector:
    matchLabels:
      app: config-test
  template:
    metadata:
      labels:
        app: config-test
    spec:
      containers:
      - name: config-test
        image: busybox
        command: ["/bin/sh", "-c"]
        args:
          - |
            echo "initial_config=value" > /tmp/config.txt;
            sleep 5;
            echo "updated_config=new_value" >> /tmp/config.txt;
            cat /tmp/config.txt;
            sleep infinity
```
This deployment starts two pods. Each pod will create a `config.txt` file, append to it, and then print the file contents. Because the write is occurring in each pod's writable layer, modifications are not propagated between replicas. If you were to examine the logs of each pod separately, each pod would have a distinct file, both containing "initial_config=value" and "updated_config=new_value". They are not shared, they are copies.

Next, let's consider how a shared volume comes into the picture:

**Snippet 2: Using a Shared volume for persistent data**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: shared-volume-test
spec:
  replicas: 2
  selector:
    matchLabels:
      app: shared-volume-test
  template:
    metadata:
      labels:
        app: shared-volume-test
    spec:
      volumes:
        - name: shared-data
          emptyDir: {}
      containers:
      - name: shared-volume-test
        image: busybox
        volumeMounts:
          - name: shared-data
            mountPath: /data
        command: ["/bin/sh", "-c"]
        args:
          - |
            echo "initial_config=value" > /data/config.txt;
            sleep 5;
            echo "updated_config=new_value" >> /data/config.txt;
            cat /data/config.txt;
            sleep infinity

```

Here, we introduce an `emptyDir` volume that the two pods will mount at `/data`. While both pods can now access the same directory, the volume itself is still not read-only by default. In this case, both pods will write to the same underlying location, resulting in one or both pods overwriting the file initially created. The final content will depend on which writes first. It's also important to note that an `emptyDir` volume will get deleted when the last pod using it is terminated.

Finally, let's demonstrate how we could *enforce* a read-only root filesystem by using the securityContext option mentioned:

**Snippet 3: Enforcing Read-only filesystem**

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: readonly-fs-test
spec:
  replicas: 2
  selector:
    matchLabels:
      app: readonly-fs-test
  template:
    metadata:
      labels:
        app: readonly-fs-test
    spec:
      securityContext:
        runAsNonRoot: true
        seccompProfile:
          type: RuntimeDefault
      containers:
      - name: readonly-fs-test
        image: busybox
        securityContext:
          readOnlyRootFilesystem: true
        command: ["/bin/sh", "-c"]
        args:
          - |
            echo "this will fail" > /tmp/test.txt;
            sleep infinity

```
In this final example, the `securityContext` field with `readOnlyRootFilesystem: true` makes the root filesystem read-only. This means the `echo "this will fail"` command will fail, as the `/tmp` directory is not inside a writable volume and cannot be modified. This is a common hardening technique. You'll often see this in conjunction with running containers as non-root users.

For a deeper dive into the nuances of container filesystems and Kubernetes volumes, I'd suggest consulting "Kubernetes in Action" by Marko Lukša for a comprehensive guide to how Kubernetes handles storage. Additionally, the official Kubernetes documentation is the go-to resource for the most up-to-date information and is kept current with each release. Finally, for a more foundational understanding of container technology and the underlying technologies, “Docker Deep Dive” by Nigel Poulton is highly recommended.

In summary, while the *image* itself is typically presented in a read-only fashion, each pod replica obtains its own independent, writable copy. Shared read-only filesystems (in the traditional sense), if desired, will often need to be implemented through persistent volumes or similar mechanisms explicitly. Understanding this distinction is vital for building reliable and maintainable applications in kubernetes, and it's a lesson I learned the hard way!
