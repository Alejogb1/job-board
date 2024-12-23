---
title: "Do Kubernetes pod replicas share a read-only file system from the underlying image?"
date: "2024-12-23"
id: "do-kubernetes-pod-replicas-share-a-read-only-file-system-from-the-underlying-image"
---

Alright, let's tackle this. I've encountered this particular point more times than I can count in the past decade, particularly when scaling microservices on Kubernetes. The short answer is yes, pod replicas *do* generally share a read-only filesystem derived from the underlying container image. However, like most things in the Kubernetes world, there are layers to unpack and nuances to understand to fully grasp the implications. It's not simply a "yes" or "no" situation, it’s more about how the container runtime manages the filesystem and how Kubernetes orchestrates this.

When a container image is built, it creates a series of layered filesystems. Each layer is essentially a set of changes made on top of the previous one. This approach is beneficial for image distribution and storage because unchanged layers can be reused across different images. Kubernetes leverages this mechanism. When a pod is created, the container runtime – typically Docker, containerd, or CRI-O – takes this layered image and creates a thin, writable layer on top of it. This writable layer is what allows modifications like logs or temporary files to be written to the container's filesystem.

The important point here is that each replica of a pod, each container instance created from the same image, gets its *own* instance of this writable layer. The read-only layers from the image itself are indeed shared among all replicas. These shared layers are generally stored on the node’s local filesystem in a way that allows them to be accessed by multiple container processes. The container runtime typically handles the details of mapping these layers to the container.

Now, where things get interesting is when you want to share files between pods, or even persist data across pod restarts. In that case, the read-only file system is not sufficient. We need volumes. Kubernetes provides various volume types – emptyDir, hostPath, configMap, secret, persistent volumes, etc. – to handle this. Let's consider a few scenarios, and how volumes solve the limitations of read-only filesystems, along with practical examples.

**Scenario 1: Handling Configuration with ConfigMaps**

Let's say you're running a web application. The application needs some configurable parameters, such as database connection strings, API keys, or feature flags. Baking these directly into the image is poor practice because it requires rebuilding the image for every single configuration change. This is where ConfigMaps come in. I've frequently seen teams struggle with this, attempting to bake configuration directly into the container build process, which leads to all sorts of deployment and management challenges.

Instead, we create a ConfigMap with these parameters, then mount it as a volume in our pods. Here is a snippet of how that can look in your pod configuration:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app
spec:
  containers:
  - name: my-app-container
    image: my-app-image:latest
    volumeMounts:
    - name: config-volume
      mountPath: /app/config
      readOnly: true
  volumes:
  - name: config-volume
    configMap:
      name: my-app-config
```

In this example, the `my-app-config` configmap (not shown, but it would consist of key-value pairs) is mounted to `/app/config` as a read-only volume inside the container. The actual files are not part of the image layers. This allows all replicas to have the same configuration, which is managed externally. If you update the ConfigMap, Kubernetes can roll out new pod instances with the updated configuration without requiring a rebuild or redeploy of the container image.

**Scenario 2: Log Aggregation with EmptyDir Volumes**

Another common use case is log aggregation. Each container instance writes its logs to a local file. While this works for basic cases, we often need to consolidate these logs for monitoring and analysis. We can use an `emptyDir` volume to facilitate this process. EmptyDir volumes are ephemeral and exist only as long as the pod does, making them suitable for temporary storage like log buffers.

Let’s illustrate how to set this up in our pod definition:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-app-logger
spec:
  containers:
  - name: my-app-container
    image: my-app-image:latest
    volumeMounts:
    - name: log-volume
      mountPath: /var/log/app
  - name: log-aggregator
    image: fluentd:latest
    volumeMounts:
      - name: log-volume
        mountPath: /var/log/app
  volumes:
  - name: log-volume
    emptyDir: {}
```

In this setup, the `my-app-container` writes its logs to `/var/log/app`, which is the same location mounted by the `log-aggregator` container. Both containers share the same `emptyDir` volume. The `emptyDir` itself isn't persistent; it's created when the pod starts, and destroyed when the pod terminates. Each pod replica has its own instance of the `emptyDir`, effectively avoiding any overlap or conflicts. This is a simpler version, but it works to highlight the read-only layer limitation.

**Scenario 3: Persistent Storage with PersistentVolumes**

Now let’s move to something where you need actual data persistence beyond the lifetime of a pod. Consider a database container, which needs to persist the data between restarts or rescheduling of pods. It's unacceptable for data to simply disappear when a pod is stopped. This is where PersistentVolumes and PersistentVolumeClaims come into play. This is critical for production systems, and it’s something that you’d quickly encounter as soon as you begin deploying anything beyond a basic application.

Here's a simplified example demonstrating how to mount a persistent volume to a database container:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: my-db
spec:
  containers:
  - name: my-db-container
    image: postgres:latest
    volumeMounts:
    - name: db-data
      mountPath: /var/lib/postgresql/data
  volumes:
  - name: db-data
    persistentVolumeClaim:
      claimName: my-db-pvc
```

In this case, `/var/lib/postgresql/data`, where Postgres stores its database files, is mapped to a PersistentVolumeClaim named `my-db-pvc`. The actual provisioning and management of the underlying storage device are handled by PersistentVolumes and StorageClasses configured in the cluster. Again, this volume is not part of the image, it’s completely separate. Each pod replica would likely have a distinct volume, or use an external database instance.

The key takeaway here is that while Kubernetes pod replicas share a read-only filesystem from the underlying image layers, this limitation is by design. It promotes image immutability, simplifies the creation of consistent container instances, and encourages the use of various volume types to handle configuration, ephemeral storage, and persistent data. The sharing of the read-only base allows for resource efficiency and faster pod deployments by allowing the reuse of image layers cached locally on the nodes.

For a deep dive into the container filesystem specifics, I would recommend "Docker Deep Dive" by Nigel Poulton for a good practical foundation. For understanding the lower level Linux primitives involved, "Understanding the Linux Kernel" by Daniel P. Bovet and Marco Cesati is exceptionally thorough. And of course, the official Kubernetes documentation and the source code itself is an invaluable resource, particularly the parts related to CRI (Container Runtime Interface) and volume plugins. Knowing these fundamentals has been absolutely crucial to my work over the years, and will serve any engineer well when navigating the complexities of containerized environments.
