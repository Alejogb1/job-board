---
title: "Do pod replicas share read-only file systems from the image?"
date: "2024-12-16"
id: "do-pod-replicas-share-read-only-file-systems-from-the-image"
---

Alright, let’s tackle this one. It’s a frequently encountered question, and a fundamental aspect of understanding containerized environments like Kubernetes pods. The core issue revolves around how containers within a pod, particularly the replicas, access the filesystem originating from the container image. The short answer is yes, pod replicas typically share read-only file systems derived directly from their respective container images. However, as is often the case in software, the devil is in the details.

My past experience on a large-scale microservices project, where we were running hundreds of pods handling various backend tasks, highlighted just how crucial understanding this mechanism is. Initially, we assumed every instance had a fully isolated writable volume, which led to some unexpected data loss issues during scaling events. We had incorrectly modified files in-place inside the container itself, which of course was not the proper design pattern. It was a painful lesson, but a great example of how even seasoned developers can fall victim to misunderstanding the underlying technology.

Here’s the breakdown: when a container image is built, it typically consists of layers. These layers represent the files and directories added at each step of the build process. Think of it like a stack of read-only slices, where each layer builds upon the previous one. When you deploy a pod and its replicas, each replica gets instantiated with an identical read-only copy of this layered filesystem as defined by the image manifest. This ensures consistency across all instances.

The crucial takeaway here is that changes made *within* a running container are usually not persistent unless they're directed into specific volumes that are mounted separately (and thus are not read-only). These volumes can be either *ephemeral*, existing only for the lifetime of the pod, or *persistent*, surviving pod restarts. Modifications done to the container's root filesystem in-place within the running instance of the container are lost on restart.

Now, let's illustrate this with some concrete examples. I’ll provide snippets in pseudo-yaml, simulating how these settings typically appear in a kubernetes pod definition, along with python code examples to demonstrate file access.

**Example 1: Basic Read-Only Filesystem**

Consider a simple pod definition that creates three identical replicas, all based on a lightweight alpine image.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: basic-read-only
spec:
  replicas: 3
  selector:
    matchLabels:
      app: basic-read
  template:
    metadata:
      labels:
        app: basic-read
    spec:
      containers:
      - name: alpine-container
        image: alpine:latest
        command: ["sh", "-c", "sleep 3600"] # Keep the container running
```

If we exec into any of these containers and try to modify a file in the root filesystem, it will either fail because of the read-only nature or, if the container does have a mounted scratch or tmpfs write layer, the changes will be lost when the container is restarted.

Let's try to modify a file that likely exists, for example, create a file in `/tmp`. We don't need code for this, the command is sufficient.

```bash
# Inside a container instance
touch /tmp/mytestfile
```

The `/tmp` directory will generally be writable, because it's often mounted as a `tmpfs`, so the file may appear to be written. If the container instance gets restarted or moved to a different machine, and the `/tmp` folder does not have associated persistent storage (which is typically is not the case) the file will not be available after that restart.

**Example 2: Using an EmptyDir Volume**

If we need to write data that should persist across container restarts within the same pod lifecycle, we use an `emptyDir` volume.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: empty-dir-example
spec:
  replicas: 3
  selector:
    matchLabels:
      app: empty-dir
  template:
    metadata:
      labels:
        app: empty-dir
    spec:
      containers:
      - name: alpine-container
        image: alpine:latest
        command: ["sh", "-c", "python /app/write_to_shared.py"] # Keep the container running
        volumeMounts:
        - name: shared-volume
          mountPath: /shared-data
      volumes:
      - name: shared-volume
        emptyDir: {}
```

And then, within the image we've created a python script, `write_to_shared.py`, like this:

```python
import time

with open('/shared-data/testfile.txt', 'a') as f:
    f.write(f"Data from container: {time.time()}\n")
    f.flush()

time.sleep(120)
```

Here, we see that any modifications made to `/shared-data` inside each replica will be persisted for the lifetime of the pod. This also provides a simple communication mechanism between containers in the pod when using a shared volume. It is crucial that any application that is expected to perform writing of data be configured to target a shared or persistent volume. The read only nature of the base image ensures that you cannot easily corrupt the base image layer.

**Example 3: Using PersistentVolumes and PersistentVolumeClaims**

For persistent storage across pod lifecycles, we utilize `PersistentVolumes` and `PersistentVolumeClaims`. This setup often involves more infrastructure considerations but is essential for stateful applications. I will not cover how to provision persistent volumes in this example as it's beyond the scope of what was asked, but here's how you would incorporate it into the pod definition.

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: persistent-volume-example
spec:
  replicas: 3
  selector:
    matchLabels:
      app: persistent-volume
  template:
    metadata:
      labels:
        app: persistent-volume
    spec:
      containers:
      - name: alpine-container
        image: alpine:latest
        command: ["sh", "-c", "python /app/write_to_persistent.py"]
        volumeMounts:
        - name: persistent-volume
          mountPath: /persistent-data
      volumes:
      - name: persistent-volume
        persistentVolumeClaim:
          claimName: my-pvc # This PVC must already exist.
```

And the application, which also resides in the image, now targets a persistent volume claim:

```python
import time
with open('/persistent-data/log.txt', 'a') as f:
        f.write(f"Entry {time.time()} from container.\n")
        f.flush()
time.sleep(120)
```

In summary, yes, pod replicas share read-only file systems from the container image. This provides immutability and consistency across instances. Writing to files within this read-only filesystem will typically be lost unless you explicitly utilize volumes, like `emptyDir` or, for persistence, `PersistentVolumes`.

If you want to gain deeper understanding about this topic, I'd highly recommend the *Kubernetes in Action* book by Marko Lukša, which provides a very comprehensive overview of the core concepts and best practices. Also, exploring the official Kubernetes documentation, especially the sections on volumes and persistent storage, is invaluable. For a broader view on container image layers, *Docker Deep Dive* by Nigel Poulton offers fantastic insights. Finally, the paper *Docker: Lightweight Linux Containers for Consistent Development and Deployment*, from the USENIX 2014 LISA conference, though slightly older, provides invaluable insight to the design concepts behind modern containerization. It's important to stay grounded in both the theoretical and practical aspects of these systems. This will enable a proper understanding of how to build efficient and reliable systems utilizing this underlying technology.
