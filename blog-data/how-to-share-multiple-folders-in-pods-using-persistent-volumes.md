---
title: "How to share multiple folders in pods using persistent volumes?"
date: "2024-12-23"
id: "how-to-share-multiple-folders-in-pods-using-persistent-volumes"
---

Okay, let's unpack this. Sharing multiple folders across pods using persistent volumes, while seemingly straightforward, often throws up some interesting challenges in practice. I've had to navigate this more than once, particularly back when I was building a microservices architecture that relied heavily on shared data stores for caching and configuration. We ran into some fairly common roadblocks, which, thankfully, led us to refine our approach. Essentially, the issue boils down to correctly structuring your persistent volume claims (pvcs) and understanding how pods interact with them within kubernetes.

The core problem is that a single persistent volume (pv), and consequently a single pvc, is designed to be mounted as a single directory within a pod. This isn't particularly helpful when you need different pods to access distinct subdirectories within that shared storage space. The naive approach, of course, is to simply mount the whole volume to each pod. This, while functional, often leads to chaos, permission conflicts, and ultimately, a hard-to-maintain system. It's better to think about dividing the persistent storage into logical units, each accessible by the pods that require it.

There are a few good ways to handle this and avoid that mess, and the approach I found most reliable involves a combination of persistent volumes with subpath mounts and a carefully considered folder structure. Another viable solution is using different pvcs with hostPath or nfs volumes, but this doesn't scale well and increases the chance of error due to manual volume configurations, particularly in a production setup. So, let's focus on leveraging the power of subpaths within the same persistent volume.

The basic concept revolves around creating a single persistent volume that houses all shared data and then using subpath mounts to grant individual pods access to only the specific directories they require. For instance, imagine you have a `shared-data` directory on your persistent volume. Inside, you have subdirectories `config`, `cache`, and `logs`. One pod might need only the `config` folder, another might need `cache`, and a logging aggregation pod would need `logs`. Instead of mounting the entire `shared-data` directory to all pods, we use subpaths during the mount definition. This significantly reduces the risk of inadvertent data corruption or permission conflicts and keeps things well-organized.

Let’s dive into some examples. Consider a scenario where we need to provide a web application pod with configuration data. This config will be stored within a specific subpath.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: shared-data-pvc
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 1Gi
  storageClassName: standard  # Or the name of your appropriate storage class

---
apiVersion: v1
kind: Pod
metadata:
  name: web-app-pod
spec:
  containers:
  - name: web-app
    image: nginx
    volumeMounts:
    - name: shared-data
      mountPath: /app/config
      subPath: config  # Mounting only the config subfolder
  volumes:
  - name: shared-data
    persistentVolumeClaim:
      claimName: shared-data-pvc

```

In this example, the `web-app-pod` mounts the `shared-data-pvc` and then, importantly, uses the `subPath: config` to access only the `/config` directory, which we assume exists in the root of our shared persistent volume. Note the `readwritemany` access mode which is crucial for allowing multiple pods to read and potentially write to the same volume simultaneously; however, if you are using it, be aware of the potential issues regarding data integrity that might come up.

Now, let’s assume another pod, say, a cache service, requires access to the `cache` directory within the same persistent volume. Here’s how that pod configuration might look:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: cache-service-pod
spec:
  containers:
  - name: cache-service
    image: redis:latest
    volumeMounts:
    - name: shared-data
      mountPath: /data
      subPath: cache  # mounting only the cache subfolder
  volumes:
  - name: shared-data
    persistentVolumeClaim:
      claimName: shared-data-pvc
```

This snippet shows the `cache-service-pod` mounting the same `shared-data-pvc` but using `subPath: cache`, so it has only access to the cache directory. This gives better control and reduces the risk of overwriting or accessing something that the pod isn’t supposed to.

Finally, let's consider a third pod that needs to read the `logs` folder within the same shared persistent volume. This could be a log aggregation pod, for example.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: log-aggregator-pod
spec:
  containers:
  - name: log-aggregator
    image: fluentd:latest
    volumeMounts:
    - name: shared-data
      mountPath: /var/log
      readOnly: true # this pod should only read the logs
      subPath: logs # Mount only the logs directory
  volumes:
  - name: shared-data
    persistentVolumeClaim:
      claimName: shared-data-pvc
```

In this final example, the `log-aggregator-pod` mounts the same `shared-data-pvc` but with a `subPath: logs`. Note here the use of `readonly: true`. This is an excellent practice to improve security and restrict access when possible. The log aggregation pod only needs to read from the folder, and we can ensure it can't accidentally or maliciously modify the data.

These examples demonstrate how you can use the same persistent volume to make different subdirectories available to distinct pods. This is a significant improvement over exposing the entire volume to all pods, because it enforces the principle of least privilege and maintains a much more manageable system. It’s important to ensure you have the correct access mode on the pvc. In the examples shown, `readwritemany` is used so all the pods can access the shared volume.

While I’ve shown basic pod configurations, this same technique translates directly to deployments, stateful sets, and other kubernetes workloads. The key is to focus on the `subPath` attribute within the `volumeMount` spec. Also make sure to think carefully about access modes and security best practices when defining your pvcs.

To solidify your understanding on kubernetes storage, I would highly recommend diving deep into the official kubernetes documentation on persistent volumes and claims. Also, for a more thorough treatment of storage in containerized environments, you should look into "Kubernetes in Action" by Marko Lukša, which covers this topic in considerable depth, as well as "Designing Data-Intensive Applications" by Martin Kleppmann. This book offers crucial principles applicable across different storage solutions. These resources offer a complete picture of the intricacies involved in designing resilient storage solutions in Kubernetes. Using persistent volumes and subpaths carefully provides the flexibility and control necessary to build robust and scalable applications.
