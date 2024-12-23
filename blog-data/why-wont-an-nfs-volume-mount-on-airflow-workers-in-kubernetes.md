---
title: "Why won't an NFS volume mount on Airflow workers in Kubernetes?"
date: "2024-12-23"
id: "why-wont-an-nfs-volume-mount-on-airflow-workers-in-kubernetes"
---

, let's dive into this. I've spent more hours than I care to remember troubleshooting NFS mounting issues in Kubernetes, specifically with Airflow workers. It’s a situation that seems straightforward on the surface, but the devils, as they often do, reside in the details. From my experience, a failed NFS mount on Airflow workers within a Kubernetes environment typically stems from a combination of networking configurations, permission challenges, and sometimes, a misunderstanding of how Kubernetes manages persistent volumes. Let's break it down systematically.

First, let’s address the networking aspect. Kubernetes workers, by default, are isolated within their network namespace. This isolation, while providing security benefits, also means that the pods might not have direct visibility to the NFS server if it's running on a different network segment or is not directly routable. I recall one particularly frustrating incident where the NFS server was hosted on-prem, and the Kubernetes cluster was running in a managed cloud environment. The firewall rules on the on-prem side were not configured to allow traffic from the pod’s IP range, leading to consistent “connection refused” errors. The solution involved meticulous review and adjustment of those rules to enable communication. In these cases, always double-check that the necessary routes and firewall rules are established for communication. Network policies in Kubernetes can also inadvertently block traffic. If you're using them, inspect their rules carefully. The `kubectl describe pod <pod_name>` command provides valuable information here, specifically looking at the events section to pinpoint network-related failures.

Then comes the permissions quandary. NFS relies heavily on user and group identifiers (UIDs/GIDs). Kubernetes pods, by default, run with a different UID/GID than what might be configured for the NFS share on the server. This can lead to the dreaded "permission denied" error, even if network connectivity is established. I encountered this often when workers started up and attempted to access a pre-existing NFS folder structure. The quickest fix was aligning the pod's UID/GID with that of the NFS share. This can be achieved by setting the `securityContext` within the pod's specification. It's generally good practice to create a dedicated user for this purpose on both the NFS server and the Kubernetes pods.

Consider this hypothetical scenario: you have an NFS share mounted at `/mnt/shared` on the server, and this share requires files to be owned by user `airflow_user` with UID 1000 and group `airflow_group` with GID 1000. Here is an example of a Kubernetes pod definition snippet illustrating the `securityContext`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: airflow-worker-example
spec:
  containers:
  - name: worker
    image: apache/airflow:latest
    volumeMounts:
    - name: shared-data
      mountPath: /opt/airflow/dags # example mount point
  volumes:
  - name: shared-data
    nfs:
      server: <nfs_server_ip>
      path: /mnt/shared
  securityContext:
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000

```

Here, the `securityContext` ensures the container processes execute with the specified UID and GID. The `fsGroup` option ensures that when the volume is mounted, it's done with the correct permissions. This is critical if you plan to write data back to the NFS volume. I recommend reading up on Kubernetes securityContexts and its granular control over pod permissions in the official Kubernetes documentation.

Moving along, another tricky aspect revolves around the way Kubernetes handles persistent volumes and persistent volume claims (PVCs). For NFS volumes, you typically define a persistent volume object that represents the actual NFS share and a PVC that represents the pod's request for that volume. The claim needs to bind successfully to the volume to ensure proper mounting. If there are issues with capacity or access modes configured on the Persistent Volume (PV), this binding can fail. A common mistake is when access modes are incorrectly configured: for instance, an NFS volume configured with `ReadWriteOnce` will not work if multiple workers try to claim it at the same time. Usually, `ReadWriteMany` is preferable for shared data.

Here's an example of a Persistent Volume (PV) definition, followed by a Persistent Volume Claim (PVC) definition for illustrative purposes:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: nfs-pv-example
spec:
  capacity:
    storage: 10Gi
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  nfs:
    server: <nfs_server_ip>
    path: /mnt/shared
```

And the corresponding PVC:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: nfs-pvc-example
spec:
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 10Gi
  selector:
    matchLabels:
      pvname: nfs-pv-example
```
In this example, the PVC attempts to claim the matching persistent volume based on the labels. If you find that the PVC stays in a `Pending` state, check the events section of the PVC via `kubectl describe pvc <pvc_name>`. This will show any errors or misconfigurations preventing binding. A detailed understanding of how Kubernetes volumes work can be obtained from "Kubernetes in Action" by Marko Luksa, an excellent resource for in-depth knowledge.

Finally, one last point which sometimes catches people out: if the NFS server has a custom export configuration (e.g., specific IP addresses allowed) or uses Kerberos, you'll need to ensure that these configurations are reflected in the network policies and the client's configuration (often this is implicit through the kubernetes network but sometimes needs attention). Also ensure that the necessary NFS client packages are installed within the worker image. Usually, distributions include `nfs-common`, but it is worth checking.

In my experience, the best approach to troubleshoot this is to go through this checklist: network connectivity, permission issues, proper Persistent Volume configuration, and proper access modes. By examining each aspect meticulously, you will likely pinpoint the exact issue. Remember to utilize Kubernetes troubleshooting tools like `kubectl describe` liberally – it's a life-saver in these situations. Furthermore, I highly recommend referring to "Linux Filesystems" by David Rusling for a detailed dive into file systems like NFS and their configurations. These experiences and resources have guided me through similar issues numerous times, and I hope they help you as well.
