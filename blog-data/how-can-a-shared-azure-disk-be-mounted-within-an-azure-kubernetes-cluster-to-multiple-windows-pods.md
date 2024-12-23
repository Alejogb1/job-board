---
title: "How can a shared Azure disk be mounted within an Azure Kubernetes cluster to multiple Windows Pods?"
date: "2024-12-23"
id: "how-can-a-shared-azure-disk-be-mounted-within-an-azure-kubernetes-cluster-to-multiple-windows-pods"
---

Alright, let's tackle this. It's a scenario I've seen crop up quite a few times, usually when dealing with legacy applications that haven't quite embraced the stateless container paradigm fully. Mounting a shared Azure disk to multiple Windows pods within an Azure Kubernetes Service (AKS) cluster requires a bit more finesse than the standard volume claim for a single pod, but it's definitely achievable. I’ll walk you through it based on some of my experiences, including a particularly stubborn legacy reporting system we had to containerize a few years back.

The core challenge here revolves around the fundamental nature of persistent volumes and how Kubernetes handles them. Standard persistent volume claims are designed for exclusive access, preventing data corruption that could arise from concurrent writes. With a shared disk, we're deliberately breaking that exclusivity, and therefore, we need to understand the underlying mechanisms and the nuances of how Azure disks are handled by Kubernetes. This isn't something you should go into blindly; proper planning and understanding are crucial to avoid data inconsistencies.

First, let’s acknowledge that Azure Disks, in their vanilla configuration, are not designed for simultaneous mounting by multiple VMs/pods within Kubernetes. If you tried it directly, Kubernetes would throw errors. However, Azure *does* offer a feature called *shared disks* that allow multiple virtual machines (and therefore, pods in our context) to access a single managed disk. This is primarily achieved using SCSI Persistent Reservations. It is this feature we'll leverage.

Now, the practical steps involve a few critical stages. First, you must create an azure managed disk with the `maxShares` setting greater than 1. That tells azure it should be prepared for shared access. Next we define an azure storage class and use that storage class to define a persistent volume and persistent volume claim. Finally, we'll ensure our pods are set to mount using the shared access mode. Here's a step by step approach to making that happen:

**1. Prepare the Azure Managed Disk**

This is arguably the most crucial step. You need to specify that the disk will be used for sharing. When creating the disk through the Azure portal, Azure cli, or terraform, set the `maxShares` parameter. I typically suggest `maxShares` greater than 1; the exact number will be driven by the expected level of concurrency and your performance needs. Keep in mind that a higher `maxShares` will likely impact I/O performance. I also suggest using premium SSDs to mitigate performance problems. For the rest of the steps, I will assume that you already have an existing managed disk, and will only focus on mounting it on multiple pods, because it's rare that the disk is being created at the same time as the pod.

**2. Configure Kubernetes Storage Class**

Next, define a storage class in your Kubernetes cluster to handle the shared Azure disk. A storage class in Kubernetes abstracts the underlying storage details and dictates how persistent volumes are provisioned. Here is an example, this would be in a `storage-class.yaml` file:

```yaml
apiVersion: storage.k8s.io/v1
kind: StorageClass
metadata:
  name: azure-shared-disk
provisioner: disk.csi.azure.com
parameters:
  skuName: "Premium_LRS" # Or Standard_LRS if needed
reclaimPolicy: Delete # Or Retain if needed
allowVolumeExpansion: true
```

This configuration defines a storage class named `azure-shared-disk` that leverages the Azure disk csi driver, specifies a premium tier storage sku, and enables expansion. I generally use "delete" in development environments but will switch this to `Retain` for any production settings. Also, I have the `allowVolumeExpansion` set to true in case more space is needed later.

**3. Define Persistent Volume (PV)**

Now, you'll create a Persistent Volume (PV) that points to your shared Azure disk and utilizes the storage class. Here's an example, which I'd put in a `persistent-volume.yaml` file:

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: shared-disk-pv
spec:
  storageClassName: azure-shared-disk
  capacity:
    storage: 1024Gi # Replace with your desired storage
  accessModes:
    - ReadWriteMany
  persistentVolumeReclaimPolicy: Retain
  csi:
    driver: disk.csi.azure.com
    readOnly: false
    volumeHandle: "/subscriptions/<your-subscription-id>/resourceGroups/<your-resource-group>/providers/Microsoft.Compute/disks/<your-disk-name>"
    volumeAttributes:
        fsType: ntfs
```

In this PV, crucial elements are `storageClassName: azure-shared-disk` linking the PV to the storage class, the `accessModes: [ReadWriteMany]` for shared access, and the `volumeHandle` which must be set with the fully qualified azure resource id for the disk being mounted. Note that `readOnly` is set to `false`, and `fsType` is set to `ntfs`. We need the filesystem to be `ntfs` since we are mounting this to a windows node.

**4. Define Persistent Volume Claim (PVC)**

Next, create a Persistent Volume Claim (PVC) that requests the PV, which you’ll use to mount it into your Pods. This would be in `persistent-volume-claim.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: shared-disk-pvc
spec:
  storageClassName: azure-shared-disk
  accessModes:
    - ReadWriteMany
  resources:
    requests:
      storage: 1024Gi # should match the PV storage value
```

Here, the PVC also specifies `accessModes: [ReadWriteMany]`, indicating that multiple pods can claim this volume for simultaneous read and write operations. The storage request should match your persistent volume configuration. The `storageClassName` will match that of the `PV`.

**5. Configure Windows Pods**

Finally, you modify your Windows pod definitions to mount the shared volume using the PVC. Here's an example of how you would mount it to a Windows deployment:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: windows-app-deployment
  labels:
    app: windows-app
spec:
  replicas: 2
  selector:
    matchLabels:
      app: windows-app
  template:
    metadata:
      labels:
        app: windows-app
    spec:
      nodeSelector:
        "kubernetes.io/os": windows
      containers:
      - name: windows-container
        image: mcr.microsoft.com/windows/servercore/iis:windowsservercore-ltsc2019
        volumeMounts:
        - name: shared-volume
          mountPath: "d:\\shared-data" # Mount path inside windows pod
      volumes:
        - name: shared-volume
          persistentVolumeClaim:
            claimName: shared-disk-pvc
```

In this deployment manifest, the `volumeMounts` section dictates where the volume will be mounted within the pod's file system, and the `volumes` section references the PVC created in the prior step. The `nodeSelector` ensures that this deployment only runs on Windows nodes in your cluster. The important part here is that multiple pods created through this deployment will now share the same azure managed disk at the specified mount point within the container.

**Important Considerations:**

*   **Data Consistency:** When multiple applications write to the same location, be aware of potential data corruption. Implement proper locking mechanisms within your application to avoid overwriting data. It's absolutely crucial to have your application built to handle concurrent access. If your application was not developed with this in mind, it may not work. In my experience, this is one of the biggest challenges and often requires more work than initially anticipated.
*   **Performance:** Be mindful that sharing the disk increases I/O demand on a single disk. Performance will often become a bottleneck if the underlying storage is not adequate for the workload. Monitor I/O and consider scaling up the underlying disk size or choosing a higher tier sku if needed.
*   **Monitoring:** Set up monitoring to observe disk I/O, performance, and any potential issues.
*   **Disk Size:** Consider the storage requirements, and ensure the initial size is large enough to handle the workload. With the current versions of the Azure disk csi driver, volume expansion is supported.
*   **Quorum:** When running multiple windows nodes, there is a possible risk for node failure and a resulting quorum loss that results in the disk being unmounted. It's imperative that you plan for this eventuality and create a strategy for your application to handle such failures.

For deep dives into this, I’d recommend referencing the official Azure documentation for managed disks and the kubernetes documentation on persistent volumes. Specifically, the Azure documentation on shared disks, found under the "Azure Managed Disks" section, will provide detail around usage and implementation. For Kubernetes, the key resources are the sections on Persistent Volumes and Storage Classes. The Kubernetes CSI documentation will also help illuminate the drivers themselves. A good book on Kubernetes concepts will also help you better understand the underlying mechanisms being used to implement this scenario. I've found that O'Reilly's "Kubernetes in Action" offers a good foundation in Kubernetes architecture for most use cases.

This setup, while functional, requires careful planning and consideration of the underlying limitations of shared storage. It is rarely the ideal long-term pattern for applications and should be used with caution. However, it is a common scenario when dealing with legacy applications. Understanding the trade-offs is important to a successful implementation. Hopefully, this provides a solid starting point for your implementation.
