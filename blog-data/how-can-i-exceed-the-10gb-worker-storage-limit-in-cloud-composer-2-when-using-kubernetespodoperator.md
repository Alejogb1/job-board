---
title: "How can I exceed the 10GB worker storage limit in Cloud Composer 2 when using KubernetesPodOperator?"
date: "2024-12-23"
id: "how-can-i-exceed-the-10gb-worker-storage-limit-in-cloud-composer-2-when-using-kubernetespodoperator"
---

,  I’ve been down this road myself, more times than I care to remember, and it's a common pain point when scaling workflows on Cloud Composer. The 10GB limit on worker storage for the `KubernetesPodOperator` can feel restrictive, especially when dealing with larger datasets or complex processing pipelines. It's not a hard limit in the sense that there's no workaround; it's more about understanding how Kubernetes volumes and resource requests interact within the Cloud Composer ecosystem.

The core issue stems from how Cloud Composer’s underlying GKE cluster provisions worker nodes. Each worker gets a default persistent disk with 10GB of storage, and that's where your pod’s workspace is typically mounted. When your `KubernetesPodOperator` tasks exceed this limit, things can get messy – tasks failing, pods being evicted due to lack of disk space, and generally a frustrating experience. Here’s how I’ve addressed this in various projects, focusing on the core concepts of volume management and data handling:

The primary strategy revolves around leveraging alternative volume types that aren’t bound by the 10GB worker disk limitation. The two most effective approaches involve using persistent volume claims (PVCs) backed by network attached storage (like Persistent Disks) and utilising cloud storage directly with the appropriate mounting strategies. Both have their benefits and caveats, so let’s explore each.

**1. Persistent Volume Claims (PVCs) for Larger Persistent Storage**

This approach involves creating a PVC that's larger than 10GB and then mounting it to your `KubernetesPodOperator`. This method is good for scenarios where you need write access to a shared storage space and the data needs to persist across tasks or even workflows. It’s essentially extending the available storage space beyond the confines of the default worker’s local disk.

Here’s a snippet demonstrating how I've done this:

```python
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
from kubernetes.client import models as k8s

pvc_name = "my-large-volume-claim"
volume_mount = k8s.V1VolumeMount(
    mount_path="/data",
    name="my-volume"
)

volume = k8s.V1Volume(
    name="my-volume",
    persistent_volume_claim=k8s.V1PersistentVolumeClaimVolumeSource(claim_name=pvc_name)
)

create_pvc_task = KubernetesPodOperator(
    task_id="create_pvc_task",
    name="create-pvc",
    namespace="default",  # Adjust to your namespace
    image="busybox",
    cmds=["sh", "-c"],
    arguments=["""
        kubectl apply -f - <<EOF
        apiVersion: v1
        kind: PersistentVolumeClaim
        metadata:
          name: {}
        spec:
          accessModes:
            - ReadWriteOnce
          resources:
            requests:
              storage: 100Gi
        EOF
    """.format(pvc_name)],
    do_xcom_push=False
)


my_large_task = KubernetesPodOperator(
    task_id="my_large_task",
    name="my-large-task",
    namespace="default", # Adjust to your namespace
    image="my-data-processor-image", #Replace with your image
    cmds=["python", "my_processor.py"], #Replace with your command
    volumes=[volume],
    volume_mounts=[volume_mount],
    dag=dag,
)

create_pvc_task >> my_large_task
```

In this snippet, a dedicated PVC is created with a specified size (100Gi in this example) before the main task executes. Then, the task references this PVC. This allows the pod to access the larger storage mounted at `/data` within the container. It's crucial to define `accessModes` correctly to align with the planned data usage. `ReadWriteOnce` works for a single pod accessing a volume, while `ReadWriteMany` allows sharing amongst multiple pods simultaneously. Be cautious with the latter as it can introduce complexity regarding locking and data consistency.

**2. Direct Cloud Storage Integration (e.g., Google Cloud Storage)**

Another robust strategy, especially for handling very large datasets, is to directly utilize cloud storage solutions like Google Cloud Storage (GCS). Instead of mounting a persistent volume, the pod directly accesses the data from GCS. This approach scales well, avoids the constraints of managing persistent volumes, and simplifies data sharing across workflows. It's particularly helpful for data that's already stored in cloud object storage.

Here's a basic example of how to achieve this with GCS using a Python library:

```python
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
import kubernetes.client as k8s

gcs_task = KubernetesPodOperator(
    task_id="gcs_task",
    name="gcs-task",
    namespace="default", # Adjust to your namespace
    image="my-data-processor-image", #Replace with your image
    cmds=["python", "my_gcs_processor.py"], #Replace with your command
    env_vars=[
        k8s.V1EnvVar(name="GOOGLE_APPLICATION_CREDENTIALS", value="/gcp-key/key.json")
    ],
    volume_mounts=[
      k8s.V1VolumeMount(
        name="gcp-key",
        mount_path="/gcp-key/"
        )
      ],
    volumes=[
        k8s.V1Volume(
            name="gcp-key",
            secret=k8s.V1SecretVolumeSource(
                secret_name="gcp-key-secret" #Secret with the GCS Service Account
                )
            )
        ],
    dag=dag,
)
```

Now, `my_gcs_processor.py` would need to include logic to interact with GCS using a client library like the Google Cloud Client Library for Python. For example:

```python
from google.cloud import storage

def process_gcs_data():
    client = storage.Client()
    bucket = client.bucket("your-bucket-name")
    blob = bucket.blob("your-file.csv")

    # Download the data.
    data = blob.download_as_text()

    #Process Data
    # ...

    #Save new blob if needed.
    new_blob = bucket.blob("your_output_file.csv")
    new_blob.upload_from_string("output data here")


if __name__ == "__main__":
    process_gcs_data()
```

In this approach, the Kubernetes pod does not rely on mounted volumes for large data. It directly retrieves data from the specified GCS bucket. This strategy excels when datasets already reside in GCS, reducing overhead and enabling easy management of very large data volumes. This also keeps the Kubernetes Pods minimal and stateless, reducing the complexity and potential issues related to the cluster.

**3. Temporary Local Storage for Transient Data**

For scenarios where you need temporary space for data processing within a pod's lifetime, you could use the `emptyDir` volume type. This creates a temporary directory that's available to the pod during its execution. It’s cleared when the pod terminates and is suited for intermediate file storage or temporary outputs that aren't required to persist after the task. However, be mindful that `emptyDir` still utilizes the underlying worker node’s disk, which does come with limitations if used excessively. Ensure you’re managing the data lifecycle to prevent the pod’s directory from exceeding its limits.

```python
from airflow.providers.cncf.kubernetes.operators.kubernetes_pod import KubernetesPodOperator
import kubernetes.client as k8s

temp_data_task = KubernetesPodOperator(
    task_id="temp_data_task",
    name="temp-data-task",
    namespace="default", # Adjust to your namespace
    image="my-data-processor-image",  # Replace with your image
    cmds=["python", "my_temp_processor.py"],  # Replace with your command
    volume_mounts=[
      k8s.V1VolumeMount(
        name="temp-volume",
        mount_path="/temp/"
        )
    ],
    volumes=[
      k8s.V1Volume(
        name="temp-volume",
        empty_dir=k8s.V1EmptyDirVolumeSource()
        )
    ],
    dag=dag,
)
```

The `my_temp_processor.py` script would then use `/temp/` as a work directory for the task.

**Considerations and Best Practices:**

*   **Cost:** Using persistent storage will have ongoing storage costs, so keep those in mind, especially with data that needs to be actively written and retained. GCS, while cost effective, will still have egress charges and storage costs, so monitor it carefully.
*   **Performance:** For large datasets, direct cloud storage integration often performs better due to its optimized data retrieval. Network performance is key here.
*   **Security:** Ensure your Kubernetes pods have appropriate access rights when accessing cloud storage. Utilize service accounts and secrets correctly.
*   **Data Management:** Implement robust data handling within your scripts. If utilising GCS or similar object storage for temporary data, ensure any temporary files are cleaned up once the task is complete.
*   **Resource Requests and Limits:** Define reasonable resource requests and limits for your pods. While this directly addresses the storage constraint, it's a crucial practice for pod stability and resource consumption within Kubernetes.
*   **Documentation:** Familiarize yourself with the official Kubernetes documentation regarding volume types and their usage, it's invaluable: Kubernetes Volumes Documentation. Also, I'd advise looking at *Kubernetes in Action* by Marko Luksa, for a deep dive into Kubernetes architecture and resource management. For GCS specific operations, the Google Cloud Client Library documentation and their related books like *Programming Google Cloud Platform* by Rui Costa et al are a good start.
*   **Monitoring:** Regularly monitor your cluster and pod storage usage. Tools like Cloud Monitoring and Kubernetes dashboard are invaluable here.

In summary, surpassing the 10GB worker storage limitation for `KubernetesPodOperator` isn’t about brute force but rather understanding and utilising the flexibility of Kubernetes’ volume management. Choosing the appropriate solution depends entirely on your specific requirements for data persistence, size, and access patterns. It’s always a trade off, so careful consideration and testing are required.
