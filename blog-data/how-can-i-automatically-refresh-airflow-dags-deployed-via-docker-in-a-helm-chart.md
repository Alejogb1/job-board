---
title: "How can I automatically refresh Airflow DAGs deployed via Docker in a Helm chart?"
date: "2024-12-23"
id: "how-can-i-automatically-refresh-airflow-dags-deployed-via-docker-in-a-helm-chart"
---

Let's dive straight into this; it's a problem I've certainly encountered myself several times, especially back when I was optimizing our data pipelines for peak performance. Automatically refreshing airflow dags deployed via docker in a helm chart can seem like a complex puzzle initially, but breaking it down reveals a few robust and reliable approaches. It's less about 'magic' and more about understanding the interplay between Kubernetes, Docker, Helm, and Airflow's internal mechanics.

The crux of the issue lies in the fact that when you deploy an Airflow instance via Helm, particularly using a Docker image, your DAG files are often baked into the image at build time. Subsequently, updating those DAGs requires a new image build and a re-deployment, which isn't ideal for fast-paced development cycles. What we need is a mechanism to propagate changes to DAG files into the running Airflow instance without rebuilding and redeploying the entire image every time. We need, in essence, to decouple DAG updates from image updates.

There are a few ways we can tackle this. Let's explore some of the methods I’ve employed with success.

**Option 1: Leveraging Persistent Volumes**

The most straightforward approach involves using a Persistent Volume (PV) to store DAG files. Instead of embedding the DAGs within the docker image, we mount a persistent volume into the docker container at a location where airflow expects them – generally, `/opt/airflow/dags`.

This approach enables us to update DAG files on the persistent volume directly, and Airflow will automatically detect these changes (after its `dag_dir_list_interval` passes, which is configurable). A big advantage here is that you only need to push updates to the volume, not rebuild the entire docker image.

Here's an example snippet of a helm values file illustrating this concept:

```yaml
# values.yaml
airflow:
  dags:
    persistence:
      enabled: true
      accessMode: ReadWriteMany
      size: 1Gi
      existingClaim: my-airflow-dags # optional; if you have an existing PV claim
      mountPath: /opt/airflow/dags # Make sure this matches your airflow settings
```

And here’s the corresponding mounting logic in our Helm template snippet (simplified for clarity):

```yaml
# templates/deployment.yaml
spec:
  template:
    spec:
      containers:
        - name: airflow-worker
          volumeMounts:
          - name: dags-volume
            mountPath: /opt/airflow/dags # Make sure this matches your airflow settings
      volumes:
        {{- if .Values.airflow.dags.persistence.enabled }}
        - name: dags-volume
          persistentVolumeClaim:
            {{- if .Values.airflow.dags.persistence.existingClaim }}
            claimName: {{ .Values.airflow.dags.persistence.existingClaim }}
            {{- else }}
            claimName: {{ .Release.Name }}-dags-pv-claim
            {{- end }}
        {{- end }}
```

This configuration, when deployed, will mount a Persistent Volume Claim (or an existing one if specified) to the Airflow worker container at the `/opt/airflow/dags` path. To update DAGs, you would then interact directly with the storage backing the PVC (e.g., by using `kubectl cp` to copy new DAG files into the volume) or employ a Continuous Integration / Continuous Delivery pipeline.

**Option 2: Git Sync Sidecar Container**

Another robust method involves deploying a sidecar container within the same pod that synchronizes your DAGs from a git repository (or other source) to the `/opt/airflow/dags` directory. The `git-sync` project or similar tools are great for this. This approach offers the benefit of version control and allows your DAGs to evolve naturally through a standard git workflow.

Here is a snippet of the values.yaml depicting this arrangement:

```yaml
# values.yaml
airflow:
  dags:
    gitSync:
      enabled: true
      repo: "git@your-repo-url.git"
      branch: "main"
      rev: "HEAD" # Commit SHA or tag if needed
      dest: /opt/airflow/dags
      period: 120 # Sync every 120 seconds

```

Here is how we would incorporate this sidecar into our helm deployment:

```yaml
# templates/deployment.yaml
spec:
  template:
    spec:
      containers:
        - name: airflow-worker
          # ... other airflow config
          volumeMounts:
            - name: dags-volume
              mountPath: /opt/airflow/dags

        - name: git-sync
          image: k8s.gcr.io/git-sync:v3.6.3 # or similar git-sync image
          args:
            - --repo={{ .Values.airflow.dags.gitSync.repo }}
            - --branch={{ .Values.airflow.dags.gitSync.branch }}
            - --rev={{ .Values.airflow.dags.gitSync.rev }}
            - --dest={{ .Values.airflow.dags.gitSync.dest }}
            - --period={{ .Values.airflow.dags.gitSync.period }}
          volumeMounts:
          - name: dags-volume
            mountPath: /opt/airflow/dags

      volumes:
        - name: dags-volume
          emptyDir: {}
```

In this example, we create an emptyDir volume which is shared between the main Airflow worker container and the `git-sync` sidecar. The sidecar container clones the repository at specified intervals and ensures the dag folder is kept in sync. Airflow will automatically load the updated DAGs once the sync is complete.

**Option 3: External DAG Storage (e.g., Cloud Storage)**

For more complex setups or when dealing with very large DAG sets, relying on an external storage mechanism such as AWS S3, Google Cloud Storage, or Azure Blob Storage becomes attractive. Airflow provides mechanisms to directly load DAG files from these cloud storage options. This completely decouples DAG management from the underlying deployment, allowing for a highly flexible and scalable setup.

Here’s an example of how to configure Airflow to read DAGs from Google Cloud Storage, assuming you've already set up appropriate permissions and connectivity:

First, configure the following settings in your Airflow configuration, typically via environment variables in your helm chart (values.yaml):

```yaml
# values.yaml
airflow:
  config:
    AIRFLOW__CORE__DAGS_FOLDER: "gs://your-gcs-bucket/dags"
    AIRFLOW__CORE__LOAD_EXAMPLES: "False" # Disable examples
    # Additional configurations for GCS connectivity, if required.
```

Then, this change is directly implemented using helm by setting the corresponding environment variables for the airflow container:

```yaml
# templates/deployment.yaml
spec:
  template:
    spec:
      containers:
        - name: airflow-worker
          env:
            - name: AIRFLOW__CORE__DAGS_FOLDER
              value: "{{ .Values.airflow.config.AIRFLOW__CORE__DAGS_FOLDER }}"
            - name: AIRFLOW__CORE__LOAD_EXAMPLES
              value: "{{ .Values.airflow.config.AIRFLOW__CORE__LOAD_EXAMPLES }}"
            # ... Other env vars
```

With this setup, your airflow instance will directly download DAG files from the GCS path you specified, automatically updating as changes are made to the bucket.

In my experience, option one (Persistent Volumes) often suits smaller teams and initial setups. Option two (Git Sync) proves more robust for collaboration and version control, while option three (Cloud Storage) becomes essential for highly scalable environments or cases where teams might want to manage DAGs separately from the Airflow deployment.

It is also critical to consider the following:

*   **Permissions:** Ensure the Airflow instance (and any sidecar containers) has the necessary permissions to access the storage mechanism you choose. This might involve Kubernetes service accounts or cloud-specific IAM roles.
*   **Performance:** Be mindful of the latency involved in accessing external storage. For instance, syncing DAGs from a remote Git repository or cloud storage can introduce delays. Adjust sync intervals accordingly.
*   **Security:** Store credentials to private git repositories or cloud providers securely, using kubernetes secrets or other secret management solutions.

For further reading, I highly recommend referring to Kubernetes documentation regarding Persistent Volumes and sidecar containers. Also, check the official Airflow documentation regarding `DAGs_FOLDER` configuration options, external storage (like S3, Google Cloud Storage or Azure Blob Storage) and integration with git. The *Kubernetes in Action* book by Marko Lukša is an excellent resource, as is the official Apache Airflow documentation.

Choosing the right strategy depends on the complexity of your infrastructure and the needs of your team. However, by understanding the core mechanics involved, automating DAG updates within a Docker-based Airflow deployment becomes a manageable and efficient process. I hope this helps you in your journey, and remember that no single approach is universally perfect; it's all about finding what best fits your specific context.
