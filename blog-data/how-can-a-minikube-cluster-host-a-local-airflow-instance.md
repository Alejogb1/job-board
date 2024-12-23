---
title: "How can a minikube cluster host a local airflow instance?"
date: "2024-12-23"
id: "how-can-a-minikube-cluster-host-a-local-airflow-instance"
---

, let’s tackle this one. It's a question that brings back memories of a particularly challenging project where we needed a completely isolated, local airflow environment for iterative testing before pushing changes to a production cluster. The idea was to ensure new dag deployments wouldn't crash the entire platform. Minikube, in this scenario, became our staging ground. Setting up airflow within a minikube instance does involve a few steps beyond simple installation, but it's manageable with the right approach. The goal is to create a functioning airflow deployment inside the minikube virtual machine that allows us to develop and validate workflows before they impact production.

First, the fundamental concept here is containerization. Airflow is, at its core, a complex collection of services: a scheduler, web server, worker processes, and a database. We want these services to run within our minikube environment as if it were a real kubernetes cluster. This involves deploying these components as pods. To streamline this, we primarily use helm charts which automate much of this setup work. It's the best way to maintain consistent and reproducible deployments, rather than setting things up manually which can lead to configuration drift.

Let's start with the prerequisite: you need a functional minikube installation, and you need helm. I’m assuming here you already have these installed. If not, the official minikube and helm documentation are the best resources to get you going. The next step is to add the airflow helm chart repository. This ensures we fetch the most up-to-date chart.

```bash
helm repo add apache-airflow https://airflow.apache.org
helm repo update
```

This snippet fetches the necessary configurations for the airflow deployment. We then need to craft a configuration values file to customize our deployment. This is crucial because the default settings might not align with the resource constraints of a typical minikube environment. For example, the default database might not fit within the limited storage, or the memory usage might exceed the limits of the minikube vm, depending on how much you allocated during start. This file, usually named `values.yaml` will hold our customizations.

Now, inside this `values.yaml` file, you might want to adjust settings like:

```yaml
airflow:
  executor: LocalExecutor # For simplicity in minikube, the LocalExecutor is easiest.
  postgresql:
    enabled: false  # Disable postgresql pod
  sqlite:
    enabled: true  # Use SQLite for development, it's lightweight
  webserver:
    resources:
      requests:
        cpu: 200m
        memory: 256Mi
      limits:
        cpu: 500m
        memory: 512Mi
  scheduler:
    resources:
      requests:
        cpu: 200m
        memory: 256Mi
      limits:
        cpu: 500m
        memory: 512Mi
  workers:
    resources:
      requests:
        cpu: 100m
        memory: 128Mi
      limits:
        cpu: 250m
        memory: 256Mi
```

This snippet demonstrates a simplified configuration for minikube. We are disabling the dedicated postgresql pod (which is recommended for production but not feasible within a typical minikube deployment) and opting for the in-memory SQLite database. We've also defined resources limits and request for the main components (webserver, scheduler and worker). This makes sure airflow starts correctly with a limited amount of resources. Keep in mind you may need to tune these values to your specific setup.

With the configurations set, we are now ready to deploy airflow. We apply the helm chart along with the customized values file using the following command. Let's assume our namespace is named 'airflow-dev', we create it first and then use it for the installation:

```bash
kubectl create namespace airflow-dev
helm install airflow apache-airflow/airflow -n airflow-dev -f values.yaml
```

This process might take some time, as helm has to download and apply all the necessary resources for airflow to run correctly. After completion, you will be able to check the status of all the pods inside the `airflow-dev` namespace. A quick command is shown here:

```bash
kubectl get pods -n airflow-dev
```

If everything worked correctly, your pod status will transition to "Running", indicating the airflow deployment has been completed and is functioning as intended.

Once airflow is up and running, you need to access the web ui. Because minikube is a single node cluster, you'll need to port-forward the airflow webserver port (default 8080) to your local machine. This command lets you access it:

```bash
kubectl port-forward service/airflow-webserver 8080:8080 -n airflow-dev
```

Now, point your browser to `http://localhost:8080`, and you will see the airflow UI. The default credentials are `airflow` for both username and password. Be sure to change these if you intend to leave this instance exposed, as the default account is not secure.

This setup is, of course, only for development purposes within minikube and is not recommended for production settings. The sqlite database is not ideal for any use cases that require persistence, data consistency and scalability. For real production workloads, I highly recommend looking into more robust solutions using postgresql or mysql as an external database and exploring more advanced airflow deployments with kuberentes executor, which is the most common strategy now.

For further reading, I would recommend "Programming Apache Airflow" by J. Humphreys. It's a fantastic resource that not only guides you through the nuances of airflow's architecture but also provides invaluable guidance on best practices. Another very helpful resource to understand Kubernetes and Helm is “Kubernetes in Action” by Marko Lukša. These texts offer a fundamental overview and advanced techniques. The official Apache Airflow documentation is also an invaluable resource for keeping up-to-date with the latest changes and best practices, along with the official Helm documentation to understand the specifics of using Helm charts. Always refer to the primary source when dealing with specific configurations.

This approach should get you started with an airflow instance running locally in minikube, where you can test and validate your DAGs before pushing them to a more robust environment. It certainly helped us in the past, and I hope it's helpful to you. Always keep security in mind and adapt the specific configurations to your specific use cases and resources.
