---
title: "How to manage Airflow DAGs from local registry on Kubernetes?"
date: "2024-12-16"
id: "how-to-manage-airflow-dags-from-local-registry-on-kubernetes"
---

Let's tackle this. I recall a particularly gnarly project a few years back where we scaled our data pipelines tenfold, and orchestrating those with airflow on kubernetes, using a local dag registry, became an absolute necessity. It wasn’t all smooth sailing, but we hammered out a process that worked reliably. I'll walk you through that, focusing on the core concepts and techniques.

Managing airflow dags from a local registry on kubernetes introduces a few key challenges. First, you're dealing with the inherent dynamism of kubernetes deployments coupled with airflow's need to discover and parse dags, which are essentially python scripts. Second, keeping those dags consistent across different pods is crucial for preventing runtime errors. We don’t want different airflow workers seeing different versions of the same dag. The solution revolves around implementing a streamlined workflow for dag development, version control, and deployment.

The core principle here is separating your dag development from the airflow deployment itself. You’ll want to treat your dags as a software artifact, managed under version control, and deployed as part of your airflow application. This prevents the common pitfalls of modifying dag files directly on the airflow worker nodes.

Let's break it down into three essential steps, focusing on version control, packaging, and deployment mechanisms, each exemplified by some python-based code.

**Step 1: Version Controlling and Organizing your DAGs**

First and foremost, you need a robust version control system—git, in our case. Each dag should reside in a dedicated directory within a `dags` directory. This keeps things organized and allows you to clearly see the dependencies and versioning for individual dags. You should use an organized repository structure, such as:

```
project/
├── dags/
│   ├── dag_one/
│   │    ├── dag_one.py
│   │    ├── requirements.txt
│   │    └── utils.py
│   ├── dag_two/
│   │    ├── dag_two.py
│   │    └── config.yaml
│   └── ...
├── airflow.cfg
├── requirements.txt
└── dockerfile
```
Having a `requirements.txt` within each dag directory is crucial if specific packages are necessary for that particular dag. This enables dependency isolation. In your top level directory, `requirements.txt` would contain base airflow and its relevant providers. For example, `apache-airflow[cncf.kubernetes,postgres,google]`.

**Code Example 1: Example Directory Structure and Imports:**

```python
# project/dags/dag_one/dag_one.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from .utils import some_helper_function # relative import to avoid name collisions

def my_task():
    some_helper_function()
    print("This is task one")

with DAG(
    dag_id='example_dag_one',
    schedule_interval=None,
    start_date=datetime(2023, 1, 1),
    catchup=False
) as dag:
    task_one = PythonOperator(
        task_id='task_one',
        python_callable=my_task
    )
```
```python
#project/dags/dag_one/utils.py

def some_helper_function():
    print("Helper function called")
```
**Step 2: Packaging your DAGs**

Once the dag development and version control are done, we need a method to package the code, so it can be consistently deployed across the cluster. I found docker containers particularly useful for this. This involves creating a docker image that contains both the airflow configuration and your DAG definitions. This container becomes the unit of deployment in Kubernetes.

This docker image can be built with a dockerfile that copies all the necessary files and sets the correct airflow environment.

**Code Example 2: Example Dockerfile:**

```dockerfile
FROM apache/airflow:2.8.1-python3.10

# Install project dependencies
COPY requirements.txt /
RUN pip install -r /requirements.txt

# Copy all project files
COPY . /app

# Set Airflow home to avoid errors
ENV AIRFLOW_HOME /app

# Copy DAGs to the DAGs folder (ensure airflow configuration points to the right location)
COPY dags /app/dags
```
Remember, in your airflow.cfg, the `dags_folder` configuration setting should be relative to the `/app` directory, which becomes the root of the container. For example:
```ini
[core]
dags_folder = /app/dags
```
With this approach, the docker image encapsulates both the airflow environment and your DAGs, ensuring consistency across deployments.

**Step 3: Deploying to Kubernetes**

Finally, the deployment to Kubernetes is carried out by creating a deployment that uses the docker image we just created. A kubernetes `deployment.yaml` config file might look similar to this:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: airflow-deployment
  labels:
    app: airflow
spec:
  replicas: 3
  selector:
    matchLabels:
      app: airflow
  template:
    metadata:
      labels:
        app: airflow
    spec:
      containers:
      - name: airflow
        image: your-registry/airflow-image:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
        env:
          - name: AIRFLOW__CORE__SQL_ALCHEMY_CONN
            valueFrom:
              secretKeyRef:
                name: airflow-secret
                key: sql_alchemy_conn
          - name: AIRFLOW__CORE__EXECUTOR
            value: KubernetesExecutor
          - name: AIRFLOW__KUBERNETES__DAG_PROCESSOR_MANAGER_POD_TEMPLATE_FILE
            value: "/app/kubernetes/dag-processor-pod-template.yaml"
          - name: AIRFLOW__KUBERNETES__WORKER_POD_TEMPLATE_FILE
            value: "/app/kubernetes/worker-pod-template.yaml"
```
Note that this is a simplified example, you need to ensure configurations such as secrets management, resource requests, persistent volume claims for logs etc, are all handled correctly. Also ensure you have a kubernetes executor configured within the airflow.cfg for running tasks. This deployment will pull your container image and deploy it to your Kubernetes cluster.

The important thing here is that the docker image contains all the dag definitions within the `/app/dags` directory as configured in our dockerfile and the airflow configuration files within the image. These dag files will be parsed by the airflow scheduler during deployment and made available within the airflow UI.

**Code Example 3: Example Kubernetes Deployment YAML Fragment:**

```yaml
apiVersion: v1
kind: Service
metadata:
  name: airflow-webserver
  labels:
    app: airflow
spec:
  type: LoadBalancer
  ports:
    - port: 8080
      targetPort: 8080
  selector:
    app: airflow
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: airflow-deployment
  labels:
    app: airflow
spec:
  replicas: 3
  selector:
    matchLabels:
      app: airflow
  template:
    metadata:
      labels:
        app: airflow
    spec:
      containers:
      - name: airflow
        image: your-registry/airflow-image:latest
        imagePullPolicy: Always
        ports:
        - containerPort: 8080
        env:
          - name: AIRFLOW__CORE__SQL_ALCHEMY_CONN
            valueFrom:
              secretKeyRef:
                name: airflow-secret
                key: sql_alchemy_conn
          - name: AIRFLOW__CORE__EXECUTOR
            value: KubernetesExecutor
          - name: AIRFLOW__KUBERNETES__DAG_PROCESSOR_MANAGER_POD_TEMPLATE_FILE
            value: "/app/kubernetes/dag-processor-pod-template.yaml"
          - name: AIRFLOW__KUBERNETES__WORKER_POD_TEMPLATE_FILE
            value: "/app/kubernetes/worker-pod-template.yaml"
```
This configuration ensures each pod running the airflow scheduler and worker has an identical environment and, more importantly, the exact same dags.

**Additional Recommendations**

For a deeper dive, I'd strongly suggest exploring "Designing Data-Intensive Applications" by Martin Kleppmann. While not specifically about airflow, it lays a great foundation for distributed systems design, which is essential for understanding the underlying complexities here. Also, the official Apache Airflow documentation and the kubernetes documentation are vital resources. For managing CI/CD pipeline for your dags, look into some advanced CI/CD patterns using gitops and tools like ArgoCD or Flux.

The approach outlined provides a consistent and reliable method to manage your airflow dags, by treating them as deployable software artifacts, not only keeping your pipelines in check but also reducing headaches when debugging issues. While this setup took time to fine-tune, the benefits of improved scalability and maintainability significantly out-weighted the initial setup overhead. I found this approach provides a solid baseline for handling the dynamic environment of airflow running within kubernetes and will set you up for success.
