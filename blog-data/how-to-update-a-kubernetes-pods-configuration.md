---
title: "How to update a Kubernetes pod's configuration?"
date: "2024-12-23"
id: "how-to-update-a-kubernetes-pods-configuration"
---

Alright, let's tackle this one. It's a common scenario, and I've seen more than my share of hiccups when dealing with Kubernetes pod updates. It's not as straightforward as, say, updating a configuration file on a single server, and it requires understanding the underlying mechanisms of Kubernetes. So, let me walk you through how I’ve approached this in the past, focusing on practical methods and avoiding common pitfalls.

First, it's crucial to understand that pods themselves are considered immutable. Directly modifying a running pod is generally discouraged, and Kubernetes is designed to operate by replacing pods rather than modifying them in place. Think of a pod as a static object rather than a dynamic one. It's defined by its manifest, and when you want to change something within that manifest, you're creating a new representation of that pod. This approach provides better consistency, reproducibility, and helps Kubernetes handle failures and rolling updates smoothly.

Therefore, when we talk about "updating a pod's configuration," what we’re *really* doing is updating the specification used to *create* those pods. The actual process involves interacting with a higher-level Kubernetes object—typically a deployment, statefulset, or daemonset—that manages your pods. These controllers are responsible for ensuring that the desired number of pods, configured as defined, are running at any given time. Any configuration change flows through these controllers down to the actual pods.

Let's say, for instance, that I needed to change the environment variables for a web application running in a deployment. This wasn't an isolated case either; during a production push, the team discovered that they'd misconfigured a critical database connection string. This required a rapid turnaround, and simply manually editing the pod config wasn't going to cut it. So, how did I handle it? Well, here's the general process and one method I've used with `kubectl apply`:

1.  **Modify the Deployment Manifest:** First, I would locate the deployment definition, typically a `.yaml` file. In that file, I'd identify the `spec.template.spec.containers` section and, within that, the container whose environment I needed to change.

2.  **Apply the Changes:** Once I modified the `.yaml` file, I'd use `kubectl apply -f <filename>.yaml` to submit the updated deployment configuration. This command sends the updated definition to the Kubernetes API server, triggering a rolling update. Kubernetes, upon detecting changes to the pod template within the deployment, would create new pods based on the updated specification. It would then gradually terminate the old pods, ensuring minimal downtime. This rolling update feature ensures that your service remains available during the update process.

Here is a simplified representation of what the manifest update might look like. Let’s assume I was working with a deployment named `my-web-app`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-web-app
  template:
    metadata:
      labels:
        app: my-web-app
    spec:
      containers:
      - name: my-container
        image: my-web-app-image:v1
        env:
        - name: DATABASE_URL
          value: "old_database_url" # I would edit this
        - name: API_KEY
          value: "old_api_key" # or this
```

After editing this file (let's assume I changed the `DATABASE_URL` value), the `kubectl apply` command would trigger the update. Crucially, I did not interact directly with individual pods.

Now, another common scenario I’ve experienced involves updating the resources allocated to a container within a pod. For example, an application might have initially been provisioned with inadequate memory and cpu, and performance issues might arise as load increases. Let’s say I had to resolve an issue that arose with memory constraints by modifying the resource limits and requests. Here’s how that update process went and the resulting manifest:

1.  **Locate the Deployment Manifest (Again):** Much like before, I'd locate the deployment yaml, and go to the same `spec.template.spec.containers` section.

2.  **Adjust the Resource Limits:** Within the container definition, I would then add or modify the `resources` section to define the required resources such as `requests` and `limits`.

3.  **Apply the Change:** Again, I’d use the `kubectl apply -f <filename>.yaml` command to push the updated configuration.

Here is an example of what the change looks like in the manifest:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-web-app
  template:
    metadata:
      labels:
        app: my-web-app
    spec:
      containers:
      - name: my-container
        image: my-web-app-image:v1
        resources:
          requests:
            memory: "256Mi" # I may edit this
            cpu: "250m" # Or this
          limits:
            memory: "512Mi" # Or this
            cpu: "500m" # Or this
```
Notice how the `resources` section is added. As before, I would not modify the pod configuration directly. The Kubernetes controllers handle the entire process of updating the deployment.

Finally, consider a scenario where I needed to update the image of the container. This is probably the most common type of change I have made over the years. For instance, during an upgrade cycle, I’d need to switch to a newer build of our application. This is done similarly to the previous example.

1.  **Locate the Deployment Manifest:** As always, the process starts with locating the correct deployment manifest file.

2.  **Modify the image tag:** Locate the container section under `spec.template.spec.containers` and modify the `image` tag value to point to the new build.

3.  **Apply the change:** Finally, the `kubectl apply -f <filename>.yaml` command is used to push the new configuration, triggering a rolling update that will result in new pods using the updated container image.

Here's what that looks like:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: my-web-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: my-web-app
  template:
    metadata:
      labels:
        app: my-web-app
    spec:
      containers:
      - name: my-container
        image: my-web-app-image:v2 # this line is the only thing changed
```

These examples demonstrate the standard method: modify the deployment, then let Kubernetes handle the rest.

A word of caution: be wary of using `kubectl edit` directly on a live deployment. While convenient for quick tweaks, it doesn’t track the history or versioning, making your changes difficult to manage or roll back later. Using `kubectl apply` from well-defined manifest files is generally the best practice. Version control your manifests; this is critical for tracking changes and rollback scenarios.

For further reading on this, I would suggest looking at:

*   **“Kubernetes in Action” by Marko Luksa**: This book gives a thorough overview of Kubernetes, including detailed sections on deployments and updates. It's very useful for understanding how the internals of Kubernetes operate.

*   **The official Kubernetes documentation**: The official documentation is comprehensive. Pay particular attention to the sections on deployments, stateful sets, and daemonsets. It's the best source of truth.

*   **“Designing Data-Intensive Applications” by Martin Kleppmann:** While not exclusively Kubernetes, this book offers foundational concepts about distributed systems, which are essential when working with Kubernetes.

Remember, the key to updating a Kubernetes pod’s configuration effectively is understanding the higher-level abstractions such as deployments and stateful sets. Working directly with pods is generally ill-advised. Treat them as ephemeral; update the manifest and let Kubernetes take care of the rest.
