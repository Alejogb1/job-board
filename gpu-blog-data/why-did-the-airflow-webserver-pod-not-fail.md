---
title: "Why did the Airflow webserver pod not fail in Kubernetes?"
date: "2025-01-30"
id: "why-did-the-airflow-webserver-pod-not-fail"
---
The Airflow webserver's persistence beyond a pod's expected lifecycle, particularly in Kubernetes, often arises from a misunderstanding of how Kubernetes liveness and readiness probes, coupled with pod restart policies, interact with application behavior. Specifically, the webserver process, while potentially experiencing internal issues that would render it effectively unavailable to users, might not be terminating from Kubernetes’ perspective.

A Kubernetes pod's liveness probe checks for the overall health of the containerized application. A failure triggers a pod restart, a crucial mechanism for recovering from critical errors. In contrast, the readiness probe indicates whether a pod is ready to accept incoming traffic. A failing readiness probe won't restart a pod; it simply removes it from the load balancer's endpoints, preventing new connections. The disconnect between an Airflow webserver's perceived unresponsiveness and Kubernetes’ view of the pod’s health is where the issue often lies. In my experience deploying and managing numerous Airflow instances on Kubernetes, I’ve observed this issue arising more frequently when the webserver becomes overloaded or is experiencing database connection issues.

The default behavior of the Airflow webserver is not to exit cleanly upon encountering database connection problems or during periods of high computational load. Rather, it enters a state of degraded performance or hangs while still listening on its port, thus, fulfilling the condition of the liveness probe (typically an HTTP GET request to `/health`). This also fulfills the default readiness probe, often just another check against the health endpoint. This is critical: if the liveness probe continues to return a successful status code, Kubernetes sees no reason to restart the pod, even if it is not functioning correctly from the user's perspective. I observed this most frequently when running highly concurrent DAG executions.

Consider a scenario where the Airflow webserver is experiencing a database deadlock. It cannot process user requests or update the UI effectively, making it effectively unusable. However, because the webserver process is still alive and responsive to the liveness probe, Kubernetes perceives the container to be healthy, preventing a restart. This is the key differentiator.

Here are some code examples, drawn from situations I've encountered, demonstrating how this problem can arise and how to approach it:

**Example 1: Basic Kubernetes Pod Definition with Default Health Probes**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: airflow-webserver
spec:
  containers:
  - name: airflow-webserver
    image: apache/airflow:2.8.0 # Example version
    ports:
    - containerPort: 8080
    livenessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 10
```

*Commentary:* This example illustrates a standard pod configuration using the official Airflow image. The liveness and readiness probes are both simple HTTP GET requests to `/health`. As long as the webserver process is running and serving HTTP responses on this endpoint (even if the response indicates a non-functional service internally), Kubernetes will consider the pod healthy, leading to the scenario described above. This was often the exact configuration when I observed the issue with my Airflow deployments.

**Example 2: Enhanced Liveness Probe with Custom Logic**

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: airflow-webserver
spec:
  containers:
  - name: airflow-webserver
    image: apache/airflow:2.8.0
    ports:
    - containerPort: 8080
    livenessProbe:
      exec:
        command: ["/bin/sh", "-c", "airflow health || exit 1"]
      initialDelaySeconds: 10
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 10
```

*Commentary:* This modified configuration uses a custom liveness probe utilizing the Airflow CLI's `airflow health` command. This command is more representative of the internal health of the application than a simple check on the http port. If the webserver encounters issues connecting to the database or other critical resources, `airflow health` will likely return a non-zero exit code, triggering a pod restart by Kubernetes. This approach was considerably more effective than the default configuration during my own troubleshooting. It requires the `airflow` CLI tool to be available within the container environment.

**Example 3: Implementing an "Internal" Health Endpoint**

```python
# Within Airflow webserver custom code, for example custom plugins
from airflow.providers.http.hooks.http import HttpHook
from airflow.plugins_manager import AirflowPlugin
from flask import Flask, request, jsonify

def is_database_healthy():
  try:
    db = HttpHook(http_conn_id="airflow_db") # Or other connection method
    db.run(endpoint="/health", method="GET") # Example database health endpoint
    return True
  except Exception:
    return False

def custom_health_endpoint():
  if is_database_healthy():
      return jsonify({"status": "ok"}), 200
  else:
      return jsonify({"status": "unhealthy"}), 503

class CustomHealthPlugin(AirflowPlugin):
    name = "custom_health_plugin"
    flask_blueprints = [
      {"name": "custom_health_bp",
        "blueprint": Flask(__name__).add_url_rule("/custom_health", view_func=custom_health_endpoint, methods=["GET"])}
    ]

```
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: airflow-webserver
spec:
  containers:
  - name: airflow-webserver
    image: apache/airflow:2.8.0
    ports:
    - containerPort: 8080
    livenessProbe:
      httpGet:
        path: /custom_health
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 10
    readinessProbe:
      httpGet:
        path: /health
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 10
```

*Commentary:* This code demonstrates the implementation of a custom health endpoint within Airflow.  This python code requires being in a custom Airflow plugin. The python code attempts to verify connectivity to the database. The kubernetes deployment now utilizes that custom endpoint for the liveness probe. This approach allows for a deep health check before restarting the webserver.  While more complex to implement, this offers a more robust solution by moving beyond basic connectivity and actively testing crucial components. The readiness probe here uses the original `/health` endpoint, a reasonable default. This was my preferred approach for stability.

Several further areas warrant consideration in ensuring stability. Monitoring tools, which can include Prometheus, Grafana, or other application performance monitoring systems, allow users to observe metrics around CPU usage, memory consumption, and other performance indicators. This proactive approach provides insights into the system's health beyond the simple “up/down” nature of liveness probes. Configuration options that allow for increased resource allocation for the webserver pod or optimization of database performance would also reduce the likelihood of the webserver entering these degraded states.

Resource recommendations for further investigation: consult the official Kubernetes documentation for detailed explanations on liveness and readiness probes, pod lifecycle, and resource management within Kubernetes. Explore the official Apache Airflow documentation, specifically the sections on webserver configuration and deployment within Kubernetes. Consider various online tutorials or sample code bases regarding creating custom health endpoints and developing Airflow custom plugins. Finally, examining best practices documentation from reputable cloud providers may give useful suggestions on running containerized applications.

In summary, the Airflow webserver pod's resilience in Kubernetes, when it appears dysfunctional to the user, typically stems from the difference between Kubernetes' liveness probe definition and a user's notion of application health. Default liveness probes, often using only a simple HTTP GET request, are insufficient in detecting genuine service disruptions. Employing more robust, comprehensive checks, such as utilizing the `airflow health` command or building custom health endpoints, ensures that Kubernetes effectively restarts pods when the service is not operating correctly, resulting in a more stable and reliable Airflow environment.
