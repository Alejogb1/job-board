---
title: "Can two Airflow servers be integrated?"
date: "2024-12-23"
id: "can-two-airflow-servers-be-integrated"
---

Okay, let’s tackle this. It's a scenario I’ve encountered firsthand more times than I’d care to count, usually in contexts where scaling out or ensuring high availability became critical. The short answer, of course, is yes, two Airflow servers can be integrated. But it's not as straightforward as just plugging them into the same network. The process involves understanding the underlying architecture of Airflow and choosing the appropriate strategy to achieve a cohesive, functional system. Instead of envisioning two entirely separate, competing instances, you're more accurately building a distributed Airflow setup.

My experience with this stems back to a large-scale data migration project several years ago. We initially started with a single Airflow instance, which quickly became a bottleneck. We were processing huge volumes of data daily, and the single scheduler, even with ample resources, was frequently overwhelmed. This prompted a deep dive into distributing the workload, and that’s when we implemented our multi-server Airflow cluster. It wasn't perfect initially, but we learned valuable lessons that I can share.

First, let's be precise about what "integration" means. We're not simply aiming for two instances that are aware of each other, but rather a system where tasks from multiple schedulers and worker nodes are executed reliably as part of a single, unified DAG (Directed Acyclic Graph). This usually boils down to sharing the same metadata database, and using a scalable execution layer to distribute the workload. The fundamental architecture comprises the webserver(s) for visualization, the scheduler(s) for managing tasks, the executor(s) to handle task execution, and of course, the metadata database.

So, the core of integrating two or more Airflow servers involves several key aspects:

1.  **Shared Metadata Database:** Both (or all) Airflow instances must point to the same database. This is absolutely crucial. The database (typically PostgreSQL or MySQL) stores all DAG definitions, task metadata, user permissions, etc. Without a shared database, each Airflow instance would operate in isolation, and the integrated view of your workflows would be fractured and unreliable. Having tried this the hard way, I can't emphasize enough how critical a single metadata store is.

2.  **Executor Selection:** The default `SequentialExecutor` is unsuitable for multi-server setups because it runs tasks locally where the scheduler resides. You need a distributed executor like `CeleryExecutor` or `KubernetesExecutor`. `CeleryExecutor` relies on a message broker like Redis or RabbitMQ to distribute tasks, while `KubernetesExecutor` leverages a Kubernetes cluster for task execution. My team ended up using `CeleryExecutor` because we had an existing message broker infrastructure in place, but the choice really depends on the specific environment and infrastructure. For environments already leveraging containers, `KubernetesExecutor` often simplifies the setup.

3.  **Load Balancing:** If you intend to use multiple schedulers (which is common for high availability), you will require a load balancer in front of your webservers, since only one webserver can be active at any given time. Also, for `CeleryExecutor`, if you have multiple Celery workers, they need to be distributed to provide efficient task processing, which may also be achieved through load balancing at your infrastructure level.

Let me illustrate these concepts with some configuration examples. Note that these aren't complete configurations but rather snippets to showcase key integration points.

**Example 1: Configuring a shared metadata database (postgresql):**

```python
# airflow.cfg
sql_alchemy_conn = postgresql://airflow_user:airflow_password@database_host:5432/airflow_db
```

This setting, residing within your `airflow.cfg` file on *both* Airflow servers, instructs them to use the same PostgreSQL database for storing their metadata. Obviously, you need to replace the placeholders with your specific database credentials. It's a simple configuration change, but one that is central to ensuring both instances are synchronized. Remember that the underlying database needs to be robust and configured for high availability as well. It’s a single point of failure if not handled correctly.

**Example 2: Using CeleryExecutor:**

```python
# airflow.cfg

executor = CeleryExecutor

broker_url = redis://redis_host:6379/0

result_backend = redis://redis_host:6379/1
```

Here, we specify that all the Airflow servers will use `CeleryExecutor`, and we provide the address of our Redis broker to the scheduler(s) and worker(s) that are part of the deployment. Both the broker URL and the results backend are pointed to a single shared redis service, such that any scheduler instance and any celery worker instance can coordinate with each other effectively. Remember that the `airflow.cfg` on each machine should be exactly the same unless there is a very specific reason to keep them different.

**Example 3: Configuring the KubernetesExecutor:**

```python
# airflow.cfg

executor = KubernetesExecutor

kubernetes_namespace = airflow

# Add any necessary kubernetes client configuration below this comment. For example:
# kubernetes_in_cluster = False
# kubernetes_config_path = /path/to/kubeconfig
# kubernetes_worker_pod_template_file = /path/to/pod-template.yaml

```

When you're running your airflow setup within a kubernetes environment, and are going to be distributing the workloads on kubernetes using pods, you can utilize the `KubernetesExecutor`. In this case, every task is executed by submitting a pod to the kubernetes cluster. The `kubernetes_namespace` setting is essential to keep everything well-organized. It's not necessary to use the `KubernetesExecutor` when running in kubernetes; however it makes it very easy to manage distributed tasks.

Beyond configuration, careful planning of your environment is key. Considerations around resource allocation, monitoring, and log management are critical. For scaling up, you'll probably be looking at options like setting up multiple worker nodes for `CeleryExecutor` or adjusting resource configurations for Kubernetes pods if you use the `KubernetesExecutor`. Monitoring your Celery workers or your kubernetes pods will be essential to make sure your tasks are being completed successfully and are scaling correctly. For monitoring Celery, you can look into tools such as `flower`.

Regarding further learning, I'd recommend several sources. "Data Pipelines with Apache Airflow" by Bas Harenslak and Julian Rutger is an excellent practical guide that dives deep into operational aspects, including distributed setups. Additionally, the official Airflow documentation (available on the Apache Airflow project website) is invaluable for its detailed explanations and configuration examples. Specifically, look into the sections on 'Executors' and 'High Availability.' Also, don't shy away from reading the documentation for your chosen executor, either Celery or Kubernetes, since their respective configuration parameters will heavily influence the reliability and efficiency of the overall setup. Finally, the Kubernetes documentation is a necessity if you are running your airflow setup using kubernetes and intend to use the `KubernetesExecutor`.

In conclusion, integrating multiple Airflow servers is not a simple point-and-click affair, but rather a process that involves careful planning, proper configuration, and a solid understanding of Airflow's architecture. It's a necessary step for scaling out workflows and building a robust, highly available platform. It requires that all schedulers and worker nodes are coordinated through a shared metadata store, and a distributed execution layer using tools such as Celery or Kubernetes. Through these means, two or more Airflow servers can effectively function as a single distributed system. My past experiences have shown me that diligent planning and careful implementation, guided by best practices and solid documentation, will result in a reliable and efficient workflow management platform that is able to handle very large-scale tasks.
