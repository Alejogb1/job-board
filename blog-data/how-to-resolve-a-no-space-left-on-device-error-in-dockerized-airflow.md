---
title: "How to resolve a 'no space left on device' error in Dockerized Airflow?"
date: "2024-12-23"
id: "how-to-resolve-a-no-space-left-on-device-error-in-dockerized-airflow"
---

,  I've definitely seen my share of “no space left on device” errors, especially with Dockerized Airflow deployments. It’s a classic scenario that tends to manifest unexpectedly, usually at the least convenient moment. The good news is, it’s almost always resolvable with a methodical approach. It typically isn't one single cause, rather an accumulation of factors that we need to methodically break down. Let me walk you through my troubleshooting process based on past experiences – it's a combination of Docker fundamentals, Airflow's peculiarities, and some real-world gotchas.

Firstly, understand the core issue isn't just about "no space." It's more precise to say, "a particular storage location managed by Docker has exhausted its allotted space," or that a disk in the container itself has run dry. Docker, as many are aware, utilizes a storage driver. This driver manages how image layers and container data are stored. When this storage fills up, operations begin to fail.

The "no space left on device" error, in the context of Dockerized Airflow, can emerge from a few primary sources: excessive log accumulation, bloated image layers, a runaway number of container instances, or a lack of disk space on the host machine dedicated to the docker daemon. Let's break these down.

**1. Container Logs:**

Airflow, by its nature, generates a significant amount of logs. Task execution, scheduler activity, webserver interactions – all of these produce logs. If these logs aren't actively managed (rotated or purged), they'll steadily consume space. In my experience, not setting appropriate log rotation policies was the single biggest contributor to this problem, particularly in long-running Airflow environments.

Here's how I'd approach log management, specifically within the docker context:

**Snippet 1: Setting Log Rotation within docker-compose.yml**

```yaml
version: "3.7"
services:
  airflow-webserver:
    image: apache/airflow:2.6.3-python3.10
    ports:
      - "8080:8080"
    volumes:
      - ./logs:/opt/airflow/logs
    environment:
       _AIRFLOW__LOGGING__LOG_FILENAME_TEMPLATE: "{{ ti.dag_id }}/{{ ti.task_id }}/{{ run_id }}/{{ try_number }}.log"
       _AIRFLOW__LOGGING__LOGGING_LEVEL: INFO
       _AIRFLOW__LOGGING__TASK_LOG_READER: airflow.providers.cncf.kubernetes.log.KubernetesLogReader
    command: webserver
  airflow-scheduler:
    image: apache/airflow:2.6.3-python3.10
    volumes:
      - ./logs:/opt/airflow/logs
    environment:
        _AIRFLOW__LOGGING__LOG_FILENAME_TEMPLATE: "{{ ti.dag_id }}/{{ ti.task_id }}/{{ run_id }}/{{ try_number }}.log"
        _AIRFLOW__LOGGING__LOGGING_LEVEL: INFO
        _AIRFLOW__LOGGING__TASK_LOG_READER: airflow.providers.cncf.kubernetes.log.KubernetesLogReader
    command: scheduler

  airflow-worker:
    image: apache/airflow:2.6.3-python3.10
    volumes:
     - ./logs:/opt/airflow/logs
    environment:
         _AIRFLOW__LOGGING__LOG_FILENAME_TEMPLATE: "{{ ti.dag_id }}/{{ ti.task_id }}/{{ run_id }}/{{ try_number }}.log"
         _AIRFLOW__LOGGING__LOGGING_LEVEL: INFO
         _AIRFLOW__LOGGING__TASK_LOG_READER: airflow.providers.cncf.kubernetes.log.KubernetesLogReader
    command: worker
```

In this example, I've ensured each service (webserver, scheduler, worker) maps a directory named "./logs" on the host to "/opt/airflow/logs" inside the container. Critically, also set the `_AIRFLOW__LOGGING__LOG_FILENAME_TEMPLATE` to organize logs by DAG, task, run, and try number. This facilitates easier log management and makes rotation strategies more feasible. Consider implementing `logrotate` on the host or other log-management tools to manage these files. It is essential to note that the above example does not include log rotation on the host and is merely a guide for the configuration inside of airflow.

**2. Docker Image Layers:**

Docker images are comprised of layers. These layers are cached to optimize build processes, but these cached layers can consume substantial space over time. Building an image with unnecessary packages or files bloats the layers, making each subsequent container larger. Periodic pruning of unused image layers, and sometimes entire images can significantly reclaim disk space.

**Snippet 2: Docker Pruning via Command-Line**

```bash
docker system prune -a
```

This single command will remove:

*   All stopped containers
*   All networks not used by at least one container
*   All dangling images
*   All build cache

While effective, use this command cautiously as it will permanently remove data. Specifically, consider the `-a` flag, which removes all unused images and not just dangling ones. It’s best practice to schedule this via cron or a similar mechanism outside of operational hours. It's also worth ensuring that you use `.dockerignore` files when building your images. This prevents unintended files from getting added to the build context and thus bloating the resulting image.

**3. Host Machine Disk Space and Docker Daemon Storage:**

Finally, and quite frankly one of the first things I check, is the host machine's disk space. Docker has its own storage area, commonly `/var/lib/docker`, and if that runs out, your containers will start exhibiting “no space” errors. It is essential to regularly check this directory. Using `df -h` command will give a quick overview. If it is close to full, consider moving this directory to a different partition with more space or resizing it (if available) if you have LVM setup. The process for this depends on the operating system and is not a trivial change in production systems, but must be considered. It is important to remember that any change to docker's storage directory will require downtime of all docker containers.

Another element often overlooked is the storage configuration of Docker itself. Docker uses a "storage driver" to manage how it stores container files and images. Depending on the driver (e.g., `overlay2`, `devicemapper`), the available space can be managed differently. Incorrectly configured drivers can lead to space issues even if it looks like you have enough disk space.

**Snippet 3: Docker Daemon Storage Driver Verification**

```bash
docker info | grep 'Storage Driver'
```

This command, while simple, will tell you what storage driver your docker daemon is currently using. Reviewing the documentation for your specific storage driver is key to understanding how Docker is managing storage. For example, the `overlay2` driver, now a common default, manages storage in layers. For older versions of docker, you might have been using `devicemapper`, and the configuration for that is quite different.

Beyond these points, here's some additional advice:

*   **Monitoring:** Implement robust disk space monitoring for your host machine and Docker storage. I've always relied on tools like Prometheus and Grafana to set up alerts for low disk space. It prevents surprises.
*   **Regular Maintenance:** Make docker pruning a regular part of your system maintenance. Automation is crucial.
*   **Image Optimization:** Pay attention to how you construct your Docker images, only add packages that are actually required.
*   **Configuration Management:** Use a configuration management tool to ensure consistent deployments. Avoid the risk of manually deployed images and potentially forgetting to add crucial configurations.
*   **Documentation:** Maintain thorough documentation for your Docker setup, including how you manage logs, images, and host storage. It is far easier to troubleshoot if documented well.

For further learning I recommend diving deep into the following materials:

*   **Docker documentation**: Docker’s official documentation is the best place to start. Focus on the sections about storage drivers, image layering, and general system maintenance.
*   **"The Docker Book" by James Turnbull**: A comprehensive introduction to Docker that covers topics from the basics to more complex setups.
*   **"Kubernetes in Action" by Marko Luksa**: If you plan to scale your Airflow deployment using Kubernetes, this is invaluable. It covers Docker concepts extensively within the Kubernetes ecosystem.
*   **"Site Reliability Engineering" by Betsy Beyer, Chris Jones, Jennifer Petoff and Niall Richard Murphy**: This book from Google provides invaluable insights into monitoring and maintaining large-scale systems, which is quite helpful for keeping a complex system like dockerized airflow running smoothly.
*   The documentation for your *specific storage driver*: Understanding how docker's storage drivers work is key to preventing issues down the road.

Dealing with "no space left on device" errors can be frustrating, but a systematic, knowledge-driven approach will almost certainly lead you to the root cause. It's a good opportunity to examine your setup and implement robust management. Remember: proactive maintenance is easier than reactive firefighting.
