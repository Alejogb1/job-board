---
title: "How do I configure Promtail for Docker Swarm?"
date: "2025-01-30"
id: "how-do-i-configure-promtail-for-docker-swarm"
---
Docker Swarm, while offering orchestration benefits, introduces complexities when deploying centralized logging solutions like Promtail. Specifically, the dynamic nature of task placement and the lack of inherent node awareness within a Swarm context necessitate a specific configuration approach for Promtail to effectively scrape logs. I've personally encountered this challenge across multiple deployments and found that a successful setup hinges on understanding Promtail's `clients` configuration and how it interacts with Docker Swarm's service discovery.

The fundamental issue is that Promtail, by default, assumes static targets for log scraping. In a Swarm environment, services can move between nodes, their container IDs change, and their exposed ports may vary. Therefore, a static configuration pointing directly to container IDs or IP addresses becomes brittle and quickly obsolete. The solution lies in leveraging Promtail's ability to discover targets dynamically through various service discovery mechanisms, primarily by utilizing Docker Swarm’s labels and service metadata. This involves configuring Promtail to look for specific labels assigned to Docker Swarm services, extracting relevant information, and then constructing the appropriate log paths for scraping.

To implement this effectively, the Promtail configuration file needs to be tailored to target the correct log files within the containerized environment. Docker Swarm maintains a consistent directory structure, allowing Promtail to find logs irrespective of where the container resides within the cluster. Log files are typically located within the `/var/lib/docker/containers/<container ID>` directory. However, rather than manually configuring paths based on specific container IDs which are impractical in a dynamic environment, we use labels to identify containers to be scraped. Promtail uses these labels to dynamically generate paths.

Consider a scenario where you have a service named `web-service` deployed using Docker Swarm. This service might have multiple replicas. You could label this service with `promtail.scrape=true` and `promtail.job=web-logs`. The Promtail configuration file should then include a scrape configuration using the docker_sd discovery. Let me demonstrate with examples.

**Example 1: Basic Promtail Configuration for Docker Swarm**

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: docker-swarm-logs
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_label_promtail_scrape']
        regex: 'true'
        action: keep
      - source_labels: ['__meta_docker_container_id']
        target_label: '__path__'
        replacement: '/var/lib/docker/containers/$1/*-json.log'
      - source_labels: ['__meta_docker_container_label_promtail_job']
        target_label: 'job'
      - source_labels: ['__meta_docker_container_name']
        target_label: 'container'
    pipeline_stages:
    - json:
        expressions:
          message: log
          stream: stream
          time: time
    - timestamp:
        source: time
        format: "2006-01-02T15:04:05.000000000Z"
    - labels:
        container:
        job:
```

This configuration utilizes `docker_sd_configs` to discover Docker containers running on the local node. The `relabel_configs` section is crucial here. The first relabel action keeps only containers that have `promtail.scrape=true` defined as a label. The second action extracts the container ID and sets the `__path__` variable, which directs Promtail to the correct log file location. The remaining relabel actions extract additional information like the job name and container name from labels. The pipeline stages parse the JSON log entries, including extracting a timestamp, before sending them to Loki. This ensures that logs are properly ingested and categorized.

**Example 2: Advanced Labeling with Multi-Line Support**

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: docker-swarm-logs
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_label_promtail_scrape']
        regex: 'true'
        action: keep
      - source_labels: ['__meta_docker_container_id']
        target_label: '__path__'
        replacement: '/var/lib/docker/containers/$1/*-json.log'
      - source_labels: ['__meta_docker_container_label_promtail_job']
        target_label: 'job'
      - source_labels: ['__meta_docker_container_label_env']
        target_label: 'environment'
      - source_labels: ['__meta_docker_container_name']
        target_label: 'container'
    pipeline_stages:
    - json:
        expressions:
          message: log
          stream: stream
          time: time
    - timestamp:
        source: time
        format: "2006-01-02T15:04:05.000000000Z"
    - multiline:
        firstline: '^\S|\s+$'
        max_wait_time: 3s
    - labels:
        container:
        job:
        environment:
```

In this example, I've added the extraction of an 'environment' label allowing for further classification of logs. I've also added a `multiline` stage, configured for a common multiline log pattern. If, for instance, application logs include stack traces across multiple lines, this configuration coalesces those into a single log entry. This makes it easier to track errors and understand context. You may need to adjust the `firstline` regex according to your log patterns.

**Example 3: Excluding Certain Containers**

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: docker-swarm-logs
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_label_promtail_scrape']
        regex: 'true'
        action: keep
      - source_labels: ['__meta_docker_container_label_promtail_ignore']
        regex: 'true'
        action: drop
      - source_labels: ['__meta_docker_container_id']
        target_label: '__path__'
        replacement: '/var/lib/docker/containers/$1/*-json.log'
      - source_labels: ['__meta_docker_container_label_promtail_job']
        target_label: 'job'
      - source_labels: ['__meta_docker_container_name']
        target_label: 'container'
    pipeline_stages:
    - json:
        expressions:
          message: log
          stream: stream
          time: time
    - timestamp:
        source: time
        format: "2006-01-02T15:04:05.000000000Z"
    - labels:
        container:
        job:
```

This configuration builds upon the previous examples but introduces an exclusion filter. If a container has a label `promtail.ignore=true`, it will be dropped from the target list. This is useful for excluding noisy or irrelevant containers from log scraping. Using a simple keep and drop approach provides a basic but useful filtering method.

For further learning on Promtail and its capabilities, I recommend consulting the official documentation, specifically focusing on the `docker_sd_configs`, `relabel_configs`, and `pipeline_stages` sections. Exploring examples in the Promtail GitHub repository can also be beneficial. Additionally, reading resources related to Docker Swarm service labels and discovery mechanisms will provide the necessary background for creating more sophisticated configurations. Finally, understanding the core concepts of Loki log parsing and filtering will enable you to develop more optimized logging infrastructure. Mastering Promtail for Docker Swarm requires a deep dive into the interdependencies between the three tools, and I hope that this response provides a useful starting point.
