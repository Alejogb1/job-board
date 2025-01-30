---
title: "How can Docker JSON logs be parsed correctly in Promtail?"
date: "2025-01-30"
id: "how-can-docker-json-logs-be-parsed-correctly"
---
Docker's JSON logging driver outputs each log message as a single JSON object, a structure that while convenient for structured logging, necessitates specific configuration within Promtail to extract meaningful fields for analysis and monitoring. The default behavior of Promtail, which treats each line as a single log entry, fails to parse these individual JSON objects correctly, leading to the entire JSON string being ingested as a single log message under a single `log` label. I've personally spent many hours debugging issues arising from this default handling, tracing incorrect data and frustrated searches through Grafana logs. Consequently, achieving effective parsing requires employing Promtail's relabeling capabilities and JSON parsing stages.

Promtail's configuration process for ingesting Docker JSON logs centers around two critical elements within the `pipeline_stages`: the `json` stage and the `relabel` stages. The `json` stage is designed to decode the JSON log message and expose its fields as labels within Promtail's processing pipeline. Without this stage, the JSON object is just opaque text. Subsequent `relabel` stages use these newly exposed labels to shape the final labels attached to each log entry and determine where that entry is ultimately stored in Loki. This multi-stage approach allows for nuanced handling of complex JSON structures and specific routing rules.

The following code examples illustrate these steps, demonstrating progressively more sophisticated approaches to parsing Docker JSON logs in Promtail.

**Example 1: Basic JSON Parsing**

This initial example represents the bare minimum required to parse a simple Docker JSON log message. Assume the Docker container emits JSON structured as `{"log":"This is a log message", "time":"2024-01-26T10:00:00Z"}`.

```yaml
scrape_configs:
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
    pipeline_stages:
      - json:
          expressions:
            log: log
            time: time
      - labels:
          __path__: /var/log/docker/container.log # Dummy path for illustration
      - timestamp:
          source: time
          format: "2006-01-02T15:04:05Z"
```

Here, the `json` stage extracts the "log" and "time" fields from the JSON object using their corresponding keys. Importantly, these fields become available as labels within the Promtail pipeline. The `labels` stage adds a dummy path label, for demonstration purposes; this would be more relevant in other configurations using file or static paths. Finally, a `timestamp` stage is used to interpret the "time" field and override the default timestamp derived from Promtail's reception time. This ensures that log entries are chronologically aligned based on the container's internal time and facilitates searching based on actual logged timestamps, rather than when the logs were received. Without the `timestamp` stage, a log event's timestamp would be determined when Promtail scrapes the log, potentially creating inconsistencies.

**Example 2: Parsing Nested JSON with Relabeling**

Consider a more complex scenario where your Docker container emits JSON logs containing a nested object, such as `{"log":"Operation started", "context": {"operation":"auth", "user":"johndoe"}}`. You need to extract the `operation` field for routing or alerting purposes, which will require manipulating the labels.

```yaml
scrape_configs:
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
    pipeline_stages:
      - json:
          expressions:
            log: log
            context: context
      - json:
          source: context
          expressions:
             operation: operation
             user: user
      - labels:
          __path__: /var/log/docker/container.log # Dummy path for illustration
      - relabel:
          source_labels: ['operation']
          target_label: 'operation'
          regex: '(.*)'
```

The first `json` stage extracts the top-level `log` and `context` fields. Then, a second `json` stage is employed, using `source: context` to specify that it should parse the content of the previously extracted 'context' label. This extracts the inner `operation` and `user` fields from the nested JSON object.  The `relabel` stage then copies the value of the newly extracted label, `operation` into a new label with the name `operation`. The regex matching and copying step (though seemingly trivial with `regex: '(.*)'`) is essential to move the label from a temporary internal label to one that can be used for Loki queries. If there were a specific operation value we were interested in, we could use more specific regular expression matching here, such as `regex: '^(auth)$'` which would filter the operations down.

**Example 3: Conditional Relabeling and Dropping Logs**

In this example, let's explore a scenario where some container logs include an "error" field while others don't, and only logs containing "error" need to be sent to the Loki logs storage with label `error_detected=true`. Logs without error must not include an error label.

```yaml
scrape_configs:
  - job_name: docker
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
    pipeline_stages:
      - json:
          expressions:
            log: log
            error: error
      - labels:
          __path__: /var/log/docker/container.log # Dummy path for illustration
      - relabel:
          source_labels: ['error']
          target_label: 'error_detected'
          regex: '(.*)'
          replacement: 'true'
      - match:
          stage: labels
          action: drop
          selector: '{error_detected=""}'
```

Initially, we parse the log and the error fields into labels. The first relabel stage then converts the error label's value into 'true' and moves it into a new `error_detected` label, ensuring consistency in labels. Here the regex is also `(.*)`, which matches any value for the error label. Next is an essential `match` stage. This stage utilizes `action: drop` with a `selector: '{error_detected=""}'`, which drops all logs where there was no error. The result is that only log entries which included a JSON key "error" will be sent to Loki, and they will have the label `error_detected=true`. The crucial aspect here is the utilization of relabeling to introduce a new label specifically for filtering purposes and then the use of the match action to selectively filter logs based on the generated label. This type of configuration is crucial for efficient logging, where only logs of interest are ingested and indexed, reducing storage and resource utilization.

To effectively use and manage Promtail configurations, I recommend consulting the following resources. The official Loki documentation provides comprehensive information regarding Promtail's pipeline stages and configuration options; this is often the first resource I turn to. Additionally, the Promtail repository itself (usually on a platform such as GitHub) often includes updated examples and explanations of the more current capabilities. Another useful source is the community forums and discussion boards surrounding Promtail and Loki, where you can find practical examples from other users who have tackled similar logging challenges. While the technical documentation is crucial for specifics, these community platforms frequently offer insights into best practices and specific troubleshooting approaches. These resources, when used together, provide a robust foundation for mastering Docker JSON log parsing with Promtail and integrating them with Loki. Through my own trial-and-error based learning experience, I have found that these are the most helpful in addressing unexpected errors in processing logs, which I believe is a shared experience among anyone who has spent time managing log collection systems.
