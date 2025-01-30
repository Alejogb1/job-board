---
title: "How can log levels be extracted using Prometheus scrape_configs?"
date: "2025-01-30"
id: "how-can-log-levels-be-extracted-using-prometheus"
---
Prometheus’s `scrape_configs` are primarily designed to ingest metrics, not log data, and therefore do not inherently possess a direct mechanism for parsing and extracting log levels. However, by combining the capabilities of Prometheus with intermediate components, log levels can be transformed into metrics and subsequently scraped. My previous work on a large-scale microservice architecture involved implementing precisely this methodology to gain granular insights into application behavior. This required careful orchestration of log aggregation, processing, and metric exposition.

The core challenge lies in that Prometheus expects numeric time-series data as output. Logs, conversely, are textual. The `scrape_configs` directive defines target endpoints from which Prometheus pulls this numerical data at configured intervals. To make log levels accessible to Prometheus, a system must be established that:

1.  **Aggregates Logs:** Typically, logs are sent from individual applications to a central log aggregator (e.g., Loki, Fluentd, Elasticsearch).
2.  **Parses Logs:** This aggregator or a downstream processor needs to extract log levels from the log messages. Common log formats like JSON or log4j typically include a designated field for the log level.
3.  **Exposes as Metrics:**  The extracted log levels must be converted into numerical metrics. For example, the frequency of different log levels (INFO, WARN, ERROR, etc.) can be counted over time and exposed as Prometheus-compatible counters.
4.  **Prometheus Scrapes:** The target that exposes these derived metrics is then configured as a scrape target within the `scrape_configs` section of the Prometheus configuration.

The process does not involve altering the standard scrape config to parse logs directly. Instead, we augment the observability pipeline to first process logs, generate derived metrics, and *then* have Prometheus scrape those specific endpoints.

Let's consider three practical scenarios using various components:

**Example 1: Using `loki` as a log aggregator and `promtail` for processing.**

`loki` is a log aggregation system designed to pair well with Prometheus. `promtail` is an agent that ships log data to loki. We can use `promtail`’s pipeline to extract log levels and generate metrics.

```yaml
# promtail config
scrape_configs:
  - job_name: application_logs
    static_configs:
      - targets:
          - localhost # path to log file
    pipeline_stages:
      - match:
          selector: '{app="my-app"}' # use labels to filter logs
          stages:
            - json:
                expressions:
                  level: level # extract the level from log json structure
            - metrics:
              level_counter:
                  type: counter
                  description: "Frequency of each log level"
                  source: level
      - labels:
         level:
```

In this `promtail` configuration, we define a `match` stage that only processes logs from our application `my-app`. The `json` stage extracts the `level` field from a JSON-formatted log entry. The `metrics` stage defines a counter, `level_counter`, incremented by one each time a log at given level appears. The extracted `level` is added as a label to the log entry. Loki will store these log entries with metrics attached, which can be queried from Grafana for visualization. Note that *this does not directly allow prometheus to scrape metrics, only allows loki to generate metrics.*

To make these level counts accessible to prometheus, we need a separate service to expose the aggregated metrics via Prometheus-compatible endpoint. We can achieve this with a small custom application that connects to Loki and exposes the metrics.

**Example 2: Utilizing a custom application to generate metrics.**

Suppose you have a custom Python application that processes logs from Kafka, parses the log level, and generates Prometheus metrics. Here is a simplified code representation using the `prometheus_client` library:

```python
import logging
from prometheus_client import start_http_server, Counter
import json
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define Prometheus metrics
level_counter = Counter('app_log_level', 'Frequency of each log level', ['level'])

# Dummy function simulating log processing from Kafka/other source
def process_log():
    # Simulate consuming log data
    log_data = '{"level": "ERROR", "message": "Something bad happened"}'
    log_entry = json.loads(log_data)

    log_level = log_entry.get('level')
    if log_level:
        level_counter.labels(level=log_level).inc()
        logging.info(f"Incremented level counter: {log_level}")

    log_data = '{"level": "INFO", "message": "All good"}'
    log_entry = json.loads(log_data)

    log_level = log_entry.get('level')
    if log_level:
        level_counter.labels(level=log_level).inc()
        logging.info(f"Incremented level counter: {log_level}")


if __name__ == '__main__':
    # Start Prometheus HTTP server on port 8000
    start_http_server(8000)
    logging.info("Prometheus metrics server started on port 8000")
    while True:
      process_log()
      time.sleep(5)

```

This Python code defines a Prometheus counter named `app_log_level`, with a label to differentiate between different log levels. The dummy `process_log()` function simulates log consumption, extracts the log level from the JSON string, and increments the corresponding counter. Finally, the code starts an HTTP server on port 8000, allowing Prometheus to scrape metrics at `localhost:8000/metrics`. This example assumes the presence of a logging framework that provides JSON formatted outputs. This endpoint can then be included in the Prometheus `scrape_configs` section.

**Example 3: Integrating Fluentd with Prometheus exporter.**

Fluentd, a log collector, can be configured to parse logs, compute aggregates, and then output data suitable for Prometheus to consume.

```conf
<source>
  @type forward
  port 24224
  bind 0.0.0.0
</source>

<match my-app.**>
  @type parser
  format json
  key_name log
  <store>
    @type prometheus
    prometheus_label_keys level
    <metric>
        name log_level_count
        type counter
        key level
    </metric>
    <service>
      host 0.0.0.0
      port 9999
    </service>
  </store>
</match>
```

In this configuration, Fluentd listens for incoming log messages tagged with `my-app.**` on port 24224.  The `parser` directive instructs Fluentd to parse the `log` field as JSON and extract log level information. The `prometheus` output plugin aggregates the counts per log level and exposes these metrics at the address `0.0.0.0:9999/metrics` which can then be scraped by Prometheus. This example integrates Fluentd using it's prometheus output plugin to achieve the goal.

**Recommendations**

To further explore this topic, consult documentation for the following:

*   **Prometheus:** Thoroughly understand the concepts of scrape configurations, labels, and metric types.
*   **Loki/Promtail:** Explore the various capabilities of these tools, particularly pipeline stages and log querying.
*   **Fluentd:** Review its flexible input and output configurations, including the prometheus plugin capabilities.
*   **Python/Go Prometheus client libraries:** Become familiar with how to define and expose custom metrics from applications.
*   **Kafka/other messaging queues:** Become familiar with common logging data transport mechanisms.
*   **Log formats:** Familiarize yourself with different logging formats, such as JSON, log4j, and CEF.

Integrating logging data with Prometheus requires a pipeline approach. Directly parsing logs with the `scrape_configs` directive is not feasible.  It's essential to use specialized tools for log collection, processing, and metric generation before exposing the numeric data to Prometheus for scraping. The presented examples provide starting points, and specific implementation details will vary based on environment, tools, and data volumes. Understanding the capabilities of each tool in the observability pipeline is critical to creating an effective monitoring solution.
