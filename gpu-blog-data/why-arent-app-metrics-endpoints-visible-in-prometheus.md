---
title: "Why aren't app metrics endpoints visible in Prometheus?"
date: "2025-01-30"
id: "why-arent-app-metrics-endpoints-visible-in-prometheus"
---
Prometheus, by default, discovers targets through service discovery or static configurations; a mismatch in these configurations, specifically concerning the defined target paths and scraping intervals, often obscures application metric endpoints. My experience troubleshooting such issues stems from maintaining a microservices environment for an e-commerce platform where we adopted Prometheus as our primary monitoring solution. We initially faced silent failures where metrics, demonstrably exposed by our services, were not being collected.

The core issue arises when the Prometheus server, operating as a pull-based system, cannot locate and retrieve the metric data exposed by your application. This retrieval depends entirely on the precise configuration within `prometheus.yml`. If the target application’s metrics endpoint deviates from what is configured in Prometheus, or if the scraping interval is inadequate for the target’s update frequency, the metrics will simply not appear in Prometheus. Prometheus will not auto-magically “find” new endpoints. You must explicitly specify the endpoints it should monitor. It operates based on the provided settings, and any deviation will lead to a lack of visibility.

Furthermore, the way an application exposes metrics is also crucial. Prometheus expects metric endpoints to be served in a specific format, typically plain text that follows the Prometheus exposition format. If your application serves metrics in a different format (e.g., JSON, XML), Prometheus will not be able to parse the data and it will remain unavailable in the dashboard. The target application must therefore implement an endpoint that complies with the Prometheus text-based exposition format which follows specific naming conventions and data types.

To understand the configuration aspects more clearly, consider three examples based on issues I encountered.

**Example 1: Incorrect Target Path**

The most common culprit is a simple typo or misunderstanding of the application’s metric endpoint. Initially, one of our services, responsible for order processing, exposed metrics on `/metrics`, as we correctly coded in the service, but the Prometheus configuration targeted `/prometheus` instead. This minor discrepancy was the source of the lack of metrics. The `prometheus.yml` file had the following configuration segment:

```yaml
scrape_configs:
  - job_name: 'order-service'
    static_configs:
      - targets: ['order-service-host:8080']
        labels:
          app: order-service
    metrics_path: /prometheus
    scrape_interval: 15s
```

Here, the `metrics_path` directive specified `/prometheus`, while our application served the metrics at `/metrics`. Changing the configuration file to reflect the actual endpoint, which we knew to exist at `/metrics`, resolved the issue:

```yaml
scrape_configs:
  - job_name: 'order-service'
    static_configs:
      - targets: ['order-service-host:8080']
        labels:
          app: order-service
    metrics_path: /metrics
    scrape_interval: 15s
```
After restarting the Prometheus server, the order service's metrics became immediately available. This emphasizes that the value of the `metrics_path` configuration is not an arbitrary value, but must exactly match the application’s exposed metric endpoint path.

**Example 2: Inadequate Scrape Interval**

Another scenario involves scraping intervals that are inadequate for the frequency at which an application updates its metrics. In another instance, our data ingestion service generated metrics in a very short period, roughly every five seconds. Our initial Prometheus configuration set a scrape interval of 30 seconds.

```yaml
scrape_configs:
  - job_name: 'data-ingestion-service'
    static_configs:
      - targets: ['data-ingestion-host:8080']
        labels:
          app: data-ingestion-service
    metrics_path: /metrics
    scrape_interval: 30s
```

While the metrics *were* visible in Prometheus, the data was not up-to-date and certain spikes and rapid changes were being missed. The lack of data was directly tied to the scrape interval. A `scrape_interval` of 30 seconds means Prometheus polled the application for metrics only every 30 seconds. Given that the application updated its metrics every five seconds, Prometheus was not collecting data as frequently as it was being produced. We adjusted the `scrape_interval` to ten seconds, slightly slower than the actual metric generation, which helped in capturing the metric fluctuation. Setting it to 5 would have put undue stress on the target and the Prometheus instance. The updated configuration:

```yaml
scrape_configs:
  - job_name: 'data-ingestion-service'
    static_configs:
      - targets: ['data-ingestion-host:8080']
        labels:
          app: data-ingestion-service
    metrics_path: /metrics
    scrape_interval: 10s
```

The key takeaway is that the `scrape_interval` has to be sufficiently frequent so that changes in metrics are not missed while not being overly frequent to put an unneeded load on either the application being monitored or the Prometheus server. It is a balance based on the requirements of the specific application.

**Example 3: Invalid Metric Exposition Format**

A third problem that we encountered concerned the format of the returned metrics. One service returned metrics in JSON format rather than the required plain-text Prometheus exposition format. The `prometheus.yml` configuration was correct regarding the path, but the service was not adhering to the necessary format, and as such, the scrape would fail. Even though Prometheus would contact the service at the `/metrics` path, it would receive data it could not interpret, resulting in an empty set of metrics for the service. There's no configuration on Prometheus to convert the format. The responsibility lies on the application exposing the metrics to do so in the format Prometheus understands. Resolving this involved refactoring the application itself to serve metrics in the proper Prometheus format.

```python
from flask import Flask, Response
from prometheus_client import generate_latest, Counter, Gauge, Histogram, Summary, CollectorRegistry

app = Flask(__name__)

# Create a custom registry
registry = CollectorRegistry()

# Define your metrics
request_count = Counter('http_requests_total', 'Total number of HTTP requests', registry=registry)
active_users = Gauge('active_users_count', 'Number of active users', registry=registry)
response_time = Summary('response_time_seconds', 'Response time of HTTP requests', registry=registry)
request_latency = Histogram('http_request_duration_seconds', 'Histogram of request duration', registry=registry)

@app.route('/')
def home():
  request_count.inc()
  with response_time.time():
    with request_latency.time():
      # simulate work
      import time
      time.sleep(0.2)

      active_users.inc(1) # Simulate an user interaction
      time.sleep(0.1)
      active_users.dec(1)
    return "Hello, world!"

@app.route('/metrics')
def metrics():
  return Response(generate_latest(registry), mimetype='text/plain')

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=8080)
```

This Python Flask example demonstrates using the `prometheus_client` library to expose metrics at `/metrics`. The key here is the `generate_latest()` function which takes the current values of defined metrics and outputs them into a plain-text format that Prometheus will correctly parse. If, instead, we had attempted to output the metric data as a JSON structure, it would have failed for Prometheus. This required an understanding of both the application and the required exposition format.

To further understand and troubleshoot these issues, I recommend reviewing the official Prometheus documentation, particularly the section detailing target configurations. The “Monitoring in the Cloud” book, a general book on monitoring and observability, provides a good theoretical understanding on the various monitoring approaches, including pull-based systems which is crucial in understanding how Prometheus interacts with the targets. Additionally, exploring tutorials that cover Prometheus best practices for application monitoring can give practical insights into various scenarios and solutions. The resource 'Site Reliability Engineering' from Google, although not explicitly about Prometheus, does have a lot of relevant material regarding the importance of monitoring and alerting and will provide the necessary context. These resources offer a broad understanding and practical guidance on Prometheus, moving past simplistic "it doesn't work" to understanding specific problems and solutions.
