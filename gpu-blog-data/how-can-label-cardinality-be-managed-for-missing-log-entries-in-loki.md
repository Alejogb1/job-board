---
title: "How can label cardinality be managed for missing log entries in Loki?"
date: "2025-01-26"
id: "how-can-label-cardinality-be-managed-for-missing-log-entries-in-loki"
---

Label cardinality, particularly when dealing with absent log lines, poses a unique challenge in Loki. Specifically, when a log stream entirely ceases, and consequently no new entries are being written, the existing labels associated with that stream can be difficult to manage effectively and ultimately contribute to high cardinality issues if not addressed properly. This stems from how Loki indexes and queries streams based on these labels. My experience maintaining a distributed microservices platform, logging to Loki using custom agents, has provided valuable insights into this problem.

The core issue arises from Loki's stream-based ingestion model. Each unique set of labels defines a distinct stream. When a microservice instance crashes, or a deployment is scaled down, the associated log stream effectively disappears. While this seems intuitive, the label sets associated with these inactive streams remain in Loki’s index. If there are a large number of such streams that are intermittently active, this can quickly inflate the index size and degrade query performance. The problem is further exacerbated when labels include high-cardinality values like instance IDs or randomly generated job IDs, as each change in these values results in a new stream. Although the streams may not have new log lines ingested, the label sets themselves persist until explicitly removed or the retention policy kicks in.

One common approach is to design labels carefully from the outset to minimize variability and avoid values that inherently have high cardinality. Consider labels like `environment`, `application`, and `component`, which are generally stable across instances and time. This foundational strategy is crucial in preventing the problem before it occurs. However, this initial design doesn't address the problem of stream disappearance leading to lingering index entries.

To effectively manage these issues, we must consider a combination of strategies. One primary technique involves actively informing Loki about stream disappearance. When a process that emits logs using a particular set of labels gracefully terminates, it should send an explicit signal indicating that the stream is no longer active. This avoids having Loki infer stream termination by absence of log lines, a detection method that introduces delays and is not always reliable.

Consider the following Python code example. This represents a hypothetical service gracefully shutting down:

```python
import time
import logging
import requests
import json

# Assume Loki client is already configured and available.
LOKI_URL = "http://loki:3100/loki/api/v1/push"
LABELS = {
    "environment": "production",
    "application": "order-service",
    "instance": "order-service-01"
}

def log_message(message):
    payload = {
        "streams": [{
            "stream": LABELS,
            "values": [
                [str(time.time_ns()), message]
            ]
        }]
    }
    try:
        response = requests.post(LOKI_URL, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error sending log to Loki: {e}")


def send_stream_inactive_signal():
    stream_inactive_labels = {k:v for k,v in LABELS.items()} # Clone Labels
    stream_inactive_labels["_stream_inactive"] = "true"
    payload = {
        "streams": [{
            "stream": stream_inactive_labels,
            "values": [
                [str(time.time_ns()), "stream inactive signal"]
            ]
        }]
    }
    try:
        response = requests.post(LOKI_URL, json=payload)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
      logging.error(f"Error sending stream inactive signal to Loki: {e}")

if __name__ == '__main__':
    log_message("Service starting up")
    time.sleep(5) # Simulate work
    log_message("Service shutting down")
    send_stream_inactive_signal()

```
This Python code demonstrates a typical log ingestion scenario, but with the addition of `send_stream_inactive_signal`. The key idea here is to introduce a new label, `_stream_inactive`, set to `true`, when the service shuts down. This signal, sent with the same base labels as the regular log streams, allows downstream components to identify that this stream is no longer generating logs and can be used in specific Loki query for cleanup operations. Notice we send a log line that’s not very relevant, the only important part here is to send this as a signal that will be registered by Loki. This requires adding an additional logic. This simple signaling approach allows other tooling to know, at the source, if this stream is no longer active.

Next, we can build tooling around these inactive signals to identify such streams and potentially utilize Loki's API or other tools to manage those indexes. We can implement a query in PromQL, the query language used by Loki, to locate these inactive streams.

Consider this query, demonstrating how to find those inactive streams within a given timeframe :

```promql
count_over_time({_stream_inactive="true", application="order-service"}[1h]) > 0
```

This query specifically filters streams which include `_stream_inactive="true"` and `application="order-service"` and checks if there was at least one log line within the last 1 hour. You can tweak the time frame to reflect your requirements. This query provides a list of label sets associated with inactive streams that can then be processed further. Note that while this helps identify streams that have *explicitly* been marked as inactive, it relies on agents sending such signals when they stop.

Building upon this query, a custom tool can iterate over the returned label sets and either remove these inactive streams from the index or trigger some other cleanup actions. While Loki does not provide a direct API call for immediate deletion of specific label sets from the index, we can achieve a similar effect via retention policies or, if appropriate, through a careful recreation of the index without those particular streams. For very high scale environments, you will likely want to leverage something like Grafana's Mimir which provides more tooling on top of Loki’s raw storage capabilities. However, this is beyond the scope of the question and it can get considerably more involved.

A more complex implementation would incorporate a job that continuously scans for `_stream_inactive` signals and updates a secondary data store with information about the stream’s status. Then, that data store can be used to identify those streams that haven’t emitted new log lines for an extended period and should be purged.

```python
import time
import logging
import requests
import json
import prometheus_client

# Assume Loki client is already configured and available.
LOKI_URL = "http://loki:3100/loki/api/v1/query_range"
INACTIVE_STREAMS_QUERY = 'count_over_time({_stream_inactive="true"}[1h]) > 0'
#Assume a basic key-value cache is accessible.
CACHE_EXPIRY = 3600 #seconds
INACTIVE_STREAMS_CACHE = {}
def query_loki(query):
    params = {
        'query': query,
        'start': str(time.time() - 3600),
        'end': str(time.time()),
        'step': '60s'
    }

    try:
        response = requests.get(LOKI_URL, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logging.error(f"Error querying Loki: {e}")
        return None
def update_inactive_streams_cache():
    query_result = query_loki(INACTIVE_STREAMS_QUERY)
    if not query_result or query_result['status'] != 'success':
         logging.error("Error querying Loki for inactive streams")
         return
    inactive_streams = []
    for result in query_result['data']['result']:
        label_set = result['metric']
        inactive_streams.append(label_set)

    for stream in inactive_streams:
         INACTIVE_STREAMS_CACHE[tuple(sorted(stream.items()))] = time.time()

    for key in list(INACTIVE_STREAMS_CACHE.keys()):
        if time.time() - INACTIVE_STREAMS_CACHE[key] > CACHE_EXPIRY:
            del INACTIVE_STREAMS_CACHE[key]
def print_inactive_streams():
    for key in INACTIVE_STREAMS_CACHE.keys():
        print(dict(key))

if __name__ == '__main__':
    while True:
        update_inactive_streams_cache()
        print_inactive_streams()
        time.sleep(60)
```

This last code snippet illustrates a periodic job that queries Loki, updates a local cache with the label sets of inactive streams, and prints the result. This cache-based approach can help with performance since it will not query Loki on every single check, and will only query if an older entry expired. This example demonstrates the essential concepts; in a production system, the cache management, deletion process, and error handling would require more robust implementation, especially in highly concurrent environments.

In summary, managing label cardinality for missing log entries in Loki requires a multifaceted approach that involves thoughtful label design, explicit signaling of stream inactivity, and the development of supporting tooling to manage these inactive stream entries. This strategy shifts the burden from simply reacting to the problem to proactively managing the label landscape, thereby preserving the performance and stability of the logging infrastructure.

I would recommend consulting the documentation for Loki’s query language (LogQL) to deepen understanding of available query operators. Furthermore, studying the principles of monitoring and observability, alongside the specific architectures of systems like Grafana’s Mimir, will provide additional context on how best to apply these techniques for production logging at scale.
