---
title: "How can I process JSON logs using Grafana/Loki?"
date: "2024-12-23"
id: "how-can-i-process-json-logs-using-grafanaloki"
---

Okay, let’s dive in. I've spent considerable time architecting logging solutions, and processing json logs with Grafana and Loki is something I've tackled more than once. It's a very practical challenge when you move beyond simple text logs, which many systems now do. You’re dealing with structured data, which can be fantastic for richer analysis, but it also demands a different approach. Here's how I’ve gone about it successfully, breaking it down into practical steps.

The core of this hinges on Loki’s ability to parse log lines, and that parsing flexibility is where the power really lies. Loki, unlike some traditional logging systems, does not index the log message content itself; instead, it indexes *labels* that you apply to those logs. This means when dealing with json, you need to extract relevant fields and promote them to labels for efficient querying. The process involves two primary stages: the log pipeline configuration in Promtail (or another agent you might be using), and then the query language (LogQL) within Grafana.

My past experience involved a microservices setup; each service outputted json logs with differing structures. We weren't simply logging to a file; these logs were being pushed out through grpc. Therefore, each service needed its own distinct parsing approach. We couldn’t get away with a generic 'one-size-fits-all' configuration. It became clear that understanding the structure of each log type was critical to creating effective parsing rules.

First, let's address the Promtail configuration— this is usually located in a yaml file, often `promtail.yaml`. Below is an example illustrating how to extract data from a straightforward json structure:

```yaml
scrape_configs:
  - job_name: 'json_app_logs'
    static_configs:
      - targets:
          - localhost # or whatever your log source is
    pipeline_stages:
      - json:
          expressions:
            message: message
            level: level
            timestamp: timestamp
            user_id: user.id # Nested field example
      - labels:
          message:
          level:
          user_id:
      - timestamp:
          source: timestamp
          format: unix # or iso8601 or whatever format the timestamp is in
```

This configuration assumes each line of your log is a complete json object like this: `{"message": "user logged in", "level": "info", "timestamp": 1678886400, "user": {"id": 123}}`.  The `json` stage extracts the specified fields. The `labels` stage then promotes these extracted values to Loki labels. Crucially, the `timestamp` stage ensures Loki correctly interprets the timestamps from the logs.

In our case, we had various systems that provided different timestamp formats. One particular subsystem used iso8601, and another used milliseconds since the epoch. Promtail handles those quite well by just adapting the `format` field. So you could have another similar section in your promtail.yaml that would use 'iso8601', or 'unix_ms' for the millisecond format.

Next, let’s tackle a more challenging scenario, one that I faced while dealing with a logging system that, for some inexplicable reason, embedded json within a string field. This is never optimal, but reality often presents these challenges. We needed to handle that correctly:

```yaml
scrape_configs:
  - job_name: 'embedded_json_logs'
    static_configs:
      - targets:
          - localhost
    pipeline_stages:
      - regex:
          expression: '.*log=(?P<log_json>{.*}).*'
      - json:
          source: log_json
          expressions:
            event_type: type
            operation_id: id
            status_code: http_code
      - labels:
          event_type:
          operation_id:
          status_code:
      - timestamp:
          source: timestamp # assuming there's another timestamp field outside the json
          format: unix
```

Here, the initial `regex` stage extracts the embedded json, using a named capture group `(?P<log_json>{.*})`, giving us the variable `log_json`. The subsequent `json` stage is applied *only* to this extracted json string. This is a fundamental technique when working with more complex log structures. The assumption in this example is that the raw log line contains a `log={...}` string like this: `[2023-03-15 10:00:00] INFO: process id 456 log={"type": "network", "id": "abc-123", "http_code": 200} timestamp=1678886400`.

Now, let’s consider a case where the structure is not consistent, perhaps due to different versions of the application. We can incorporate the `match` stage in Promtail's pipeline to handle this. This situation arose during a gradual rollout of a new application version that output slightly different log formats, forcing us to carefully manage both concurrently.

```yaml
scrape_configs:
  - job_name: 'versioned_json_logs'
    static_configs:
      - targets:
          - localhost
    pipeline_stages:
      - match:
          selector: '{log_version="1"}' # This can be a label you set earlier or default label
          stages:
            - json:
                expressions:
                  message: msg
                  request_id: req_id
            - labels:
                message:
                request_id:

      - match:
          selector: '{log_version="2"}'
          stages:
            - json:
                expressions:
                  message: log_message
                  transaction_id: txn_id
            - labels:
                 message:
                 transaction_id:

```

In this configuration, based on a label, `{log_version="1"}` or `{log_version="2"}`, we apply separate json extraction stages. It is crucial to understand that labels are set outside this process. For example, you may use a label setting stage that relies on a regex to determine the version from the log message itself, or perhaps rely on the container labels, if using something like kubernetes. This demonstrates that the promtail pipeline is extremely configurable and can be tailored to manage many situations.

Once the logs are ingested with the correct labels, you use LogQL in Grafana to query. For example, to view all messages with a specific user id, using the first example above, the query would look like this:  `{job="json_app_logs", user_id="123"} | unwrap message`. The `unwrap message` is used because you are interested in displaying the content, but the query was targeted via the labels. Likewise, if you wanted to see all "error" messages, assuming you labelled them as level, you could use ` {job="json_app_logs", level="error"} | unwrap message`. You can perform far more complex queries and aggregations with LogQL once your labels are correctly assigned.

To get a deeper grasp of these concepts, I’d suggest looking at the official Loki documentation, particularly the sections on Promtail configuration and pipeline stages. Also, the "Programming in Go" book by Mark Summerfield is a great resource that covers regex thoroughly, which is often vital for custom log parsing. Finally, "Site Reliability Engineering" by Betsy Beyer et al. offers valuable insight into designing a logging strategy that's both robust and scalable.

This, in my experience, is a very practical approach to processing json logs with Loki and Grafana. You're not just looking at raw text; you're building a structured, queryable dataset from your log stream, making your monitoring and troubleshooting much more efficient. It requires a careful understanding of your log formats and the capabilities of your tools, but the results are definitely worth the effort.
