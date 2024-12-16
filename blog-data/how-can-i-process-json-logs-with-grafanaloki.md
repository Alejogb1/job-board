---
title: "How can I process JSON logs with Grafana/Loki?"
date: "2024-12-16"
id: "how-can-i-process-json-logs-with-grafanaloki"
---

Alright, let’s talk about wrangling json logs with Grafana and Loki; it's a topic I've certainly spent considerable time navigating in various projects. Specifically, I remember a particularly challenging deployment involving a microservices architecture that spewed out verbose json logs. We weren't getting much value from them in their raw form, and that led me down the rabbit hole of effectively using Grafana and Loki together. The key to success here lies in understanding how Loki parses the log streams and how Grafana can leverage those parsed fields.

Loki, at its core, doesn’t *parse* json logs in the traditional sense when it initially ingests them. Instead, it treats the entire log line as a string. This is different from something like Elasticsearch, which indexes each field from the get-go. Loki's strength is its efficiency in indexing only the *labels* associated with the log streams. Therefore, for us to effectively use those json structures, we need to perform the parsing *after* the log is ingested, when querying. This is done through Loki’s powerful query language LogQL and, specifically, with the `json` parser function. This function allows us to specify which log lines should be treated as json, enabling us to extract fields and use them in further filtering and analysis.

The first critical step is to ensure your log entries are consistently structured json. Inconsistency will break parsing attempts. For example, different services might use slightly different field names, which would be detrimental to effective querying. One early mistake I recall making was letting different logging libraries introduce subtly varying json schemas, causing headaches with querying downstream. Consistency, therefore, is paramount, ideally with a well-defined logging schema across your application ecosystem. This helps maintain efficiency and prevents unexpected parsing errors.

Now, let's get to concrete examples. Suppose we have log lines coming in that look like this:

```json
{"timestamp": "2024-10-27T10:00:00Z", "service": "auth-service", "level": "info", "message": "User logged in successfully", "user_id": 123}
```

Here's how you would query this in Grafana using LogQL:

```logql
{job="my-app"} | json | level="error"
```

This LogQL expression first selects log entries labeled with `job="my-app"`. Then, `json` tells Loki to parse each line as json. Finally, it filters those entries further by only including those where the extracted json field `level` is equal to “error”. This simple example shows how we’ve transitioned from a stream of log strings to something workable.

Here's a slightly more complex scenario with a nested json structure. Let's say your logs now look like this:

```json
{"timestamp": "2024-10-27T10:00:00Z", "service": "api-gateway", "level": "info", "request_details": {"method": "GET", "path": "/users", "status_code": 200}}
```

To query for failed requests (status code > 399) from this structure:

```logql
{job="api-gateway"} | json | request_details.status_code > 399
```

In this case, we utilize the dot notation `request_details.status_code` to access nested json values. Note that if the log line doesn’t contain a `request_details` field or the `status_code` field inside `request_details`, this query will return no results for those log lines. This illustrates why consistent structure is crucial. It's not just about parsing; it’s about reliable data extraction.

Lastly, let’s explore aggregating parsed fields. If we want to see the distribution of status codes over time, we can do that using aggregations within LogQL. Consider the same log structure as the second example. We might want to analyze how response codes are spread out within a specific time range:

```logql
rate({job="api-gateway"} | json | unwrap request_details.status_code [5m])
```

Here, `unwrap request_details.status_code` extracts the status code field, transforms it into a numerical value, and makes it available for the rate aggregation within a 5-minute window, effectively giving you a time series of status code changes.

Now, while LogQL's `json` parser is powerful, remember it's processing the logs at query time, not at ingestion. So, if you are dealing with very high-volume logs, heavy parsing could impact query performance. This was an issue I encountered when initially trying this out on our production system. For very high throughput, pre-processing the logs before they are sent to Loki via a pipeline could improve performance (but this adds another layer of complexity). You might use something like Promtail to extract and relabel the fields *before* they enter Loki. This involves some configuration, but could ultimately provide performance benefits. I’ve seen this help significantly when dealing with multiple services, each with their own json logging quirks.

For further study on this topic, I recommend exploring the official Loki documentation on query languages and specifically the `json` parser details. Additionally, “Effective Logging” by Adrian Cole is a beneficial read which talks about structured logging from a broader perspective. In a more data-centric angle, “Designing Data-Intensive Applications” by Martin Kleppmann is helpful to understand the trade-offs of various data processing approaches, including those applicable to logs.

In conclusion, processing json logs with Grafana and Loki is a powerful combination, but success depends on both a deep understanding of LogQL and consistent log structure. Utilizing the json parser and its various options allows granular filtering and analysis. Remember that pre-processing options exist for high-throughput systems, but most users can begin effectively using the parser as part of their LogQL queries. The key lesson, in my experience, has always been careful log structuring, consistently implemented. Doing so will save countless hours of troubleshooting and frustration down the line.
