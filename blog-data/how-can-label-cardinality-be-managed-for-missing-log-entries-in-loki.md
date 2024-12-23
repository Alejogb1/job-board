---
title: "How can label cardinality be managed for missing log entries in Loki?"
date: "2024-12-23"
id: "how-can-label-cardinality-be-managed-for-missing-log-entries-in-loki"
---

Let's dive straight in; managing label cardinality, especially when faced with missing log entries in Loki, is a challenge I've grappled with extensively over the years. It's one of those areas where a seemingly small oversight can quickly cascade into performance and storage issues. I’ve seen it first hand, working on a distributed system where a seemingly benign new field suddenly ballooned the label set to unmanageable sizes, and a good portion of our log data vanished from sight as a consequence.

The root of the problem lies in the nature of labels in Loki; they're essentially dimensions that index and allow querying of log streams. High label cardinality – meaning a large number of unique label values – directly translates to a higher indexing overhead and larger memory footprint, and it slows down retrieval. When you introduce missing log entries, particularly if these missing entries were meant to carry certain labels, you’re facing two distinct but related issues: a cardinality problem and a visibility issue.

Missing logs, unfortunately, exacerbate the cardinality issue because the absence of expected data does not necessarily mean the absence of the *labels* that would have been present. If your system emits a specific label only on certain events, and those events are sporadic or go completely missing for stretches of time, then Loki can end up creating *empty* streams identified only by these sometimes-present labels. This contributes to cardinality bloat, and these “ghost” streams can lead to inefficient storage and query times.

Here’s how I approach mitigating this issue, combining some careful schema design with practical query techniques:

**1. Label Schema Design and Pre-Processing:**

Firstly, and this is the most crucial, a well-defined label schema is critical. Avoid labels that represent highly variable data, such as timestamps or unique identifiers. Treat labels as metadata; use them to categorize, not to record transactions. For instance, if you have log messages that contain an `order_id`, do not use `order_id` as a label. Rather, consider a coarser `order_type` or `application_component` label. A cardinal sin I've seen far too often is logging errors where the error message is the label. It’s tempting, but that creates a label for each unique error, sending cardinality through the roof.

Prior to ingestion into Loki, I often employ a pre-processing stage (using Logstash or a custom script) to clean and standardize log data. This process includes converting specific high-cardinality fields into *log lines* instead of labels.

**Example Code Snippet 1 (Python Pre-Processing):**

```python
import json
import re

def preprocess_log(log_line):
    log_data = json.loads(log_line)

    # Check for and handle 'order_id'
    if 'order_id' in log_data:
        log_data['log_message'] += f" order_id={log_data.pop('order_id')}"

    # Handle a date or time field
    if 'timestamp' in log_data:
      log_data['log_message'] += f" timestamp={log_data.pop('timestamp')}"


    return json.dumps(log_data)

# Example Usage:
log_input = '{"timestamp": "2024-10-27T12:30:00", "order_id": "abc123xyz", "message": "Order received"}'
processed_log = preprocess_log(log_input)
print(processed_log)
# Output: {"message": "Order received order_id=abc123xyz timestamp=2024-10-27T12:30:00"}
```

In the example above, fields such as `order_id` and the `timestamp` are moved into the log message, preventing them from becoming high-cardinality labels. The key here is moving dynamic data into the log line, and reserving labels for more static values.

**2. Utilizing Loki's Label Filtering and Querying:**

Loki's query language (LogQL) is quite powerful, and you can use it to mitigate the effects of "ghost" streams. When I encounter missing logs related to a certain operation, I don’t just blindly query by the labels I *think* should be present. Instead, I use label existence operators to be sure.

For example, instead of a query like `{job="my_service", operation="create"}`, I would prefer: `{job="my_service"} |= "operation=create"`. The second query is less reliant on the label `operation` always being present; it pulls any logs where the term `operation=create` exists within the log line, regardless of whether `operation` is a label or a logged value. This allows flexibility to search for instances of this operation that do not contain the dedicated label. This method helps mitigate cases where the “operation” label might be sporadically missing while still letting us find the relevant logs within our data streams.

**3. Exploring Absent Data with `absent_over_time`:**

Loki provides the `absent_over_time` function. This is a tool I’ve found invaluable when trying to understand why specific log streams are missing at certain time periods. `absent_over_time` returns a vector indicating the period where a specified stream is absent. This is useful for setting up alerts (when an expected log isn’t being generated) or when investigating incidents where logs have disappeared unexpectedly.

**Example Code Snippet 2 (LogQL query):**

Let's say I expect a log stream containing labels like `{job="background_worker", component="processor"}`. If those are missing for an extended period, we can detect that.

```logql
absent_over_time(
  {job="background_worker", component="processor"}[1h]
)
```

This LogQL query will return a vector of time series, set to '1' at times when the log stream is missing within each one-hour interval. Examining this data can expose if the `background_worker` is failing, or if the process was simply stopped, enabling quick investigation and allowing for targeted corrective actions. I use this frequently when monitoring critical infrastructure components.

**4. Strategic Label Dropping:**

Loki supports the ability to drop certain labels during the ingestion process via relabeling rules in the Loki configuration. This should be done very carefully; I tend to apply label dropping only after careful analysis and only to known superfluous or redundant labels. A dropped label means you can no longer query using it, so the decision must be made after thoroughly examining the implications.

**Example Code Snippet 3 (Loki Configuration - excerpt of relabeling rules):**

```yaml
# Sample configuration snippet (loki.yml):
scrape_configs:
  - job_name: my_application_logs
    static_configs:
      - targets: ['localhost:3100'] # Where logs are coming from
    relabel_configs:
      - source_labels: ['instance'] # From the incoming log
        regex: '.+'       # Match everything
        action: drop       # Delete the instance label
      - source_labels: ['trace_id']
        regex: '.+'
        action: drop
```
Here, I'm showing a (somewhat extreme) example where I’m dropping the `instance` and `trace_id` labels, if they exist.  In a real world scenario you need to be very careful when choosing what labels to drop as this action is irreversible and should only be used if the label provides no analytical value, or is a significant source of cardinality issues. Usually, I wouldn't drop a label like instance. This is simply to illustrate the relabeling mechanism.

**Further Reading:**

For deeper insight into these topics, I highly recommend exploring the following:

*   **"Designing Data-Intensive Applications" by Martin Kleppmann:** This book offers an excellent overview of database concepts, including indexing, cardinality, and data modeling best practices, all crucial to handling logging data effectively.
*   **The official Loki documentation:** This is, of course, your primary reference. Pay close attention to sections on configuration, querying, and best practices regarding labels and cardinality.
*   **Papers on time-series databases:** Understanding the underlying principles of time-series databases can greatly improve your understanding of how Loki operates. Look for publications focusing on storage optimization and query efficiency in time-series data.

In conclusion, effectively managing label cardinality and missing log entries is an ongoing process that requires careful schema design, pre-processing of log data, and a thorough understanding of Loki’s querying capabilities. Missing data highlights the flaws in your design as much as it presents a problem of itself, and it’s critical to approach the issue with both a technical and analytical mindset to avoid escalating challenges.
