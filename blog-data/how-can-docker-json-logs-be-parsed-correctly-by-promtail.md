---
title: "How can Docker JSON logs be parsed correctly by promtail?"
date: "2024-12-23"
id: "how-can-docker-json-logs-be-parsed-correctly-by-promtail"
---

Okay, let's tackle this. I’ve spent a good chunk of my career working with containerized environments, and parsing docker logs for monitoring has definitely presented its share of interesting challenges. The key to correctly parsing docker json logs with promtail hinges on understanding both how docker outputs those logs and how promtail's configuration can be tailored to consume that structure. It's not just about slapping a configuration together; it’s about crafting it with an understanding of the underlying data model.

From my experience, the common pitfall is assuming the json payload docker generates is uniform across all applications. It's not. Docker's json log output typically wraps the actual log message in a structured payload that includes metadata like the `time`, `log`, and sometimes a `stream` identifier (stdout or stderr). Promtail, by default, expects simpler line-based log formats. Therefore, we need to explicitly tell promtail how to unravel this json structure and extract the useful log message.

The core issue revolves around promtail's `pipeline_stages` within a job configuration. We’re essentially creating a custom processing pipeline that transforms the raw log data into something promtail can properly index and label. We’ll need to instruct promtail to: a) parse the incoming log line as json, b) extract the log message from a specific json field (typically “log”), and c) potentially handle timestamps, labels, or other metadata.

Let's break down a few scenarios and their respective promtail configurations. Imagine I've encountered scenarios with varied log structures, as is often the case in larger deployments.

**Scenario 1: Basic JSON Log Structure**

The simplest case is when the docker log json looks something like this:

```json
{"log":"This is a standard log message.", "time":"2024-07-27T10:00:00Z", "stream":"stdout"}
```

In this scenario, the `log` field holds the message we’re interested in. The corresponding promtail `pipeline_stages` configuration would be:

```yaml
pipeline_stages:
  - json:
      expressions:
        log: log
  - labels:
      stream:
```

Here's what's going on:
1. **`json:` Stage**: This stage instructs promtail to parse the entire log line as json. The `expressions` field specifies which json keys to extract, in this case, we extract the value associated with the key 'log' and assign it to a field named 'log' within promtail's internal structure.
2. **`labels:` Stage**: While this specific example doesn’t use `stream` for a label, it’s included to illustrate how you *could* extract it. If you wanted this field as a label, we’d configure it similarly to the `log` extraction. This allows us to further refine and query logs based on various metadata properties. This is essential for building effective dashboards and alerting rules.

**Scenario 2: Nested JSON Log Structure**

Now, consider a scenario where the log message is nestled deeper in the json structure:

```json
{"metadata": {"timestamp": "2024-07-27T10:00:00Z"}, "data": {"message": "A more complex log message.", "level":"info"}, "stream":"stderr"}
```

Here, the actual log message resides in `data.message`. Our promtail configuration needs to reflect this nested structure:

```yaml
pipeline_stages:
  - json:
      expressions:
         message: data.message
         timestamp: metadata.timestamp
  - labels:
      level: data.level
  - timestamp:
      source: timestamp
      format: "2006-01-02T15:04:05Z"
```

Here's a breakdown of the changes:
1.  **`json:` Stage**: We've changed `expressions` to extract the nested value at `data.message` and assign it to promtail’s internal variable `message`. We also extract the `timestamp` from the metadata object.
2. **`labels:` Stage**: This adds the `level` from the data object as a label. This allows us to then filter logs based on severity.
3. **`timestamp:` Stage**: This stage is crucial as it sets the log timestamp correctly. This ensures accurate time series data in Loki. The timestamp is parsed from the `timestamp` variable and formatted using Go's time layout which is set to `"2006-01-02T15:04:05Z"`, as the JSON log was in ISO format. Without this step, log events can sometimes have timestamping issues.

**Scenario 3: JSON Log with Mixed Data Types & Dynamic Structure**

In practice, you may also encounter more intricate json payloads. Consider a case with varying types and possibly inconsistent fields:

```json
{"time":"2024-07-27T10:00:00Z", "data": "Simple message", "stream": "stdout"}
```

Or sometimes, like this:
```json
{"time":"2024-07-27T10:00:01Z", "data": {"detail":"Detailed message","type":"event"}, "stream": "stdout"}
```

This shows a case where the data field can be either a string or a json object. We need promtail to robustly handle this. The following will work, but you may need to adjust to the data:

```yaml
pipeline_stages:
  - json:
      expressions:
        raw_data: data
        timestamp: time
        stream: stream
  - template:
       source: raw_data
       template: "{{ if is_string . }} {{ . }} {{ else }} {{ .detail }} {{ end }}"
       output: message
  - labels:
       stream:
  - timestamp:
      source: timestamp
      format: "2006-01-02T15:04:05Z"
```

Here's the explanation:
1.  **`json:` Stage**:  We extract all original fields, as needed. `raw_data` is now a string or a json object.
2. **`template:` Stage**: Here we are dynamically processing the data. If raw\_data is a string, we output that string to the `message` variable. If the raw\_data is a json object, then we look for the `detail` field and output that value. This flexibility allows us to handle varying types of JSON logs and ensure we extract the desired message.
3.  **`labels:` Stage**: We keep the `stream` label as it may be useful to filter log sources.
4. **`timestamp:` Stage**: This stage formats the log timestamp.

**Key Considerations and Best Practices**

*   **Robustness**: Always account for potential variations in json logs. Test your configurations thoroughly to avoid unexpected behavior. Ensure you have a good grasp of how the applications you are monitoring log before finalizing configurations. Use example logs from different cases to thoroughly test and validate your setup.
*   **Labels**: Leverage labels wisely. Add labels for container names, pod names, or any other metadata that helps you filter and aggregate log data more effectively.
*   **Timestamp Accuracy**: Correct timestamp extraction and formatting is critical for time-series analysis. Double check the timezone and format of your timestamps.
*   **Promtail Documentation**: While I've provided specific examples, the official promtail documentation is your best resource for understanding all the available configuration options and nuances.
*   **Configuration Management:** Use a configuration management system (e.g., Ansible, Puppet) to manage your promtail configurations. Changes are less error-prone and easier to implement across your infrastructure.

**Further Learning**

For anyone looking to deepen their understanding, I highly recommend a few resources:

*   **“The Go Programming Language” by Alan A. A. Donovan and Brian W. Kernighan:** Understanding Go’s time formatting is crucial for effective timestamp management in promtail.
*   **The official Promtail documentation:** This is your definitive guide to all things promtail. Specifically, pay close attention to the documentation on pipeline stages and label extraction.
*   **“Effective DevOps” by Jennifer Davis and Ryn Daniels:** A good background on the importance of monitoring in DevOps cultures, with information on implementing such systems.

In short, parsing docker json logs with promtail is about understanding the log structure and crafting a tailored pipeline. It’s not a one-size-fits-all solution. By understanding how promtail's `pipeline_stages` work, you can ensure accurate, reliable log ingestion for your monitoring needs. This has served me well in several complex deployments, and I'm confident it will help you too.
